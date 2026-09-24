use ir::{
    hir::{HirExpr, HirMatchArm, HirStmt, HirType, StrId},
    ir_hasher::{FxHashMap, HashMap},
};

use crate::hir_lowerer::monomorphization::{
    Monomorphizer, assertions::contains_unresolved_generic, instantiate_struct_for_types,
    substitute_type,
};

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub fn monomorphize_stmt<'subs>(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        substitutions: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirStmt<'a, 'bump> {
        match stmt {
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                let new_init = init.map(|i| {
                    let s = self.monomorphize_stmt(i, substitutions);
                    self.bump.alloc_value_immutable(s)
                });
                let new_cond = condition.map(|c| {
                    let e = self.monomorphize_expr(c, substitutions);
                    self.bump.alloc_value_immutable(e)
                });
                let new_incr = increment.map(|i| {
                    let e = self.monomorphize_expr(i, substitutions);
                    self.bump.alloc_value_immutable(e)
                });
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::For {
                    init: new_init,
                    condition: new_cond,
                    increment: new_incr,
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_then: Vec<HirStmt> = then_block
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                let then_slice = self.bump.alloc_slice(&new_then);
                let new_else = else_block.map(|e| {
                    let new_stmt = self.monomorphize_stmt(e, substitutions);
                    self.bump.alloc_value_immutable(new_stmt)
                });
                HirStmt::If {
                    cond: *self.bump.alloc_value_immutable(new_cond),
                    then_block: then_slice,
                    else_block: new_else,
                    span: *span,
                }
            }
            HirStmt::While { cond, body } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::While {
                    cond: self.bump.alloc_value_immutable(new_cond),
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Let {
                name,
                ty,
                value,
                mutable,
                is_static,
                catch_pattern,
                else_block,
                span,
            } => {
                let subd_ty = substitute_type(ty, substitutions, &self.bump);

                let new_value = self
                    .try_monomorphize_assoc_call(value, &subd_ty, substitutions)
                    .unwrap_or_else(|| {
                        self.monomorphize_expr_with_expected_type(value, &subd_ty, substitutions)
                    });

                let new_ty = self.instantiate_type_recursively(subd_ty, *span);
                self.ctx.variable_types.borrow_mut().insert(*name, new_ty);

                HirStmt::Let {
                    name: *name,
                    ty: new_ty,
                    value: *self.bump.alloc_value_immutable(new_value),
                    mutable: *mutable,
                    is_static: *is_static,
                    catch_pattern: *catch_pattern,
                    else_block: *else_block,
                    span: *span,
                }
            }
            HirStmt::Return(opt, span) => {
                let new_opt = opt.map(|e| {
                    let ret_ty = *self.current_return_type.borrow();
                    let new_expr = match ret_ty {
                        Some(rt) => {
                            self.monomorphize_expr_with_expected_type(e, &rt, substitutions)
                        }
                        None => self.monomorphize_expr(e, substitutions),
                    };
                    self.bump.alloc_value_immutable(new_expr)
                });
                HirStmt::Return(new_opt, *span)
            }
            HirStmt::Expr(e) => {
                let new_expr = self.monomorphize_expr(e, substitutions);
                HirStmt::Expr(self.bump.alloc_value_immutable(new_expr))
            }
            HirStmt::UnsafeBlock { body } => {
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::UnsafeBlock {
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Block { body, span } => {
                let new_body: Vec<HirStmt> = body
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                HirStmt::Block {
                    body: self.bump.alloc_slice(&new_body),
                    span: *span,
                }
            }
            HirStmt::Match { expr, arms, span } => {
                let new_expr = self.monomorphize_expr(expr, substitutions);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, substitutions))
                        }),
                        body: self
                            .bump
                            .alloc_value_immutable(self.monomorphize_stmt(arm.body, substitutions)),
                    })
                    .collect();
                HirStmt::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                    span: *span,
                }
            }
            HirStmt::Defer(stmt) => {
                let new_stmt = self.monomorphize_stmt(stmt, substitutions);
                HirStmt::Defer(self.bump.alloc_value_immutable(new_stmt))
            }

            _ => stmt.clone(),
        }
    }

    pub(crate) fn try_monomorphize_assoc_call<'subs>(
        &self,
        value: &HirExpr<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        outer_subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<HirExpr<'a, 'bump>> {
        let HirExpr::Call {
            callee,
            args,
            type_args: None,
            span,
        } = value
        else {
            return None;
        };
        let HirExpr::ModuleAccess(acc) = &**callee else {
            return None;
        };
        let HirType::Struct {
            type_args: expected_targs,
            ..
        } = expected_ty
        else {
            return None;
        };
        if expected_targs.is_empty() {
            return None;
        }

        let (&struct_name, module_path) = acc.path.split_last().unwrap_or_else(|| {
            panic!(
                "try_monomorphize_assoc_call: empty module path in static call `.{}` at {span}",
                self.context.resolve_string(&acc.member)
            )
        });

        let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
            struct_name
        } else {
            let resolved = self
                .ctx
                .resolve_type_path_name(module_path, struct_name, *span);
            if !self.ctx.structs.borrow().contains_key(&resolved) {
                panic!(
                    "[try_monomorphize_assoc_call] resolve_type_path_name resolved bare name `{}` \
                 to `{}` (using ctx.module_idx = {}) at {span}, but no struct is registered \
                 under that key at all.",
                    struct_name, resolved, self.ctx.module_idx,
                );
            }
            resolved
        };

        let struct_methods_binding = self.ctx.struct_methods.borrow();
        let method_map = struct_methods_binding.get(&target_key).unwrap_or_else(|| {
        panic!(
            "[try_monomorphize_assoc_call] no methods registered for struct key `{}` \
             (resolved from bare name `{}` in static call `.{}` at {span}). This almost \
             always means target_key resolution at the call site doesn't match the key \
             used when the `impl` block for this struct was registered. Known struct_methods keys: {:?}",
            target_key,
            struct_name,
            acc.member,
            struct_methods_binding
                .keys()
                .map(|k| self.context.resolve_string(k).to_string())
                .collect::<Vec<_>>(),
        )
    });
        let base_method_name = *method_map.get(&acc.member).unwrap_or_else(|| {
            panic!(
                "try_monomorphize_assoc_call: struct `{}` has no method `{}` (static call at \
             {span}). Known methods on this struct: {:?}",
                self.context.resolve_string(&target_key),
                self.context.resolve_string(&acc.member),
                method_map
                    .keys()
                    .map(|k| self.context.resolve_string(k).to_string())
                    .collect::<Vec<_>>(),
            )
        });
        drop(struct_methods_binding);

        let functions_binding = self.functions.borrow();
        let base_func = functions_binding
            .get(&base_method_name)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: method `{}` resolved to function `{}` but \
                 that name isn't registered in the function table",
                    acc.member, base_method_name,
                )
            })
            .clone();
        drop(functions_binding);

        let struct_generics = self
            .ctx
            .structs
            .borrow()
            .get(&target_key)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: struct key `{}` has a registered method `{}` \
                 but no struct declaration exists for it",
                    target_key, acc.member,
                )
            })
            .generics
            .clone();

        let type_params = base_func
            .generics
            .as_ref()
            .or(struct_generics.as_ref())
            .unwrap_or_else(|| {
                panic!(
                    "[try_monomorphize_assoc_call] call site `{}.{}(...)` at {span} expects {} \
                 concrete type argument(s) (target type: {:?}), but neither method `{}` nor \
                 struct `{}` declares any generic parameters",
                    struct_name,
                    acc.member,
                    expected_targs.len(),
                    expected_ty,
                    acc.member,
                    target_key,
                )
            });

        if type_params.len() != expected_targs.len() {
            panic!(
                "[try_monomorphize_assoc_call] `{}.{}` declares {} generic parameter(s) but the \
             call site at {span} supplies {} type argument(s) via its expected type {:?}",
                struct_name,
                acc.member,
                type_params.len(),
                expected_targs.len(),
                expected_ty,
            );
        }

        let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
        for (p, a) in type_params.iter().zip(expected_targs.iter()) {
            let resolved = substitute_type(a, outer_subs, &self.bump);
            if contains_unresolved_generic(&resolved) {
                panic!(
                    "try_monomorphize_assoc_call: type argument `{:?}` for parameter `{}` in \
                 `{}.{}` at {span} still contains an unresolved generic after substitution \
                 against outer scope {:?}",
                    resolved,
                    p.name,
                    struct_name,
                    acc.member,
                    outer_subs.keys().collect::<Vec<_>>(),
                );
            }
            inner_subs.insert(p.name, resolved);
        }

        let new_args: Vec<HirExpr> = args
            .iter()
            .map(|a| self.monomorphize_expr(a, outer_subs))
            .collect();
        let args_slice = self.bump.alloc_slice(&new_args);

        let concrete_recv_ty = instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            target_key,
            expected_targs,
            &self.bump,
        )
        .unwrap_or_else(|| {
            panic!(
                "[try_monomorphize_assoc_call] failed to instantiate `{}` with type args {:?} \
             for call `{}.{}` at {span}, instantiate_struct_for_types returned None even \
             though every arg was confirmed concrete above",
                target_key, expected_targs, struct_name, acc.member,
            )
        });
        let field_types: Vec<HirType> = concrete_recv_ty
            .fields
            .iter()
            .map(|f| f.field_type)
            .collect();
        let concrete_recv_ty = HirType::Struct {
            name: concrete_recv_ty.name,
            field_types: self.bump.alloc_slice(&field_types),
            type_args: &[],
        };

        let prev_self = self.current_this.replace(Some(concrete_recv_ty));
        let new_name = self
            .monomorphize_function(&base_func, &inner_subs)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: monomorphize_function returned None for `{}.{}` \
                 at {span}",
                    struct_name, acc.member,
                )
            });
        self.current_this.replace(prev_self);

        Some(HirExpr::Call {
            callee: self
                .bump
                .alloc_value_immutable(HirExpr::Ident(new_name, acc.span)),
            args: args_slice,
            type_args: None,
            span: *span,
        })
    }

    pub(crate) fn monomorphize_stmt_with_expected_type<'subs>(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        substitutions: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirStmt<'a, 'bump> {
        match stmt {
            HirStmt::Expr(e) => {
                let new_expr = self
                    .try_monomorphize_enum_init_with_expected_type(e, expected_ty, substitutions)
                    .unwrap_or_else(|| {
                        self.monomorphize_expr_with_expected_type(e, expected_ty, substitutions)
                    });
                HirStmt::Expr(self.bump.alloc_value_immutable(new_expr))
            }
            HirStmt::Block { body, span } => {
                let Some((last, rest)) = body.split_last() else {
                    return HirStmt::Block { body, span: *span };
                };
                let mut new_body: Vec<HirStmt> = rest
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                new_body.push(self.monomorphize_stmt_with_expected_type(
                    last,
                    expected_ty,
                    substitutions,
                ));
                HirStmt::Block {
                    body: self.bump.alloc_slice(&new_body),
                    span: *span,
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_then: Vec<HirStmt> = match then_block.split_last() {
                    Some((last, rest)) => {
                        let mut v: Vec<HirStmt> = rest
                            .iter()
                            .map(|s| self.monomorphize_stmt(s, substitutions))
                            .collect();
                        v.push(self.monomorphize_stmt_with_expected_type(
                            last,
                            expected_ty,
                            substitutions,
                        ));
                        v
                    }
                    None => Vec::new(),
                };
                let new_else = else_block.map(|e| {
                    let new_stmt =
                        self.monomorphize_stmt_with_expected_type(e, expected_ty, substitutions);
                    self.bump.alloc_value_immutable(new_stmt)
                });
                HirStmt::If {
                    cond: new_cond,
                    then_block: self.bump.alloc_slice(&new_then),
                    else_block: new_else,
                    span: *span,
                }
            }
            HirStmt::Match { expr, arms, span } => {
                let new_expr = self.monomorphize_expr(expr, substitutions);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, substitutions))
                        }),
                        body: self.bump.alloc_value_immutable(
                            self.monomorphize_stmt_with_expected_type(
                                arm.body,
                                expected_ty,
                                substitutions,
                            ),
                        ),
                    })
                    .collect();
                HirStmt::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                    span: *span,
                }
            }
            HirStmt::UnsafeBlock { body } => {
                let new_body =
                    self.monomorphize_stmt_with_expected_type(body, expected_ty, substitutions);
                HirStmt::UnsafeBlock {
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            other => self.monomorphize_stmt(other, substitutions),
        }
    }
}
