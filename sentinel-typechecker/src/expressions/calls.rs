use ir::{
    ast::FuncSafety,
    borrow_checker::{BorrowKind, LoanId, ReadTemplate, RefTemplate},
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{HirExpr, HirFunc, HirParam, HirType, RefKind, StrId, ThisPassingKind},
    ir_hasher::FxHashMap,
    span::SourceSpan,
};

use crate::{
    naming::{str_id_to_string, type_to_string},
    type_checker::SLICE_PRIMITIVES,
    TypeChecker,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn check_module_access_expr(
        &mut self,
        access: &&ir::hir::HirModuleAccess<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let member_name = access.member.to_string();

        let is_named_import = if access.path.len() == 1 {
            self.imports_by_module
                .get(&self.context.current_module_idx)
                .map(|imp| imp.named.contains_key(&access.path[0]))
                .unwrap_or(false)
        } else {
            false
        };

        let alias_module_idx: Option<usize> = if access.path.len() == 1 {
            self.imports_by_module
                .get(&self.context.current_module_idx)
                .and_then(|imp| {
                    imp.named
                        .get(&access.path[0])
                        .or_else(|| imp.module_aliases.get(&access.path[0]))
                })
                .copied()
        } else {
            None
        };

        let (resolved_module_idx, assoc_type_name): (Option<usize>, Option<StrId>) =
            if let Some(midx) = alias_module_idx {
                if is_named_import {
                    (Some(midx), Some(access.path[0]))
                } else {
                    (Some(midx), None)
                }
            } else {
                match self
                    .context
                    .dep_graph
                    .borrow()
                    .resolve_module_path(access.path)
                {
                    Some(midx) => (Some(midx), None),
                    None => match access.path.split_last() {
                        Some((&type_seg, module_path)) => {
                            let midx = self
                                .context
                                .dep_graph
                                .borrow()
                                .resolve_module_path(module_path);
                            (midx, midx.map(|_| type_seg))
                        }
                        None => (None, None),
                    },
                }
            };

        let free_func = resolved_module_idx
            .and_then(|midx| self.context.get_module_function(midx, &member_name));

        let mangled_type_name: Option<String> = assoc_type_name.and_then(|t| {
            let midx = resolved_module_idx?;
            let pkg = self.context.dep_graph.borrow().get_module_package(midx)?;
            Some(format!("{}_{}", pkg.to_string(), t.to_string()))
        });

        let method_func = if free_func.is_none() {
            mangled_type_name
                .or_else(|| access.path.last().map(|s| s.to_string()))
                .and_then(|tn| self.context.get_method(&tn, &member_name).copied())
        } else {
            None
        };

        let func = match free_func.or(method_func) {
            Some(f) => f,
            None => {
                let path_str = access
                    .path
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let qualified_name = format!("{}.{}", path_str, member_name);
                let candidate_modules = self
                    .context
                    .dep_graph
                    .borrow()
                    .find_function_by_name_anywhere(access.member);
                if candidate_modules.is_empty() {
                    self.record(TypeErrorKind::UndefinedFunction(qualified_name));
                } else {
                    let suggestion_paths: Vec<String> = candidate_modules
                        .iter()
                        .filter_map(|&midx| {
                            self.context.dep_graph.borrow().get_module_package(midx)
                        })
                        .map(|pkg| pkg.to_string())
                        .collect();
                    self.record(TypeErrorKind::UndefinedFunctionWithSuggestion {
                        name: qualified_name,
                        suggested_modules: suggestion_paths,
                    });
                }

                return HirType::Unknown;
            }
        };

        self.check_unsafe_call(&func, &member_name);

        self.check_module_path_imported(access.path);

        let param_types: Vec<HirType<'a, 'bump>> = func
            .params
            .unwrap_or(&[])
            .iter()
            .filter_map(|p| p.get_type().copied())
            .collect();
        HirType::Lambda {
            params: self.context.bump.alloc_slice(&param_types),
            return_type: self
                .context
                .bump
                .alloc_value(func.return_type.unwrap_or(HirType::Void)),
        }
    }

    pub fn check_this_expr(&mut self, span: &SourceSpan<'a>) -> HirType<'a, 'bump> {
        match self.context.get_variable("this") {
            Some((symbol_id, ty)) => {
                self.occurrences.push((
                    *span,
                    self.this_id,
                    ty,
                    self.context.current_module_idx,
                    symbol_id,
                    false,
                ));
                ty
            }
            None => {
                self.record(TypeErrorKind::Generic(
                    "`this` used outside of a method that has a `this` receiver".to_string(),
                ));
                HirType::Unknown
            }
        }
    }

    pub fn check_all_func_args(
        &mut self,
        args: &[HirExpr<'a, 'bump>],
        params: &[HirParam<'a, 'bump>],
        templated_base_param: Option<usize>,
        callee: Option<HirFunc<'a, 'bump>>,
    ) -> Vec<LoanId> {
        let mut arg_loans: Vec<LoanId> = Vec::new();
        let write_template: Option<Vec<bool>> = callee.map(|f| self.analyze_definite_writes(&f));
        let read_template: Option<Vec<ReadTemplate>> =
            callee.map(|f| self.analyze_read_templates(&f));

        for (i, (arg, param)) in args.iter().zip(params.iter()).enumerate() {
            let param_type = param.get_type();

            if let (
                HirParam::Normal {
                    multi_place: Some(accesses),
                    ..
                },
                HirExpr::Ref {
                    expr,
                    ref_kind,
                    span,
                },
            ) = (param, arg)
            {
                let arg_type = self.check_ref_expr(expr, *ref_kind, *span, false);
                if let Some(pt) = param_type {
                    self.recover(self.types_compatible(pt, &arg_type), ());
                }
                if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                    self.optimistically_mark_mut_target_init(expr);
                }
                arg_loans.extend(self.register_multi_place_loans(expr, accesses));
                continue;
            }

            if Some(i) == templated_base_param {
                if let HirExpr::Ref {
                    expr,
                    ref_kind,
                    span,
                } = arg
                {
                    let ignores_init = *ref_kind != RefKind::Shared
                        && Self::callee_ignores_initial_contents(
                            write_template.as_deref(),
                            read_template.as_deref(),
                            i,
                        );
                    let arg_type = self.check_ref_expr_deferring_init(
                        expr,
                        *ref_kind,
                        *span,
                        false,
                        ignores_init,
                    );
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(pt, &arg_type), ());
                    }
                    if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                        self.optimistically_mark_mut_target_init(expr);
                    }
                } else {
                    let arg_type = match param_type {
                        Some(_) if matches!(arg, HirExpr::Lambda { .. }) && callee.is_some() => {
                            self.check_arg_expr(arg, param_type, callee.as_ref())
                        }
                        Some(pt) => self.check_expr_expected(arg, pt),
                        None => self.check_expr(arg),
                    };
                    self.check_and_record_value_use(arg, &arg_type);
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(pt, &arg_type), ());
                    }
                }
                continue;
            }

            if let (
                HirParam::Normal {
                    multi_place: None, ..
                },
                HirExpr::Ref {
                    expr,
                    ref_kind: rk @ (RefKind::Unique | RefKind::Alias),
                    span,
                },
            ) = (param, arg)
            {
                let ignores_init = Self::callee_ignores_initial_contents(
                    write_template.as_deref(),
                    read_template.as_deref(),
                    i,
                );
                let arg_type =
                    self.check_ref_expr_deferring_init(expr, *rk, *span, true, ignores_init);
                if let Some(pt) = param_type {
                    self.recover(self.types_compatible(pt, &arg_type), ());
                }
                if ignores_init {
                    self.optimistically_mark_mut_target_init(expr);
                }
                if let Some(place) = self.resolve_place(expr) {
                    if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                        arg_loans.push(loan_id);
                    }
                }
                continue;
            }

            let arg_type = self.check_arg_expr(arg, param_type, callee.as_ref());
            self.check_and_record_value_use(arg, &arg_type);
            if let Some(pt) = param_type {
                self.recover(self.types_compatible(pt, &arg_type), ());
            }

            if let HirExpr::Ref { expr, .. } = arg {
                if let Some(place) = self.resolve_place(expr) {
                    if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                        arg_loans.push(loan_id);
                    }
                }
            }
        }

        arg_loans
    }

    pub fn callee_ignores_initial_contents(
        write_template: Option<&[bool]>,
        read_template: Option<&[ReadTemplate]>,
        i: usize,
    ) -> bool {
        let definitely_written = write_template
            .and_then(|t| t.get(i))
            .copied()
            .unwrap_or(false);
        let never_reads_contents = read_template
            .and_then(|t| t.get(i))
            .is_some_and(|rt| !Self::read_template_touches_contents(rt));
        definitely_written || never_reads_contents
    }

    pub fn check_unsafe_call(&mut self, func: &HirFunc<'a, 'bump>, display_name: &str) {
        if matches!(func.function_metadata.func_safety, FuncSafety::Unsafe) && !self.in_unsafe() {
            self.record(TypeErrorKind::Generic(format!(
                "call to unsafe function `{}` requires an `unsafe` block",
                display_name
            )));
        }
    }

    pub fn check_interface_call_expr(
        &mut self,
        callee: &&HirExpr<'a, 'bump>,
        interface: &StrId,
        args: &&[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        let (object, field) = match callee {
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                (object, field)
            }
            other => {
                self.record(TypeErrorKind::Generic(format!(
                    "interface call callee must be a field access, found {:?}",
                    other
                )));
                return HirType::Unknown;
            }
        };
        let _ = self.check_expr(object);

        let iface_name = interface.to_string();
        let Some(iface) = self.context.get_interface(&iface_name) else {
            self.record(TypeErrorKind::UndefinedType(iface_name));
            return HirType::Unknown;
        };

        let method_name = field.to_string();
        let Some(method) = iface.methods.and_then(|methods| {
            methods
                .iter()
                .find(|m| m.unmangled_name.to_string() == method_name)
        }) else {
            self.record(TypeErrorKind::Generic(format!(
                "no method `{}` on interface `{}`",
                method_name, iface_name
            )));
            return HirType::Unknown;
        };

        let total_params = method.params.map(|p| p.len()).unwrap_or(0);
        let expected_args = total_params.saturating_sub(1);
        // exclude `this`
        if args.len() != expected_args {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args,
                found_args: args.len(),
            });
        }

        if let Some(params) = method.params {
            for (arg, param) in args.iter().zip(params.iter().skip(1)) {
                let arg_type = match param.get_type() {
                    Some(pt) => self.check_expr_expected(arg, pt),
                    None => self.check_expr(arg),
                };
                self.check_and_record_value_use(arg, &arg_type);
                if let Some(pt) = param.get_type() {
                    self.recover(self.types_compatible(pt, &arg_type), ());
                }
            }
        }

        method.return_type.unwrap_or(HirType::Void)
    }

    pub fn check_call_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        callee: &&HirExpr<'a, 'bump>,
        args: &&[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
        type_args: &Option<&[HirType<'a, 'bump>]>,
    ) -> HirType<'a, 'bump> {
        match &callee {
            HirExpr::Ident(func_name, ident_span) => {
                self.set_span(*ident_span);
                let lookup_name = str_id_to_string(*func_name);

                if self.context.is_local_binding(&lookup_name) {
                    let callee_type = self.check_expr(callee);
                    return match callee_type {
                        HirType::Lambda { return_type, .. } => *return_type,
                        HirType::Generic(g) if self.fn_closure_constraints.contains_key(&g) => {
                            self.check_closure_call(g, args)
                        }
                        _ => {
                            self.record(TypeErrorKind::Generic(format!(
                                "Expression of type `{}` is not callable",
                                type_to_string(&callee_type)
                            )));
                            HirType::Unknown
                        }
                    };
                }

                let func = match self.context.get_function(&lookup_name) {
                    Some(f) => f,
                    None => {
                        self.record(TypeErrorKind::UndefinedFunction(lookup_name));
                        return HirType::Unknown;
                    }
                };
                self.check_unsafe_call(&func, &lookup_name);

                self.record_item_occurrence(
                    *ident_span,
                    *func_name,
                    func.return_type.unwrap_or(HirType::Void),
                    func.declaring_module_idx,
                );

                self.recover(
                    self.check_visibility(
                        func.function_metadata.visibility,
                        func.declaring_module_idx,
                        "function",
                        &lookup_name,
                    ),
                    (),
                );
                self.check_name_import_visibility(*func_name, &lookup_name);

                let mut substitutions: FxHashMap<StrId, HirType<'a, 'bump>> = match func.generics {
                    Some(tp) if !tp.is_empty() => match type_args {
                        Some(ta) => {
                            let mut full_args = ta.to_vec();
                            if full_args.len() < tp.len() {
                                let mut temp_map = FxHashMap::default();
                                for (&p, &a) in tp.iter().zip(full_args.iter()) {
                                    temp_map.insert(p.name, a);
                                }
                                let start = full_args.len();
                                for param in &tp[start..] {
                                    if let Some(ref def_ty) = param.default_type {
                                        let resolved =
                                            self.substitute_type_local(def_ty, &temp_map);
                                        temp_map.insert(param.name, resolved);
                                        full_args.push(resolved);
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if full_args.len() != tp.len() {
                                self.record(TypeErrorKind::Generic(format!(
                                    "function `{}` expects {} type argument(s), found {}",
                                    lookup_name,
                                    tp.len(),
                                    ta.len()
                                )));
                            }
                            let mut map = FxHashMap::default();
                            tp.iter()
                                .zip(full_args.iter())
                                .map(|(&p, &a)| (p, a))
                                .for_each(|(p, a)| {
                                    map.insert(p.name, a);
                                });
                            map
                        }
                        None => {
                            // Apply defaults where they exist; everything else is left for
                            // inference and verified after the arguments have been checked.
                            let mut temp_map = FxHashMap::default();
                            for param in tp.iter() {
                                if let Some(ref def_ty) = param.default_type {
                                    let resolved = self.substitute_type_local(def_ty, &temp_map);
                                    temp_map.insert(param.name, resolved);
                                }
                            }
                            temp_map
                        }
                    },
                    _ => {
                        if type_args.is_some() {
                            self.record(TypeErrorKind::Generic(format!(
                                "function `{}` is not generic; no type arguments expected",
                                lookup_name
                            )));
                        }
                        FxHashMap::default()
                    }
                };

                if let Some(params) = func.params {
                    substitutions.extend(self.pre_infer_generics(Some(&func), args, params));
                }
                self.closure_pre_subs = substitutions.clone();
                self.closure_generic_subs.clear();

                let expected_args = func.params.map(|p| p.len()).unwrap_or(0);
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }

                let Some(params) = func.params else {
                    let ret_ty = func.return_type.unwrap_or(HirType::Void);
                    if type_args.is_none() {
                        self.check_generics_inferred(
                            &func,
                            &substitutions,
                            &lookup_name,
                            *ident_span,
                        );
                    }
                    return if substitutions.is_empty() {
                        ret_ty
                    } else {
                        self.substitute_type_local(&ret_ty, &substitutions)
                    };
                };

                let params = if substitutions.is_empty() {
                    params
                } else {
                    self.substitute_params_local(params, &substitutions)
                };

                let unsubstituted_ret_ty = func.return_type.unwrap_or(HirType::Void);
                let ret_ty = if substitutions.is_empty() {
                    unsubstituted_ret_ty
                } else {
                    self.substitute_type_local(&unsubstituted_ret_ty, &substitutions)
                };

                if let Some(value) =
                    self.check_potential_this_param_for_move(args, func, params, ret_ty)
                {
                    return value;
                }

                let read_templates = self.analyze_read_templates(&func);
                for (arg_idx, arg) in args.iter().enumerate() {
                    if let HirExpr::Ref {
                        expr: inner,
                        ref_kind: RefKind::Shared,
                        ..
                    } = arg
                    {
                        if let Some(template) = read_templates.get(arg_idx) {
                            self.check_call_arg_read_effects(inner, template, args);
                        }
                    }
                }

                let arg_loans = self.check_all_func_args(args, params, None, Some(func));

                substitutions.extend(std::mem::take(&mut self.closure_generic_subs));
                if type_args.is_none() {
                    self.check_generics_inferred(&func, &substitutions, &lookup_name, *ident_span);
                }
                let ret_ty = if substitutions.is_empty() {
                    ret_ty
                } else {
                    self.substitute_type_local(&ret_ty, &substitutions)
                };

                if !self.return_type_may_alias(&ret_ty) {
                    for loan in arg_loans {
                        self.borrow_checker.end_loan_now(loan);
                    }
                }
                ret_ty
            }

            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                let obj_type = self.check_expr(object);
                let stripped = Self::strip_ref(&obj_type);

                if let Some(elem) = match *stripped {
                    HirType::Slice(e) | HirType::Array(e, _) => Some(*e),
                    _ => None,
                } {
                    let name = field.as_str();
                    if SLICE_PRIMITIVES.contains(&name) {
                        return self.check_slice_primitive_call(object, elem, name, args);
                    }
                }

                let interface_name = match stripped {
                    HirType::DynInterface(name, _) => Some(name.to_string()),
                    HirType::Dyn { bounds } => bounds.iter().find_map(|b| match b {
                        HirType::DynInterface(name, _) => Some(name.to_string()),
                        HirType::Struct { name, .. } => {
                            let name_str = name.to_string();
                            self.context.get_interface(&name_str).map(|_| name_str)
                        }
                        _ => None,
                    }),
                    _ => None,
                };

                if let Some(iface_name) = interface_name {
                    let method_name = field.to_string();
                    let iface = match self.context.get_interface(&iface_name) {
                        Some(i) => i,
                        None => {
                            self.record(TypeErrorKind::UndefinedType(iface_name));
                            return HirType::Unknown;
                        }
                    };

                    let method = iface.methods.and_then(|methods| {
                        methods
                            .iter()
                            .find(|m| m.unmangled_name.to_string() == method_name)
                    });
                    let Some(method) = method else {
                        self.record(TypeErrorKind::Generic(format!(
                            "no method `{}` on interface `{}`",
                            method_name, iface_name
                        )));
                        return HirType::Unknown;
                    };

                    let total_params = method.params.map(|p| p.len()).unwrap_or(0);
                    let expected_args = total_params.saturating_sub(1);
                    if args.len() != expected_args {
                        self.record(TypeErrorKind::InvalidFunctionCall {
                            expected_args,
                            found_args: args.len(),
                        });
                    }

                    if let Some(params) = method.params {
                        if let Some(HirParam::This {
                            kind, multi_place, ..
                        }) = params.first()
                        {
                            let requires_mut = match kind {
                                ThisPassingKind::RefMut
                                | ThisPassingKind::MutSafePtr
                                | ThisPassingKind::MoveMut => true,
                                ThisPassingKind::MultiPlace => multi_place.is_some_and(|a| {
                                    a.iter().any(|x| x.ref_kind == RefKind::Unique)
                                }),
                                _ => false,
                            };
                            if requires_mut {
                                self.recover(
                                    self.check_receiver_is_mutable(object, field.as_str()),
                                    (),
                                );
                            }

                            if let (ThisPassingKind::MultiPlace, Some(accesses)) =
                                (kind, multi_place)
                            {
                                let loans = self.register_multi_place_loans(object, accesses);
                                for loan in loans {
                                    self.borrow_checker.end_loan_now(loan);
                                }
                            } else if matches!(
                                kind,
                                ThisPassingKind::Move | ThisPassingKind::MoveMut
                            ) {
                                self.check_and_record_value_use(object, &obj_type);
                            } else if let Some(place) = self.resolve_place(object) {
                                let borrow_kind = if requires_mut {
                                    BorrowKind::Mutable
                                } else {
                                    BorrowKind::Shared
                                };
                                self.check_borrow_use(object, place, borrow_kind);
                            }
                        }
                        for (arg, param) in args.iter().zip(params.iter().skip(1)) {
                            let arg_type = self.check_expr(arg);
                            self.check_and_record_value_use(arg, &arg_type);
                            if let Some(param_type) = param.get_type() {
                                let result = self.types_compatible(param_type, &arg_type);
                                self.recover(result, ());
                            }
                        }
                    }

                    return method.return_type.unwrap_or(HirType::Void);
                }

                let (struct_name_id, type_name, func) =
                    match self.resolve_callable_method(stripped, &field.to_string()) {
                        Some(found) => found,
                        None => {
                            self.record(TypeErrorKind::Generic(format!(
                                "no method `{}` on `{}`",
                                field,
                                type_to_string(&obj_type)
                            )));
                            return HirType::Unknown;
                        }
                    };
                self.check_unsafe_call(&func, &format!("{}.{}", type_name, field));

                let total_params = func.params.map(|p| p.len()).unwrap_or(0);
                let expected_args = total_params.saturating_sub(1);
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }

                let struct_type_args: &[HirType<'a, 'bump>] = match stripped {
                    HirType::Struct { type_args, .. } => type_args,
                    _ => &[],
                };
                let method_subs =
                    self.generic_substitutions_for_struct(struct_name_id, struct_type_args);

                let unsubstituted_ret_ty = func.return_type.unwrap_or(HirType::Void);
                let ret_ty = if method_subs.is_empty() {
                    unsubstituted_ret_ty
                } else {
                    self.substitute_type_local(&unsubstituted_ret_ty, &method_subs)
                };
                self.record_method_occurrence(*span, *field, ret_ty, struct_name_id);

                let template = if self.return_type_may_alias(&ret_ty) {
                    Some(self.analyze_ref_template(&func))
                } else {
                    None
                };

                if let Some(params) = func.params {
                    let mut receiver_multi_place_loans: Vec<LoanId> = Vec::new();

                    if let Some(HirParam::This {
                        kind,
                        multi_place,
                        span: _,
                    }) = params.first()
                    {
                        let requires_mut = match kind {
                            ThisPassingKind::RefMut
                            | ThisPassingKind::MutSafePtr
                            | ThisPassingKind::MoveMut => true,
                            ThisPassingKind::MultiPlace => multi_place
                                .is_some_and(|a| a.iter().any(|x| x.ref_kind == RefKind::Unique)),
                            _ => false,
                        };
                        if requires_mut {
                            self.recover(
                                self.check_receiver_is_mutable(object, field.as_str()),
                                (),
                            );
                        }

                        if let (ThisPassingKind::MultiPlace, Some(accesses)) = (kind, multi_place) {
                            receiver_multi_place_loans =
                                self.register_multi_place_loans(object, accesses);
                        } else {
                            let has_precise_template =
                                matches!(template, Some(RefTemplate::Path { .. }));

                            if matches!(kind, ThisPassingKind::Move | ThisPassingKind::MoveMut) {
                                self.check_and_record_value_use(object, &obj_type);
                            } else if !has_precise_template {
                                if let Some(place) = self.resolve_place(object) {
                                    let borrow_kind = if requires_mut {
                                        BorrowKind::Mutable
                                    } else {
                                        BorrowKind::Shared
                                    };

                                    if !requires_mut && !self.return_type_may_alias(&ret_ty) {
                                        self.check_borrow_use_shell(expr, place, borrow_kind);
                                    } else {
                                        self.check_borrow_use(expr, place, borrow_kind);
                                    }
                                }
                            }
                        }

                        // defer entirely to finalize_call_loans below, which checks
                        // the precise resolved place (such as ptr[Const(1)] vs
                        // ptr[Const(2)]) and can prove index-disjointness that the
                        // whole-receiver check can't.
                    }

                    let method_params: &[HirParam<'a, 'bump>] = if method_subs.is_empty() {
                        params
                    } else {
                        self.substitute_params_local(params, &method_subs)
                    };
                    let normal_params: &[HirParam<'a, 'bump>] =
                        method_params.get(1..).unwrap_or(&[]);

                    let read_templates = self.analyze_read_templates(&func);
                    for (arg_idx, arg) in args.iter().enumerate() {
                        if let HirExpr::Ref {
                            expr: inner,
                            ref_kind: RefKind::Shared,
                            ..
                        } = arg
                        {
                            if let Some(rt) = read_templates.get(arg_idx) {
                                self.check_call_arg_read_effects(inner, rt, args);
                            }
                        }
                    }

                    let arg_loans = self.check_all_func_args(args, normal_params, None, Some(func));

                    if let Some(loan_id) =
                        self.finalize_call_loans(Some(object), args, arg_loans, &ret_ty, template)
                    {
                        self.call_loans.insert(Self::expr_key(expr), loan_id);
                        for loan in receiver_multi_place_loans {
                            self.borrow_checker.end_loan_now(loan);
                        }
                    }
                }

                ret_ty
            }
            HirExpr::ModuleAccess(access) => {
                self.set_span(access.span);
                let member_name = access.member.to_string();

                let is_named_import = if access.path.len() == 1 {
                    self.imports_by_module
                        .get(&self.context.current_module_idx)
                        .map(|imp| imp.named.contains_key(&access.path[0]))
                        .unwrap_or(false)
                } else {
                    false
                };

                let alias_module_idx: Option<usize> = if access.path.len() == 1 {
                    self.imports_by_module
                        .get(&self.context.current_module_idx)
                        .and_then(|imp| {
                            imp.named
                                .get(&access.path[0])
                                .or_else(|| imp.module_aliases.get(&access.path[0]))
                        })
                        .copied()
                } else {
                    None
                };

                let (resolved_module_idx, assoc_type_name): (Option<usize>, Option<StrId>) =
                    if let Some(midx) = alias_module_idx {
                        if is_named_import {
                            (Some(midx), Some(access.path[0]))
                        } else {
                            (Some(midx), None)
                        }
                    } else {
                        match self
                            .context
                            .dep_graph
                            .borrow()
                            .resolve_module_path(access.path)
                        {
                            Some(midx) => (Some(midx), None),
                            None => match access.path.split_last() {
                                Some((&type_seg, module_path)) => {
                                    let midx = self
                                        .context
                                        .dep_graph
                                        .borrow()
                                        .resolve_module_path(module_path);
                                    (midx, midx.map(|_| type_seg))
                                }
                                None => (None, None),
                            },
                        }
                    };

                let free_func = resolved_module_idx
                    .and_then(|midx| self.context.get_module_function(midx, &member_name));

                let mangled_type_name: Option<String> = assoc_type_name.and_then(|t| {
                    let midx = resolved_module_idx?;
                    Some(
                        self.context
                            .dep_graph
                            .borrow()
                            .mangle_type_name(midx, t, &self.context.string_pool)
                            .to_string(),
                    )
                });

                let method_func = if free_func.is_none() {
                    mangled_type_name
                        .or_else(|| access.path.last().map(|s| s.to_string()))
                        .and_then(|tn| self.context.get_method(&tn, &member_name).copied())
                } else {
                    None
                };

                let func = match free_func.or(method_func) {
                    Some(f) => f,
                    None => {
                        let path_str = access
                            .path
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        let qualified_name = format!("{}::{}", path_str, member_name);
                        let candidate_modules = self
                            .context
                            .dep_graph
                            .borrow()
                            .find_function_by_name_anywhere(access.member);
                        if candidate_modules.is_empty() {
                            self.record(TypeErrorKind::UndefinedFunction(qualified_name));
                        } else {
                            let suggestion_paths: Vec<String> = candidate_modules
                                .iter()
                                .filter_map(|&midx| {
                                    self.context.dep_graph.borrow().get_module_package(midx)
                                })
                                .map(|pkg| pkg.to_string())
                                .collect();
                            self.record(TypeErrorKind::UndefinedFunctionWithSuggestion {
                                name: qualified_name,
                                suggested_modules: suggestion_paths,
                            });
                        }

                        return HirType::Unknown;
                    }
                };

                self.check_module_path_imported(access.path);

                let expected_args = func.params.map(|p| p.len()).unwrap_or(0);
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }
                if let Some(params) = func.params {
                    let read_templates = self.analyze_read_templates(&func);
                    for (arg_idx, arg) in args.iter().enumerate() {
                        if let HirExpr::Ref {
                            expr: inner,
                            ref_kind: RefKind::Shared,
                            ..
                        } = arg
                        {
                            if let Some(template) = read_templates.get(arg_idx) {
                                self.check_call_arg_read_effects(inner, template, args);
                            }
                        }
                    }

                    let arg_loans = self.check_all_func_args(args, params, None, Some(func));

                    let ret_ty = func.return_type.unwrap_or(HirType::Void);
                    if !self.return_type_may_alias(&ret_ty) {
                        for loan in arg_loans {
                            self.borrow_checker.end_loan_now(loan);
                        }
                    }
                }
                let ret_ty = func.return_type.unwrap_or(HirType::Void);
                if let Some(midx) = resolved_module_idx {
                    if free_func.is_some() {
                        self.record_item_occurrence(access.span, access.member, ret_ty, midx);
                    } else if let Some(target_type) = assoc_type_name {
                        self.record_method_occurrence(
                            access.span,
                            access.member,
                            ret_ty,
                            target_type,
                        );
                    }
                }
                ret_ty
            }

            other => {
                self.set_span(*span);
                let callee_type = self.check_expr(other);
                match callee_type {
                    HirType::Lambda { return_type, .. } => *return_type,
                    _ => {
                        self.record(TypeErrorKind::Generic(format!(
                            "Expression of type `{}` is not callable",
                            type_to_string(&callee_type)
                        )));
                        HirType::Unknown
                    }
                }
            }
        }
    }

    pub fn check_slice_primitive_call(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        elem: HirType<'a, 'bump>,
        method: &str,
        args: &[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        let ret = if method == "get_unchecked" {
            elem
        } else {
            HirType::Void
        };

        if !self.in_unsafe() {
            self.record(TypeErrorKind::Generic(format!(
                "`{}` requires an unsafe block: it performs no bounds check",
                method
            )));
        }

        let (expected_args, needs_mut) = match method {
            "write_uninit" => (2, true),
            "write_uninit_all" => (1, true),
            _ => (1, false),
        };
        if args.len() != expected_args {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args,
                found_args: args.len(),
            });
            for a in args {
                self.check_expr(a);
            }
            return ret;
        }

        if needs_mut {
            self.recover(self.check_receiver_is_mutable(object, method), ());
        }
        if let Some(place) = self.resolve_place(object) {
            let kind = if needs_mut {
                BorrowKind::Mutable
            } else {
                BorrowKind::Shared
            };
            self.check_borrow_use(object, place, kind);
        }

        let idx_or_src_ty = |this: &mut Self, e: &HirExpr<'a, 'bump>| {
            let t = this.check_expr_expected(e, &HirType::Usize);
            if !this.is_integer(&t) {
                this.record(TypeErrorKind::Generic(format!(
                    "index must be an integer, found `{}`",
                    type_to_string(&t)
                )));
            }
        };

        match method {
            "write_uninit" => {
                idx_or_src_ty(self, &args[0]);
                let val_ty = self.check_expr_expected(&args[1], &elem);
                self.check_and_record_value_use(&args[1], &val_ty);
                self.recover(self.types_compatible(&elem, &val_ty), ());
            }
            "get_unchecked" => idx_or_src_ty(self, &args[0]),
            "write_uninit_all" => {
                let src_ty = self.check_expr(&args[0]);
                match *Self::strip_ref(&src_ty) {
                    HirType::Slice(e) | HirType::Array(e, _) => {
                        self.recover(self.types_compatible(&elem, e), ());
                    }
                    _ => self.record(TypeErrorKind::Generic(format!(
                        "`write_uninit_all` expects a slice or array source, found `{}`",
                        type_to_string(&src_ty)
                    ))),
                }
                if let Some(p) = self.resolve_place(&args[0]) {
                    self.check_borrow_use(&args[0], p, BorrowKind::Shared);
                }
            }
            _ => unreachable!(),
        }
        ret
    }

    pub fn check_name_import_visibility(&mut self, name: StrId, name_str: &str) {
        let current = self.context.current_module_idx;

        if self
            .functions_by_module
            .get(&current)
            .is_some_and(|s| s.contains(&name))
        {
            return;
        }

        if let Some(imports) = self.imports_by_module.get(&current) {
            if let Some(&target_module) = imports.named.get(&name) {
                if self
                    .functions_by_module
                    .get(&target_module)
                    .is_some_and(|s| s.contains(&name))
                {
                    return;
                }
            }

            let candidates: Vec<usize> = imports
                .wildcard
                .iter()
                .copied()
                .filter(|m| {
                    self.functions_by_module
                        .get(m)
                        .is_some_and(|s| s.contains(&name))
                })
                .collect();

            if candidates.len() == 1 {
                return;
            }
            if candidates.len() > 1 {
                let candidate_pkgs: Vec<String> = candidates
                    .iter()
                    .filter_map(|&m| self.context.dep_graph.borrow().get_module_package(m))
                    .map(|p| p.to_string())
                    .collect();
                self.record(TypeErrorKind::Generic(format!(
                    "`{}` is ambiguous: it is auto-imported from multiple packages ({}); \
                     add an explicit `import` to disambiguate",
                    name_str,
                    candidate_pkgs.join(", "),
                )));
                return;
            }
        }

        self.record(TypeErrorKind::Generic(format!(
            "`{}` is not declared in this module and has not been imported",
            name_str,
        )));
    }

    pub fn check_module_path_imported(&mut self, path_segments: &[StrId]) {
        let current = self.context.current_module_idx;
        let Some(target) = self
            .context
            .dep_graph
            .borrow()
            .resolve_module_path(path_segments)
        else {
            return;
        };
        if target == current {
            return;
        }

        let imported = self
            .imports_by_module
            .get(&current)
            .is_some_and(|imp| imp.modules.contains(&target));
        if !imported {
            let path_str = path_segments
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            self.record(TypeErrorKind::Generic(format!(
                "module `{}` used without an `import {};` declaration",
                path_str, path_str,
            )));
        }
    }

    pub fn resolve_callable_method(
        &self,
        ty: &HirType<'a, 'bump>,
        method_name: &str,
    ) -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
        let try_name = |n: String| -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
            let id = StrId(self.context.string_pool.intern(&n));
            self.context
                .get_method(&n, method_name)
                .map(|f| (id, n, *f))
        };

        match ty {
            HirType::Struct { name, .. } => {
                let struct_name_str = name.to_string();
                if let Some(hit) = try_name(struct_name_str.clone()) {
                    return Some(hit);
                }
                self.resolve_default_interface_method(*name, &struct_name_str, method_name)
            }
            HirType::Slice(elem) | HirType::Array(elem, _) => {
                if let Some(elem_name) = self.builtin_element_name(elem) {
                    if let Some(hit) = try_name(format!("slice_{}", elem_name)) {
                        return Some(hit);
                    }
                }
                try_name("slice".to_string())
            }
            other => try_name(self.builtin_element_name(other)?),
        }
    }

    pub fn resolve_default_interface_method(
        &self,
        struct_name: StrId,
        struct_name_str: &str,
        method_name: &str,
    ) -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
        let interfaces = self.context.struct_interfaces.get(struct_name_str)?;
        for iface_name in interfaces {
            let Some(iface) = self.context.get_interface(iface_name) else {
                continue;
            };
            let Some(methods) = iface.methods else {
                continue;
            };
            if let Some(m) = methods
                .iter()
                .find(|m| m.unmangled_name.as_str() == method_name && m.body.is_some())
            {
                return Some((struct_name, struct_name_str.to_string(), *m));
            }
        }
        None
    }

    pub fn builtin_element_name(&self, ty: &HirType<'a, 'bump>) -> Option<String> {
        Some(match ty {
            HirType::I8 => "i8".into(),
            HirType::I16 => "i16".into(),
            HirType::I32 => "i32".into(),
            HirType::I64 => "i64".into(),
            HirType::I128 => "i128".into(),
            HirType::U8 => "u8".into(),
            HirType::U16 => "u16".into(),
            HirType::U32 => "u32".into(),
            HirType::U64 => "u64".into(),
            HirType::U128 => "u128".into(),
            HirType::Usize => "usize".into(),
            HirType::Isize => "isize".into(),
            HirType::F32 => "f32".into(),
            HirType::F64 => "f64".into(),
            HirType::Boolean => "bool".into(),
            HirType::String => "str".into(),
            HirType::Char => "char".into(),
            HirType::Struct { name, .. } => name.to_string(),
            _ => return None,
        })
    }

    pub fn check_receiver_is_mutable(
        &self,
        receiver: &HirExpr<'a, 'bump>,
        method_name: &str,
    ) -> TypeCheckResult<'a, ()> {
        let Some(root_name) = self.find_root_local_ident(receiver) else {
            return Ok(());
        };

        if !self.context.is_mutable(&root_name) {
            return Err(TypeErrorKind::Generic(format!(
                "cannot call `{}` on `{}`: `{}` is not declared `mut`",
                method_name, root_name, root_name
            ))
            .at(self.current_span));
        }

        Ok(())
    }

    pub fn check_generics_inferred(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
        display_name: &str,
        span: SourceSpan<'a>,
    ) {
        let Some(generics) = func.generics else {
            return;
        };
        let unresolved = generics.iter().any(|g| {
            !subs.contains_key(&g.name)
                && !g
                    .constraints
                    .iter()
                    .any(|c| matches!(c, HirType::Lambda { .. }))
        });
        if unresolved {
            self.set_span(span);
            self.record(TypeErrorKind::Generic(format!(
                "generic function `{}` requires explicit type arguments, e.g. `{}<Type>(...)`",
                display_name, display_name
            )));
        }
    }
}
