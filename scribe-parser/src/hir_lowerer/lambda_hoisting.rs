use std::sync::Arc;

use ir::ast::{ExternModifier, FuncSafety, InlineModifier, Visibility};
use ir::hir::{
    CaptureMode, ClosureLowering, EffectIndexKey, FuncModifiers, Hir, HirEffectSegment, HirExpr,
    HirField, HirFieldInit, HirFunc, HirLambdaParam, HirModule, HirParam, HirStmt, HirStruct,
    HirType, RefKind, StrId, static_effect_path,
};
use ir::ir_hasher::FxHashMap;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub struct LambdaHoister<'a, 'bump> {
    bump: &'bump GrowableBump<'bump>,
    context: Arc<StringPool>,
    counter: usize,
    module_name: StrId,
    hoisted: Vec<Hir<'a, 'bump>>,
    closures: FxHashMap<usize, ClosureLowering<'a, 'bump>>,
    capture_frames: Vec<Vec<(StrId, Vec<HirEffectSegment<'bump>>, HirExpr<'a, 'bump>)>>,
    env_to_fn: FxHashMap<StrId, StrId>,
}

impl<'a, 'bump> LambdaHoister<'a, 'bump> {
    pub fn new(
        bump: &'bump GrowableBump<'bump>,
        context: Arc<StringPool>,
        module_name: StrId,
        closures: FxHashMap<usize, ClosureLowering<'a, 'bump>>,
    ) -> Self {
        Self {
            bump,
            context,
            counter: 0,
            module_name,
            hoisted: Vec::new(),
            closures,
            capture_frames: Vec::new(),
            env_to_fn: FxHashMap::default(),
        }
    }

    fn this_id(&self) -> StrId {
        StrId::from_static("this")
    }

    pub fn run(
        mut self,
        module: HirModule<'a, 'bump>,
    ) -> (HirModule<'a, 'bump>, FxHashMap<StrId, StrId>) {
        let mut new_items: Vec<Hir<'a, 'bump>> = Vec::with_capacity(module.items.len());

        for item in module.items {
            let rewritten = self.rewrite_item(*item);
            new_items.push(rewritten);
        }

        new_items.extend(self.hoisted.drain(..));

        let out = HirModule {
            name: module.name,
            imports: module.imports,
            items: self.bump.alloc_slice(&new_items),
        };
        let env_to_fn = std::mem::take(&mut self.env_to_fn);
        (out, env_to_fn)
    }

    fn rewrite_item(&mut self, item: Hir<'a, 'bump>) -> Hir<'a, 'bump> {
        match item {
            Hir::Func(f) => {
                let rewritten = self.rewrite_func(*f);
                Hir::Func(self.bump.alloc_value(rewritten))
            }
            Hir::Impl(i) => {
                if let Some(methods) = i.methods {
                    let rewritten_methods: Vec<HirFunc<'a, 'bump>> =
                        methods.iter().map(|m| self.rewrite_func(*m)).collect();
                    let methods_slice = self.bump.alloc_slice(&rewritten_methods);
                    let mut new_impl = *i;
                    new_impl.methods = Some(methods_slice);
                    Hir::Impl(self.bump.alloc_value(new_impl))
                } else {
                    item
                }
            }
            other => other,
        }
    }

    fn rewrite_func(&mut self, func: HirFunc<'a, 'bump>) -> HirFunc<'a, 'bump> {
        let Some(body) = func.body else {
            return func;
        };
        let new_body = self.rewrite_stmt(body);
        HirFunc {
            body: Some(new_body),
            ..func
        }
    }

    fn try_substitute(
        &self,
        root: StrId,
        path: &[HirEffectSegment<'bump>],
    ) -> Option<HirExpr<'a, 'bump>> {
        let frame = self.capture_frames.last()?;
        frame
            .iter()
            .find(|(r, p, _)| {
                *r == root
                    && p.len() == path.len()
                    && p.iter()
                        .zip(path.iter())
                        .all(|(a, b)| ir::hir::effect_segment_eq(a, b))
            })
            .map(|(_, _, repl)| *repl)
    }

    fn rewrite_stmt(&mut self, stmt: HirStmt<'a, 'bump>) -> HirStmt<'a, 'bump> {
        match stmt {
            HirStmt::Let {
                name,
                ty,
                value,
                is_static,
                mutable,
                catch_pattern,
                else_block,
                span,
            } => {
                let new_value = self.rewrite_expr(value);
                let new_else = else_block.map(|e| {
                    let r = self.rewrite_stmt(*e);
                    &*self.bump.alloc_value_immutable(r)
                });
                HirStmt::Let {
                    name,
                    ty,
                    value: new_value,
                    mutable,
                    is_static,
                    catch_pattern,
                    else_block: new_else,
                    span,
                }
            }
            HirStmt::Return(Some(expr), span) => {
                let new_expr = self.rewrite_expr(*expr);
                HirStmt::Return(Some(self.bump.alloc_value_immutable(new_expr)), span)
            }
            HirStmt::Expr(expr) => {
                let new_expr = self.rewrite_expr(*expr);
                HirStmt::Expr(self.bump.alloc_value_immutable(new_expr))
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                let new_cond = self.rewrite_expr(cond);
                let new_then: Vec<HirStmt<'a, 'bump>> =
                    then_block.iter().map(|s| self.rewrite_stmt(*s)).collect();
                let new_then_slice = self.bump.alloc_slice(&new_then);
                let new_else = else_block.map(|e| {
                    let rewritten = self.rewrite_stmt(*e);
                    self.bump.alloc_value_immutable(rewritten)
                });
                HirStmt::If {
                    cond: new_cond,
                    then_block: new_then_slice,
                    else_block: new_else,
                    span,
                }
            }
            HirStmt::While { cond, body } => {
                let new_cond = self.rewrite_expr(*cond);
                let new_body = self.rewrite_stmt(*body);
                HirStmt::While {
                    cond: self.bump.alloc_value_immutable(new_cond),
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                let new_init = init.map(|i| {
                    let r = self.rewrite_stmt(*i);
                    self.bump.alloc_value_immutable(r)
                });
                let new_cond = condition.map(|c| {
                    let r = self.rewrite_expr(*c);
                    self.bump.alloc_value_immutable(r)
                });
                let new_inc = increment.map(|i| {
                    let r = self.rewrite_expr(*i);
                    self.bump.alloc_value_immutable(r)
                });
                let new_body = self.rewrite_stmt(*body);
                HirStmt::For {
                    init: new_init,
                    condition: new_cond,
                    increment: new_inc,
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Match { expr, arms, span } => {
                let new_expr = self.rewrite_expr(*expr);
                let new_arms: Vec<ir::hir::HirMatchArm<'a, 'bump>> = arms
                    .iter()
                    .map(|arm| {
                        let new_guard = arm.guard.map(|g| {
                            let r = self.rewrite_expr(*g);
                            self.bump.alloc_value_immutable(r)
                        });
                        let new_body = self.rewrite_stmt(*arm.body);
                        ir::hir::HirMatchArm {
                            pattern: arm.pattern,
                            guard: new_guard,
                            body: self.bump.alloc_value_immutable(new_body),
                        }
                    })
                    .collect();
                HirStmt::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                    span,
                }
            }
            HirStmt::UnsafeBlock { body } => {
                let new_body = self.rewrite_stmt(*body);
                HirStmt::UnsafeBlock {
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Block { body, span } => {
                let new_body: Vec<HirStmt<'a, 'bump>> =
                    body.iter().map(|s| self.rewrite_stmt(*s)).collect();
                HirStmt::Block {
                    body: self.bump.alloc_slice(&new_body),
                    span,
                }
            }
            HirStmt::Defer(inner) => {
                let new_inner = self.rewrite_stmt(*inner);
                HirStmt::Defer(self.bump.alloc_value_immutable(new_inner))
            }
            HirStmt::Break(Some(expr), span) => {
                let new_expr = self.rewrite_expr(*expr);
                HirStmt::Break(Some(self.bump.alloc_value_immutable(new_expr)), span)
            }
            other => other,
        }
    }

    fn rewrite_expr(&mut self, expr: HirExpr<'a, 'bump>) -> HirExpr<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, span) => self
                .try_substitute(name, &[])
                .unwrap_or(HirExpr::Ident(name, span)),
            HirExpr::This { span } => self
                .try_substitute(self.this_id(), &[])
                .unwrap_or(HirExpr::This { span }),
            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                let node = HirExpr::FieldAccess {
                    object,
                    field,
                    span,
                };
                if let Some((root, path)) = static_effect_path(&node, self.this_id(), &self.bump) {
                    if let Some(sub) = self.try_substitute(root, &path) {
                        return sub;
                    }
                }
                let new_obj = self.rewrite_expr(*object);
                HirExpr::FieldAccess {
                    object: self.bump.alloc_value(new_obj),
                    field,
                    span,
                }
            }
            HirExpr::Get {
                object,
                field,
                span,
            } => {
                let node = HirExpr::Get {
                    object,
                    field,
                    span,
                };
                if let Some((root, path)) = static_effect_path(&node, self.this_id(), &self.bump) {
                    if let Some(sub) = self.try_substitute(root, &path) {
                        return sub;
                    }
                }
                let new_obj = self.rewrite_expr(*object);
                HirExpr::Get {
                    object: self.bump.alloc_value(new_obj),
                    field,
                    span,
                }
            }
            HirExpr::Index {
                object,
                index,
                span,
            } => {
                let node = HirExpr::Index {
                    object,
                    index,
                    span,
                };
                if let Some((root, path)) = static_effect_path(&node, self.this_id(), &self.bump) {
                    if let Some(sub) = self.try_substitute(root, &path) {
                        return sub;
                    }
                }
                let new_object = self.rewrite_expr(*object);
                let new_index = self.rewrite_expr(*index);
                HirExpr::Index {
                    object: self.bump.alloc_value(new_object),
                    index: self.bump.alloc_value(new_index),
                    span,
                }
            }

            HirExpr::Tuple(exprs, span) => {
                let new_exprs: Vec<HirExpr<'a, 'bump>> =
                    exprs.iter().map(|e| self.rewrite_expr(*e)).collect();
                HirExpr::Tuple(self.bump.alloc_slice(&new_exprs), span)
            }
            HirExpr::ArrayLiteral { elements, span } => {
                let new_elements: Vec<HirExpr<'a, 'bump>> =
                    elements.iter().map(|e| self.rewrite_expr(*e)).collect();
                HirExpr::ArrayLiteral {
                    elements: self.bump.alloc_slice(&new_elements),
                    span,
                }
            }
            HirExpr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let new_start = self.rewrite_expr(*start);
                let new_end = self.rewrite_expr(*end);
                HirExpr::Range {
                    start: self.bump.alloc_value(new_start),
                    end: self.bump.alloc_value(new_end),
                    inclusive,
                    span,
                }
            }
            HirExpr::Slice {
                object,
                start,
                end,
                inclusive,
                span,
            } => {
                let new_object = self.rewrite_expr(*object);
                let new_start = self.rewrite_expr(*start);
                let new_end = self.rewrite_expr(*end);
                HirExpr::Slice {
                    object: self.bump.alloc_value(new_object),
                    start: self.bump.alloc_value(new_start),
                    end: self.bump.alloc_value(new_end),
                    inclusive,
                    span,
                }
            }
            HirExpr::Cast {
                expr,
                target_type,
                span,
            } => {
                let new_expr = self.rewrite_expr(*expr);
                HirExpr::Cast {
                    expr: self.bump.alloc_value(new_expr),
                    target_type,
                    span,
                }
            }
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => {
                let new_args: Vec<HirExpr<'a, 'bump>> =
                    args.iter().map(|a| self.rewrite_expr(*a)).collect();
                HirExpr::Intrinsic {
                    kind,
                    type_args,
                    args: self.bump.alloc_slice(&new_args),
                    span,
                }
            }
            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => {
                let new_args: Vec<HirExpr<'a, 'bump>> =
                    args.iter().map(|a| self.rewrite_expr(*a)).collect();
                HirExpr::EnumInit {
                    enum_name,
                    variant,
                    args: self.bump.alloc_slice(&new_args),
                    type_args,
                    span,
                }
            }
            HirExpr::InterpolatedString(parts) => {
                let new_parts: Vec<ir::hir::InterpolationPart<'a, 'bump>> = parts
                    .iter()
                    .map(|p| match p {
                        ir::hir::InterpolationPart::Expr(e) => {
                            let new_e = self.rewrite_expr(**e);
                            ir::hir::InterpolationPart::Expr(self.bump.alloc_value_immutable(new_e))
                        }
                        other => *other,
                    })
                    .collect();
                HirExpr::InterpolatedString(self.bump.alloc_slice(&new_parts))
            }
            HirExpr::If { if_stmt, span } => {
                let new_if_stmt = self.rewrite_stmt(*if_stmt);
                HirExpr::If {
                    if_stmt: self.bump.alloc_value_immutable(new_if_stmt),
                    span,
                }
            }
            HirExpr::Match { expr, arms, span } => {
                let new_expr = self.rewrite_expr(*expr);
                let new_arms: Vec<ir::hir::HirMatchArm<'a, 'bump>> = arms
                    .iter()
                    .map(|arm| {
                        let new_guard = arm.guard.map(|g| {
                            let r = self.rewrite_expr(*g);
                            self.bump.alloc_value_immutable(r)
                        });
                        let new_body = self.rewrite_stmt(*arm.body);
                        ir::hir::HirMatchArm {
                            pattern: arm.pattern,
                            guard: new_guard,
                            body: self.bump.alloc_value_immutable(new_body),
                        }
                    })
                    .collect();
                HirExpr::Match {
                    expr: self.bump.alloc_value(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                    span,
                }
            }
            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                let new_body: Vec<HirStmt<'a, 'bump>> =
                    body.iter().map(|s| self.rewrite_stmt(*s)).collect();
                HirExpr::Block {
                    body: self.bump.alloc_slice(&new_body),
                    is_unsafe,
                    span,
                }
            }

            HirExpr::Lambda {
                modifier: _,
                params,
                return_type,
                body,
                span,
            } => self.hoist_lambda(params, *return_type, body, span),

            HirExpr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let l = self.rewrite_expr(*left);
                let r = self.rewrite_expr(*right);
                HirExpr::Binary {
                    left: self.bump.alloc_value(l),
                    op,
                    right: self.bump.alloc_value(r),
                    span,
                }
            }
            HirExpr::Comparison {
                left,
                op,
                right,
                span,
            } => {
                let l = self.rewrite_expr(*left);
                let r = self.rewrite_expr(*right);
                HirExpr::Comparison {
                    left: self.bump.alloc_value(l),
                    op,
                    right: self.bump.alloc_value(r),
                    span,
                }
            }
            HirExpr::Call {
                callee,
                args,
                span,
                type_args,
            } => {
                let new_callee = self.rewrite_expr(*callee);
                let new_args: Vec<HirExpr<'a, 'bump>> =
                    args.iter().map(|a| self.rewrite_expr(*a)).collect();
                HirExpr::Call {
                    callee: self.bump.alloc_value(new_callee),
                    args: self.bump.alloc_slice(&new_args),
                    span,
                    type_args,
                }
            }
            HirExpr::InterfaceCall {
                callee,
                args,
                interface,
                span,
            } => {
                let new_callee = self.rewrite_expr(*callee);
                let new_args: Vec<HirExpr<'a, 'bump>> =
                    args.iter().map(|a| self.rewrite_expr(*a)).collect();
                HirExpr::InterfaceCall {
                    callee: self.bump.alloc_value(new_callee),
                    args: self.bump.alloc_slice(&new_args),
                    interface,
                    span,
                }
            }
            HirExpr::Assignment {
                target,
                op,
                value,
                span,
            } => {
                let new_target = self.rewrite_expr(*target);
                let new_value = self.rewrite_expr(*value);
                HirExpr::Assignment {
                    target: self.bump.alloc_value(new_target),
                    op,
                    value: self.bump.alloc_value(new_value),
                    span,
                }
            }
            HirExpr::StructInit {
                name,
                args,
                span,
                type_args,
            } => {
                let new_name = self.rewrite_expr(*name);
                let new_args: Vec<HirFieldInit<'a, 'bump>> = args
                    .iter()
                    .map(|a| HirFieldInit {
                        name: a.name,
                        name_span: a.name_span,
                        value: self.rewrite_expr(a.value),
                    })
                    .collect();
                HirExpr::StructInit {
                    name: self.bump.alloc_value(new_name),
                    args: self.bump.alloc_slice(&new_args),
                    span,
                    type_args,
                }
            }
            HirExpr::ExprList { list, span } => {
                let new_list: Vec<HirExpr<'a, 'bump>> =
                    list.iter().map(|e| self.rewrite_expr(*e)).collect();
                HirExpr::ExprList {
                    list: self.bump.alloc_slice(&new_list),
                    span,
                }
            }
            HirExpr::Ref {
                expr,
                ref_kind: mutable,
                span,
            } => {
                let new_inner = self.rewrite_expr(*expr);
                HirExpr::Ref {
                    expr: self.bump.alloc_value(new_inner),
                    ref_kind: mutable,
                    span,
                }
            }
            HirExpr::Deref { expr, span } => {
                let new_inner = self.rewrite_expr(*expr);
                HirExpr::Deref {
                    expr: self.bump.alloc_value(new_inner),
                    span,
                }
            }
            other => other,
        }
    }

    fn expr_for_index_key(
        &self,
        key: &EffectIndexKey<'bump>,
        span: ir::span::SourceSpan<'a>,
    ) -> HirExpr<'a, 'bump> {
        match key {
            EffectIndexKey::Const(n) => HirExpr::Number(*n, span),
            EffectIndexKey::Place { root, path } => {
                let mut e = if *root == self.this_id() {
                    HirExpr::This { span }
                } else {
                    HirExpr::Ident(*root, span)
                };
                for f in path.iter() {
                    e = HirExpr::FieldAccess {
                        object: self.bump.alloc_value(e),
                        field: *f,
                        span,
                    };
                }
                e
            }
            EffectIndexKey::Dynamic => {
                unreachable!("a finalized closure capture never contains a Dynamic index")
            }
        }
    }

    fn expr_from_capture_path(
        &self,
        root: StrId,
        path: &[HirEffectSegment<'bump>],
        span: ir::span::SourceSpan<'a>,
    ) -> HirExpr<'a, 'bump> {
        let mut e = if root == self.this_id() {
            HirExpr::This { span }
        } else {
            HirExpr::Ident(root, span)
        };
        for seg in path {
            e = match seg {
                HirEffectSegment::Field(f) => HirExpr::FieldAccess {
                    object: self.bump.alloc_value(e),
                    field: *f,
                    span,
                },
                HirEffectSegment::Index(key) => {
                    let idx = self.expr_for_index_key(key, span);
                    HirExpr::Index {
                        object: self.bump.alloc_value(e),
                        index: self.bump.alloc_value(idx),
                        span,
                    }
                }
            };
        }
        e
    }

    fn hoist_lambda(
        &mut self,
        params: &'bump [HirLambdaParam<'a, 'bump>],
        return_type: HirType<'a, 'bump>,
        body: &'bump HirStmt<'a, 'bump>,
        span: ir::span::SourceSpan<'a>,
    ) -> HirExpr<'a, 'bump> {
        let key = body as *const HirStmt<'a, 'bump> as usize;

        let Some(closure) = self.closures.get(&key) else {
            return self.hoist_plain_lambda(params, return_type, body, span);
        };
        let closure = closure.clone();

        let env_ident = StrId::from_static("__env");
        let mut frame = Vec::with_capacity(closure.captures.len());
        for cap in &closure.captures {
            let field_access = HirExpr::FieldAccess {
                object: self.bump.alloc_value(HirExpr::Ident(env_ident, span)),
                field: cap.name,
                span,
            };
            let access = match cap.mode {
                CaptureMode::ByValue => field_access,
                CaptureMode::ByRef(_) => HirExpr::Deref {
                    expr: self.bump.alloc_value(field_access),
                    span,
                },
            };
            frame.push((cap.source, cap.source_path.to_vec(), access));
        }

        self.capture_frames.push(frame);
        let rewritten_body = self.rewrite_stmt(*body);
        self.capture_frames.pop();

        let env_param_ty = match closure.kind {
            ir::hir::ClosureKind::FnOnce => closure.env_ty,
            _ => HirType::Ref {
                inner: self.bump.alloc_value(closure.env_ty),
                ref_kind: RefKind::Shared,
                provenance: None,
            },
        };
        let mut hir_params: Vec<HirParam<'a, 'bump>> = Vec::with_capacity(params.len() + 1);
        hir_params.push(HirParam::Normal {
            name: env_ident,
            param_type: env_param_ty,
            span,
            multi_place: None,
        });
        for (p, ty) in params.iter().zip(closure.param_tys.iter()) {
            hir_params.push(HirParam::Normal {
                name: p.name,
                param_type: *ty,
                span: p.span,
                multi_place: p.multi_place,
            });
        }
        let params_slice = self.bump.alloc_slice(&hir_params);

        let synthetic_func = HirFunc {
            name: closure.fn_name,
            function_metadata: FuncModifiers {
                visibility: Visibility::Private,
                extern_modifier: ExternModifier::None,
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Safe,
            },
            generics: None,
            params: Some(params_slice),
            return_type: Some(closure.ret_ty),
            body: Some(rewritten_body),
            unmangled_name: closure.fn_name,
            declaring_module_idx: 0,
            impl_target: None,
            span,
        };
        self.hoisted
            .push(Hir::Func(self.bump.alloc_value(synthetic_func)));

        let field_types: &[HirType<'a, 'bump>] = match closure.env_ty {
            HirType::Struct { field_types, .. } => field_types,
            _ => unreachable!("ClosureLowering::env_ty is always a Struct"),
        };
        let env_fields: Vec<HirField<'a, 'bump>> = closure
            .captures
            .iter()
            .zip(field_types.iter())
            .map(|(cap, ty)| HirField {
                name: cap.name,
                field_type: *ty,
                visibility: Visibility::Private,
            })
            .collect();
        let env_struct = HirStruct {
            name: closure.env_name,
            visibility: Visibility::Private,
            generics: None,
            fields: self.bump.alloc_slice(&env_fields),
        };
        self.hoisted
            .push(Hir::Struct(self.bump.alloc_value(env_struct)));

        self.env_to_fn.insert(closure.env_name, closure.fn_name);

        let field_inits: Vec<HirFieldInit<'a, 'bump>> = closure
            .captures
            .iter()
            .map(|cap| {
                let root_expr = self.expr_from_capture_path(cap.source, cap.source_path, span);
                let root_expr = self.rewrite_expr(root_expr); // picks up outer substitution if nested
                let value = match cap.mode {
                    CaptureMode::ByValue => root_expr,
                    CaptureMode::ByRef(rk) => HirExpr::Ref {
                        expr: self.bump.alloc_value(root_expr),
                        ref_kind: rk,
                        span,
                    },
                };
                HirFieldInit {
                    name: cap.name,
                    name_span: span,
                    value,
                }
            })
            .collect();

        HirExpr::StructInit {
            name: self
                .bump
                .alloc_value(HirExpr::Ident(closure.env_name, span)),
            args: self.bump.alloc_slice(&field_inits),
            span,
            type_args: None,
        }
    }

    fn hoist_plain_lambda(
        &mut self,
        params: &'bump [HirLambdaParam<'a, 'bump>],
        return_type: HirType<'a, 'bump>,
        body: &'bump HirStmt<'a, 'bump>,
        span: ir::span::SourceSpan<'a>,
    ) -> HirExpr<'a, 'bump> {
        let inner_rewritten_body = self.rewrite_stmt(*body);
        let synthetic_name = self.fresh_lambda_name();

        let hir_params: Vec<HirParam<'a, 'bump>> = params
            .iter()
            .map(|p: &HirLambdaParam<'a, 'bump>| HirParam::Normal {
                name: p.name,
                param_type: p.param_type.unwrap_or_else(|| {
                    panic!(
                        "lambda parameter {:?} has no resolved type at hoisting time \
                         hoisting must run after type inference",
                        p.name
                    )
                }),
                multi_place: p.multi_place,
                span: p.span,
            })
            .collect();
        let params_slice = self.bump.alloc_slice(&hir_params);

        let synthetic_func = HirFunc {
            name: synthetic_name,
            function_metadata: FuncModifiers {
                visibility: Visibility::Private,
                extern_modifier: ExternModifier::None,
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Safe,
            },
            generics: None,
            params: Some(params_slice),
            return_type: Some(return_type),
            body: Some(inner_rewritten_body),
            unmangled_name: synthetic_name,
            declaring_module_idx: 0,
            impl_target: None,
            span,
        };

        self.hoisted
            .push(Hir::Func(self.bump.alloc_value(synthetic_func)));

        HirExpr::Ident(synthetic_name, span)
    }

    fn fresh_lambda_name(&mut self) -> StrId {
        let module_str = self.context.resolve_string(&self.module_name);
        let name = format!("__lambda_{}_{}", module_str, self.counter);
        self.counter += 1;
        StrId(self.context.intern(&name))
    }
}
