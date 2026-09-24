use ir::hir::HirType;
#[cfg(debug_assertions)]
use ir::hir::{Hir, HirExpr, HirStmt, StrId};

use crate::hir_lowerer::monomorphization::Monomorphizer;

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    #[cfg(debug_assertions)]
    pub(super) fn assert_closures_fully_concrete(&self, items: &[Hir<'a, 'bump>]) {
        for item in items {
            match item {
                Hir::Struct(s) if self.env_structs.contains_key(&s.name) => {
                    for f in s.fields.iter() {
                        if contains_unresolved_generic(&f.field_type) {
                            panic!(
                                "closure environment `{}` still has an unresolved generic in \
                                 field `{}`. Closures must be fully concrete by the time \
                                 monomorphization runs; hoisting should only ever run on capture \
                                 types the enclosing scope has already resolved.",
                                s.name, f.name,
                            );
                        }
                    }
                }
                Hir::Func(f) if self.env_structs.values().any(|fname| *fname == f.name) => {
                    if let Some(params) = f.params {
                        for p in params.iter() {
                            use ir::hir::HirParam;

                            if let HirParam::Normal {
                                name, param_type, ..
                            } = p
                            {
                                if contains_unresolved_generic(param_type) {
                                    panic!(
                                        "closure function `{}` still has an unresolved generic \
                                         in param `{}`.",
                                        f.name, name,
                                    );
                                }
                            }
                        }
                    }
                    if let Some(rt) = f.return_type {
                        if contains_unresolved_generic(&rt) {
                            panic!(
                                "closure function `{}` still has an unresolved generic in its \
                                 return type.",
                                f.name,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn assert_fully_monomorphized(&self, items: &[Hir<'a, 'bump>]) {
        for item in items {
            match item {
                Hir::Func(f) => {
                    if let Some(body) = f.body {
                        self.assert_stmt_monomorphized(&body, f.name);
                    }
                }
                Hir::Impl(i) => {
                    if let Some(methods) = i.methods {
                        for m in methods.iter() {
                            if let Some(body) = m.body {
                                self.assert_stmt_monomorphized(&body, m.name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[cfg(debug_assertions)]
    fn assert_stmt_monomorphized(&self, stmt: &HirStmt<'a, 'bump>, owner: StrId) {
        use ir::hir::HirStmt;

        match stmt {
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.assert_expr_monomorphized(cond, owner);
                for s in then_block.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
                if let Some(e) = else_block {
                    self.assert_stmt_monomorphized(e, owner);
                }
            }
            HirStmt::While { cond, body } => {
                self.assert_expr_monomorphized(cond, owner);
                self.assert_stmt_monomorphized(body, owner);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.assert_stmt_monomorphized(i, owner);
                }
                if let Some(c) = condition {
                    self.assert_expr_monomorphized(c, owner);
                }
                if let Some(i) = increment {
                    self.assert_expr_monomorphized(i, owner);
                }
                self.assert_stmt_monomorphized(body, owner);
            }
            HirStmt::Let { value, .. } => self.assert_expr_monomorphized(value, owner),
            HirStmt::Return(Some(e), _span) => self.assert_expr_monomorphized(e, owner),
            HirStmt::Expr(e) => self.assert_expr_monomorphized(e, owner),
            HirStmt::UnsafeBlock { body } => self.assert_stmt_monomorphized(body, owner),
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.assert_expr_monomorphized(expr, owner);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.assert_expr_monomorphized(g, owner);
                    }
                    self.assert_stmt_monomorphized(arm.body, owner);
                }
            }
            HirStmt::Defer(s) => self.assert_stmt_monomorphized(s, owner),
            _ => {}
        }
    }

    #[cfg(debug_assertions)]
    fn assert_expr_monomorphized(&self, expr: &HirExpr<'a, 'bump>, owner: StrId) {
        use ir::hir::HirExpr;

        match expr {
            HirExpr::Call {
                callee,
                type_args,
                span,
                args,
            } => {
                if let HirExpr::ModuleAccess(acc) = &**callee {
                    let (&struct_name, module_path) =
                        acc.path.split_last().unwrap_or((&acc.member, &[]));
                    let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
                        struct_name
                    } else {
                        self.ctx
                            .resolve_type_path_name(module_path, struct_name, *span)
                    };
                    let still_generic = self
                        .ctx
                        .structs
                        .borrow()
                        .get(&target_key)
                        .map(|s| s.generics.is_some())
                        .unwrap_or(false);
                    if still_generic && type_args.is_none() {
                        panic!(
                            "[assert_fully_monomorphized] function `{}` still contains an \
                                 unresolved generic static call `.{}` on struct `{}` at {span}.",
                            owner, acc.member, target_key,
                        );
                    }
                }
                self.assert_expr_monomorphized(callee, owner);
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }

            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => {
                let still_generic = self
                    .ctx
                    .enums
                    .borrow()
                    .get(enum_name)
                    .map(|e| e.generics.is_some())
                    .unwrap_or(false);
                if still_generic && type_args.is_none() {
                    panic!(
                        "[assert_fully_monomorphized] function `{}` still contains an unresolved \
                         generic enum construction `{}.{}` at {span}.",
                        owner, enum_name, variant,
                    );
                }
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }

            HirExpr::Match { expr, arms, .. } => {
                self.assert_expr_monomorphized(expr, owner);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.assert_expr_monomorphized(g, owner);
                    }
                    self.assert_stmt_monomorphized(arm.body, owner);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
            }
            HirExpr::If { if_stmt, .. } => self.assert_stmt_monomorphized(if_stmt, owner),
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.assert_expr_monomorphized(left, owner);
                self.assert_expr_monomorphized(right, owner);
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.assert_expr_monomorphized(object, owner);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.assert_expr_monomorphized(target, owner);
                self.assert_expr_monomorphized(value, owner);
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.assert_expr_monomorphized(&f.value, owner);
                }
            }
            HirExpr::Cast { expr, .. }
            | HirExpr::Deref { expr, .. }
            | HirExpr::Ref { expr, .. } => self.assert_expr_monomorphized(expr, owner),
            HirExpr::Index { object, index, .. } => {
                self.assert_expr_monomorphized(object, owner);
                self.assert_expr_monomorphized(index, owner);
            }
            HirExpr::ArrayLiteral { elements, .. } => {
                for e in elements.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }
            HirExpr::Tuple(exprs, _) => {
                for e in exprs.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.assert_expr_monomorphized(object, owner);
                self.assert_expr_monomorphized(start, owner);
                self.assert_expr_monomorphized(end, owner);
            }
            HirExpr::Range { start, end, .. } => {
                self.assert_expr_monomorphized(start, owner);
                self.assert_expr_monomorphized(end, owner);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    use ir::hir::InterpolationPart;

                    if let InterpolationPart::Expr(e) = p {
                        self.assert_expr_monomorphized(e, owner);
                    }
                }
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }
            HirExpr::InterfaceCall { callee, args, .. } => {
                self.assert_expr_monomorphized(callee, owner);
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }

            _ => {}
        }
    }
}

pub fn contains_unresolved_generic(ty: &HirType) -> bool {
    match ty {
        HirType::Generic(_) => true,
        HirType::Slice(inner) | HirType::Array(inner, _) => contains_unresolved_generic(inner),
        HirType::Nullable(inner) => contains_unresolved_generic(inner),
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::UnsafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. } => contains_unresolved_generic(inner),
        HirType::Struct {
            field_types,
            type_args,
            ..
        } => {
            field_types.iter().any(contains_unresolved_generic)
                || type_args.iter().any(contains_unresolved_generic)
        }
        HirType::DynInterface(_, args)
        | HirType::Enum {
            type_args: args, ..
        } => args.iter().any(contains_unresolved_generic),
        HirType::Lambda {
            params,
            return_type,
        } => {
            params.iter().any(contains_unresolved_generic)
                || contains_unresolved_generic(return_type)
        }
        HirType::Tuple(elems) => elems.iter().any(contains_unresolved_generic),
        HirType::Dyn { bounds } => bounds.iter().any(contains_unresolved_generic),
        _ => false,
    }
}

pub(super) fn peel_to_struct_owned<'a, 'bump>(ty: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
    match ty {
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. }
        | HirType::UnsafePointer { inner, .. } => peel_to_struct_owned(*inner),
        _ => ty,
    }
}

pub(super) fn peel_to_struct<'b, 'a, 'bump>(ty: &'b HirType<'a, 'bump>) -> &'b HirType<'a, 'bump> {
    match ty {
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. }
        | HirType::UnsafePointer { inner, .. } => peel_to_struct(inner),
        _ => ty,
    }
}
