use ir::{
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{HirExpr, HirType, RefKind},
    span::SourceSpan,
};

use crate::{closures, str_id_to_string, TypeChecker};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn is_reference_like(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::Ref { .. }
                | HirType::SafePointer { .. }
                | HirType::UnsafePointer { .. }
                | HirType::OwnedPointer { .. }
        )
    }

    pub fn expr_is_dangling(&self, expr: &HirExpr<'a, 'bump>) -> bool {
        match expr {
            HirExpr::Ref { expr: inner, .. } => match self.find_root_local_ident(inner) {
                Some(root_name) => match self.context.get_variable(&root_name) {
                    Some((_, root_type)) => !root_type.is_pointer_semantics(),
                    None => false,
                },
                None => false,
            },
            HirExpr::Ident(name, _) => {
                let var_name = str_id_to_string(*name);
                self.context.is_dangling(&var_name)
            }
            _ => false,
        }
    }

    pub fn check_no_dangling_pointer(&self, expr: &HirExpr<'a, 'bump>) -> TypeCheckResult<'a, ()> {
        if let HirExpr::Ref { expr: inner, .. } = expr {
            if let Some(root_name) = self.find_root_local_ident(inner) {
                if let Some((_, root_type)) = self.context.get_variable(&root_name) {
                    if !root_type.is_pointer_semantics() {
                        return Err(TypeErrorKind::Generic(format!(
                            "cannot return a pointer to local variable `{}`: its storage does not outlive this function",
                            root_name
                        )).at(self.current_span));
                    }
                }
            }
            return Ok(());
        }

        if let HirExpr::Ident(name, _) = expr {
            let var_name = str_id_to_string(*name);
            if self.context.is_dangling(&var_name) {
                return Err(TypeErrorKind::Generic(format!(
                    "cannot return `{}`: it holds a pointer to local stack memory that does not outlive this function",
                    var_name
                )).at(self.current_span));
            }
        }

        Ok(())
    }

    pub fn find_root_local_ident(&self, expr: &HirExpr<'a, 'bump>) -> Option<String> {
        match expr {
            HirExpr::Ident(name, _) => Some(str_id_to_string(*name)),
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.find_root_local_ident(object)
            }
            HirExpr::Deref { expr: inner, .. } => self.find_root_local_ident(inner),
            _ => None,
        }
    }

    pub fn check_ref_expr_deferring_init(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
        register_loan: bool,
        skip_init: bool,
    ) -> HirType<'a, 'bump> {
        let prev = self.skip_slice_init_check;
        // Only arm the flag for a top-level slice operand, so it can't be
        // picked up by an unrelated nested `&mut x[..]`.
        self.skip_slice_init_check = skip_init && matches!(expr, HirExpr::Slice { .. });
        let ty = self.check_ref_expr(expr, ref_kind, span, register_loan);
        self.skip_slice_init_check = prev;
        ty
    }

    pub fn check_ref_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
        register_loan: bool,
    ) -> HirType<'a, 'bump> {
        self.set_span(span);

        let inner_ty = if ref_kind != RefKind::Shared {
            self.check_expr_as_place(expr)
        } else {
            self.check_expr(expr)
        };

        let provenance = self.infer_provenance(expr);

        if register_loan {
            if let Some((root, path)) = self.effect_path_of(expr) {
                let lvl = match ref_kind {
                    RefKind::Unique => closures::UseLevel::Mut,
                    RefKind::Alias => closures::UseLevel::Alias,
                    RefKind::Shared => closures::UseLevel::Read,
                };
                self.note_capture_use(root, &path, lvl);
            }
            if let Some(place) = self.resolve_place(expr) {
                let result = match ref_kind {
                    RefKind::Unique => self.borrow_checker.borrow_mut(place),
                    RefKind::Alias => self.borrow_checker.borrow_alias(place),
                    RefKind::Shared => self.borrow_checker.borrow_shared(place),
                };
                if let Err(e) = result {
                    let msg = self.describe_borrow_error(&e, provenance.as_ref());
                    self.record(TypeErrorKind::Generic(msg));
                }
            }
        }

        HirType::Ref {
            inner: self.context.bump.alloc_value(inner_ty),
            ref_kind,
            provenance,
        }
    }
}
