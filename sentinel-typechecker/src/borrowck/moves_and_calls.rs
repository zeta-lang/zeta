use ir::{
    borrow_checker::{BorrowKind, LoanId, RefTemplate, TemplateBase},
    errors::type_error::TypeErrorKind,
    hir::{
        HirExpr, HirFunc, HirParam, HirType, HirTypeKind, ProvenanceAnnotation,
        ProvenancePathSegment, ProvenanceRoot, RefKind, StrId, ThisPassingKind,
    },
};

use crate::{TypeChecker, closures, dataflow::Effect, naming::type_to_string};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Records that `root` (or `root.field`) is moved out while skipping types implementing `Copy`.
    pub fn record_move(
        &mut self,
        root: StrId,
        field: Option<StrId>,
        field_ty: &HirType<'a, 'bump>,
        container_ty: Option<&HirType<'a, 'bump>>,
    ) {
        if self.copy_analysis.borrow().type_is_copy(field_ty) {
            return;
        }
        let local = self.local_table.register(root);
        let path: Vec<ir::hir::HirEffectSegment<'bump>> = field
            .map(|f| vec![ir::hir::HirEffectSegment::Field(f)])
            .unwrap_or_default();
        self.note_capture_use(root, &path, closures::UseLevel::Move);

        if let Some(&base_place) = self.borrow_checker.local_place(root) {
            let moved_place = match field {
                None => base_place,
                Some(f) => self.borrow_checker.project_field(base_place, f),
            };
            // Is the move legal?
            if let Err(e) = self.borrow_checker.check_move(moved_place) {
                let path = match field {
                    None => &[][..],
                    Some(f) => &[ProvenancePathSegment::Field(f)][..],
                };
                let provenance = ProvenanceAnnotation {
                    root: ProvenanceRoot::Var(root),
                    path: self.context.bump.alloc_slice(path),
                };
                let msg = self.describe_borrow_error(&e, Some(&provenance));
                self.record(TypeErrorKind::Generic(msg));
            }
        }

        // No partial moves for what implements Drop!
        match field {
            None => {
                self.log_effect(Effect::Move { local, field: None });
                self.move_state.mark_whole_moved(local);
            }
            Some(f) => {
                let blocks_partial_move = container_ty.is_some_and(|cty| match cty.kind {
                    HirTypeKind::Struct { name, .. } | HirTypeKind::Enum { name, .. } => {
                        self.copy_analysis.borrow().implements_drop(name)
                    }
                    _ => false,
                });
                if blocks_partial_move {
                    self.record(TypeErrorKind::Generic(format!(
                        "cannot partially move out of `{}`, which implements `Drop`",
                        container_ty
                            .map(|t| type_to_string(t, &self.context.string_pool))
                            .unwrap_or_default()
                    )));
                    return;
                }
                self.log_effect(Effect::Move {
                    local,
                    field: Some(f),
                });
                self.move_state.mark_field_moved(local, f);
            }
        }
    }

    /// Reports "use of moved value" if `root` (whole) or `root.field` was already moved.
    /// Whole-variable uses are blocked by *any* prior move; field uses only by a move of that field.
    pub fn check_use(&mut self, root: StrId, field: Option<StrId>, root_ty: &HirType<'a, 'bump>) {
        if self.copy_analysis.borrow().type_is_copy(root_ty) {
            return;
        }
        let Some(local_id) = self.local_table.get(root) else {
            return; // never moved, so nothing to report
        };
        match field {
            None => {
                if self.move_state.blocks_whole_use(local_id) {
                    self.record(TypeErrorKind::Generic(format!(
                        "use of moved value: `{}`",
                        root
                    )));
                }
            }
            Some(f) => {
                if self.move_state.is_field_moved(local_id, f) {
                    self.record(TypeErrorKind::Generic(format!(
                        "use of moved value: `{}.{}`",
                        root, f
                    )));
                }
            }
        }
    }

    /// Entry point for "this expression is read by value". Static places are ignored.
    pub fn check_and_record_value_use(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ty: &HirType<'a, 'bump>,
    ) {
        if self.is_static_place(expr) {
            return;
        }

        if let Some((root, path)) = self.effect_path_of(expr) {
            self.note_capture_use(root, &path, closures::UseLevel::Read);
        }
        if let Some(place) = self.resolve_place(expr) {
            self.check_borrow_use(expr, place, BorrowKind::Shared);
        }
        self.check_no_move_out_of_borrow(expr, ty);
        match expr {
            HirExpr::Ident(name, _) => {
                self.check_use(*name, None, ty);
                self.record_move(*name, None, ty, None);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::Ident(root, _) = &**object {
                    let container_ty = self.context.get_variable(root);
                    self.check_use(*root, Some(*field), ty);
                    self.record_move(*root, Some(*field), ty, container_ty.map(|f| f.1).as_ref());
                }
            }
            _ => {}
        }
    }

    /// A by-value read of a non-Copy value out of a place we don't own leaves two
    /// owners (the old place and the new binding), this prevents that from ever happening.
    fn check_no_move_out_of_borrow(&mut self, expr: &HirExpr<'a, 'bump>, ty: &HirType<'a, 'bump>) {
        if !matches!(
            expr,
            HirExpr::Deref { .. }
                | HirExpr::Index { .. }
                | HirExpr::FieldAccess { .. }
                | HirExpr::Get { .. }
        ) {
            return;
        }
        if self.copy_analysis.borrow().type_is_copy(ty) {
            return;
        }
        if self.place_is_behind_borrow(expr) {
            self.record(TypeErrorKind::Generic(format!(
                "cannot move a value of type `{}` out of a reference, pointer or slice; \
                 it is not `Copy`. Use `$replace` to take it explicitly",
                ty
            )));
        }
    }

    fn is_pointer_like(ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty.kind,
            HirTypeKind::Ref { .. }
                | HirTypeKind::SafePointer { .. }
                | HirTypeKind::UnsafePointer { .. }
        )
    }

    /// Check if `expr` names a place that lives behind a reference or pointer, directly or
    /// through owned field/index projections.
    fn place_is_behind_borrow(&mut self, expr: &HirExpr<'a, 'bump>) -> bool {
        match expr {
            HirExpr::Ident(name, _) => self.local_ref_kind.contains_key(name),
            HirExpr::This { .. } => self.local_ref_kind.contains_key(&self.this_id),
            HirExpr::Deref { expr: inner, .. } => {
                Self::is_pointer_like(&self.peek_type(inner)) || self.place_is_behind_borrow(inner)
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                Self::is_pointer_like(&self.peek_type(object))
                    || self.place_is_behind_borrow(object)
            }
            HirExpr::Index { object, .. } => {
                let oty = self.peek_type(object);
                Self::is_pointer_like(&oty)
                    || matches!(oty.kind, HirTypeKind::Slice(_))
                    || self.place_is_behind_borrow(object)
            }
            _ => false,
        }
    }

    /// A method taking `this` by value would move the receiver; that's illegal if the receiver is
    /// reached through a reference/pointer (auto-deref) or sits behind a borrow, this function fixes it but skips the entire check for values implementing `Copy`.
    pub fn check_move_receiver_not_borrowed(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        method_name: &str,
    ) {
        let peek_ty = self.peek_type(object);
        let (value_ty, auto_deref) = match peek_ty.kind {
            HirTypeKind::Ref { inner, .. }
            | HirTypeKind::SafePointer { inner, .. }
            | HirTypeKind::UnsafePointer { inner, .. } => (*inner, true),
            _ => (peek_ty, false),
        };
        if self.copy_analysis.borrow().type_is_copy(&value_ty) {
            return;
        }
        if auto_deref || self.place_is_behind_borrow(object) {
            self.record(TypeErrorKind::Generic(format!(
                "cannot call `{}` on a value behind a reference: it takes `this` by value, \
                 which would move `{}` out of borrowed content",
                method_name, value_ty
            )));
        }
    }

    /// Checks the receiver holds enough access for the method's `this` kind.
    /// References can never be *upgraded*: `&mut` satisfies everything, `&alias` satisfies only `&alias`, and `&` satisfies
    /// neither.
    /// By-value `this` kinds delegate to [`Self::check_move_receiver_not_borrowed`].
    /// If the receiver isn't itself a `Ref` type, the kind is looked up from the local it is rooted in.
    pub fn check_receiver_ref_kind(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        kind: &ThisPassingKind,
        method_name: &str,
    ) {
        let needed = match kind {
            ThisPassingKind::Move | ThisPassingKind::MoveMut => {
                self.check_move_receiver_not_borrowed(object, method_name);
                return;
            }
            ThisPassingKind::RefMut | ThisPassingKind::MutSafePtr => RefKind::Unique,
            ThisPassingKind::RefAlias => RefKind::Alias,
            _ => return,
        };
        let held = match self.peek_type(object).kind {
            HirTypeKind::Ref { ref_kind, .. } => ref_kind,
            _ => {
                let Some((root, _)) = self.static_field_path(object) else {
                    return;
                };
                let Some(&k) = self.local_ref_kind.get(&root) else {
                    return;
                };
                k
            }
        };
        let ok = match (held, needed) {
            (RefKind::Unique, _) => true,
            (RefKind::Alias, RefKind::Alias) => true,
            _ => false,
        };
        if !ok {
            let name = self
                .static_field_path(object)
                .map(|(r, _)| r)
                .unwrap_or_else(|| StrId::from_static("this value"));
            let (has, wants) = (
                match held {
                    RefKind::Shared => "`&`",
                    RefKind::Alias => "`&alias`",
                    RefKind::Unique => "`&mut`",
                },
                match needed {
                    RefKind::Alias => "`&alias`",
                    _ => "`&mut`",
                },
            );
            self.record(TypeErrorKind::Generic(format!(
                "cannot call `{}` on `{}`: it needs {} access but `{}` is only {}; \
                 a reference cannot be upgraded",
                method_name, name, wants, name, has
            )));
        }
    }

    /// Checks an ordinary (non-method) call.
    ///
    /// It derives a ref template for the calle if the return type may alias an argument.
    ///
    /// This also checks each `&arg` against the callee's read effects
    /// ([`Self::check_call_arg_read_effects`]) by checking all args.
    /// The the templated base param is handled specially.
    ///
    /// And finally decides which loan, if any,  the result holds ([`Self::finalize_call_loans`]).
    pub fn check_potential_this_param_for_move(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
        func: HirFunc<'a, 'bump>,
        params: &[HirParam<'a, 'bump>],
        ret_ty: HirType<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        let template = if self.return_type_may_alias(&ret_ty) {
            Some(self.analyze_ref_template(&func))
        } else {
            None
        };

        let templated_base_param = match &template {
            Some(RefTemplate::Path {
                base: TemplateBase::Param(i),
                ..
            }) => Some(*i),
            _ => None,
        };

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

        let arg_loans = self.check_all_func_args(args, params, templated_base_param, Some(func));
        self.register_call_ensures(expr, &func, args);
        self.apply_callee_concurrency(&func, args);

        if let Some(loan_id) = self.finalize_call_loans(None, args, arg_loans, &ret_ty, template) {
            self.call_loans.insert(Self::expr_key(expr), loan_id);
        }

        Some(ret_ty)
    }

    /// The strongest reference kind appearing in a return type (`Unique` > `Alias` > `Shared`),
    /// looking through `T?`, arrays and tuples. `None` if the type holds no `&`-family reference.
    fn ret_ref_kind(ty: &HirType<'a, 'bump>) -> Option<RefKind> {
        match ty.kind {
            HirTypeKind::Ref { ref_kind, .. } => Some(ref_kind),
            HirTypeKind::Nullable(inner) | HirTypeKind::Array(inner, _) => {
                Self::ret_ref_kind(inner)
            }
            HirTypeKind::Tuple(elems) => {
                elems
                    .iter()
                    .filter_map(Self::ret_ref_kind)
                    .max_by_key(|k| match k {
                        RefKind::Unique => 2,
                        RefKind::Alias => 1,
                        RefKind::Shared => 0,
                    })
            }
            _ => None,
        }
    }

    /// The template couldn't say which part of the receiver the result points into,
    /// so assume it may point anywhere in it. Only for real `&`-family results: raw
    /// and owned pointers are not borrow-tracked.
    fn opaque_call_loan(
        &mut self,
        receiver: Option<&HirExpr<'a, 'bump>>,
        ret_ty: &HirType<'a, 'bump>,
    ) -> Option<LoanId> {
        let kind = Self::ret_ref_kind(ret_ty)?;
        let place = self.resolve_place(receiver?)?;
        let result = match kind {
            RefKind::Unique => self.borrow_checker.borrow_mut(place),
            RefKind::Alias => self.borrow_checker.borrow_alias(place),
            RefKind::Shared => self.borrow_checker.borrow_shared(place),
        };
        result.ok()
    }

    /// Decides which loan a call's result holds, and ends the temporary loans taken for its args.
    ///
    /// * **No template** (result can't alias an arg): end all arg loans, result holds nothing.
    /// * **`Path` template** (result borrows through a known param/path): the arg loans were only
    ///   needed during the call, so end them, then take one new loan on the templated place with
    ///   the return type's ref kind. Other template kinds leave the arg loans alive (conservatively,
    ///   the result may borrow from any of them).
    /// * **Template place unresolvable**: fall back to [`Self::opaque_call_loan`] (borrow the
    ///   whole receiver).
    ///
    /// A conflict while taking the result loan is reported with the template's provenance.
    pub fn finalize_call_loans(
        &mut self,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
        arg_loans: Vec<LoanId>,
        ret_ty: &HirType<'a, 'bump>,
        template: Option<RefTemplate>,
    ) -> Option<LoanId> {
        let Some(template) = template else {
            for loan in arg_loans {
                self.borrow_checker.end_loan_now(loan);
            }
            return None;
        };

        if matches!(template, RefTemplate::Path { .. }) {
            for &loan in &arg_loans {
                self.borrow_checker.end_loan_now(loan);
            }
        }

        let place = match self.resolve_template_place(&template, receiver, args) {
            Some(p) => p,
            None => return self.opaque_call_loan(receiver, ret_ty),
        };

        let ret_ref = Self::peel_nullable(ret_ty);
        let result = if matches!(
            ret_ref.kind,
            HirTypeKind::Ref {
                ref_kind: RefKind::Unique,
                ..
            }
        ) {
            self.borrow_checker.borrow_mut(place)
        } else if matches!(
            ret_ref.kind,
            HirTypeKind::Ref {
                ref_kind: RefKind::Alias,
                ..
            }
        ) {
            self.borrow_checker.borrow_alias(place)
        } else {
            self.borrow_checker.borrow_shared(place)
        };

        match result {
            Ok(loan_id) => Some(loan_id),
            Err(e) => {
                let provenance = self.provenance_from_template(&template, receiver, args);
                let msg = self.describe_borrow_error(&e, provenance.as_ref());
                self.record(TypeErrorKind::Generic(msg));
                None
            }
        }
    }
}
