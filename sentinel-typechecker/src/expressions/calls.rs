use ir::hir::{JoinStr, Operator};

use ir::{
    ast::{FuncSafety, MutabilityState},
    borrow_checker::{BorrowKind, LoanId, ReadTemplate, RefTemplate},
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{HirExpr, HirFunc, HirParam, HirType, HirTypeKind, RefKind, StrId, ThisPassingKind},
    ir_hasher::FxHashMap,
    span::SourceSpan,
};

use crate::{
    TypeChecker,
    initialization::BareImportKind,
    naming::type_to_string,
    type_checker::{Occurrence, SLICE_PRIMITIVES, SymbolId},
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// `&alias this` is a shared-mutable access: it must be checked as `Alias`
    /// (compatible with other aliases), never as `Shared` (which conflicts with
    /// an existing alias loan) nor `Mutable`.
    fn receiver_borrow_kind(kind: &ThisPassingKind, requires_mut: bool) -> BorrowKind {
        match kind {
            ThisPassingKind::RefAlias => BorrowKind::Alias,
            _ if requires_mut => BorrowKind::Mutable,
            _ => BorrowKind::Shared,
        }
    }

    /// Loan currently attached to the place behind each `&x` / `&mut x` / `&alias x`
    /// argument, taken BEFORE the arguments are checked.
    fn ref_arg_loans_before(&mut self, args: &[HirExpr<'a, 'bump>]) -> Vec<Option<LoanId>> {
        args.iter()
            .map(|a| match a {
                HirExpr::Ref { expr, .. } => self
                    .resolve_place(expr)
                    .and_then(|p| self.borrow_checker.loan_for_place(p).copied()),
                _ => None,
            })
            .collect()
    }

    /// End loans the argument borrows created, for calls (closure calls) whose result
    /// cannot keep them alive. Without this the loan lives to the end of the block and
    /// a later `&alias`/`&mut` use of the same local is reported as a conflict.
    fn end_new_ref_arg_loans(&mut self, args: &[HirExpr<'a, 'bump>], before: &[Option<LoanId>]) {
        for (arg, prev) in args.iter().zip(before.iter()) {
            let HirExpr::Ref { expr, .. } = arg else {
                continue;
            };
            let Some(place) = self.resolve_place(expr) else {
                continue;
            };
            if let Some(loan) = self.borrow_checker.loan_for_place(place).copied() {
                if Some(loan) != *prev {
                    self.borrow_checker.end_loan_now(loan);
                }
            }
        }
    }

    pub fn check_module_access_expr(
        &mut self,
        access: &&ir::hir::HirModuleAccess<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        if let Some(ty) = self.check_module_const_access(access) {
            return ty;
        }

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

        if let Some(midx) = resolved_module_idx {
            let konst = self
                .module_consts
                .get(&midx)
                .and_then(|m| m.get(&access.member))
                .map(|c| (c.ty, c.visibility, c.declaring_module_idx, c.name, "const"));
            let stat = self
                .module_statics
                .get(&midx)
                .and_then(|m| m.get(&access.member))
                .map(|s| (s.ty, s.visibility, s.declaring_module_idx, s.name, "static"));
            if let Some((ty, vis, decl_idx, name, kind)) = konst.or(stat) {
                self.recover(
                    self.check_visibility(vis, decl_idx, kind, name.as_str()),
                    (),
                );
                self.check_module_path_imported(access.path);
                if kind == "static" {
                    self.note_static_access(access.member);
                    self.check_ident_init_read(access.member, access.member.as_str(), &ty);
                }
                self.record_item_occurrence(access.span, access.member, ty, midx);
                return ty;
            }
        }

        let free_func = resolved_module_idx
            .and_then(|midx| self.context.get_module_function(midx, &member_name));

        let mangled_type_name: Option<StrId> = assoc_type_name.and_then(|_| {
            let midx = resolved_module_idx?;
            let pkg = self.context.dep_graph.borrow().get_module_package(midx)?;
            let mangled = self.context.dep_graph.borrow().mangle_type_name(
                midx,
                pkg,
                &self.context.string_pool,
            );
            Some(mangled)
        });

        let method_func = if free_func.is_none() {
            mangled_type_name
                .or_else(|| access.path.last().map(|p| *p))
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

                return HirType::synthetic(HirTypeKind::Unknown);
            }
        };

        self.check_unsafe_call(&func, &member_name);
        self.check_call_visibility(&func, "function", &member_name);

        self.check_module_path_imported(access.path);

        let param_types: Vec<HirType<'a, 'bump>> = func
            .params
            .unwrap_or(&[])
            .iter()
            .filter_map(|p| p.get_type())
            .collect();
        let ret_ty = func
            .return_type
            .unwrap_or(HirType::synthetic(HirTypeKind::Void));
        let lambda_ty = HirType::synthetic(HirTypeKind::Lambda {
            params: self.context.bump.alloc_slice(&param_types),
            return_type: self.context.bump.alloc_value(ret_ty),
        });

        if let Some(_) = resolved_module_idx {
            if free_func.is_some() {
                self.record_item_occurrence_named(
                    access.span,
                    access.member,
                    func.name,
                    ret_ty,
                    func.declaring_module_idx,
                );
            } else if let Some(target_type) = assoc_type_name {
                self.record_method_occurrence(access.span, access.member, lambda_ty, target_type);
            }
        }
        lambda_ty
    }

    pub fn check_this_expr(&mut self, span: &SourceSpan<'a>) -> HirType<'a, 'bump> {
        match self.context.get_variable("this") {
            Some((symbol_id, ty)) => {
                self.occurrences.push(Occurrence {
                    span: *span,
                    name: self.this_id,
                    hir_name: self.this_id,
                    ty,
                    module_idx: self.context.current_module_idx,
                    symbol: symbol_id,
                    is_decl: false,
                });
                ty
            }
            None => {
                self.record(TypeErrorKind::Generic(
                    "`this` used outside of a method that has a `this` receiver".to_string(),
                ));
                HirType::synthetic(HirTypeKind::Unknown)
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
        // An extern declaration (`extern "C" func ...;`) has no body to analyze, so the
        // write/read templates above are empty for it. The call is already inside `unsafe`,
        // so trust a mutable/alias pointer argument to be written through the FFI boundary
        // rather than demanding proof we structurally can't produce.
        let is_extern_no_body = callee.map(|f| f.body.is_none()).unwrap_or(false);

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
                    self.recover(self.types_compatible(&pt, &arg_type), ());
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
                            is_extern_no_body,
                        );
                    let arg_type = self.check_ref_expr_deferring_init(
                        expr,
                        *ref_kind,
                        *span,
                        false,
                        ignores_init,
                    );
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(&pt, &arg_type), ());
                    }
                    if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                        self.optimistically_mark_mut_target_init(expr);
                    }
                } else {
                    let arg_type = match param_type {
                        Some(_) if matches!(arg, HirExpr::Lambda { .. }) && callee.is_some() => {
                            self.check_arg_expr(arg, param_type.as_ref(), callee.as_ref())
                        }
                        Some(pt) => self.check_expr_expected(arg, &pt),
                        None => self.check_expr(arg),
                    };
                    self.check_and_record_value_use(arg, &arg_type);
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(&pt, &arg_type), ());
                    }
                }
                continue;
            }

            if self.check_write_only_mut_arg(
                arg,
                param,
                i,
                write_template.as_deref(),
                read_template.as_deref(),
                is_extern_no_body,
                &mut arg_loans,
            ) {
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
                let accepts_uninit = Self::callee_ignores_initial_contents(
                    write_template.as_deref(),
                    read_template.as_deref(),
                    i,
                    is_extern_no_body,
                );
                let proven_written = write_template
                    .as_deref()
                    .and_then(|t| t.get(i))
                    .copied()
                    .unwrap_or(false);

                let arg_type =
                    self.check_ref_expr_deferring_init(expr, *rk, *span, true, accepts_uninit);
                if let Some(pt) = param_type {
                    self.recover(self.types_compatible(&pt, &arg_type), ());
                }
                if proven_written {
                    self.optimistically_mark_mut_target_init(expr);
                }
                if let Some(place) = self.resolve_place(expr) {
                    if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                        arg_loans.push(loan_id);
                    }
                }
                continue;
            }

            let arg_type = self.check_arg_expr(arg, param_type.as_ref(), callee.as_ref());
            self.check_and_record_value_use(arg, &arg_type);
            if let Some(pt) = param_type {
                self.recover(self.types_compatible(&pt, &arg_type), ());
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
        is_extern_no_body: bool,
    ) -> bool {
        if is_extern_no_body {
            return true;
        }
        let definitely_written = write_template
            .and_then(|t| t.get(i))
            .copied()
            .unwrap_or(false);
        let never_reads_contents = read_template
            .and_then(|t| t.get(i))
            .is_some_and(|rt| !Self::read_template_touches_contents(rt));
        definitely_written || never_reads_contents
    }

    pub(crate) fn check_call_visibility(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        kind: &str,
        name: &str,
    ) {
        self.recover(
            self.check_visibility(
                func.function_metadata.visibility,
                func.declaring_module_idx,
                kind,
                name,
            ),
            (),
        );
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
        call: &HirExpr<'a, 'bump>,
        callee: &&HirExpr<'a, 'bump>,
        interface: &StrId,
        args: &&[HirExpr<'a, 'bump>],
        span: SourceSpan<'a>,
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
                return HirType::synthetic(HirTypeKind::Unknown);
            }
        };
        let _ = self.check_expr(object);

        let iface_name = interface.to_string();
        let Some(iface) = self.context.get_interface(&iface_name) else {
            self.record(TypeErrorKind::UndefinedType(iface_name));
            return HirType::synthetic(HirTypeKind::Unknown);
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
            return HirType::synthetic(HirTypeKind::Unknown);
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
                    Some(pt) => self.check_expr_expected(arg, &pt),
                    None => self.check_expr(arg),
                };
                self.check_and_record_value_use(arg, &arg_type);
                if let Some(param_type) = param.get_type() {
                    let result = self.types_compatible(&param_type, &arg_type);
                    self.recover(result, ());
                }
            }
        }

        // Every implementation is checked against the interface method's `ensures` (see
        // `check_ensures_conditions_at_exit`), so callers may rely on them whichever type is
        // behind the interface.
        self.register_call_ensures(call, method, args);

        let ret = method
            .return_type
            .unwrap_or(HirType::synthetic(HirTypeKind::Void));
        self.record_method_occurrence(span, *field, ret, *interface);
        ret
    }

    /// True when `expr` reaches its value through an `&alias` reference that is not the
    /// current method's own `this` (a method declared `&alias this` is responsible for itself;
    /// its callers are checked here).
    fn receiver_through_alias(&self, expr: &HirExpr<'a, 'bump>) -> bool {
        if matches!(expr, HirExpr::This { .. }) {
            return false;
        }
        if matches!(
            self.peek_type(expr).kind,
            HirTypeKind::Ref {
                ref_kind: RefKind::Alias,
                ..
            }
        ) {
            return true;
        }
        match expr {
            HirExpr::Ident(n, _) => self.local_ref_kind.get(n) == Some(&RefKind::Alias),
            HirExpr::FieldAccess { object, .. }
            | HirExpr::Get { object, .. }
            | HirExpr::Index { object, .. } => self.receiver_through_alias(object),
            HirExpr::Deref { expr: inner, .. } => self.receiver_through_alias(inner),
            _ => false,
        }
    }

    /// Future hook: collections implementing a dedicated interface that handles concurrent
    /// invalidation themselves. Nothing is exempt for now.
    fn alias_invalidation_exempt(&self, _recv_ty: &HirType<'a, 'bump>) -> bool {
        false
    }

    fn check_alias_receiver_invalidation(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        owner: StrId,
        func: &HirFunc<'a, 'bump>,
        method_name: &str,
    ) {
        if !self.receiver_through_alias(object) {
            return;
        }
        let paths = self.analyze_invalidations(owner, func);
        let Some(first) = paths.first() else { return };
        if self.alias_invalidation_exempt(&self.peek_type(object)) {
            return;
        }
        let what = first.join(".");
        self.record(TypeErrorKind::Generic(format!(
            "cannot call `{method_name}` through an `&alias` reference: it may invalidate \
             `this.{what}` (reallocate it or change its length), which would invalidate other \
             `&alias` references into the same value, possibly on another thread. Use `&mut` \
             access instead"
        )));
    }

    pub fn check_call_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        callee: &&HirExpr<'a, 'bump>,
        args: &&[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
        type_args: Option<&'bump [HirType<'a, 'bump>]>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        if let HirExpr::FieldAccess { .. } = &**callee {
            let fty = self.peek_type(callee);
            if let HirTypeKind::Lambda {
                params,
                return_type,
            } = Self::strip_ref(&fty).kind
            {
                if args.len() != params.len() {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: params.len(),
                        found_args: args.len(),
                    });
                }
                for (a, pt) in args.iter().zip(params.iter()) {
                    let at = self.check_expr_expected(a, pt);
                    self.recover(self.types_compatible(pt, &at), ());
                }
                return *return_type;
            }
        }

        match &callee {
            HirExpr::Ident(func_name, ident_span) => {
                self.set_span(*ident_span);
                if self.context.is_local_binding(func_name) {
                    let callee_type = self.check_expr(callee);
                    let loans_before = self.ref_arg_loans_before(args);
                    return match callee_type.kind {
                        HirTypeKind::Lambda {
                            params,
                            return_type,
                        } => {
                            if args.len() != params.len() {
                                self.record(TypeErrorKind::InvalidFunctionCall {
                                    expected_args: params.len(),
                                    found_args: args.len(),
                                });
                            }
                            for (arg, pt) in args.iter().zip(params.iter()) {
                                let arg_type = self.check_expr_expected(arg, pt);
                                self.check_and_record_value_use(arg, &arg_type);
                                self.recover(self.types_compatible(pt, &arg_type), ());
                            }
                            if !self.return_type_may_alias(return_type) {
                                self.end_new_ref_arg_loans(args, &loans_before);
                            }
                            *return_type
                        }
                        HirTypeKind::Generic(g) if self.fn_closure_constraints.contains_key(&g) => {
                            let ret = self.check_closure_call(g, args, *span);
                            if !self.return_type_may_alias(&ret) {
                                self.end_new_ref_arg_loans(args, &loans_before);
                            }
                            ret
                        }
                        _ => {
                            self.record(TypeErrorKind::Generic(format!(
                                "Expression of type `{}` is not callable",
                                type_to_string(&callee_type, &self.context.string_pool)
                            )));
                            HirType::synthetic(HirTypeKind::Unknown)
                        }
                    };
                }

                let func = self.context.get_function(func_name).or_else(|| {
                    self.functions_by_module
                        .iter()
                        .find(|(_, s)| s.contains(func_name))
                        .and_then(|(&m, _)| self.context.get_module_function(m, func_name))
                });
                let func = match func {
                    Some(f) => f,
                    None => {
                        self.record(TypeErrorKind::UndefinedFunction(func_name.to_string()));
                        return HirType::synthetic(HirTypeKind::Unknown);
                    }
                };
                self.check_unsafe_call(&func, func_name.as_str());
                self.check_call_visibility(&func, "function", func_name.as_str());

                self.record_item_occurrence_named(
                    *ident_span,
                    func.unmangled_name,
                    func.name,
                    func.return_type
                        .unwrap_or(HirType::synthetic(HirTypeKind::Void)),
                    func.declaring_module_idx,
                );

                self.check_name_import_visibility(
                    *func_name,
                    func.unmangled_name,
                    func_name.as_str(),
                );

                self.check_resolved_function_call(
                    expr,
                    func,
                    func_name.as_str(),
                    *ident_span,
                    args,
                    type_args,
                    expected,
                )
            }

            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                if let HirExpr::Ident(tname, t_usage_span) = &**object {
                    if self.context.get_variable(tname).is_none() {
                        let midx = self.context.current_module_idx;
                        let mangled = self.context.dep_graph.borrow().mangle_type_name(
                            midx,
                            *tname,
                            &self.context.string_pool,
                        );
                        for key in [mangled, *tname] {
                            if let Some(func) = self.context.get_method(key, field).copied() {
                                // Enums/builtins have no struct def; only record the type occurrence for structs.
                                let ty_struct = self
                                    .context
                                    .get_struct(key)
                                    .or_else(|| self.context.get_struct(mangled))
                                    .or_else(|| self.context.get_struct(*tname));

                                let call_style = format!("{}.{}", tname, field);
                                self.check_unsafe_call(&func, &call_style);
                                self.check_call_visibility(&func, "method", &call_style);
                                let ret = self.check_resolved_function_call(
                                    expr,
                                    func,
                                    &call_style,
                                    *span,
                                    args,
                                    type_args,
                                    expected,
                                );
                                if let Some(ty_struct) = ty_struct {
                                    self.record_item_occurrence_named(
                                        *t_usage_span,
                                        ty_struct.unmangled_name,
                                        ty_struct.name,
                                        HirType::struct_type(
                                            ty_struct.name,
                                            ty_struct.unmangled_name,
                                            self.context.bump.alloc_slice(
                                                ty_struct
                                                    .fields
                                                    .iter()
                                                    .map(|f| f.field_type)
                                                    .collect::<Vec<_>>()
                                                    .as_slice(),
                                            ),
                                            type_args.unwrap_or(&[]),
                                            ty_struct.span,
                                        ),
                                        self.context.current_module_idx,
                                    );
                                }
                                self.record_method_occurrence(*span, *field, ret, key);
                                return ret;
                            }
                        }
                    }
                }

                let recv_peek = self.peek_type(object);
                let skip_recv_init = self
                    .resolve_callable_method(Self::strip_ref(&recv_peek), &field.to_string())
                    .is_some_and(|(_, _, f)| {
                        matches!(f.function_metadata.func_safety, FuncSafety::Unsafe)
                            && matches!(
                                f.params.and_then(|p| p.first()),
                                Some(HirParam::This { kind, .. })
                                    if !matches!(kind, ThisPassingKind::Move | ThisPassingKind::MoveMut)
                            )
                    });
                let prev_si = self.suppress_init_read;
                if skip_recv_init {
                    self.suppress_init_read = true;
                }
                let obj_type = self.check_expr(object);
                self.suppress_init_read = prev_si;
                let stripped = Self::strip_ref(&obj_type);

                if let Some(elem) = match stripped.kind {
                    HirTypeKind::Slice(e) | HirTypeKind::Array(e, _) => Some(*e),
                    _ => None,
                } {
                    let name = field.as_str();
                    if SLICE_PRIMITIVES.contains(&name) {
                        return self.check_slice_primitive_call(object, elem, name, args);
                    }
                }

                if let HirTypeKind::Struct {
                    name: struct_name,
                    type_args,
                    ..
                } = stripped.kind
                {
                    if let Some(struct_def) = self.context.get_struct(struct_name) {
                        if let Some(field_idx) =
                            struct_def.fields.iter().position(|f| f.name == *field)
                        {
                            let field_ty = if type_args.is_empty() {
                                struct_def.fields[field_idx].field_type
                            } else {
                                self.instantiate_struct(struct_name, type_args)
                                    .map(|fields| fields[field_idx])
                                    .unwrap_or(struct_def.fields[field_idx].field_type)
                            };

                            let callable = match field_ty.kind {
                                HirTypeKind::Lambda {
                                    params,
                                    return_type,
                                } => Some((params, return_type)),
                                HirTypeKind::Generic(g) => {
                                    self.fn_closure_constraints.get(&g).copied().and_then(|c| {
                                        match c.kind {
                                            HirTypeKind::Lambda {
                                                params,
                                                return_type,
                                            } => Some((params, return_type)),
                                            _ => None,
                                        }
                                    })
                                }
                                _ => None,
                            };

                            if let Some((params, return_type)) = callable {
                                self.check_bare_name_import(
                                    self.context.struct_owner(struct_name),
                                    struct_name,
                                    struct_name.as_str(),
                                    BareImportKind::Struct,
                                );
                                self.occurrences.push(Occurrence {
                                    span: *span,
                                    name: *field,
                                    hir_name: *field,
                                    ty: field_ty,
                                    module_idx: self.context.current_module_idx,
                                    symbol: SymbolId::Field {
                                        struct_name,
                                        field_name: *field,
                                    },
                                    is_decl: false,
                                });
                                if args.len() != params.len() {
                                    self.record(TypeErrorKind::InvalidFunctionCall {
                                        expected_args: params.len(),
                                        found_args: args.len(),
                                    });
                                }
                                for (arg, pt) in args.iter().zip(params.iter()) {
                                    let arg_type = self.check_expr_expected(arg, pt);
                                    self.check_and_record_value_use(arg, &arg_type);
                                    self.recover(self.types_compatible(pt, &arg_type), ());
                                }
                                return *return_type;
                            }
                        }
                    }
                }

                let interface_name = match &stripped.kind {
                    HirTypeKind::DynInterface { name, .. } => Some(name.to_string()),
                    HirTypeKind::Dyn { bounds } => bounds.iter().find_map(|b| match &b.kind {
                        HirTypeKind::DynInterface { name, .. } => Some(name.to_string()),
                        HirTypeKind::Struct { name, .. } => {
                            let name_str = name.to_string();
                            self.context.get_interface(&name_str).map(|_| name_str)
                        }
                        _ => None,
                    }),
                    HirTypeKind::Generic(g) => self.generic_bound_with_method(*g, field.as_str()),
                    _ => None,
                };

                if let Some(iface_name) = interface_name {
                    let method_name = field.to_string();
                    let iface = match self.context.get_interface(&iface_name) {
                        Some(i) => i,
                        None => {
                            self.record(TypeErrorKind::UndefinedType(iface_name));
                            return HirType::synthetic(HirTypeKind::Unknown);
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
                        return HirType::synthetic(HirTypeKind::Unknown);
                    };

                    let total_params = method.params.map(|p| p.len()).unwrap_or(0);
                    let expected_args = total_params.saturating_sub(1);
                    if args.len() != expected_args {
                        self.record(TypeErrorKind::InvalidFunctionCall {
                            expected_args,
                            found_args: args.len(),
                        });
                    }

                    let mut subs: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
                    if let (Some(ta), Some(gs)) = (type_args, method.generics) {
                        for (g, a) in gs.iter().zip(ta.iter()) {
                            subs.insert(g.name, *a);
                        }
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
                            self.check_receiver_ref_kind(object, kind, field.as_str());

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
                                let borrow_kind = Self::receiver_borrow_kind(kind, requires_mut);
                                self.check_borrow_use(object, place, borrow_kind);
                            }
                        }
                        for (arg, param) in args.iter().zip(params.iter().skip(1)) {
                            let pt: Option<HirType<'a, 'bump>> = param.get_type().map(|t| {
                                if subs.is_empty() {
                                    t
                                } else {
                                    self.substitute_type_local(&t, &subs)
                                }
                            });
                            let arg_type = match &pt {
                                Some(pt) => self.check_expr_expected(arg, pt),
                                None => self.check_expr(arg),
                            };
                            self.check_and_record_value_use(arg, &arg_type);
                            if let Some(pt) = &pt {
                                let result = self.types_compatible(pt, &arg_type);
                                self.recover(result, ());
                            }
                        }
                    }

                    let ret = method
                        .return_type
                        .unwrap_or(HirType::synthetic(HirTypeKind::Void));
                    let ret = if subs.is_empty() {
                        ret
                    } else {
                        self.substitute_type_local(&ret, &subs)
                    };
                    let iface_id = StrId(self.context.string_pool.intern(&iface_name));
                    self.record_method_occurrence(*span, *field, ret, iface_id);
                    return ret;
                }

                let (struct_name_id, type_name, func) =
                    match self.resolve_callable_method(stripped, &field.to_string()) {
                        Some(found) => found,
                        None => {
                            self.record(TypeErrorKind::Generic(format!(
                                "no method `{}` on `{}`",
                                field,
                                type_to_string(&obj_type, &self.context.string_pool)
                            )));
                            return HirType::synthetic(HirTypeKind::Unknown);
                        }
                    };
                let call_style = format!("{}.{}", type_name, field);
                self.check_unsafe_call(&func, &call_style);
                self.check_call_visibility(&func, "method", &call_style);
                self.check_alias_receiver_invalidation(
                    object,
                    struct_name_id,
                    &func,
                    field.as_str(),
                );

                let total_params = func.params.map(|p| p.len()).unwrap_or(0);
                let expected_args = total_params.saturating_sub(1);
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }

                let struct_type_args: &[HirType<'a, 'bump>] = match &stripped.kind {
                    HirTypeKind::Struct { type_args, .. } | HirTypeKind::Enum { type_args, .. } => {
                        type_args
                    }
                    _ => &[],
                };
                let mut method_subs =
                    self.generic_substitutions_for_type(struct_name_id, struct_type_args);

                if let Some(exp) = expected {
                    if let Some(ret) = func.return_type {
                        self.unify_generic(&ret, exp, &mut method_subs);
                    }
                }
                if let Some(params) = func.params {
                    let normal_params: &[HirParam<'a, 'bump>] = params.get(1..).unwrap_or(&[]);
                    for (k, v) in self.pre_infer_generics(Some(&func), args, normal_params) {
                        method_subs.entry(k).or_insert(v);
                    }
                }

                let unsubstituted_ret_ty = func
                    .return_type
                    .unwrap_or(HirType::synthetic(HirTypeKind::Void));
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

                        self.check_receiver_ref_kind(object, kind, field.as_str());

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
                                    let borrow_kind =
                                        Self::receiver_borrow_kind(kind, requires_mut);

                                    // Pass the RECEIVER, not the call expression: provenance (and
                                    // capture effects) are derived from the place being accessed.
                                    // With the call expr, `access_provenance` is None, so a use
                                    // through `let state: &mut T = ..; state.m()` looked like a
                                    // competing access to the very loan `state` holds.
                                    if !requires_mut && !self.return_type_may_alias(&ret_ty) {
                                        self.check_borrow_use_shell(object, place, borrow_kind);
                                    } else {
                                        self.check_borrow_use(object, place, borrow_kind);
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

                    let read_templates = self.analyze_read_templates_for_borrows(&func);
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
                    self.register_call_ensures(expr, &func, args);
                    self.apply_callee_concurrency(&func, args);

                    let loan =
                        self.finalize_call_loans(Some(object), args, arg_loans, &ret_ty, template);

                    if let Some(loan_id) = loan {
                        let key = Self::expr_key(expr);

                        if let Some(old_id) = self.call_loans.insert(key, loan_id) {
                            if old_id != loan_id {
                                self.borrow_checker.end_loan_now(old_id);
                            }
                        }

                        self.call_loan_order.push((key, loan_id));
                    }

                    for l in receiver_multi_place_loans {
                        self.borrow_checker.end_loan_now(l);
                    }

                    self.apply_call_invalidations(object, struct_name_id, &func);
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

                let path_str = access.path.join("::");
                let qualified_name = format!("{}::{}", path_str, member_name);
                let func = match free_func.or(method_func) {
                    Some(f) => f,
                    None => {
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

                        return HirType::synthetic(HirTypeKind::Unknown);
                    }
                };

                self.check_module_path_imported(access.path);
                self.check_call_visibility(&func, "function", &qualified_name);

                if free_func.is_some() && func.generics.is_some_and(|g| !g.is_empty()) {
                    let display = format!("{}.{}", access.path.join("::"), member_name);

                    let ret_ty = self.check_resolved_function_call(
                        expr,
                        func,
                        &display,
                        access.span,
                        args,
                        type_args,
                        expected,
                    );
                    if let Some(midx) = resolved_module_idx {
                        self.record_item_occurrence(access.span, access.member, ret_ty, midx);
                    }
                    return ret_ty;
                }

                let expected_args = func.params.map(|p| p.len()).unwrap_or(0);
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }
                if let Some(params) = func.params {
                    let read_templates = self.analyze_read_templates_for_borrows(&func);
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
                    self.register_call_ensures(expr, &func, args);
                    self.apply_callee_concurrency(&func, args);

                    let ret_ty = func
                        .return_type
                        .unwrap_or(HirType::synthetic(HirTypeKind::Void));
                    if !self.return_type_may_alias(&ret_ty) {
                        for loan in arg_loans {
                            self.borrow_checker.end_loan_now(loan);
                        }
                    }
                }
                let ret_ty = func
                    .return_type
                    .unwrap_or(HirType::synthetic(HirTypeKind::Void));
                if let Some(_) = resolved_module_idx {
                    if free_func.is_some() {
                        self.record_item_occurrence_named(
                            access.span,
                            access.member,
                            func.name,
                            ret_ty,
                            func.declaring_module_idx,
                        );
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
                match callee_type.kind {
                    HirTypeKind::Lambda { return_type, .. } => *return_type,
                    _ => {
                        self.record(TypeErrorKind::Generic(format!(
                            "Expression of type `{}` is not callable",
                            type_to_string(&callee_type, &self.context.string_pool)
                        )));
                        HirType::synthetic(HirTypeKind::Unknown)
                    }
                }
            }
        }
    }

    fn check_resolved_function_call(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        func: HirFunc<'a, 'bump>,
        display_name: &str,
        span: SourceSpan<'a>,
        args: &&[HirExpr<'a, 'bump>],
        type_args: Option<&[HirType<'a, 'bump>]>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
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
                                let resolved = self.substitute_type_local(def_ty, &temp_map);
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
                            display_name,
                            tp.len(),
                            ta.len()
                        )));
                    }
                    let mut map = FxHashMap::default();
                    for (&p, &a) in tp.iter().zip(full_args.iter()) {
                        map.insert(p.name, a);
                    }
                    map
                }
                None => {
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
                        display_name
                    )));
                }
                FxHashMap::default()
            }
        };

        if let Some(exp) = expected {
            if let Some(ret) = func.return_type {
                self.unify_generic(&ret, exp, &mut substitutions);
            }
        }
        if let Some(params) = func.params {
            for (k, v) in self.pre_infer_generics(Some(&func), args, params) {
                substitutions.entry(k).or_insert(v);
            }
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
            let ret_ty = func
                .return_type
                .unwrap_or(HirType::synthetic(HirTypeKind::Void));
            if type_args.is_none() {
                self.check_generics_inferred(&func, &substitutions, display_name, span);
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

        let unsubstituted_ret_ty = func
            .return_type
            .unwrap_or(HirType::synthetic(HirTypeKind::Void));
        let ret_ty = if substitutions.is_empty() {
            unsubstituted_ret_ty
        } else {
            self.substitute_type_local(&unsubstituted_ret_ty, &substitutions)
        };

        if let Some(value) =
            self.check_potential_this_param_for_move(expr, args, func, params, ret_ty)
        {
            self.register_call_ensures(expr, &func, args);
            return value;
        }

        let read_templates = self.analyze_read_templates_for_borrows(&func);
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
        self.register_call_ensures(expr, &func, args);
        self.apply_callee_concurrency(&func, args);

        substitutions.extend(std::mem::take(&mut self.closure_generic_subs));
        if type_args.is_none() {
            self.check_generics_inferred(&func, &substitutions, display_name, span);
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
            HirType::synthetic(HirTypeKind::Void)
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

        match method {
            "write_uninit" => {
                let t = self.check_expr_expected(&args[0], &HirType::synthetic(HirTypeKind::Usize));
                if !self.is_integer(&t) {
                    self.record(TypeErrorKind::Generic(format!(
                        "index must be an integer, found `{}`",
                        type_to_string(&t, &self.context.string_pool)
                    )));
                }
                let val_ty = self.check_expr_expected(&args[1], &elem);
                self.check_and_record_value_use(&args[1], &val_ty);
                self.recover(self.types_compatible(&elem, &val_ty), ());
            }
            "get_unchecked" => {
                let t = self.check_expr_expected(&args[0], &HirType::synthetic(HirTypeKind::Usize));
                if !self.is_integer(&t) {
                    self.record(TypeErrorKind::Generic(format!(
                        "index must be an integer, found `{}`",
                        type_to_string(&t, &self.context.string_pool)
                    )));
                }
            }
            "write_uninit_all" => {
                let src_ty = self.check_expr(&args[0]);
                match Self::strip_ref(&src_ty).kind {
                    HirTypeKind::Slice(e) | HirTypeKind::Array(e, _) => {
                        self.recover(self.types_compatible(&elem, e), ());
                    }
                    _ => self.record(TypeErrorKind::Generic(format!(
                        "`write_uninit_all` expects a slice or array source, found `{}`",
                        type_to_string(&src_ty, &self.context.string_pool)
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

    pub fn check_name_import_visibility(&mut self, name: StrId, unmangled: StrId, name_str: &str) {
        let current = self.context.current_module_idx;

        if self
            .functions_by_module
            .get(&current)
            .is_some_and(|s| s.contains(&name))
        {
            return;
        }

        if let Some(imports) = self.imports_by_module.get(&current) {
            for key in [name, unmangled] {
                if let Some(&target) = imports.named.get(&key) {
                    let real = self
                        .context
                        .dep_graph
                        .borrow()
                        .canonical_member_module(target, unmangled);
                    for m in [target, real] {
                        if self
                            .functions_by_module
                            .get(&m)
                            .is_some_and(|s| s.contains(&name))
                        {
                            return;
                        }
                    }
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

        match &ty.kind {
            HirTypeKind::Struct { name, .. } => {
                let struct_name_str = name.to_string();
                if let Some(hit) = try_name(struct_name_str.clone()) {
                    return Some(hit);
                }
                self.resolve_default_interface_method(*name, &struct_name_str, method_name)
            }
            HirTypeKind::Enum { name, .. } => try_name(name.to_string()),
            HirTypeKind::Slice(elem) | HirTypeKind::Array(elem, _) => {
                if let Some(elem_name) = self.builtin_element_name(elem) {
                    if let Some(hit) = try_name(format!("slice_{}", elem_name)) {
                        return Some(hit);
                    }
                }
                try_name("slice".to_string())
            }
            _ => try_name(self.builtin_element_name(ty)?),
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
        Some(match ty.kind {
            HirTypeKind::I8 => "i8".into(),
            HirTypeKind::I16 => "i16".into(),
            HirTypeKind::I32 => "i32".into(),
            HirTypeKind::I64 => "i64".into(),
            HirTypeKind::I128 => "i128".into(),
            HirTypeKind::U8 => "u8".into(),
            HirTypeKind::U16 => "u16".into(),
            HirTypeKind::U32 => "u32".into(),
            HirTypeKind::U64 => "u64".into(),
            HirTypeKind::U128 => "u128".into(),
            HirTypeKind::Usize => "usize".into(),
            HirTypeKind::Isize => "isize".into(),
            HirTypeKind::F32 => "f32".into(),
            HirTypeKind::F64 => "f64".into(),
            HirTypeKind::Boolean => "bool".into(),
            HirTypeKind::String => "str".into(),
            HirTypeKind::Char => "char".into(),
            HirTypeKind::Struct { name, .. } => name.to_string(),
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
            let through_ref = self
                .context
                .get_variable(&root_name)
                .is_some_and(|(_, ty)| {
                    matches!(
                        ty.kind,
                        HirTypeKind::Ref {
                            ref_kind: RefKind::Unique | RefKind::Alias,
                            ..
                        } | HirTypeKind::SafePointer {
                            mutability_state: MutabilityState::Mut,
                            ..
                        } | HirTypeKind::UnsafePointer {
                            mutability_state: MutabilityState::Mut,
                            ..
                        }
                    )
                });

            if through_ref {
                return Ok(());
            }
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
                    .any(|c| matches!(c.kind, HirTypeKind::Lambda { .. }))
        });
        if unresolved {
            self.set_span(span);
            self.record(TypeErrorKind::Generic(format!(
                "generic function `{}` requires explicit type arguments, e.g. `{}<Type>(...)`",
                display_name, display_name
            )));
        }
    }

    fn resolve_interface_key(&self, name: &str) -> Option<String> {
        if self.context.get_interface(name).is_some() {
            return Some(name.to_string());
        }
        let id = StrId(self.context.string_pool.intern(name));
        let cur = self.context.current_module_idx;
        let mut mods = vec![cur];
        if let Some(imp) = self.imports_by_module.get(&cur) {
            mods.extend(imp.named.get(&id).copied());
            mods.extend(imp.modules.iter().copied());
            mods.extend(imp.wildcard.iter().copied());
        }
        for m in mods {
            let key = self
                .context
                .dep_graph
                .borrow()
                .mangle_type_name(m, id, &self.context.string_pool)
                .to_string();
            if self.context.get_interface(&key).is_some() {
                return Some(key);
            }
        }
        None
    }

    fn generic_bound_with_method(&self, g: StrId, method: &str) -> Option<String> {
        for name in self.generic_bounds.get(&g)? {
            let Some(key) = self.resolve_interface_key(name) else {
                continue;
            };
            let declares = self
                .context
                .get_interface(&key)
                .and_then(|i| i.methods)
                .is_some_and(|ms| ms.iter().any(|m| m.unmangled_name.as_str() == method));
            if declares {
                return Some(key);
            }
        }
        None
    }

    /// `f(&mut arr[a..<b])` / `f(&mut arr[i])` where `f` never reads through that param:
    /// don't demand the range be initialized, and treat it as initialized afterwards.
    fn check_write_only_mut_arg(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param: &HirParam<'a, 'bump>,
        i: usize,
        write_template: Option<&[bool]>,
        read_template: Option<&[ReadTemplate]>,
        is_extern_no_body: bool,
        arg_loans: &mut Vec<LoanId>,
    ) -> bool {
        let (
            HirParam::Normal {
                multi_place: None, ..
            },
            HirExpr::Ref {
                expr,
                ref_kind: rk @ (RefKind::Unique | RefKind::Alias),
                span,
            },
        ) = (param, arg)
        else {
            return false;
        };
        if !matches!(expr, HirExpr::Slice { .. } | HirExpr::Index { .. }) {
            return false;
        }
        if !Self::callee_ignores_initial_contents(
            write_template,
            read_template,
            i,
            is_extern_no_body,
        ) {
            return false;
        }

        let arg_type = self.check_ref_expr_deferring_init(expr, *rk, *span, true, true);
        if let Some(pt) = param.get_type() {
            self.recover(self.types_compatible(&pt, &arg_type), ());
        }
        self.optimistically_mark_mut_target_init(expr);
        if let Some(place) = self.resolve_place(expr) {
            if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                arg_loans.push(loan_id);
            }
        }
        true
    }

    /// `libc_time.X`: type + occurrence on `X`.
    pub(crate) fn check_module_const_access(
        &mut self,
        acc: &ir::hir::HirModuleAccess<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        if acc.member.is_empty() {
            return None;
        }
        let cur = self.context.current_module_idx;
        let (decl, owner) = {
            let dg = self.context.dep_graph.borrow();
            let module = acc
                .path
                .last()
                .and_then(|l| {
                    self.imports_by_module
                        .get(&cur)?
                        .module_aliases
                        .get(l)
                        .copied()
                })
                .or_else(|| dg.resolve_module_path(acc.path))?;
            let direct = self
                .module_consts
                .get(&module)
                .and_then(|m| m.get(&acc.member))
                .copied()
                .map(|d| (d, module));
            direct.or_else(|| {
                let real = dg.canonical_member_module(module, acc.member);
                self.module_consts
                    .get(&real)?
                    .get(&acc.member)
                    .copied()
                    .map(|d| (d, real))
            })?
        };
        self.recover(
            self.check_visibility(
                decl.visibility,
                decl.declaring_module_idx,
                "const",
                decl.name.as_str(),
            ),
            (),
        );
        if let Some(span) = Self::field_ident_span(acc.span, acc.member) {
            self.record_item_occurrence(span, acc.member, decl.ty, owner);
        }
        Some(decl.ty)
    }

    /// Integer value of a literal or module const (`3`, `N`, `N - 1`).
    pub(crate) fn const_int_value(&self, e: &HirExpr<'a, 'bump>, depth: u32) -> Option<i64> {
        if depth > 16 {
            return None;
        }
        match e {
            HirExpr::Number(n, _) => Some(*n),
            HirExpr::Ident(name, _) if !self.context.is_local_binding(name) => {
                let (decl, _) = self.lookup_module_const_with_owner(*name)?;
                self.const_int_value(&decl.value, depth + 1)
            }
            HirExpr::Cast { expr, .. } => self.const_int_value(expr, depth + 1),
            HirExpr::Binary {
                left, op, right, ..
            } => {
                let (l, r) = (
                    self.const_int_value(left, depth + 1)?,
                    self.const_int_value(right, depth + 1)?,
                );
                match op {
                    Operator::Add => l.checked_add(r),
                    Operator::Subtract => l.checked_sub(r),
                    Operator::Multiply => l.checked_mul(r),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
