use ir::{
    borrow_checker::BorrowKind,
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirType, IntrinsicKind, ProvenanceRoot, RefKind},
};

use crate::{naming::type_to_string, str_id_to_string, TypeChecker};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn check_intrinsic_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        kind: &IntrinsicKind,
        type_args: &&[HirType<'a, 'bump>],
        args: &&[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        match kind {
            IntrinsicKind::AssumeInit => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$assume_init takes no type arguments".into(),
                    ));
                }
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                    "$assume_init requires an unsafe block: marking a place initialized that is not \
                     is undefined behavior if it is later read"
                        .into(),
                ));
                }
                if args.len() != 1 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 1,
                        found_args: args.len(),
                    });
                    return HirType::Void;
                }
                if !matches!(
                    &args[0],
                    HirExpr::Ident(..)
                        | HirExpr::FieldAccess { .. }
                        | HirExpr::Get { .. }
                        | HirExpr::Index { .. }
                        | HirExpr::Slice { .. }
                ) {
                    self.record(TypeErrorKind::Generic(
                    "$assume_init's argument must be a place (variable, field, index, or slice)".into(),
                ));
                    return HirType::Void;
                }
                // Suppressed so naming the uninit place isn't itself an error. Not a move: no value-use.
                let _ = self.check_expr_suppressed(&args[0]);
                self.optimistically_mark_mut_target_init(&args[0]);
                HirType::Void
            }
            IntrinsicKind::FnPtr => {
                let t = self.check_expr(&args[0]);
                let HirType::Struct { name, .. } = t else {
                    self.record(TypeErrorKind::Generic("$fn_ptr expects a closure".into()));
                    return HirType::Unknown;
                };
                let Some(lowering) = self.closure_table.values().find(|c| c.env_name == name)
                else {
                    self.record(TypeErrorKind::Generic("$fn_ptr expects a closure".into()));
                    return HirType::Unknown;
                };
                if !lowering.captures.is_empty() {
                    let desc = self.describe_path(
                        lowering.captures[0].source,
                        lowering.captures[0].source_path,
                    );
                    self.record(TypeErrorKind::Generic(format!(
                        "`$fn_ptr` requires a non-capturing closure, but it captures `{}`",
                        desc
                    )));
                    return HirType::Unknown;
                }
                HirType::Lambda {
                    params: self.context.bump.alloc_slice(&lowering.param_tys),
                    return_type: self.context.bump.alloc_value(lowering.ret_ty),
                }
            }
            IntrinsicKind::Leak => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$leak takes no type arguments".to_string(),
                    ));
                }
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "$leak requires an unsafe block: it gives up ownership without running \
                     drop glue or freeing, so the caller becomes responsible for the allocation"
                            .to_string(),
                    ));
                }
                if args.len() != 1 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 1,
                        found_args: args.len(),
                    });
                    return HirType::Unknown;
                }

                let arg_ty = self.check_expr(&args[0]);
                // Records the move, so the source local is not dropped at scope exit.
                self.check_and_record_value_use(&args[0], &arg_ty);

                match &arg_ty {
                    HirType::OwnedPointer { inner, .. } => match **inner {
                        // ^[T] -> [T]
                        HirType::Slice(elem) => HirType::Slice(elem),
                        // ^T -> [*]mut T
                        pointee => HirType::UnsafePointer {
                            inner: self.context.bump.alloc_value(pointee),
                            mutability_state: ir::ast::MutabilityState::Mut,
                        },
                    },
                    other => {
                        self.record(TypeErrorKind::Generic(format!(
                            "$leak expects an owned pointer `^T`, found `{}` type {other:?}",
                            type_to_string(&arg_ty)
                        )));
                        HirType::Unknown
                    }
                }
            }
            IntrinsicKind::MemForget => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$mem_forget takes no type arguments".to_string(),
                    ));
                }
                if args.len() != 1 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 1,
                        found_args: args.len(),
                    });
                    return HirType::Void;
                }
                let ty = self.check_expr(&args[0]);
                // Records the move, so the source is neither used again nor dropped at scope exit.
                self.check_and_record_value_use(&args[0], &ty);
                HirType::Void
            }
            IntrinsicKind::Replace => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$replace takes no type arguments".to_string(),
                    ));
                }
                if args.len() != 2 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 2,
                        found_args: args.len(),
                    });
                    return HirType::Unknown;
                }

                let place_expr: &HirExpr<'a, 'bump> = match &args[0] {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                };
                if !matches!(
                    place_expr,
                    HirExpr::Ident(..)
                        | HirExpr::FieldAccess { .. }
                        | HirExpr::Get { .. }
                        | HirExpr::Deref { .. }
                        | HirExpr::Index { .. }
                ) {
                    self.record(TypeErrorKind::Generic(
                        "$replace's first argument must be a place (variable, field, deref, or index)"
                            .to_string(),
                    ));
                    return HirType::Unknown;
                }
                if matches!(&args[1], HirExpr::Uninit { .. }) {
                    self.record(TypeErrorKind::Generic(
                        "$replace cannot write `uninit`: the slot must always hold a valid value"
                            .to_string(),
                    ));
                    return HirType::Unknown;
                }
                if let HirExpr::FieldAccess { object, field, .. }
                | HirExpr::Get { object, field, .. } = place_expr
                {
                    let f = field.as_str();
                    if (f == "len" || f == "cap")
                        && Self::slice_field_owned(&self.peek_type(object)).is_some()
                    {
                        self.record(TypeErrorKind::Generic(
                            "$replace cannot target a slice's `.len`/`.cap`".to_string(),
                        ));
                        return HirType::Unknown;
                    }
                }

                let target_ty = self.check_expr_as_place(place_expr);
                // The old value is read out, so the slot must be initialized.
                self.check_read_for_compound_target(place_expr, &target_ty);

                let value_ty = self.check_expr_expected(&args[1], &target_ty);
                self.check_and_record_value_use(&args[1], &value_ty); // moves the new value in
                self.recover(self.types_compatible(&target_ty, &value_ty), ());

                if let HirExpr::Ident(name, _) = place_expr {
                    let var_name = str_id_to_string(*name);
                    if self.context.is_local_binding(&var_name)
                        && !self.context.is_mutable(&var_name)
                    {
                        self.record(TypeErrorKind::Generic(format!(
                            "cannot $replace `{}`: it is not declared `mut`",
                            var_name
                        )));
                    }
                }
                if let Some(place) = self.resolve_place(place_expr) {
                    self.check_borrow_use(place_expr, place, BorrowKind::Mutable);
                }

                if let Some((root, path)) = self.static_field_path(place_expr) {
                    if matches!(value_ty, HirType::Nullable(_) | HirType::Null) {
                        self.clear_non_null(root, &path);
                    } else {
                        self.mark_non_null(root, &path);
                    }
                    self.mark_field_init(root, &path);
                }

                target_ty // the old value, moved out to the caller
            }
            IntrinsicKind::Reinterpret => {
                if type_args.len() != 1 {
                    self.record(TypeErrorKind::Generic(format!(
                        "$reinterpret expects exactly 1 type argument, found {}",
                        type_args.len()
                    )));
                    return HirType::Unknown;
                }
                if args.len() != 1 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 1,
                        found_args: args.len(),
                    });
                    return HirType::Unknown;
                }
                let target_ty = type_args[0];
                let source_ty = self.check_expr(&args[0]);
                self.check_and_record_value_use(&args[0], &source_ty);
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "$reinterpret requires an unsafe block: it reinterprets a value's bit \
                     representation as a different type with no conversion or validation"
                            .to_string(),
                    ));
                }
                target_ty
            }
            IntrinsicKind::Unreachable => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$unreachable takes no type arguments".to_string(),
                    ));
                }
                if !args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$unreachable takes no value arguments".to_string(),
                    ));
                }
                HirType::Never
            }
            IntrinsicKind::SizeOf | IntrinsicKind::AlignOf | IntrinsicKind::TypeName => {
                if type_args.len() != 1 {
                    self.record(TypeErrorKind::Generic(format!(
                        "intrinsic expects exactly 1 type argument, found {}",
                        type_args.len()
                    )));
                }
                if !args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "this intrinsic takes no value arguments".to_string(),
                    ));
                }
                match kind {
                    IntrinsicKind::SizeOf | IntrinsicKind::AlignOf => HirType::Usize,
                    IntrinsicKind::TypeName => HirType::String,
                    _ => unreachable!(),
                }
            }
            IntrinsicKind::Own => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$own takes no type arguments".to_string(),
                    ));
                }
                if args.is_empty() || args.len() > 4 {
                    self.record(TypeErrorKind::Generic(format!(
                        "$own expects 1 argument (ptr), 2 (ptr, allocator or len), 3 (ptr, allocator, len), \
                         or 4 (ptr, allocator, len, cap) for owned slices, found {}",
                         args.len()
                    )));
                    return HirType::Unknown;
                }

                let ptr_ty = self.check_expr(&args[0]);
                self.check_and_record_value_use(&args[0], &ptr_ty);
                let pointee = match Self::strip_ref(&ptr_ty) {
                    HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. } => {
                        *inner
                    }
                    _ => {
                        self.record(TypeErrorKind::Generic(format!(
                            "$own expects a `*T` or `[*]T`, found `{}`",
                            type_to_string(&ptr_ty)
                        )));
                        return HirType::Unknown;
                    }
                };

                let (alloc_arg, len_arg, cap_arg) = if args.len() == 4 {
                    (Some(&args[1]), Some(&args[2]), Some(&args[3]))
                } else if args.len() == 3 {
                    (Some(&args[1]), Some(&args[2]), None)
                } else if args.len() == 2 {
                    let arg1_ty = self.check_expr(&args[1]);
                    if self.is_integer(&arg1_ty) {
                        (None, Some(&args[1]), None)
                    } else {
                        (Some(&args[1]), None, None)
                    }
                } else {
                    (None, None, None)
                };

                let allocator = if let Some(alloc_expr) = alloc_arg {
                    let alloc_ty = self.check_expr(alloc_expr);
                    let Some(allocator) = self.infer_provenance(alloc_expr) else {
                        self.record(TypeErrorKind::Generic(
                            "$own's allocator argument must be a place with stable provenance (a global, `this`, a param, or a projection through them)".to_string(),
                        ));
                        return HirType::Unknown;
                    };

                    let satisfies_allocator = match Self::strip_ref(&alloc_ty) {
                        HirType::Struct { name, .. } => {
                            let struct_name = str_id_to_string(*name);
                            self.context.struct_implements(&struct_name, "RawAllocator")
                                || self.context.struct_implements(&struct_name, "Allocator")
                        }
                        HirType::Generic(p) => self
                            .generic_bounds
                            .get(p)
                            .is_some_and(|names| names.iter().any(|n| n.ends_with("Allocator"))),
                        _ => false,
                    };

                    if !satisfies_allocator {
                        self.record(TypeErrorKind::Generic(format!(
                            "$own's allocator argument must be a struct or generic type parameter implementing \
                             `RawAllocator` or `Allocator`, found `{}`",
                            type_to_string(&alloc_ty)
                        )));
                        return HirType::Unknown;
                    }

                    if !matches!(allocator.root, ProvenanceRoot::Global { .. }) {
                        if let Some(place) = self.resolve_place(alloc_expr) {
                            match self.borrow_checker.borrow_shared(place) {
                                Ok(loan_id) => {
                                    self.call_loans.insert(Self::expr_key(expr), loan_id);
                                }
                                Err(e) => {
                                    let msg = self.describe_borrow_error(&e, Some(&allocator));
                                    self.record(TypeErrorKind::Generic(msg));
                                }
                            }
                        }
                    }
                    allocator
                } else {
                    let this_expr = HirExpr::This {
                        span: self.current_span,
                    };
                    let Some(allocator) = self.infer_provenance(&this_expr) else {
                        self.record(TypeErrorKind::Generic(
                            "$own without explicit allocator requires a `this` allocator in scope"
                                .to_string(),
                        ));
                        return HirType::Unknown;
                    };
                    allocator
                };

                let result_inner = if let Some(cap_expr) = cap_arg {
                    let len_expr = len_arg
                        .expect("cap_arg implies len_arg due to the 4-arg-only branch above");
                    let len_ty = self.check_expr(len_expr);
                    self.check_and_record_value_use(len_expr, &len_ty);
                    if !self.is_integer(&len_ty) {
                        self.record(TypeErrorKind::Generic(format!(
                            "$own's len argument must be an integer, found `{}`",
                            type_to_string(&len_ty)
                        )));
                    }
                    let cap_ty = self.check_expr(cap_expr);
                    self.check_and_record_value_use(cap_expr, &cap_ty);
                    if !self.is_integer(&cap_ty) {
                        self.record(TypeErrorKind::Generic(format!(
                            "$own's cap argument must be an integer, found `{}`",
                            type_to_string(&cap_ty)
                        )));
                    }
                    HirType::Slice(self.context.bump.alloc_value(pointee))
                } else if let Some(len_expr) = len_arg {
                    let len_ty = self.check_expr(len_expr);
                    self.check_and_record_value_use(len_expr, &len_ty);
                    if !self.is_integer(&len_ty) {
                        self.record(TypeErrorKind::Generic(format!(
                            "$own's argument must be an integer, found `{}`",
                            type_to_string(&len_ty)
                        )));
                    }
                    self.record(TypeErrorKind::Generic(
                        "$own for an owned slice requires both `len` and `cap`: use \
                         `$own(ptr, allocator, len, cap)`"
                            .to_string(),
                    ));
                    HirType::Slice(self.context.bump.alloc_value(pointee))
                } else {
                    *pointee
                };

                HirType::OwnedPointer {
                    inner: self.context.bump.alloc_value(result_inner),
                    allocator: Some(allocator),
                }
            }
            IntrinsicKind::AssertAlign => {
                if !type_args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$assert_align takes no type arguments".to_string(),
                    ));
                }
                if args.len() != 2 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 2,
                        found_args: args.len(),
                    });
                } else {
                    let ptr_ty = self.check_expr(&args[0]);
                    self.check_and_record_value_use(&args[0], &ptr_ty);
                    if !matches!(
                        Self::strip_ref(&ptr_ty),
                        HirType::SafePointer { .. }
                            | HirType::UnsafePointer { .. }
                            | HirType::OwnedPointer { .. }
                    ) {
                        self.record(TypeErrorKind::Generic(format!(
                            "$assert_align expects a pointer, found `{}`",
                            type_to_string(&ptr_ty)
                        )));
                    }

                    let align_ty = self.check_expr(&args[1]);
                    if !self.is_integer(&align_ty) {
                        self.record(TypeErrorKind::Generic(format!(
                            "$assert_align expects an integer alignment, found `{}`",
                            type_to_string(&align_ty)
                        )));
                    }
                    if let HirExpr::Number(n, _) = &args[1] {
                        if *n <= 0 || (*n as u64) & ((*n as u64) - 1) != 0 {
                            self.record(TypeErrorKind::Generic(format!(
                                "alignment must be a positive power of two, found {}",
                                n
                            )));
                        }
                    }
                }
                HirType::Void
            }
            IntrinsicKind::DropInPlace => {
                if type_args.len() > 1 {
                    self.record(TypeErrorKind::Generic(
                        "$drop_in_place takes at most one type argument".to_string(),
                    ));
                }
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "$drop_in_place requires an unsafe block: it runs the destructor of whatever the \
                         pointer refers to, and the caller must guarantee the value is initialized and \
                         never dropped again"
                            .to_string(),
                    ));
                }
                if args.len() != 1 {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args: 1,
                        found_args: args.len(),
                    });
                    return HirType::Void;
                }

                let ptr_ty = self.check_expr(&args[0]);
                if matches!(
                    &args[0],
                    HirExpr::Index { .. } | HirExpr::FieldAccess { .. } | HirExpr::Get { .. }
                ) && !matches!(
                    Self::strip_ref(&ptr_ty),
                    HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
                ) {
                    self.record(TypeErrorKind::Generic(
                        "$drop_in_place needs an address: write `&mut place`, not the place itself"
                            .into(),
                    ));
                    return HirType::Void;
                }

                let pointee = match &ptr_ty {
                    HirType::UnsafePointer {
                        inner,
                        mutability_state,
                    }
                    | HirType::SafePointer {
                        inner,
                        mutability_state,
                    } if matches!(mutability_state, ir::ast::MutabilityState::Mut) => **inner,
                    HirType::Ref {
                        inner,
                        ref_kind: RefKind::Unique | RefKind::Alias,
                        ..
                    } => **inner,
                    HirType::OwnedPointer { .. } => {
                        self.record(TypeErrorKind::Generic(
                            "$drop_in_place cannot take an owned pointer `^T`: its scope would free it \
                             again. Pass `&mut` / `[*]mut` to the pointee"
                                .to_string(),
                        ));
                        return HirType::Void;
                    }
                    other => {
                        self.record(TypeErrorKind::Generic(format!(
                            "$drop_in_place expects a `[*]mut T` or `&mut T`, found `{}`",
                            type_to_string(other)
                        )));
                        return HirType::Void;
                    }
                };
                if let Some(t) = type_args.first() {
                    self.recover(self.types_compatible(t, &pointee), ());
                }
                HirType::Void
            }

            IntrinsicKind::CpuRelax => {
                if !args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$cpu_relax takes no arguments".to_string(),
                    ));
                }
                HirType::Void
            }

            IntrinsicKind::AtomicLoad => self
                .check_atomic_intrinsic("$atomic_load", type_args, args, 0, 1, false)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicStore => {
                self.check_atomic_intrinsic("$atomic_store", type_args, args, 1, 1, false);
                HirType::Void
            }
            IntrinsicKind::AtomicSwap => self
                .check_atomic_intrinsic("$atomic_swap", type_args, args, 1, 1, false)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicCas => self
                .check_atomic_intrinsic("$atomic_cas", type_args, args, 2, 2, false)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFetchAdd => self
                .check_atomic_intrinsic("$atomic_fetch_add", type_args, args, 1, 1, true)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFetchSub => self
                .check_atomic_intrinsic("$atomic_fetch_sub", type_args, args, 1, 1, true)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFetchAnd => self
                .check_atomic_intrinsic("$atomic_fetch_and", type_args, args, 1, 1, true)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFetchOr => self
                .check_atomic_intrinsic("$atomic_fetch_or", type_args, args, 1, 1, true)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFetchXor => self
                .check_atomic_intrinsic("$atomic_fetch_xor", type_args, args, 1, 1, true)
                .unwrap_or(HirType::Unknown),
            IntrinsicKind::AtomicFence => {
                if !type_args.is_empty() || !args.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "$atomic_fence takes no arguments".to_string(),
                    ));
                }
                HirType::Void
            }
        }
    }

    fn check_atomic_intrinsic(
        &mut self,
        name: &str,
        type_args: &[HirType<'a, 'bump>],
        args: &[HirExpr<'a, 'bump>],
        value_args: usize,
        ordering_args: usize,
        integer_only: bool,
    ) -> Option<HirType<'a, 'bump>> {
        if type_args.len() != 1 {
            self.record(TypeErrorKind::Generic(format!(
                "{name} expects exactly 1 type argument, found {}",
                type_args.len()
            )));
            return None;
        }
        if args.len() != 1 + value_args + ordering_args {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args: 1 + value_args,
                found_args: args.len(),
            });
            return None;
        }

        let t = type_args[0];
        let ok = matches!(t, HirType::Generic(_))
            || self.is_integer(&t)
            || (!integer_only
                && matches!(
                    t,
                    HirType::Boolean | HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
                ));
        if !ok {
            self.record(TypeErrorKind::Generic(format!(
                "{name} requires {}, found `{}`",
                if integer_only {
                    "an integer type"
                } else {
                    "an integer, `bool` or pointer type"
                },
                type_to_string(&t)
            )));
            return None;
        }

        let mutates = value_args > 0;
        let ptr_ty = self.check_expr(&args[0]);
        self.check_and_record_value_use(&args[0], &ptr_ty);
        let pointee = match &ptr_ty {
            HirType::Ref {
                inner, ref_kind, ..
            } => {
                if mutates && *ref_kind != RefKind::Alias {
                    self.record(TypeErrorKind::Generic(format!(
                        "{name} mutates through its pointer: pass `&alias place` \
                         (not `&mut`/`&`; atomics are shared mutation)"
                    )));
                    return Some(t); // keep the result type so callers don't cascade
                }
                **inner
            }
            HirType::SafePointer {
                inner,
                mutability_state,
            }
            | HirType::UnsafePointer {
                inner,
                mutability_state,
            } => {
                if mutates && *mutability_state != ir::ast::MutabilityState::Mut {
                    self.record(TypeErrorKind::Generic(format!(
                        "{name} needs a `mut` pointer"
                    )));
                    return Some(t);
                }
                **inner
            }
            _ => {
                self.record(TypeErrorKind::Generic(format!(
                    "{name} expects `&alias`, `&`, or a raw pointer, found `{}`",
                    type_to_string(&ptr_ty)
                )));
                return Some(t);
            }
        };
        self.recover(self.types_compatible(&t, &pointee), ());

        for a in &args[1 + value_args..] {
            let ot = self.check_expr(a);
            let ok = matches!(Self::strip_ref(&ot), HirType::Enum { name, .. } if name.as_str().ends_with("Ordering"))
                || matches!(ot, HirType::Unknown);
            if !ok {
                self.record(TypeErrorKind::Generic(format!(
                    "{name}: expected an `Ordering`, found `{}`",
                    type_to_string(&ot)
                )));
            }
        }
        Some(t)
    }
}
