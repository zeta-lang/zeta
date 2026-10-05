use ir::{
    attributes::AttrTarget,
    hir::{AssignmentOperator, DropKind, HirExpr, HirType, IntrinsicKind, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{AtomicOrdering, BinOp, Instruction, IntrinsicOp, Operand, SsaType, Value, cast_kind},
};
use smallvec::SmallVec;

use crate::midend::ir::mir_lowering::{FunctionLowerer, lowerer::OrdSel};

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
    pub(super) fn lower_replace_intrinsic(
        &mut self,
        place: &HirExpr<'a, 'bump>,
        new_expr: &HirExpr<'a, 'bump>,
        span: &SourceSpan<'a>,
    ) -> Value {
        self.record_move_if_any(new_expr);

        // Plain local: SSA rebind. Marking it moved first makes handle_ident skip
        // the drop of the old value (it now belongs to the caller) and re-init it.
        if let HirExpr::Ident(name, ident_span) = place {
            let rhs = self.lower_expr(new_expr);
            let old = self.lower_expr(place);
            self.record_move_if_any(place);
            self.handle_ident(AssignmentOperator::Assign, rhs, *name, *ident_span);
            return old;
        }

        let (addr, ty) = self.lower_place_addr(place, span);
        let init = self.lower_init_operand(new_expr, &ty);

        // replaces matches with a match expression
        let inline = Self::is_aggregate_ssa_type(&ty);

        let old = if inline {
            // Slot is overwritten in place, so snapshot the old bytes first.
            let size = ir::layout::sizeof_ssa(&ty, TargetInfo { ptr_bytes: 8 })
                .expect("$replace: slot type has no known size");
            let (alloc_ty, count) = match &ty {
                SsaType::Array(inner, len) => ((**inner).clone(), *len),
                _ => (ty.clone(), 1),
            };
            let tmp = self.new_value();
            self.emit(Instruction::StackAlloc {
                dest: tmp,
                ty: alloc_ty,
                count,
            });
            self.current_block_data.value_types.insert(tmp, ty.clone());

            let n = self.new_value();
            self.emit(Instruction::Const {
                dest: n,
                ty: SsaType::Usize,
                value: Operand::ConstInt(size as i64),
            });
            self.current_block_data
                .value_types
                .insert(n, SsaType::Usize);
            self.emit_memcpy(tmp, addr, n);
            tmp
        } else {
            let v = self.new_value();
            self.emit(Instruction::Load {
                dest: v,
                ptr: Operand::Value(addr),
            });
            self.current_block_data.value_types.insert(v, ty.clone());
            v
        };

        // Deliberately no drop of the old contents: ownership moved into `old`.
        self.store_init(addr, 0, &ty, init);

        match self.static_field_path_mir(place) {
            Some((root, path)) => self
                .narrowed_fields
                .retain(|(r, p), _| !(*r == root && p.starts_with(&path))),
            None => self.narrowed_fields.clear(),
        }
        if let HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } =
            place
        {
            let owner = match &**object {
                HirExpr::This { .. } => Some(StrId::from_static("this")),
                HirExpr::Ident(root, _) => Some(*root),
                _ => None,
            };
            if let Some(o) = owner {
                self.drop_state.mark_field_initialized(o, *field);
            }
        }

        old
    }

    pub(super) fn lower_intrinsic_expr(
        &mut self,
        kind: IntrinsicKind,
        type_args: &[HirType<'a, 'bump>],
        args: &[HirExpr<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        use ir::hir::IntrinsicKind;
        use ir::ssa_ir::IntrinsicOp;

        match kind {
            IntrinsicKind::DropInPlace => {
                let ptr = self.lower_expr(&args[0]);
                let kind = match type_args.first() {
                    Some(t) => t.drop_kind(),
                    None => {
                        let pointee = match self.value_type(ptr).cloned() {
                            Some(SsaType::Pointer(_, inner)) => *inner,
                            Some(other) => other,
                            None => SsaType::Void,
                        };
                        match pointee {
                            SsaType::User(name, _, _) => DropKind::Type(name),
                            _ => DropKind::Undroppable,
                        }
                    }
                };
                self.emit_indexed_element_drop(&kind, ptr, span);
                ptr
            }
            IntrinsicKind::FnPtr => {
                panic!("fn_ptr intrinsic is not supported");
            }
            IntrinsicKind::Leak => {
                // Mark `ctx` moved so emit_scope_drops skips it, and clear array flags if any.
                self.record_arg_move(&args[0]);

                let mut src = self.lower_expr(&args[0]);
                let mut src_ty = self.current_block_data.value_types[&src].clone();

                let dst_ty = match &src_ty {
                    // ^[T]: take the data pointer (word 0), like lower_cast_expr does
                    SsaType::Owned(inner) if matches!(inner.as_ref(), SsaType::Slice(_)) => {
                        let SsaType::Slice(elem) = inner.as_ref() else {
                            unreachable!()
                        };
                        let ptr = self.current_block_data.fresh_value();
                        self.emit(Instruction::LoadField {
                            dest: ptr,
                            base: Operand::Value(src),
                            offset: 0,
                        });
                        let ptr_ty = SsaType::Pointer(
                            ir::ssa_ir::SsaPointerKind::UnsafeMut,
                            Box::new((**elem).clone()),
                        );
                        self.current_block_data
                            .value_types
                            .insert(ptr, ptr_ty.clone());
                        src = ptr;
                        src_ty = ptr_ty.clone();
                        ptr_ty
                    }
                    SsaType::Owned(inner) => {
                        SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, inner.clone())
                    }
                    other => panic!("$leak: expected owned pointer, got {:?}", other),
                };

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Cast {
                    dest,
                    value: Operand::Value(src),
                    kind: cast_kind(&src_ty, &dst_ty),
                });
                self.current_block_data.value_types.insert(dest, dst_ty);
                dest
            }
            IntrinsicKind::Replace => {
                let place_expr: &HirExpr<'a, 'bump> = match &args[0] {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                };
                self.lower_replace_intrinsic(place_expr, &args[1], &span)
            }
            IntrinsicKind::Reinterpret => {
                let src = self.lower_expr(&args[0]);
                let src_ty = self
                    .current_block_data
                    .value_types
                    .get(&src)
                    .cloned()
                    .expect("$reinterpret: source value has no known type");
                let target_ty = lower_type_hir(&type_args[0], self.enums, self.structs);

                if src_ty == target_ty {
                    src
                } else {
                    let kind = cast_kind(&src_ty, &target_ty);
                    let dest = self.current_block_data.fresh_value();
                    self.emit(Instruction::Cast {
                        dest,
                        value: Operand::Value(src),
                        kind,
                    });
                    self.current_block_data.value_types.insert(dest, target_ty);
                    dest
                }
            }
            IntrinsicKind::Unreachable => {
                let msg = self.current_block_data.fresh_value();
                let msg_str = StrId::from_static("entered unreachable code");
                self.emit(Instruction::Const {
                    dest: msg,
                    ty: SsaType::String,
                    value: Operand::ConstString(msg_str),
                });
                self.current_block_data
                    .value_types
                    .insert(msg, SsaType::String);

                self.emit_debug_panic(msg);

                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }
            IntrinsicKind::SizeOf | IntrinsicKind::AlignOf | IntrinsicKind::TypeName => {
                let query_ty = lower_type_hir(&type_args[0], self.enums, self.structs);
                let op = match kind {
                    IntrinsicKind::SizeOf => IntrinsicOp::SizeOf,
                    IntrinsicKind::AlignOf => IntrinsicOp::AlignOf,
                    IntrinsicKind::TypeName => IntrinsicOp::TypeName,
                    _ => unreachable!(),
                };

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Intrinsic {
                    dest: Some(dest),
                    op,
                    query_ty: Some(query_ty),
                    args: SmallVec::new(),
                });

                let result_ty = match kind {
                    IntrinsicKind::SizeOf | IntrinsicKind::AlignOf => SsaType::Usize,
                    IntrinsicKind::TypeName => SsaType::String,
                    _ => unreachable!(),
                };
                self.current_block_data.value_types.insert(dest, result_ty);
                dest
            }

            IntrinsicKind::AssertAlign => {
                let ptr_val = self.lower_expr(&args[0]);
                let align_val = self.lower_expr(&args[1]);

                // mask = align - 1; misaligned if (ptr & mask) != 0
                let one = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: one,
                    ty: SsaType::Usize,
                    value: Operand::ConstInt(1),
                });
                self.current_block_data
                    .value_types
                    .insert(one, SsaType::Usize);

                let mask = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: mask,
                    op: BinOp::Sub,
                    left: Operand::Value(align_val),
                    right: Operand::Value(one),
                });
                self.current_block_data
                    .value_types
                    .insert(mask, SsaType::Usize);

                let masked = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: masked,
                    op: BinOp::BitAnd,
                    left: Operand::Value(ptr_val),
                    right: Operand::Value(mask),
                });
                self.current_block_data
                    .value_types
                    .insert(masked, SsaType::Usize);

                let zero = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: zero,
                    ty: SsaType::Usize,
                    value: Operand::ConstInt(0),
                });
                self.current_block_data
                    .value_types
                    .insert(zero, SsaType::Usize);

                let is_misaligned = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: is_misaligned,
                    op: BinOp::Ne,
                    left: Operand::Value(masked),
                    right: Operand::Value(zero),
                });
                self.current_block_data
                    .value_types
                    .insert(is_misaligned, SsaType::Bool);

                let panic_bb = self.current_block_data.new_block();
                let cont_bb = self.current_block_data.new_block();

                self.emit(Instruction::Branch {
                    cond: Operand::Value(is_misaligned),
                    then_bb: panic_bb,
                    else_bb: cont_bb,
                });

                self.current_block_data.switch_to(panic_bb);
                let msg = self.current_block_data.fresh_value();
                let msg_str = StrId::from_static("alignment assertion failed");
                self.emit(Instruction::Const {
                    dest: msg,
                    ty: SsaType::String,
                    value: Operand::ConstString(msg_str),
                });
                self.current_block_data
                    .value_types
                    .insert(msg, SsaType::String);
                self.emit_debug_panic(msg);

                self.current_block_data.switch_to(cont_bb);
                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }

            IntrinsicKind::AssumeInit => {
                if let HirExpr::FieldAccess { object, field, .. }
                | HirExpr::Get { object, field, .. } = &args[0]
                {
                    let owner = match &**object {
                        HirExpr::This { .. } => Some(StrId::from_static("this")),
                        HirExpr::Ident(root, _) => Some(*root),
                        _ => None,
                    };
                    if let Some(o) = owner {
                        self.drop_state.mark_field_initialized(o, *field);
                    }
                }
                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }

            IntrinsicKind::MemForget => {
                // Mark moved so emit_scope_drops skips it, then evaluate and discard.
                // No instruction is needed: the forgetting is the absence of a drop.
                self.record_arg_move(&args[0]);
                let _ = self.lower_expr(&args[0]);

                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }

            IntrinsicKind::Own => {
                let ptr_val = self.lower_expr(&args[0]);
                let ptr_ty = self
                    .current_block_data
                    .value_types
                    .get(&ptr_val)
                    .cloned()
                    .expect("$own: pointer arg has no known type");
                let pointee_ty = match &ptr_ty {
                    SsaType::Pointer(_, inner) => (**inner).clone(),
                    other => {
                        panic!("$own: expected pointer-typed first arg, got {:?}", other)
                    }
                };

                let len_cap_exprs = if args.len() == 4 {
                    Some((&args[2], &args[3]))
                } else {
                    None
                };

                match len_cap_exprs {
                    // Owned slice: {ptr, len, cap} fat pointer, 24 bytes.
                    Some((len_expr, cap_expr)) => {
                        let len_val = self.lower_expr(len_expr);
                        let cap_val = self.lower_expr(cap_expr);

                        let fat_ptr = self.current_block_data.fresh_value();
                        let fat_ptr_layout_ty = SsaType::Tuple(vec![
                            SsaType::Pointer(
                                ir::ssa_ir::SsaPointerKind::UnsafeMut,
                                Box::new(pointee_ty.clone()),
                            ),
                            SsaType::Usize, // len
                            SsaType::Usize, // cap
                        ]);
                        self.emit(Instruction::StackAlloc {
                            dest: fat_ptr,
                            ty: fat_ptr_layout_ty,
                            count: 1,
                        });

                        let slice_ty =
                            SsaType::Owned(Box::new(SsaType::Slice(Box::new(pointee_ty))));
                        self.current_block_data
                            .value_types
                            .insert(fat_ptr, slice_ty);

                        self.emit(Instruction::StoreField {
                            base: Operand::Value(fat_ptr),
                            offset: 0,
                            value: Operand::Value(ptr_val),
                        });
                        self.emit(Instruction::StoreField {
                            base: Operand::Value(fat_ptr),
                            offset: 8,
                            value: Operand::Value(len_val),
                        });
                        self.emit(Instruction::StoreField {
                            base: Operand::Value(fat_ptr),
                            offset: 16,
                            value: Operand::Value(cap_val),
                        });

                        fat_ptr
                    }

                    None => {
                        let owned_ty = SsaType::Owned(Box::new(pointee_ty));
                        let dest = self.current_block_data.fresh_value();
                        self.emit(Instruction::Cast {
                            dest,
                            value: Operand::Value(ptr_val),
                            kind: cast_kind(&ptr_ty, &owned_ty),
                        });
                        self.current_block_data.value_types.insert(dest, owned_ty);
                        dest
                    }
                }
            }
            IntrinsicKind::AtomicLoad
            | IntrinsicKind::AtomicStore
            | IntrinsicKind::AtomicSwap
            | IntrinsicKind::AtomicCas
            | IntrinsicKind::AtomicFetchAdd
            | IntrinsicKind::AtomicFetchSub
            | IntrinsicKind::AtomicFetchAnd
            | IntrinsicKind::AtomicFetchOr
            | IntrinsicKind::AtomicFetchXor
            | IntrinsicKind::AtomicFence => self.lower_atomic(kind, type_args, args),

            IntrinsicKind::CpuRelax => {
                self.emit(Instruction::Intrinsic {
                    dest: None,
                    op: IntrinsicOp::CpuRelax,
                    query_ty: None,
                    args: SmallVec::new(),
                });

                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }
        }
    }

    pub(super) fn lower_atomic(
        &mut self,
        kind: IntrinsicKind,
        type_args: &[HirType<'a, 'bump>],
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        // index of the (success) ordering arg, and of the failure arg for CAS
        let (ord_idx, fail_idx) = match kind {
            IntrinsicKind::AtomicLoad => (1, None),
            IntrinsicKind::AtomicStore
            | IntrinsicKind::AtomicSwap
            | IntrinsicKind::AtomicFetchAdd
            | IntrinsicKind::AtomicFetchSub
            | IntrinsicKind::AtomicFetchAnd
            | IntrinsicKind::AtomicFetchOr
            | IntrinsicKind::AtomicFetchXor => (2, None),
            IntrinsicKind::AtomicCas => (3, Some(4)),
            IntrinsicKind::AtomicFence => (0, None),
            _ => unreachable!(),
        };

        // Static if it resolves to a constant variant, otherwise lowered as a runtime value.
        let succ = self.lower_ordering_sel(&args[ord_idx]);
        let fail = fail_idx.map(|i| self.lower_ordering_sel(&args[i]));

        let mut ops: SmallVec<Operand, 4> = SmallVec::new();

        let query_ty = if kind == IntrinsicKind::AtomicFence {
            None
        } else {
            let ty = lower_type_hir(&type_args[0], self.enums, self.structs);

            let size = ir::layout::sizeof_ssa(&ty, TargetInfo { ptr_bytes: 8 })
                .expect("atomic: type has no known size");
            assert!(
                matches!(size, 1 | 2 | 4 | 8),
                "atomic: unsupported width {size}"
            );

            let ptr = self.lower_expr(&args[0]);
            ops.push(Operand::Value(ptr));

            // Ordering args are NOT operands; they're handled via succ/fail above.
            match kind {
                IntrinsicKind::AtomicLoad => {}
                IntrinsicKind::AtomicStore
                | IntrinsicKind::AtomicSwap
                | IntrinsicKind::AtomicFetchAdd
                | IntrinsicKind::AtomicFetchSub
                | IntrinsicKind::AtomicFetchAnd
                | IntrinsicKind::AtomicFetchOr
                | IntrinsicKind::AtomicFetchXor => {
                    let value = self.lower_expr_expected(&args[1], &ty);
                    ops.push(Operand::Value(value));
                }
                IntrinsicKind::AtomicCas => {
                    let expected = self.lower_expr_expected(&args[1], &ty);
                    let desired = self.lower_expr_expected(&args[2], &ty);
                    ops.push(Operand::Value(expected));
                    ops.push(Operand::Value(desired));
                }
                _ => unreachable!(),
            }

            Some(ty)
        };

        let result_ty = match (&query_ty, kind) {
            (Some(t), k)
                if Self::build_atomic_op(k, AtomicOrdering::SeqCst, AtomicOrdering::SeqCst).1 =>
            {
                t.clone()
            }
            _ => SsaType::Void,
        };
        let returns = result_ty != SsaType::Void;

        match (succ, fail) {
            (OrdSel::Static(s), None) => {
                let (s, f) = Self::check_static_ordering(kind, s, None);
                let (op, _) = Self::build_atomic_op(kind, s, f);
                let dest = self.new_value();
                self.emit(Instruction::Intrinsic {
                    dest: returns.then_some(dest),
                    op,
                    query_ty,
                    args: ops,
                });
                self.current_block_data.value_types.insert(dest, result_ty);
                dest
            }
            (OrdSel::Static(s), Some(OrdSel::Static(f))) => {
                let (s, f) = Self::check_static_ordering(kind, s, Some(f));
                let (op, _) = Self::build_atomic_op(kind, s, f);
                let dest = self.new_value();
                self.emit(Instruction::Intrinsic {
                    dest: returns.then_some(dest),
                    op,
                    query_ty,
                    args: ops,
                });
                self.current_block_data.value_types.insert(dest, result_ty);
                dest
            }
            (OrdSel::Static(s), fail @ Some(OrdSel::Dynamic(_))) => {
                let (s, _) = Self::check_static_ordering(kind, s, None);
                self.lower_atomic_dynamic(
                    kind,
                    OrdSel::Static(s),
                    fail,
                    ops,
                    query_ty,
                    result_ty,
                    returns,
                )
            }
            (succ @ OrdSel::Dynamic(_), fail) => {
                self.lower_atomic_dynamic(kind, succ, fail, ops, query_ty, result_ty, returns)
            }
        }
    }

    fn ordering_variants(&self) -> Vec<AtomicOrdering> {
        let Some(AttrTarget::Enum(ordering_enum)) =
            self.known_type_target(StrId::from_static("Ordering"))
        else {
            unreachable!()
        };
        // tag -> AtomicOrdering, from the real variant order
        self.enums[&ordering_enum]
            .variants
            .iter()
            .map(|v| Self::ordering_from_name(v.name.as_str()))
            .collect()
    }

    fn lower_atomic_dynamic(
        &mut self,
        kind: IntrinsicKind,
        succ: OrdSel,
        fail: Option<OrdSel>,
        ops: SmallVec<Operand, 4>,
        query_ty: Option<SsaType>,
        result_ty: SsaType,
        returns: bool,
    ) -> Value {
        let variants = self.ordering_variants();

        // result slot (join without phi)
        let slot = if returns {
            let s = self.new_value();
            self.emit(Instruction::StackAlloc {
                dest: s,
                ty: result_ty.clone(),
                count: 1,
            });
            self.current_block_data
                .value_types
                .insert(s, result_ty.clone());
            Some(s)
        } else {
            None
        };

        match succ {
            OrdSel::Static(s) => {
                self.dispatch_failure(kind, s, fail, &variants, &ops, &query_ty, &result_ty, slot);
            }
            OrdSel::Dynamic(sv) => {
                self.emit_ordering_dispatch(
                    sv,
                    &variants,
                    |o| Self::validate_ordering(kind, o),
                    |this, s| {
                        this.dispatch_failure(
                            kind, s, fail, &variants, &ops, &query_ty, &result_ty, slot,
                        );
                    },
                );
            }
        }

        let out = self.new_value();
        match slot {
            Some(s) => {
                self.emit(Instruction::LoadField {
                    dest: out,
                    base: Operand::Value(s),
                    offset: 0,
                });
                self.current_block_data.value_types.insert(out, result_ty);
            }
            None => {
                self.current_block_data
                    .value_types
                    .insert(out, SsaType::Void);
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_failure(
        &mut self,
        kind: IntrinsicKind,
        s: AtomicOrdering,
        fail: Option<OrdSel>,
        variants: &[AtomicOrdering],
        ops: &SmallVec<Operand, 4>,
        query_ty: &Option<SsaType>,
        result_ty: &SsaType,
        slot: Option<Value>,
    ) {
        match fail {
            None => self.emit_atomic_leaf(kind, s, s, ops, query_ty, result_ty, slot),
            Some(OrdSel::Static(f)) => {
                // `s` may only be known at runtime (we're inside a dispatch arm),
                // so an illegal pair is a runtime panic in this arm, not a compile error.
                match Self::validate_failure(s, f) {
                    Some(f) => self.emit_atomic_leaf(kind, s, f, ops, query_ty, result_ty, slot),
                    None => self.emit_atomic_bad_order(),
                }
            }
            Some(OrdSel::Dynamic(fv)) => {
                self.emit_ordering_dispatch(
                    fv,
                    variants,
                    |f| Self::validate_failure(s, f),
                    |this, f| this.emit_atomic_leaf(kind, s, f, ops, query_ty, result_ty, slot),
                );
            }
        }
    }

    /// Emit one concrete atomic intrinsic and store its result into `slot`.
    #[allow(clippy::too_many_arguments)]
    fn emit_atomic_leaf(
        &mut self,
        kind: IntrinsicKind,
        s: AtomicOrdering,
        f: AtomicOrdering,
        ops: &SmallVec<Operand, 4>,
        query_ty: &Option<SsaType>,
        result_ty: &SsaType,
        slot: Option<Value>,
    ) {
        let (op, _) = Self::build_atomic_op(kind, s, f);
        let d = self.new_value();
        self.emit(Instruction::Intrinsic {
            dest: slot.map(|_| d),
            op,
            query_ty: query_ty.clone(),
            args: ops.clone(),
        });
        if let Some(slot) = slot {
            self.current_block_data
                .value_types
                .insert(d, result_ty.clone());
            self.emit(Instruction::StoreField {
                base: Operand::Value(slot),
                offset: 0,
                value: Operand::Value(d),
            });
        }
    }

    /// Branch on the runtime `Ordering` value. `map` returns `None` for variants that
    /// are illegal for this operation; those, and any tag matching no variant, go to a
    /// block that panics. Legal variants mapping to the same ordering share one block,
    /// so `arm` runs once per distinct ordering. The join block is left as current.
    fn emit_ordering_dispatch<M, A>(
        &mut self,
        ord_val: Value,
        variants: &[AtomicOrdering],
        map: M,
        mut arm: A,
    ) where
        M: Fn(AtomicOrdering) -> Option<AtomicOrdering>,
        A: FnMut(&mut Self, AtomicOrdering),
    {
        let mut targets: Vec<AtomicOrdering> = Vec::new();
        let mut group_of: Vec<Option<usize>> = Vec::with_capacity(variants.len());
        for &v in variants {
            group_of.push(map(v).map(|m| match targets.iter().position(|t| *t == m) {
                Some(g) => g,
                None => {
                    targets.push(m);
                    targets.len() - 1
                }
            }));
        }

        // Only when every variant is legal AND they all agree is the check redundant.
        if group_of.iter().all(Option::is_some) && targets.len() == 1 {
            arm(self, targets[0]);
            return;
        }

        let tag = self.new_value();
        self.emit(Instruction::LoadField {
            dest: tag,
            base: Operand::Value(ord_val),
            offset: 0,
        });
        self.current_block_data
            .value_types
            .insert(tag, SsaType::I64);

        let arm_blocks: Vec<_> = targets
            .iter()
            .map(|_| self.current_block_data.new_block())
            .collect();
        let bad_bb = self.current_block_data.new_block();
        let join_bb = self.current_block_data.new_block();

        // Compare chain over legal variants only; the final else is the panic block.
        let legal: Vec<(usize, usize)> = group_of
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.map(|g| (i, g)))
            .collect();

        for (n, &(i, g)) in legal.iter().enumerate() {
            let else_bb = if n + 1 == legal.len() {
                bad_bb
            } else {
                self.current_block_data.new_block()
            };
            let k = self.new_value();
            self.emit(Instruction::Const {
                dest: k,
                ty: SsaType::I64,
                value: Operand::ConstInt(i as i64),
            });
            self.current_block_data.value_types.insert(k, SsaType::I64);
            let eq = self.new_value();
            self.emit(Instruction::Binary {
                dest: eq,
                op: BinOp::Eq,
                left: Operand::Value(tag),
                right: Operand::Value(k),
            });
            self.current_block_data
                .value_types
                .insert(eq, SsaType::Bool);
            self.emit(Instruction::Branch {
                cond: Operand::Value(eq),
                then_bb: arm_blocks[g],
                else_bb,
            });
            if else_bb != bad_bb {
                self.current_block_data.switch_to(else_bb);
            }
        }

        self.current_block_data.switch_to(bad_bb);
        self.emit_atomic_bad_order();

        for (g, &t) in targets.iter().enumerate() {
            self.current_block_data.switch_to(arm_blocks[g]);
            arm(self, t);
            // An arm may already have terminated (e.g. it ended in a panic).
            if !self.block_terminated() {
                self.emit(Instruction::Jump { target: join_bb });
            }
        }

        self.current_block_data.switch_to(join_bb);
    }

    fn emit_atomic_bad_order(&mut self) {
        let msg = self.new_value();
        self.emit(Instruction::Const {
            dest: msg,
            ty: SsaType::String,
            value: Operand::ConstString(StrId::from_static("invalid atomic ordering")),
        });
        self.current_block_data
            .value_types
            .insert(msg, SsaType::String);
        self.emit_debug_panic(msg); // emits the terminating Ret
    }

    fn try_static_ordering(&self, expr: &HirExpr<'a, 'bump>) -> Option<AtomicOrdering> {
        let Some(AttrTarget::Enum(ordering_enum)) =
            self.known_type_target(StrId::from_static("Ordering"))
        else {
            panic!("atomic: known type `Ordering` is not an enum");
        };
        let (enum_name, variant) = self.resolve_constant_enum_variant(expr)?;
        assert_eq!(enum_name, ordering_enum);
        Some(Self::ordering_from_name(variant.as_str()))
    }

    fn ordering_from_name(n: &str) -> AtomicOrdering {
        match n {
            "Relaxed" => AtomicOrdering::Relaxed,
            "Acquire" => AtomicOrdering::Acquire,
            "Release" => AtomicOrdering::Release,
            "AcqRel" => AtomicOrdering::AcqRel,
            "SeqCst" => AtomicOrdering::SeqCst,
            other => panic!("unknown atomic ordering variant: {other}"),
        }
    }

    fn lower_ordering_sel(&mut self, expr: &HirExpr<'a, 'bump>) -> OrdSel {
        match self.try_static_ordering(expr) {
            Some(o) => OrdSel::Static(o),
            None => OrdSel::Dynamic(self.lower_expr(expr)),
        }
    }

    fn build_atomic_op(
        kind: IntrinsicKind,
        s: AtomicOrdering,
        f: AtomicOrdering,
    ) -> (IntrinsicOp, bool) {
        match kind {
            IntrinsicKind::AtomicLoad => (IntrinsicOp::AtomicLoad { ordering: s }, true),
            IntrinsicKind::AtomicStore => (IntrinsicOp::AtomicStore { ordering: s }, false),
            IntrinsicKind::AtomicSwap => (IntrinsicOp::AtomicSwap { ordering: s }, true),
            IntrinsicKind::AtomicCas => (
                IntrinsicOp::AtomicCas {
                    success: s,
                    failure: f,
                },
                true,
            ),
            IntrinsicKind::AtomicFetchAdd => (IntrinsicOp::AtomicFetchAdd { ordering: s }, true),
            IntrinsicKind::AtomicFetchSub => (IntrinsicOp::AtomicFetchSub { ordering: s }, true),
            IntrinsicKind::AtomicFetchAnd => (IntrinsicOp::AtomicFetchAnd { ordering: s }, true),
            IntrinsicKind::AtomicFetchOr => (IntrinsicOp::AtomicFetchOr { ordering: s }, true),
            IntrinsicKind::AtomicFetchXor => (IntrinsicOp::AtomicFetchXor { ordering: s }, true),
            IntrinsicKind::AtomicFence => (IntrinsicOp::AtomicFence { ordering: s }, false),
            _ => unreachable!(),
        }
    }

    /// Legal success/single ordering for the op (mirrors the C runtime's switch cases).
    fn validate_ordering(kind: IntrinsicKind, o: AtomicOrdering) -> Option<AtomicOrdering> {
        use AtomicOrdering::*;
        let ok = match kind {
            IntrinsicKind::AtomicLoad => matches!(o, Relaxed | Acquire | SeqCst),
            IntrinsicKind::AtomicStore => matches!(o, Relaxed | Release | SeqCst),
            IntrinsicKind::AtomicFence => !matches!(o, Relaxed),
            _ => true,
        };
        ok.then_some(o)
    }

    /// Legal CAS failure ordering for a given success ordering (the table in the C file).
    fn validate_failure(s: AtomicOrdering, f: AtomicOrdering) -> Option<AtomicOrdering> {
        use AtomicOrdering::*;
        let ok = match s {
            Relaxed | Release => matches!(f, Relaxed),
            Acquire | AcqRel => matches!(f, Relaxed | Acquire),
            SeqCst => matches!(f, Relaxed | Acquire | SeqCst),
        };
        ok.then_some(f)
    }

    /// Compile-time check for orderings that are statically known.
    fn check_static_ordering(
        kind: IntrinsicKind,
        s: AtomicOrdering,
        f: Option<AtomicOrdering>,
    ) -> (AtomicOrdering, AtomicOrdering) {
        let s = Self::validate_ordering(kind, s)
            .unwrap_or_else(|| panic!("atomic: ordering {s:?} is invalid for {kind:?}"));
        match f {
            None => (s, s),
            Some(f) => {
                let f = Self::validate_failure(s, f).unwrap_or_else(|| {
                    panic!("atomic: failure ordering {f:?} is invalid with success ordering {s:?}")
                });
                (s, f)
            }
        }
    }
}
