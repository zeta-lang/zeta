use ir::{
    hir::{AssignmentOperator, HirExpr, HirType, IntrinsicKind, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value, cast_kind},
};
use smallvec::SmallVec;
use smallvec::smallvec;

use crate::midend::ir::mir_lowering::FunctionLowerer;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
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

        let inline = matches!(
            ty,
            SsaType::User(..)
                | SsaType::Enum { .. }
                | SsaType::Tuple(_)
                | SsaType::Array(..)
                | SsaType::Slice(_)
        ) || matches!(&ty, SsaType::Owned(i) if matches!(i.as_ref(), SsaType::Slice(_)))
            || ty.is_tagged_nullable();

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
                let target_ty = lower_type_hir(&type_args[0], self.enums);

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
                let msg_str = self
                    .context
                    .thread_local()
                    .intern("entered unreachable code");
                self.emit(Instruction::Const {
                    dest: msg,
                    ty: SsaType::String,
                    value: Operand::ConstString(StrId(msg_str)),
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
                let query_ty = lower_type_hir(&type_args[0], self.enums);
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
                let msg_str = self
                    .context
                    .thread_local()
                    .intern("alignment assertion failed");
                self.emit(Instruction::Const {
                    dest: msg,
                    ty: SsaType::String,
                    value: Operand::ConstString(StrId(msg_str)),
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

            IntrinsicKind::Own => {
                let ptr_val = self.lower_expr(&args[0]);
                let ptr_ty = self
                    .current_block_data
                    .value_types
                    .get(&ptr_val)
                    .cloned()
                    .expect("$own: pointer arg has no known type");
                let pointee_ty = match &ptr_ty {
                    SsaType::Pointer(inner) => (**inner).clone(),
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
                            SsaType::Pointer(Box::new(pointee_ty.clone())),
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
                        self.current_block_data
                            .value_types
                            .insert(ptr_val, owned_ty);
                        ptr_val
                    }
                }
            }
            IntrinsicKind::AtomicCasU32 => {
                let ptr_val = self.lower_expr(&args[0]);
                let expected_val = self.lower_expr_as_u32(&args[1]);
                let new_val = self.lower_expr_as_u32(&args[2]);

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Intrinsic {
                    dest: Some(dest),
                    op: IntrinsicOp::AtomicCasU32,
                    query_ty: None,
                    args: smallvec![
                        Operand::Value(ptr_val),
                        Operand::Value(expected_val),
                        Operand::Value(new_val)
                    ],
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::U32);
                dest
            }

            IntrinsicKind::AtomicLoadU32 => {
                let ptr_val = self.lower_expr(&args[0]);

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Intrinsic {
                    dest: Some(dest),
                    op: IntrinsicOp::AtomicLoadU32,
                    query_ty: None,
                    args: smallvec![Operand::Value(ptr_val)],
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::U32);
                dest
            }
            IntrinsicKind::AtomicStoreU32 => {
                let ptr_val = self.lower_expr(&args[0]);
                let val_val = self.lower_expr_as_u32(&args[1]);

                self.emit(Instruction::Intrinsic {
                    dest: None,
                    op: IntrinsicOp::AtomicStoreU32,
                    query_ty: None,
                    args: smallvec![Operand::Value(ptr_val), Operand::Value(val_val)],
                });

                let dest = self.current_block_data.fresh_value();
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Void);
                dest
            }

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
}
