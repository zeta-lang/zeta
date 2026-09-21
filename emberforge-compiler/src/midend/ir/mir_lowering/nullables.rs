use ir::{
    hir::{HirExpr, HirType, Operator, StrId},
    ir_conversion::lower_type_hir,
    layout::{TargetInfo, alignof_ssa, round_up_to_align},
    ssa_ir::{Instruction, Operand, SsaType, Value, cast_kind},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn coerce_into_tagged_nullable(&mut self, val: Value, expected: &SsaType) -> Value {
        if !expected.is_tagged_nullable() {
            return val; // pointer-optimized nullables share bits with the pointer
        }
        match self.value_type(val) {
            Some(SsaType::Null) | Some(SsaType::Nullable(_)) | Some(SsaType::Void) | None => {
                return val;
            }
            _ => {}
        }
        let slot = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: slot,
            ty: expected.clone(),
            count: 1,
        });
        self.current_block_data
            .value_types
            .insert(slot, expected.clone());
        self.store_field_value(slot, 0, expected, val); // writes tag=1, then the payload
        slot
    }

    pub(super) fn null_compatible_with(ty: &SsaType) -> bool {
        match ty {
            SsaType::Pointer(_) => true,
            SsaType::Nullable(inner) => inner.is_pointer(),
            _ => false,
        }
    }

    pub(super) fn cond_may_narrow_null(cond: &HirExpr<'a, 'bump>) -> bool {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return false;
        };
        if !matches!(op, Operator::Equals | Operator::NotEquals) {
            return false;
        }
        let is_narrowable_side = |e: &HirExpr<'a, 'bump>| {
            matches!(
                e,
                HirExpr::Ident(_, _) | HirExpr::FieldAccess { .. } | HirExpr::Get { .. }
            )
        };
        (matches!(right, HirExpr::Null(_)) && is_narrowable_side(left))
            || (matches!(left, HirExpr::Null(_)) && is_narrowable_side(right))
    }

    pub(super) fn store_null_field(&mut self, base: Value, offset: usize, field_ty: &SsaType) {
        if field_ty.is_tagged_nullable() {
            self.store_const_u8(base, offset, 0); // null tag
            return;
        }
        // pointer-optimized nullable: null is all-zero bits
        let zero = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: zero,
            ty: SsaType::I64,
            value: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(zero, SsaType::I64);
        self.emit(Instruction::StoreField {
            base: Operand::Value(base),
            offset,
            value: Operand::Value(zero),
        });
    }

    pub(super) fn retype_nullable_ptr_bits(&mut self, val: Value, field_ty: &SsaType) -> Value {
        let Some(pointee) = field_ty.nullable_pointer_repr() else {
            return val;
        };
        let ptr_ty = SsaType::Pointer(Box::new(pointee.clone()));
        match self.value_type(val).cloned() {
            Some(SsaType::Nullable(_)) | Some(SsaType::Null) => {
                let dest = self.new_value();
                self.emit(Instruction::Cast {
                    dest,
                    value: Operand::Value(val),
                    kind: cast_kind(&ptr_ty, &ptr_ty),
                });
                self.current_block_data.value_types.insert(dest, ptr_ty);
                dest
            }
            _ => val,
        }
    }

    pub(super) fn lower_null_as(&mut self, expected: Option<HirType<'a, 'bump>>) -> Value {
        let v = self.current_block_data.fresh_value();

        match expected {
            Some(HirType::Nullable(inner)) => {
                let inner_ssa = lower_type_hir(inner, self.enums);

                if inner_ssa.is_pointer() {
                    // Pointer-optimized nullable: null is just 0.
                    self.emit(Instruction::Const {
                        dest: v,
                        ty: SsaType::I64,
                        value: Operand::ConstInt(0),
                    });
                    self.current_block_data
                        .value_types
                        .insert(v, SsaType::Nullable(Box::new(inner_ssa)));
                } else {
                    let _payload_align =
                        alignof_ssa(&inner_ssa, TargetInfo { ptr_bytes: 8 }).unwrap_or(1);
                    let tag_offset = 0usize; // tag at offset 0 per layout_of_ssa for Nullable
                    let nullable_ty = SsaType::Nullable(Box::new(inner_ssa.clone()));
                    let size = ir::layout::sizeof_ssa(&nullable_ty, TargetInfo { ptr_bytes: 8 })
                        .unwrap_or(16);

                    self.emit(Instruction::StackAlloc {
                        dest: v,
                        ty: nullable_ty.clone(),
                        count: size,
                    });

                    // Write null tag (0) at offset 0.
                    let null_tag_val = self.current_block_data.fresh_value();
                    self.emit(Instruction::Const {
                        dest: null_tag_val,
                        ty: SsaType::U8,
                        value: Operand::ConstInt(0),
                    });
                    self.emit(Instruction::StoreField {
                        base: Operand::Value(v),
                        offset: tag_offset,
                        value: Operand::Value(null_tag_val),
                    });

                    self.current_block_data.value_types.insert(v, nullable_ty);
                }
            }
            _ => {
                self.emit(Instruction::Const {
                    dest: v,
                    ty: SsaType::I64,
                    value: Operand::ConstInt(0),
                });
                self.current_block_data.value_types.insert(v, SsaType::Null);
            }
        }

        v
    }

    pub(super) fn unwrap_known_nonnull(&mut self, val: Value, ty: &SsaType) -> Value {
        if let Some(pointee) = ty.nullable_pointer_repr() {
            // Pointer-optimized nullable: the bits are already exactly the
            // pointer we want, just re-typed without the Nullable wrapper.
            let pointee = pointee.clone();
            let dest = self.current_block_data.fresh_value();
            let ptr_ty = SsaType::Pointer(Box::new(pointee));
            self.emit(Instruction::Cast {
                dest,
                value: Operand::Value(val),
                kind: cast_kind(&ptr_ty, &ptr_ty),
            });
            self.current_block_data.value_types.insert(dest, ptr_ty);
            dest
        } else if let SsaType::Nullable(inner) = ty {
            // Tag-based nullable: load the payload out from behind the tag byte.
            let payload_align =
                alignof_ssa(inner, TargetInfo { ptr_bytes: 8 }).unwrap_or_else(|e| {
                    panic!("failed to compute alignment for nullable payload: {:?}", e)
                });
            let payload_offset = round_up_to_align(1, payload_align);

            let payload = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: payload,
                base: Operand::Value(val),
                offset: payload_offset,
            });
            self.current_block_data
                .value_types
                .insert(payload, (**inner).clone());
            payload
        } else {
            val
        }
    }

    pub(super) fn narrow_nonnull(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        branch_is_true: bool,
    ) -> Option<(StrId, Value, Value)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };

        let name = match (left, right) {
            // `x == null` / `x != null`: narrows in whichever branch the
            // non-null case holds -- true branch for `!=`, false branch for `==`.
            (HirExpr::Ident(n, _), HirExpr::Null(_)) | (HirExpr::Null(_), HirExpr::Ident(n, _)) => {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                *n
            }
            // `x == <non-null value>` (nullable equality): only the *true*
            // branch implies non-null; `x != value` proves nothing, since
            // `x == null` also satisfies it.
            (HirExpr::Ident(n, _), other) if !matches!(other, HirExpr::Null(_)) => {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                *n
            }
            (other, HirExpr::Ident(n, _)) if !matches!(other, HirExpr::Null(_)) => {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                *n
            }
            _ => return None,
        };

        let cur = *self.var_map.get(&name)?;
        let ty = self.current_block_data.value_type(cur)?.clone();
        if !matches!(ty, SsaType::Nullable(_)) {
            return None;
        }
        let unwrapped = self.unwrap_known_nonnull(cur, &ty);
        self.var_map.insert(name, unwrapped);
        Some((name, cur, unwrapped))
    }

    pub(super) fn narrow_nonnull_path(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        branch_is_true: bool,
    ) -> Option<(StrId, Vec<StrId>)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };

        let is_field_access =
            |e: &HirExpr<'a, 'bump>| matches!(e, HirExpr::FieldAccess { .. } | HirExpr::Get { .. });

        let target: &HirExpr<'a, 'bump> =
            if matches!(right, HirExpr::Null(_)) && is_field_access(left) {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                left
            } else if matches!(left, HirExpr::Null(_)) && is_field_access(right) {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                right
            } else if is_field_access(left) && !matches!(right, HirExpr::Null(_)) {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                left
            } else if is_field_access(right) && !matches!(left, HirExpr::Null(_)) {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                right
            } else {
                return None;
            };

        let (HirExpr::FieldAccess {
            object,
            field,
            span,
        }
        | HirExpr::Get {
            object,
            field,
            span,
        }) = target
        else {
            unreachable!()
        };

        let (addr, ty) = self.lower_field_addr(object, *field, span);
        let val = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: val,
            ptr: Operand::Value(addr),
        });
        self.current_block_data.value_types.insert(val, ty.clone());
        if !matches!(ty, SsaType::Nullable(_)) {
            return None;
        }

        let unwrapped = self.unwrap_known_nonnull(val, &ty);
        let (root, mut path) = self.static_field_path_mir(object)?;
        path.push(*field);
        self.narrowed_fields.insert((root, path.clone()), unwrapped);
        Some((root, path))
    }

    /// A `null` of the given nullable type: 0 bits for pointer-optimized,
    /// a stack slot with tag = 0 for tagged nullables.
    pub(super) fn lower_null_ssa(&mut self, ty: &SsaType) -> Value {
        if ty.nullable_pointer_repr().is_none() && ty.is_tagged_nullable() {
            let slot = self.current_block_data.fresh_value();
            self.emit(Instruction::StackAlloc {
                dest: slot,
                ty: ty.clone(),
                count: 1,
            });
            self.current_block_data.value_types.insert(slot, ty.clone());
            self.store_const_u8(slot, 0, 0);
            slot
        } else {
            let v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: v,
                ty: SsaType::I64,
                value: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(v, ty.clone());
            v
        }
    }

    /// `T` -> `T?`. Pointer-optimized: the bits already are the value.
    /// Tagged: fresh slot, tag = some, payload written after the tag.
    pub(super) fn wrap_into_nullable(&mut self, val: Value, ty: &SsaType) -> Value {
        if ty.nullable_pointer_repr().is_some() {
            return val;
        }
        let slot = self.current_block_data.fresh_value();
        self.emit(Instruction::StackAlloc {
            dest: slot,
            ty: ty.clone(),
            count: 1,
        });
        self.current_block_data.value_types.insert(slot, ty.clone());
        self.store_field_value(slot, 0, ty, val);
        slot
    }

    pub(super) fn auto_unwrap_receiver(&mut self, val: Value) -> Value {
        let Some(ty) = self.current_block_data.value_types.get(&val).cloned() else {
            return val;
        };
        match &ty {
            SsaType::Nullable(inner)
                if matches!(
                    inner.as_ref(),
                    SsaType::Pointer(_) | SsaType::Owned(_) | SsaType::User(_, _)
                ) =>
            {
                self.unwrap_known_nonnull(val, &ty)
            }
            _ => val,
        }
    }
}
