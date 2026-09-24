use ir::{
    hir::{HirExpr, Operator},
    ir_conversion::lower_operator_bin,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;
use smallvec::smallvec;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn lower_tagged_nullable_eq(
        &mut self,
        nullable: Value,
        ty: &SsaType,
        rhs: &HirExpr<'a, 'bump>,
        is_eq: bool,
    ) -> Value {
        let SsaType::Nullable(inner) = ty else {
            unreachable!()
        };
        let rv = self.lower_expr_expected(rhs, inner);

        let tag = self.new_value();
        self.emit(Instruction::LoadField {
            dest: tag,
            base: Operand::Value(nullable),
            offset: 0,
        });
        self.current_block_data.value_types.insert(tag, SsaType::U8);
        let is_some = self.new_value();
        self.emit(Instruction::Binary {
            dest: is_some,
            op: BinOp::Ne,
            left: Operand::Value(tag),
            right: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(is_some, SsaType::Bool);

        // payload load is harmless when null: the slot is valid memory
        let payload = self.unwrap_known_nonnull(nullable, ty);
        let peq = self.new_value();
        self.emit(Instruction::Binary {
            dest: peq,
            op: BinOp::Eq,
            left: Operand::Value(payload),
            right: Operand::Value(rv),
        });
        self.current_block_data
            .value_types
            .insert(peq, SsaType::Bool);

        let both = self.and_conds(Some(is_some), peq);
        if is_eq {
            both
        } else {
            let v = self.new_value();
            self.emit(Instruction::Binary {
                dest: v,
                op: BinOp::Eq,
                left: Operand::Value(both),
                right: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(v, SsaType::Bool);
            v
        }
    }

    pub(super) fn lower_short_circuit_and(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let lhs = self.lower_expr(left);

        let rhs_bb = self.current_block_data.new_block();
        let false_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(lhs),
            then_bb: rhs_bb,
            else_bb: false_bb,
        });

        // RHS
        self.current_block_data.switch_to(rhs_bb);
        let rhs = self.lower_expr(right);
        self.emit(Instruction::Jump { target: merge_bb });

        // FALSE
        self.current_block_data.switch_to(false_bb);
        let false_val = self.current_block_data.fresh_value();

        self.emit(Instruction::Const {
            dest: false_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(false),
        });

        self.emit(Instruction::Jump { target: merge_bb });

        // MERGE
        self.current_block_data.switch_to(merge_bb);

        let result = self.current_block_data.fresh_value();

        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(rhs_bb, rhs), (false_bb, false_val),],
        });

        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    pub(super) fn lower_short_circuit_or(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let lhs = self.lower_expr(left);

        let true_bb = self.current_block_data.new_block();
        let rhs_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(lhs),
            then_bb: true_bb,
            else_bb: rhs_bb,
        });

        // TRUE
        self.current_block_data.switch_to(true_bb);

        let true_val = self.current_block_data.fresh_value();

        self.emit(Instruction::Const {
            dest: true_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(true),
        });

        self.emit(Instruction::Jump { target: merge_bb });

        // RHS
        self.current_block_data.switch_to(rhs_bb);

        let rhs = self.lower_expr(right);

        self.emit(Instruction::Jump { target: merge_bb });

        // MERGE
        self.current_block_data.switch_to(merge_bb);

        let result = self.current_block_data.fresh_value();

        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(true_bb, true_val), (rhs_bb, rhs),],
        });

        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    pub(crate) fn lower_null_comparison(
        &mut self,
        operand: &HirExpr<'a, 'bump>,
        is_eq: bool,
    ) -> Value {
        enum Src {
            Val(Value),
            Addr(Value),
        }
        let cmp_op = if is_eq { BinOp::Eq } else { BinOp::Ne };

        let (src, ty) = match operand {
            HirExpr::FieldAccess {
                object,
                field,
                span,
            }
            | HirExpr::Get {
                object,
                field,
                span,
            } if self.narrowed_field_value(operand).is_none() => {
                let (addr, ty) = self.lower_field_addr(object, *field, span);
                (Src::Addr(addr), ty)
            }
            _ => {
                let v = self.lower_expr(operand);
                let ty = self
                    .current_block_data
                    .value_types
                    .get(&v)
                    .cloned()
                    .unwrap_or(SsaType::I64);
                (Src::Val(v), ty)
            }
        };

        // `null == null`
        if ty == SsaType::Null {
            let v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: v,
                ty: SsaType::Bool,
                value: Operand::ConstBool(is_eq),
            });
            self.current_block_data.value_types.insert(v, SsaType::Bool);
            return v;
        }

        let pointee: Option<SsaType> = if let Some(p) = ty.nullable_pointer_repr() {
            Some(p.clone())
        } else if let SsaType::Pointer(inner) = &ty {
            Some((**inner).clone())
        } else {
            None
        };

        if let Some(pointee) = pointee {
            let val = match src {
                Src::Val(v) => v,
                Src::Addr(a) => {
                    let loaded = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest: loaded,
                        ptr: Operand::Value(a),
                    });
                    self.current_block_data
                        .value_types
                        .insert(loaded, ty.clone());
                    loaded
                }
            };
            let ptr_ty = SsaType::Pointer(Box::new(pointee));
            let zero = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: zero,
                ty: ptr_ty.clone(),
                value: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(zero, ptr_ty);

            let cmp = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cmp,
                op: cmp_op,
                left: Operand::Value(val),
                right: Operand::Value(zero),
            });
            self.current_block_data
                .value_types
                .insert(cmp, SsaType::Bool);
            return cmp;
        }

        if ty.is_tagged_nullable() {
            let base = match src {
                Src::Val(v) | Src::Addr(v) => v,
            };
            let tag = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: tag,
                base: Operand::Value(base),
                offset: 0,
            });
            self.current_block_data.value_types.insert(tag, SsaType::U8);

            let cmp = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cmp,
                op: cmp_op,
                left: Operand::Value(tag),
                right: Operand::ConstInt(0),
            });
            self.current_block_data
                .value_types
                .insert(cmp, SsaType::Bool);
            return cmp;
        }

        // Legacy fallback: compare the raw value against 0.
        let val = match src {
            Src::Val(v) => v,
            Src::Addr(a) => {
                let loaded = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest: loaded,
                    ptr: Operand::Value(a),
                });
                self.current_block_data
                    .value_types
                    .insert(loaded, ty.clone());
                loaded
            }
        };
        let zero = self.lower_expr_null();
        let cmp = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: cmp,
            op: cmp_op,
            left: Operand::Value(val),
            right: Operand::Value(zero),
        });
        self.current_block_data
            .value_types
            .insert(cmp, SsaType::Bool);
        cmp
    }

    pub(crate) fn lower_comparison_expr(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        if matches!(op, Operator::Equals | Operator::NotEquals) {
            let (l_expr, r_expr): (&HirExpr<'a, 'bump>, &HirExpr<'a, 'bump>) = (left, right);
            let null_other = match (l_expr, r_expr) {
                (HirExpr::Null(_), HirExpr::Null(_)) => None,
                (o, HirExpr::Null(_)) | (HirExpr::Null(_), o) => Some(o),
                _ => None,
            };
            if let Some(other) = null_other {
                return self.lower_null_comparison(other, matches!(op, Operator::Equals));
            }
        }

        let l = self.lower_expr(left);
        let l_ty = self
            .current_block_data
            .value_types
            .get(&l)
            .cloned()
            .unwrap_or(SsaType::I64);
        if matches!(op, Operator::Equals | Operator::NotEquals) && l_ty.is_tagged_nullable() {
            return self.lower_tagged_nullable_eq(l, &l_ty, right, matches!(op, Operator::Equals));
        }
        let r = self.lower_expr_expected(right, &l_ty);
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: v,
            op: lower_operator_bin(&op),
            left: Operand::Value(l),
            right: Operand::Value(r),
        });
        self.current_block_data.value_types.insert(v, SsaType::Bool);
        v
    }
}
