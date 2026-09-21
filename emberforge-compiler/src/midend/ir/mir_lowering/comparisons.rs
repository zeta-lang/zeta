use ir::{
    hir::HirExpr,
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
}
