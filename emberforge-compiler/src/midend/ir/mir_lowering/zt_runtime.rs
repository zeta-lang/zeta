use ir::{
    hir::StrId,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;
use smallvec::smallvec;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    /// Emits a call to the runtime's `__zeta_memset(ptr, value, size)`.
    pub(super) fn emit_memset(&mut self, ptr: Value, value: i64, size: usize) {
        let memset_fn = StrId(self.context.intern("__zeta_memset"));

        let val_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: val_v,
            ty: SsaType::I32,
            value: Operand::ConstInt(value),
        });
        self.current_block_data
            .value_types
            .insert(val_v, SsaType::I32);

        let size_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: size_v,
            ty: SsaType::Usize,
            value: Operand::ConstInt(size as i64),
        });
        self.current_block_data
            .value_types
            .insert(size_v, SsaType::Usize);

        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(memset_fn),
            args: smallvec![
                Operand::Value(ptr),
                Operand::Value(val_v),
                Operand::Value(size_v),
            ],
        });
    }

    pub(super) fn emit_debug_panic(&mut self, msg: Value) {
        let panic_fn = StrId::from_static("zeta_debug_debug_panic");
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(panic_fn),
            args: smallvec![Operand::Value(msg)],
        });
        self.emit(Instruction::Ret { value: None });
    }

    /// `__zeta_memcpy(dst, src, size_bytes)`
    pub(super) fn emit_memcpy(&mut self, dst: Value, src: Value, size: Value) {
        let f = StrId(self.context.intern("__zeta_memcpy"));
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(f),
            args: smallvec![
                Operand::Value(dst),
                Operand::Value(src),
                Operand::Value(size)
            ],
        });
    }
}
