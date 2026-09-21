use ir::{
    hir::{HirExpr, StrId},
    layout::TargetInfo,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::{FunctionLowerer, lowerer::SlicePrimitive};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn lower_slice_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        start: &HirExpr<'a, 'bump>,
        end: &HirExpr<'a, 'bump>,
        inclusive: bool,
    ) -> Value {
        let (base_addr, elem_ty, len) = self.lower_index_base_len(object);
        let start_v = self.lower_expr(start);
        let end_v = self.lower_expr(end);

        // Fold `inclusive` into `end_v` up front so bound-checking and length
        // computation both just work in exclusive-end terms afterward.
        let end_v = if inclusive {
            let one = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: one,
                ty: SsaType::Usize,
                value: Operand::ConstInt(1),
            });
            self.current_block_data
                .value_types
                .insert(one, SsaType::Usize);
            let bumped = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: bumped,
                op: BinOp::Add,
                left: Operand::Value(end_v),
                right: Operand::Value(one),
            });
            self.current_block_data
                .value_types
                .insert(bumped, SsaType::Usize);
            bumped
        } else {
            end_v
        };

        self.emit_bounds_check(Operand::Value(start_v), Operand::Value(end_v), true); // start <= end
        if let Some(len_operand) = len {
            self.emit_bounds_check(Operand::Value(end_v), len_operand, true); // end <= len
        }

        let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
            .expect("slice element has no known size") as i64;

        let size_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: size_v,
            ty: SsaType::I64,
            value: Operand::ConstInt(elem_size),
        });
        self.current_block_data
            .value_types
            .insert(size_v, SsaType::I64);

        let offset_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: offset_v,
            op: BinOp::Mul,
            left: Operand::Value(start_v),
            right: Operand::Value(size_v),
        });
        self.current_block_data
            .value_types
            .insert(offset_v, SsaType::I64);

        let ptr_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: ptr_v,
            op: BinOp::Add,
            left: Operand::Value(base_addr),
            right: Operand::Value(offset_v),
        });
        self.current_block_data
            .value_types
            .insert(ptr_v, SsaType::Pointer(Box::new(elem_ty.clone())));

        let len_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: len_v,
            op: BinOp::Sub,
            left: Operand::Value(end_v),
            right: Operand::Value(start_v),
        });
        self.current_block_data
            .value_types
            .insert(len_v, SsaType::Usize);

        let fat_ptr = self.current_block_data.fresh_value();
        let fat_ty = SsaType::Tuple(vec![
            SsaType::Pointer(Box::new(elem_ty.clone())),
            SsaType::Usize,
        ]);
        self.emit(Instruction::StackAlloc {
            dest: fat_ptr,
            ty: fat_ty,
            count: 1,
        });
        self.current_block_data
            .value_types
            .insert(fat_ptr, SsaType::Slice(Box::new(elem_ty)));

        self.emit(Instruction::StoreField {
            base: Operand::Value(fat_ptr),
            offset: 0,
            value: Operand::Value(ptr_v),
        });
        self.emit(Instruction::StoreField {
            base: Operand::Value(fat_ptr),
            offset: 8,
            value: Operand::Value(len_v),
        });
        fat_ptr
    }

    pub(super) fn lower_index_base_len(
        &mut self,
        object: &HirExpr<'a, 'bump>,
    ) -> (Value, SsaType, Option<Operand>) {
        let base = self.lower_expr(object);
        self.split_indexable(base)
    }

    pub(super) fn split_indexable(&mut self, base: Value) -> (Value, SsaType, Option<Operand>) {
        let base_ty = self
            .current_block_data
            .value_types
            .get(&base)
            .cloned()
            .expect("split_indexable: base value has no known type");

        match &base_ty {
            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Slice(_)) => {
                let elem_inner = match inner.as_ref() {
                    SsaType::Slice(e) => (**e).clone(),
                    _ => unreachable!(),
                };
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new(elem_inner.clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, elem_inner, Some(Operand::Value(len_v)))
            }

            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Owned(_)) => {
                let SsaType::Owned(slice_inner) = inner.as_ref() else {
                    unreachable!()
                };
                let SsaType::Slice(elem_ty) = slice_inner.as_ref() else {
                    panic!(
                        "lower_index_base_len: Pointer(Owned(_)) base whose inner Owned isn't a Slice: {:?}",
                        inner
                    );
                };
                let elem_inner = (**elem_ty).clone();
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new(elem_inner.clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, elem_inner, Some(Operand::Value(len_v)))
            }

            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Array(_, _)) => {
                let SsaType::Array(elem, len) = inner.as_ref() else {
                    unreachable!()
                };
                (base, (**elem).clone(), Some(Operand::ConstInt(*len as i64)))
            }

            SsaType::Pointer(inner) => (base, (**inner).clone(), None),

            SsaType::Array(inner, len) => (
                base,
                (**inner).clone(),
                Some(Operand::ConstInt(*len as i64)),
            ),

            SsaType::Slice(inner) => {
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new((**inner).clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, (**inner).clone(), Some(Operand::Value(len_v)))
            }

            SsaType::Owned(inner) => match inner.as_ref() {
                SsaType::Slice(inner) => {
                    let data_ptr = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: data_ptr,
                        base: Operand::Value(base),
                        offset: 0,
                    });
                    self.current_block_data
                        .value_types
                        .insert(data_ptr, SsaType::Pointer(Box::new((**inner).clone())));
                    let len_v = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: len_v,
                        base: Operand::Value(base),
                        offset: 8,
                    });
                    self.current_block_data
                        .value_types
                        .insert(len_v, SsaType::Usize);
                    (data_ptr, (**inner).clone(), Some(Operand::Value(len_v)))
                }
                _ => panic!("[lower_index_base_len] cannot index into {:?}", inner),
            },

            other => panic!("[lower_index_base_len] cannot index into {:?}", other),
        }
    }

    pub(super) fn emit_bounds_check(&mut self, idx: Operand, len: Operand, inclusive_upper: bool) {
        let cond = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: cond,
            op: if inclusive_upper {
                BinOp::Le
            } else {
                BinOp::Lt
            },
            left: idx,
            right: len,
        });
        self.current_block_data
            .value_types
            .insert(cond, SsaType::Bool);

        let ok_bb = self.current_block_data.new_block();
        let panic_bb = self.current_block_data.new_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb: ok_bb,
            else_bb: panic_bb,
        });

        self.current_block_data.switch_to(panic_bb);
        let msg = self.current_block_data.fresh_value();
        let msg_str = StrId::from_static("index out of bounds");
        self.emit(Instruction::Const {
            dest: msg,
            ty: SsaType::String,
            value: Operand::ConstString(msg_str),
        });
        self.current_block_data
            .value_types
            .insert(msg, SsaType::String);
        self.emit_debug_panic(msg);

        self.current_block_data.switch_to(ok_bb);
    }

    pub(super) fn align_up(value: usize, align: usize) -> usize {
        if align == 0 {
            return value;
        }
        (value + align - 1) & !(align - 1)
    }

    pub(super) fn slice_kind(&self, ty: &SsaType) -> Option<bool> {
        match ty {
            SsaType::Pointer(inner) => self.slice_kind(inner),
            SsaType::Owned(inner) => match inner.as_ref() {
                SsaType::Slice(_) => Some(true),
                _ => self.slice_kind(inner).map(|_| true),
            },
            SsaType::Slice(_) => Some(false),
            _ => None,
        }
    }

    pub(super) fn resolve_slice_pseudo_field(
        &self,
        ty: &SsaType,
        field: StrId,
    ) -> Option<(usize, SsaType)> {
        let is_owned = self.slice_kind(ty)?;
        match self.context.resolve_string(&field) {
            "len" => Some((8, SsaType::Usize)),
            "cap" if is_owned => Some((16, SsaType::Usize)),
            _ => None,
        }
    }

    pub(super) fn lower_index_addr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        index: &HirExpr<'a, 'bump>,
    ) -> (Value, SsaType) {
        let idx = self.lower_expr(index);
        let (base_ptr, elem_ty, len) = self.lower_index_base_len(object);

        if let Some(len_operand) = len {
            self.emit_bounds_check(Operand::Value(idx), len_operand, false);
        }

        let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
            .expect("[lower_index_addr] element type has no known size")
            as i64;

        let size_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: size_v,
            ty: SsaType::I64,
            value: Operand::ConstInt(elem_size),
        });
        self.current_block_data
            .value_types
            .insert(size_v, SsaType::I64);

        let offset_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: offset_v,
            op: BinOp::Mul,
            left: Operand::Value(idx),
            right: Operand::Value(size_v),
        });
        self.current_block_data
            .value_types
            .insert(offset_v, SsaType::I64);

        let addr_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: addr_v,
            op: BinOp::Add,
            left: Operand::Value(base_ptr),
            right: Operand::Value(offset_v),
        });
        self.current_block_data
            .value_types
            .insert(addr_v, SsaType::Pointer(Box::new(elem_ty.clone())));

        (addr_v, elem_ty)
    }

    pub(super) fn slice_primitive_of(&self, field: StrId) -> Option<SlicePrimitive> {
        match self.context.resolve_string(&field) {
            "write_uninit" => Some(SlicePrimitive::WriteUninit),
            "write_uninit_all" => Some(SlicePrimitive::WriteUninitAll),
            "get_unchecked" => Some(SlicePrimitive::GetUnchecked),
            _ => None,
        }
    }

    pub(super) fn lower_slice_primitive(
        &mut self,
        prim: SlicePrimitive,
        object: &HirExpr<'a, 'bump>,
        obj_val: Value,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        let (base_ptr, elem_ty, _len) = self.split_indexable(obj_val);

        match prim {
            SlicePrimitive::GetUnchecked => {
                let idx = self.lower_expr_expected(&args[0], &SsaType::Usize);
                let addr = self.emit_elem_addr(base_ptr, idx, &elem_ty);
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest,
                    ptr: Operand::Value(addr),
                });
                self.current_block_data.value_types.insert(dest, elem_ty);
                dest
            }

            SlicePrimitive::WriteUninit => {
                if Self::is_move_by_value(&elem_ty) {
                    self.record_move_if_any(&args[1]);
                }
                let idx = self.lower_expr_expected(&args[0], &SsaType::Usize);
                let val = self.lower_expr_expected(&args[1], &elem_ty);
                let addr = self.emit_elem_addr(base_ptr, idx, &elem_ty);
                self.emit(Instruction::Store {
                    ptr: Operand::Value(addr),
                    value: Operand::Value(val),
                });
                if let (HirExpr::Ident(root, _), HirExpr::Number(n, _)) = (object, &args[0]) {
                    self.drop_state.mark_index_initialized(*root, *n);
                }
                self.unit_value()
            }

            SlicePrimitive::WriteUninitAll => {
                let src_val = self.lower_expr(&args[0]);
                let (src_ptr, _src_elem, src_len) = self.split_indexable(src_val);
                let len_op = src_len.expect("write_uninit_all: source has no length");

                let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
                    .expect("write_uninit_all: element type has no known size")
                    as i64;

                let bytes = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: bytes,
                    op: BinOp::Mul,
                    left: len_op,
                    right: Operand::ConstInt(elem_size),
                });
                self.current_block_data
                    .value_types
                    .insert(bytes, SsaType::Usize);

                self.emit_memcpy(base_ptr, src_ptr, bytes);
                self.unit_value()
            }
        }
    }
}
