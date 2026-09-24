use ir::{
    hir::{HirExpr, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::{FunctionLowerer, lowerer::fun_name};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn lower_place_addr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        span: &SourceSpan<'a>,
    ) -> (Value, SsaType) {
        match expr {
            HirExpr::Index {
                object,
                index,
                span: _,
            } => {
                if let HirExpr::Range {
                    start,
                    end,
                    inclusive,
                    span: _,
                } = index
                {
                    let obj_val = self.lower_slice_expr(object, start, end, *inclusive);
                    let slice_ty = self
                        .current_block_data
                        .value_types
                        .get(&obj_val)
                        .cloned()
                        .expect("lower_index_base_len: base value has no known type");
                    (obj_val, slice_ty)
                } else {
                    self.lower_index_addr(object, index)
                }
            }

            HirExpr::FieldAccess {
                object,
                field,
                span,
            }
            | HirExpr::Get {
                object,
                field,
                span,
            } => self.lower_field_addr(object, *field, span),

            HirExpr::Deref {
                expr: inner,
                span: deref_span,
            } => {
                let ptr = match self.narrowed_field_value(inner) {
                    Some(v) => v,
                    None => self.lower_expr(inner),
                };
                let pointee_ty = match self.current_block_data.value_types.get(&ptr).cloned() {
                    Some(SsaType::Pointer(inner_ty)) | Some(SsaType::Owned(inner_ty)) => *inner_ty,
                    other => panic!(
                        "lower_place_addr: Deref of non-pointer {:?} at span {deref_span}",
                        other
                    ),
                };
                (ptr, pointee_ty)
            }

            other => {
                if let HirExpr::Ident(name, _) = other {
                    if let Some(&cur) = self.var_map.get(name) {
                        if !self.promoted_to_stack.contains(name) {
                            if let Some(ty) = self.current_block_data.value_types.get(&cur).cloned()
                            {
                                let already_addressable = matches!(
                                    ty,
                                    SsaType::Pointer(_)
                                        | SsaType::User(_, _)
                                        | SsaType::Enum { .. }
                                        | SsaType::Slice(_)
                                        | SsaType::Owned(_)
                                        | SsaType::Tuple(_)
                                        | SsaType::Array(_, _)
                                );
                                if !already_addressable {
                                    let slot = self.current_block_data.fresh_value();
                                    self.emit(Instruction::StackAlloc {
                                        dest: slot,
                                        ty: ty.clone(),
                                        count: 1,
                                    });
                                    self.current_block_data
                                        .value_types
                                        .insert(slot, SsaType::Pointer(Box::new(ty.clone())));
                                    self.emit(Instruction::Store {
                                        ptr: Operand::Value(slot),
                                        value: Operand::Value(cur),
                                    });
                                    self.var_map.insert(*name, slot);
                                    self.promoted_to_stack.insert(*name);
                                    return (slot, ty);
                                }
                            }
                        }
                    }
                }

                let val = self.lower_expr(other);
                match self.current_block_data.value_types.get(&val).cloned() {
                    Some(SsaType::Pointer(inner_ty)) => (val, *inner_ty),

                    Some(
                        ty @ (SsaType::User(_, _)
                        | SsaType::Enum { .. }
                        | SsaType::Slice(_)
                        | SsaType::Owned(_)
                        | SsaType::Tuple(_)
                        | SsaType::Array(_, _)),
                    ) => (val, ty),

                    Some(ty) => panic!(
                        "[lower_place_addr] cannot take address of a non-pointer-backed value of type {:?}, \
                         scalar locals must be stack-allocated to be referenced, not yet implemented, span {}",
                        ty, span
                    ),
                    None => panic!("[lower_place_addr] value has no known type"),
                }
            }
        }
    }

    pub(crate) fn emit_elem_addr(
        &mut self,
        base_ptr: Value,
        idx: Value,
        elem_ty: &SsaType,
    ) -> Value {
        let elem_size = ir::layout::sizeof_ssa(elem_ty, TargetInfo { ptr_bytes: 8 })
            .expect("[emit_elem_addr] element type has no known size")
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
        addr_v
    }

    pub(crate) fn lower_field_addr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
        span: &SourceSpan<'a>,
    ) -> (Value, SsaType) {
        let obj_val = self.lower_expr_as_receiver(object);
        let obj_val = self.auto_unwrap_receiver(obj_val);

        let obj_ty = self.current_block_data.value_types.get(&obj_val).cloned();
        if let Some(ty) = &obj_ty {
            let slice_inner = match ty {
                SsaType::Slice(inner) => Some((inner, false)),
                SsaType::Owned(inner) => match inner.as_ref() {
                    SsaType::Slice(elem) => Some((elem, true)),
                    _ => None,
                },
                SsaType::Pointer(p) => match p.as_ref() {
                    SsaType::Slice(inner) => Some((inner, false)),
                    SsaType::Owned(inner) => match inner.as_ref() {
                        SsaType::Slice(elem) => Some((elem, true)),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            };

            if let Some((elem_ty, is_owned)) = slice_inner {
                let (offset, field_ty) = match field.as_str() {
                    "len" => (8, SsaType::I64),
                    "cap" if is_owned => (16, SsaType::I64),
                    "cap" => panic!("field `cap` does not exist on borrowed slice at {span}"),
                    "ptr" => (0, SsaType::Pointer(elem_ty.clone())),
                    other => panic!("unknown builtin field `{}` on slice type at {span}", other),
                };

                let addr = self.current_block_data.fresh_value();
                self.emit(Instruction::FieldAddr {
                    dest: addr,
                    base: Operand::Value(obj_val),
                    offset,
                });
                self.current_block_data
                    .value_types
                    .insert(addr, SsaType::Pointer(Box::new(field_ty.clone())));

                return (addr, field_ty);
            }
        }

        let cls_name = match self.current_block_data.value_types.get(&obj_val) {
            Some(SsaType::User(name, _)) => *name,
            Some(SsaType::Pointer(inner)) | Some(SsaType::Owned(inner)) => fun_name(inner),
            other => panic!(
                "lower_field_addr: could not determine object's struct: {:?}",
                other
            ),
        };

        let offsets = self
            .struct_field_offsets
            .get(&cls_name)
            .unwrap_or_else(|| panic!("Unknown struct {} in FieldAccess at {span}", cls_name));

        let offset = *offsets
            .get(&field)
            .unwrap_or_else(|| panic!("Unknown field {} on struct {}", field, cls_name));

        let field_ty = self
            .structs
            .get(&cls_name)
            .and_then(|hc| hc.fields.iter().find(|f| f.name == field))
            .map(|f| lower_type_hir(&f.field_type, self.enums))
            .unwrap_or(SsaType::I64);

        let addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: addr,
            base: Operand::Value(obj_val),
            offset,
        });
        self.current_block_data
            .value_types
            .insert(addr, SsaType::Pointer(Box::new(field_ty.clone())));

        (addr, field_ty)
    }
}
