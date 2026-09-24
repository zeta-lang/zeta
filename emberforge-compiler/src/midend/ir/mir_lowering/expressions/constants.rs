use ir::{
    hir::{HirExpr, Operator, StrId},
    ir_conversion::lower_operator_bin,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value, cast_kind},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn store_const_u8(&mut self, base: Value, offset: usize, v: i64) {
        let c = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: c,
            ty: SsaType::U8,
            value: Operand::ConstInt(v),
        });
        self.current_block_data.value_types.insert(c, SsaType::U8);
        self.emit(Instruction::StoreField {
            base: Operand::Value(base),
            offset,
            value: Operand::Value(c),
        });
    }

    pub(crate) fn field_addr(&mut self, base: Value, offset: usize, ty: &SsaType) -> Value {
        let a = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: a,
            base: Operand::Value(base),
            offset,
        });
        self.current_block_data
            .value_types
            .insert(a, SsaType::Pointer(Box::new(ty.clone())));
        a
    }

    pub(crate) fn lower_this_expr(&mut self) -> Value {
        let this_name = StrId::from_static("this");
        *self.var_map.get(&this_name).unwrap()
    }

    pub(crate) fn lower_expr_list_expr(&mut self, list: &[HirExpr<'a, 'bump>]) -> Value {
        if list.is_empty() {
            let v = self.current_block_data.fresh_value();
            self.emit(Instruction::Undef {
                dest: v,
                ty: SsaType::Void,
            });
            self.current_block_data.value_types.insert(v, SsaType::Void);
            v
        } else {
            let mut result = self.lower_expr(&list[0]);
            for expr in &list[1..] {
                result = self.lower_expr(expr);
            }
            result
        }
    }

    pub(crate) fn lower_interpolated_str(&mut self) -> Value {
        let v = self.current_block_data.fresh_value();
        let empty_str = StrId::from_static("");
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::String,
            value: Operand::ConstString(empty_str),
        });
        self.current_block_data
            .value_types
            .insert(v, SsaType::String);
        v
    }

    pub(crate) fn lower_tuple_expr(&mut self, elements: &[HirExpr<'a, 'bump>]) -> Value {
        if elements.is_empty() {
            let v = self.current_block_data.fresh_value();
            self.current_block_data.value_types.insert(v, SsaType::I64);
            v
        } else {
            self.lower_expr(&elements[0])
        }
    }

    pub(crate) fn lower_decimal_expr(&mut self, d: &f64) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::F64,
            value: Operand::ConstFloat(*d),
        });
        self.current_block_data.value_types.insert(v, SsaType::F64);
        v
    }

    pub(crate) fn lower_bool_expr(&mut self, b: &bool) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I8,
            value: Operand::ConstInt(if *b { 1 } else { 0 }),
        });
        self.current_block_data.value_types.insert(v, SsaType::I8);
        v
    }

    pub(crate) fn lower_string_expr(&mut self, s: &StrId) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::String,
            value: Operand::ConstString(*s),
        });
        self.current_block_data
            .value_types
            .insert(v, SsaType::String);
        v
    }

    pub(crate) fn lower_ident_expr(&mut self, name: &StrId, span: &SourceSpan<'a>) -> Value {
        if let Some(&v) = self.var_map.get(name) {
            if self.promoted_to_stack.contains(name) {
                let pointee_ty = match self.current_block_data.value_types.get(&v) {
                    Some(SsaType::Pointer(inner)) => (**inner).clone(),
                    other => panic!(
                        "Ident `{}` marked stack-promoted but its value type isn't a pointer: {:?}",
                        name, other
                    ),
                };
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest,
                    ptr: Operand::Value(v),
                });
                self.current_block_data.value_types.insert(dest, pointee_ty);
                dest
            } else {
                v
            }
        } else if let Some(const_expr) = self.constants.get(name) {
            self.lower_expr(const_expr)
        } else if let Some(func) = self.funcs.get(name).or_else(|| self.global_funcs.get(name)) {
            let dest = self.current_block_data.fresh_value();
            let ty = SsaType::FuncPointer {
                params: func.params.iter().map(|(_, t)| t.clone()).collect(),
                return_type: Box::new(func.ret_type.clone()),
            };
            self.emit(Instruction::Const {
                dest,
                ty: ty.clone(),
                value: Operand::FunctionRef(*name),
            });
            self.current_block_data.value_types.insert(dest, ty);
            dest
        } else {
            panic!(
                "lower_expr: variable `{}` (StrId {:?}) referenced before definition at span {}",
                self.context.resolve_string(name),
                name,
                span
            )
        }
    }

    pub(crate) fn lower_uninit_value(&mut self, ssa_ty: &SsaType) -> Value {
        match ssa_ty {
            SsaType::Array(inner, len) => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::StackAlloc {
                    dest,
                    ty: (**inner).clone(),
                    count: *len,
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Array(inner.clone(), *len));
                dest
            }

            ty if Self::is_aggregate_ssa_type(ty) => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::StackAlloc {
                    dest,
                    ty: ty.clone(),
                    count: 1,
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Pointer(Box::new(ty.clone())));
                dest
            }

            _ => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Undef {
                    dest,
                    ty: ssa_ty.clone(),
                });
                self.current_block_data
                    .value_types
                    .insert(dest, ssa_ty.clone());
                dest
            }
        }
    }

    pub(crate) fn lower_array_literal(&mut self, elements: &[HirExpr<'a, 'bump>]) -> Value {
        let elem_values: Vec<Value> = elements.iter().map(|e| self.lower_expr(e)).collect();

        let elem_ty = self
            .current_block_data
            .value_types
            .get(&elem_values[0])
            .cloned()
            .expect("lower_array_literal: element value has no known type");

        let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
            .expect("lower_array_literal: element type has no known size")
            as i64;

        let arr_v = self.current_block_data.fresh_value();
        self.emit(Instruction::StackAlloc {
            dest: arr_v,
            ty: elem_ty.clone(),
            count: elem_values.len(),
        });
        self.current_block_data.value_types.insert(
            arr_v,
            SsaType::Array(Box::new(elem_ty.clone()), elem_values.len()),
        );

        for (i, val) in elem_values.into_iter().enumerate() {
            let addr_v = self.current_block_data.fresh_value();
            self.emit(Instruction::FieldAddr {
                dest: addr_v,
                base: Operand::Value(arr_v),
                offset: (i as i64 * elem_size) as usize,
            });
            self.current_block_data
                .value_types
                .insert(addr_v, SsaType::Pointer(Box::new(elem_ty.clone())));

            self.emit(Instruction::Store {
                ptr: Operand::Value(addr_v),
                value: Operand::Value(val),
            });
        }

        arr_v
    }

    pub(crate) fn lower_zeroed_value(&mut self, ssa_ty: &SsaType) -> Value {
        match ssa_ty {
            SsaType::Array(inner, len) => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::StackAlloc {
                    dest,
                    ty: (**inner).clone(),
                    count: *len,
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Array(inner.clone(), *len));
                let elem_size = ir::layout::sizeof_ssa(inner, TargetInfo { ptr_bytes: 8 })
                    .expect("lower_zeroed_value: array element type has no known size");
                self.emit_memset(dest, 0, elem_size * len);
                dest
            }

            SsaType::User(_, _) | SsaType::Tuple(_) => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::StackAlloc {
                    dest,
                    ty: ssa_ty.clone(),
                    count: 1,
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::Pointer(Box::new(ssa_ty.clone())));

                let size = ir::layout::sizeof_ssa(ssa_ty, TargetInfo { ptr_bytes: 8 })
                    .expect("lower_zeroed_value: aggregate type has no known size");
                self.emit_memset(dest, 0, size);

                dest
            }

            _ => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest,
                    ty: ssa_ty.clone(),
                    value: Operand::ConstInt(0),
                });
                self.current_block_data
                    .value_types
                    .insert(dest, ssa_ty.clone());
                dest
            }
        }
    }

    pub(crate) fn lower_expr_null(&mut self) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I64,
            value: Operand::ConstInt(0),
        });
        self.current_block_data.value_types.insert(v, SsaType::Null);
        v
    }

    pub(crate) fn lower_expr_number(&mut self, n: i64) -> Value {
        self.lower_expr_number_inner(n, SsaType::Usize)
    }

    pub(crate) fn lower_expr_number_inner(&mut self, n: i64, ty: SsaType) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: ty.clone(),
            value: Operand::ConstInt(n),
        });
        self.current_block_data.value_types.insert(v, ty);
        v
    }

    pub(crate) fn lower_expr_as_u32(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
        if let HirExpr::Number(n, _) = expr {
            let v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: v,
                ty: SsaType::U32,
                value: Operand::ConstInt(*n),
            });
            self.current_block_data.value_types.insert(v, SsaType::U32);
            return v;
        }

        let v = self.lower_expr(expr);
        let src_ty = self
            .current_block_data
            .value_types
            .get(&v)
            .cloned()
            .unwrap_or(SsaType::I64);
        if src_ty == SsaType::U32 {
            return v;
        }

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Cast {
            dest,
            value: Operand::Value(v),
            kind: cast_kind(&src_ty, &SsaType::U32),
        });
        self.current_block_data
            .value_types
            .insert(dest, SsaType::U32);
        dest
    }

    pub(crate) fn lower_expr_binary(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: &Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        self.lower_expr_binary_expected(left, op, right, None)
    }

    pub(crate) fn lower_expr_binary_expected(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: &Operator,
        right: &HirExpr<'a, 'bump>,
        expected: Option<&SsaType>,
    ) -> Value {
        match op {
            Operator::LogicalAnd => self.lower_short_circuit_and(left, right),
            Operator::LogicalOr => self.lower_short_circuit_or(left, right),
            _ => {
                let l = match expected {
                    Some(exp) => self.lower_expr_expected(left, exp),
                    None => self.lower_expr(left),
                };
                let l_ty = self
                    .current_block_data
                    .value_types
                    .get(&l)
                    .cloned()
                    .unwrap_or(SsaType::I64);
                let r = self.lower_expr_expected(right, &l_ty);

                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: v,
                    op: lower_operator_bin(op),
                    left: Operand::Value(l),
                    right: Operand::Value(r),
                });
                self.current_block_data.value_types.insert(v, l_ty);
                v
            }
        }
    }
}
