use ir::{
    hir::{HirExpr, Operator, StrId},
    ir_conversion::lower_operator_bin,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
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
        self.current_block_data.value_types.insert(
            a,
            SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, Box::new(ty.clone())),
        );
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
            self.emit(Instruction::Const {
                dest: v,
                ty: SsaType::I64,
                value: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(v, SsaType::I64);
            return v;
        }

        let mut elem_vals: Vec<(Value, SsaType)> = Vec::with_capacity(elements.len());
        for elem in elements {
            let val = self.lower_expr(elem);
            let ty = self
                .current_block_data
                .value_types
                .get(&val)
                .cloned()
                .unwrap_or(SsaType::I64);
            elem_vals.push((val, ty));
        }

        let field_types: Vec<SsaType> = elem_vals.iter().map(|(_, ty)| ty.clone()).collect();
        let tuple_ty = SsaType::Tuple(field_types.clone());

        let obj = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: obj,
            ty: tuple_ty.clone(),
            count: 0,
        });
        self.current_block_data
            .value_types
            .insert(obj, tuple_ty.clone());

        // Compute field offsets using the same alignment rules as layout_of_ssa.
        let target = ir::layout::TargetInfo { ptr_bytes: 8 };
        let mut cursor = 0usize;
        for (i, (val, fty)) in elem_vals.iter().enumerate() {
            let falign = ir::layout::alignof_ssa(fty, target).unwrap_or(8);
            // Round up cursor to field alignment.
            cursor = (cursor + falign - 1) & !(falign - 1);

            if Self::is_aggregate_ssa_type(fty) {
                // Aggregate: copy via memcpy (store_init handles this).
                let addr = self.current_block_data.fresh_value();
                self.emit(Instruction::FieldAddr {
                    dest: addr,
                    base: Operand::Value(obj),
                    offset: cursor,
                });
                self.current_block_data.value_types.insert(
                    addr,
                    SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, Box::new(fty.clone())),
                );
                self.store_init(
                    obj,
                    cursor,
                    fty,
                    crate::midend::ir::mir_lowering::lowerer::FieldInitVal::Val(*val),
                );
            } else {
                self.emit(Instruction::StoreField {
                    base: Operand::Value(obj),
                    offset: cursor,
                    value: Operand::Value(*val),
                });
            }

            let fsize = ir::layout::sizeof_ssa(fty, target).unwrap_or(8);
            cursor += fsize;
            let _ = i; // suppress unused warning
        }

        obj
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
                    Some(SsaType::Pointer(_, inner)) => (**inner).clone(),
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
        } else if let Some((addr, ty)) = self.lower_static_addr(*name) {
            if Self::is_aggregate_ssa_type(&ty) || matches!(ty, SsaType::Array(..)) {
                addr // aggregates are carried by address
            } else {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest,
                    ptr: Operand::Value(addr),
                });
                self.current_block_data.value_types.insert(dest, ty);
                dest
            }
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
                self.current_block_data.value_types.insert(
                    dest,
                    SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, Box::new(ty.clone())),
                );
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
            self.current_block_data.value_types.insert(
                addr_v,
                SsaType::Pointer(
                    ir::ssa_ir::SsaPointerKind::UnsafeMut,
                    Box::new(elem_ty.clone()),
                ),
            );

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

            SsaType::User(_, _, _) | SsaType::Tuple(_) => {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::StackAlloc {
                    dest,
                    ty: ssa_ty.clone(),
                    count: 1,
                });
                self.current_block_data.value_types.insert(
                    dest,
                    SsaType::Pointer(
                        ir::ssa_ir::SsaPointerKind::UnsafeMut,
                        Box::new(ssa_ty.clone()),
                    ),
                );

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
        let value = if matches!(ty, SsaType::F32 | SsaType::F64) {
            Operand::ConstFloat(n as f64)
        } else {
            Operand::ConstInt(n)
        };
        self.emit(Instruction::Const {
            dest: v,
            ty: ty.clone(),
            value,
        });
        self.current_block_data.value_types.insert(v, ty);
        v
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
                let left_is_lit = matches!(left, HirExpr::Number(..) | HirExpr::Decimal(..));
                let right_is_lit = matches!(right, HirExpr::Number(..) | HirExpr::Decimal(..));

                let (l, r, l_ty) = if expected.is_none() && left_is_lit && !right_is_lit {
                    let r = self.lower_expr(right);
                    let r_ty = self
                        .current_block_data
                        .value_types
                        .get(&r)
                        .cloned()
                        .unwrap_or(SsaType::I64);
                    let l = self.lower_expr_expected(left, &r_ty);
                    (l, r, r_ty)
                } else {
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
                    (l, r, l_ty)
                };

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

    pub(crate) fn lower_static_addr(&mut self, name: StrId) -> Option<(Value, SsaType)> {
        let ty = self.registry.statics.borrow().get(&name)?.ty.clone();
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::GlobalAddr { dest: v, name });
        let vt = match &ty {
            SsaType::Array(..) => ty.clone(),
            _ => SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, Box::new(ty.clone())),
        };
        self.current_block_data.value_types.insert(v, vt);
        Some((v, ty))
    }
}
