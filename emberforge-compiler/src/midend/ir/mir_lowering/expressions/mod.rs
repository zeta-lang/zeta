pub mod addr;
pub mod calls;
pub mod constants;
pub mod fields;
pub mod initialization;
pub mod names;

use ir::{
    hir::{AssignmentOperator, HirExpr, HirType, StrId},
    ir_conversion::{assign_op_to_bin_op, lower_type_hir},
    layout::{TargetInfo, alignof_ssa, round_up_to_align},
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value, cast_kind},
};

use crate::midend::{
    copy_analysis::drop_tracking::Tri,
    ir::mir_lowering::{
        FunctionLowerer,
        lowerer::{FieldInitVal, IndexedContainer},
    },
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn lower_range_expr(
        &mut self,
        start: &HirExpr<'a, 'bump>,
        end: &HirExpr<'a, 'bump>,
    ) -> Value {
        let start_v = self.lower_expr(start);
        let end_v = self.lower_expr(end);

        let dest = self.current_block_data.fresh_value();
        let ty = SsaType::Tuple(vec![SsaType::Usize, SsaType::Usize]);
        self.emit(Instruction::StackAlloc {
            dest,
            ty: ty.clone(),
            count: 1,
        });
        self.current_block_data.value_types.insert(dest, ty);

        self.emit(Instruction::StoreField {
            base: Operand::Value(dest),
            offset: 0,
            value: Operand::Value(start_v),
        });
        self.emit(Instruction::StoreField {
            base: Operand::Value(dest),
            offset: 8,
            value: Operand::Value(end_v),
        });
        dest
    }

    pub(crate) fn lower_cast_expr(
        &mut self,
        expr: &&HirExpr<'a, 'bump>,
        target_type: &HirType<'a, 'bump>,
    ) -> Value {
        let mut src = self.lower_expr(expr);

        let mut src_ty = self.current_block_data.value_types[&src].clone();
        let dst_ty = lower_type_hir(target_type, self.enums);

        if dst_ty.is_pointer() {
            match &src_ty {
                SsaType::Slice(inner) => {
                    let ptr = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: ptr,
                        base: Operand::Value(src),
                        offset: 0,
                    });
                    let ptr_ty = SsaType::Pointer(Box::new((**inner).clone()));
                    self.current_block_data
                        .value_types
                        .insert(ptr, ptr_ty.clone());
                    src = ptr;
                    src_ty = ptr_ty;
                }

                SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Slice(_)) => {
                    let SsaType::Slice(elem) = inner.as_ref() else {
                        unreachable!()
                    };
                    let ptr = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: ptr,
                        base: Operand::Value(src),
                        offset: 0,
                    });
                    let ptr_ty = SsaType::Pointer(Box::new((**elem).clone()));
                    self.current_block_data
                        .value_types
                        .insert(ptr, ptr_ty.clone());
                    src = ptr;
                    src_ty = ptr_ty;
                }

                SsaType::Owned(inner) => {
                    if let SsaType::Slice(elem) = inner.as_ref() {
                        let ptr = self.current_block_data.fresh_value();
                        self.emit(Instruction::LoadField {
                            dest: ptr,
                            base: Operand::Value(src),
                            offset: 0,
                        });
                        let ptr_ty = SsaType::Pointer(Box::new((**elem).clone()));
                        self.current_block_data
                            .value_types
                            .insert(ptr, ptr_ty.clone());
                        src = ptr;
                        src_ty = ptr_ty;
                    }
                }

                SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Owned(o) if matches!(o.as_ref(), SsaType::Slice(_))) =>
                {
                    let SsaType::Owned(owned_inner) = inner.as_ref() else {
                        unreachable!()
                    };
                    let SsaType::Slice(elem) = owned_inner.as_ref() else {
                        unreachable!()
                    };
                    let ptr = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: ptr,
                        base: Operand::Value(src),
                        offset: 0,
                    });
                    let ptr_ty = SsaType::Pointer(Box::new((**elem).clone()));
                    self.current_block_data
                        .value_types
                        .insert(ptr, ptr_ty.clone());
                    src = ptr;
                    src_ty = ptr_ty;
                }

                SsaType::Array(inner, _) => {
                    src_ty = SsaType::Pointer(Box::new((**inner).clone()));
                }

                _ => {}
            }
        }

        let kind = cast_kind(&src_ty, &dst_ty);

        let dest = self.current_block_data.fresh_value();

        self.emit(Instruction::Cast {
            dest,
            value: Operand::Value(src),
            kind,
        });

        self.current_block_data.value_types.insert(dest, dst_ty);

        dest
    }

    pub(crate) fn lower_deref_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
        let ptr = match self.narrowed_field_value(expr) {
            Some(v) => v,
            None => self.lower_expr(expr),
        };

        let dest = self.current_block_data.fresh_value();

        self.emit(Instruction::Load {
            dest,
            ptr: Operand::Value(ptr),
        });

        let pointee_ty = match self.current_block_data.value_types[&ptr].clone() {
            SsaType::Pointer(inner) | SsaType::Owned(inner) => *inner,
            other => panic!("cannot dereference {:?}", other),
        };

        self.current_block_data.value_types.insert(dest, pointee_ty);

        dest
    }

    pub(crate) fn lower_index_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        index: &HirExpr<'a, 'bump>,
    ) -> Value {
        if let HirExpr::Range {
            start,
            end,
            inclusive,
            ..
        } = index
        {
            return self.lower_slice_expr(object, start, end, *inclusive);
        }

        let (addr_v, elem_ty) = self.lower_index_addr(object, index);
        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest,
            ptr: Operand::Value(addr_v),
        });
        self.current_block_data.value_types.insert(dest, elem_ty);
        dest
    }

    pub(crate) fn lower_expr_assignment(
        &mut self,
        target: &HirExpr<'a, 'bump>,
        op: AssignmentOperator,
        value: &HirExpr<'a, 'bump>,
    ) -> Value {
        if matches!(op, AssignmentOperator::Assign) {
            self.record_move_if_any(value);
        }
        let rhs = self.lower_expr(value);

        match target {
            HirExpr::Ident(name, span) => self.handle_ident(op, rhs, *name, *span),

            HirExpr::FieldAccess {
                object,
                field,
                span,
            }
            | HirExpr::Get {
                object,
                field,
                span,
            } => self.handle_field_access(op, rhs, object, *field, *span),

            HirExpr::Deref { expr, span: _ } => {
                let ptr = self.lower_expr(expr);
                self.handle_deref_assign(ptr, rhs, op)
            }

            HirExpr::Index {
                object,
                index,
                span,
            } => {
                if let HirExpr::Range { span, .. } = index {
                    panic!("Cannot assign a range of elements in a slice/array at {span}.")
                }

                let idx_v = self.lower_expr(index);
                let (base_ptr, elem_ty, slice_len) = self.lower_index_base_len(object);
                if let Some(len_operand) = slice_len {
                    self.emit_bounds_check(Operand::Value(idx_v), len_operand, false);
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
                    left: Operand::Value(idx_v),
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

                let rhs_v = match op {
                    AssignmentOperator::Assign => {
                        let val = self.lower_expr(value);
                        let elem_drop_kind = self.drop_kind_for_ssa_type(&elem_ty, object);
                        if elem_drop_kind.is_droppable() {
                            let base_val = self.lower_expr(object);
                            let base_ty =
                                self.current_block_data.value_types.get(&base_val).cloned();
                            let container =
                                base_ty.as_ref().and_then(Self::classify_indexed_container);

                            if let Some(container) = container {
                                let safe_to_drop = match (&container, index) {
                                    (IndexedContainer::Array(len), HirExpr::Number(n, _)) => {
                                        *n >= 0 && (*n as usize) < *len
                                    }

                                    (IndexedContainer::BorrowedSlice, _)
                                    | (IndexedContainer::OwnedSlice, _) => true,

                                    (IndexedContainer::Array(_), _) => false,
                                };

                                if safe_to_drop {
                                    let const_idx = match (object, index) {
                                        (HirExpr::Ident(root, _), HirExpr::Number(n, _)) => {
                                            Some((*root, *n))
                                        }
                                        _ => None,
                                    };
                                    let status = const_idx.map_or(Tri::Yes, |(r, n)| {
                                        self.drop_state.index_status(r, n)
                                    });
                                    match status {
                                        Tri::No => {}
                                        Tri::Yes => self.emit_indexed_element_drop(
                                            &elem_drop_kind,
                                            addr_v,
                                            *span,
                                        ),
                                        Tri::Maybe => {
                                            let (r, n) = const_idx.unwrap();
                                            let (flags, _) = self.array_flags[&r];
                                            self.emit_if_flag(flags, n as usize, |s| {
                                                s.emit_indexed_element_drop(
                                                    &elem_drop_kind,
                                                    addr_v,
                                                    *span,
                                                )
                                            });
                                        }
                                    }
                                    if let Some((r, n)) = const_idx {
                                        self.drop_state.mark_index_initialized(r, n);
                                        if let Some(&(flags, _)) = self.array_flags.get(&r) {
                                            self.store_const_u8(flags, n as usize, 1);
                                        }
                                    }
                                }
                            }
                        }
                        val
                    }
                    _ => {
                        // compound assignment: arr[i] += x  =>  load, binop, store
                        let cur = self.current_block_data.fresh_value();
                        self.emit(Instruction::Load {
                            dest: cur,
                            ptr: Operand::Value(addr_v),
                        });
                        self.current_block_data
                            .value_types
                            .insert(cur, elem_ty.clone());

                        let rhs = self.lower_expr(value);
                        let result = self.current_block_data.fresh_value();
                        self.emit(Instruction::Binary {
                            dest: result,
                            op: assign_op_to_bin_op(op),
                            left: Operand::Value(cur),
                            right: Operand::Value(rhs),
                        });
                        self.current_block_data
                            .value_types
                            .insert(result, elem_ty.clone());
                        result
                    }
                };

                self.emit(Instruction::Store {
                    ptr: Operand::Value(addr_v),
                    value: Operand::Value(rhs_v),
                });

                rhs_v
            }

            _ => unimplemented!("Assignment target {:?} not yet supported", target),
        }
    }

    pub(crate) fn lower_expr_as_receiver_raw(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
        if let HirExpr::Ident(name, _) = object {
            if let Some(&v) = self.var_map.get(name) {
                if matches!(
                    self.current_block_data.value_types.get(&v),
                    Some(SsaType::Pointer(_))
                ) {
                    return v;
                }
            }
        }
        if let HirExpr::This { .. } = object {
            let this_name = StrId::from_static("this");
            if let Some(&v) = self.var_map.get(&this_name) {
                if matches!(
                    self.current_block_data.value_types.get(&v),
                    Some(SsaType::Pointer(_))
                ) {
                    return v;
                }
            }
        }
        if let HirExpr::FieldAccess {
            object: base_obj,
            field,
            span,
        }
        | HirExpr::Get {
            object: base_obj,
            field,
            span,
        } = object
        {
            let (addr, _) = self.lower_field_addr(base_obj, *field, span);
            return self.auto_unwrap_receiver(addr);
        }
        let v = self.lower_expr(object);
        self.auto_unwrap_receiver(v)
    }

    pub(crate) fn lower_expr_as_receiver(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
        let v = self.lower_expr_as_receiver_raw(object);
        self.canonicalize_receiver(v)
    }

    pub(crate) fn canonicalize_receiver(&mut self, v: Value) -> Value {
        match self.value_type(v).cloned() {
            Some(SsaType::Owned(inner)) if !matches!(*inner, SsaType::Slice(_)) => {
                let ptr_ty = SsaType::Pointer(inner);
                let dest = self.new_value();
                self.emit(Instruction::Cast {
                    dest,
                    value: Operand::Value(v),
                    kind: cast_kind(&ptr_ty, &ptr_ty),
                });
                self.current_block_data.value_types.insert(dest, ptr_ty);
                dest
            }
            _ => v,
        }
    }

    pub(crate) fn handle_deref_assign(
        &mut self,
        ptr: Value,
        rhs: Value,
        op: AssignmentOperator,
    ) -> Value {
        let value_to_store = match op {
            AssignmentOperator::Assign => rhs,

            _ => {
                let current = self.new_value();

                self.emit(Instruction::Load {
                    dest: current,
                    ptr: Operand::Value(ptr),
                });

                if let Some(SsaType::Pointer(inner)) =
                    self.current_block_data.value_types.get(&ptr).cloned()
                {
                    self.current_block_data.value_types.insert(current, *inner);
                }

                let dest = self.new_value();
                let bin_op = assign_op_to_bin_op(op);

                self.emit(Instruction::Binary {
                    dest,
                    op: bin_op,
                    left: Operand::Value(current),
                    right: Operand::Value(rhs),
                });

                let result_ty = self
                    .current_block_data
                    .value_types
                    .get(&current)
                    .cloned()
                    .or_else(|| self.current_block_data.value_types.get(&rhs).cloned())
                    .unwrap_or(SsaType::I64);

                self.current_block_data.value_types.insert(dest, result_ty);

                dest
            }
        };

        self.emit(Instruction::Store {
            ptr: Operand::Value(ptr),
            value: Operand::Value(value_to_store),
        });

        value_to_store
    }

    pub(crate) fn struct_field_ssa_type(&self, obj: Value, field: StrId) -> Option<SsaType> {
        let mut ty = self.value_type(obj)?;
        loop {
            match ty {
                SsaType::User(name, _) => {
                    return self
                        .structs
                        .get(name)?
                        .fields
                        .iter()
                        .find(|f| f.name == field)
                        .map(|f| lower_type_hir(&f.field_type, self.enums));
                }
                SsaType::Pointer(i) | SsaType::Owned(i) => ty = i.as_ref(),
                _ => return None,
            }
        }
    }

    pub(crate) fn lower_init_operand(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        field_ty: &SsaType,
    ) -> FieldInitVal {
        match expr {
            HirExpr::Null(_) => FieldInitVal::Null,
            HirExpr::Uninit { .. } => FieldInitVal::Uninit,
            other => FieldInitVal::Val(self.lower_expr_expected(other, field_ty)),
        }
    }

    pub(crate) fn store_init(
        &mut self,
        base: Value,
        offset: usize,
        field_ty: &SsaType,
        init: FieldInitVal,
    ) {
        match init {
            FieldInitVal::Uninit => {}
            FieldInitVal::Null => self.store_null_field(base, offset, field_ty),
            FieldInitVal::Val(v) => self.store_field_value(base, offset, field_ty, v),
        }
    }

    pub(crate) fn store_field_value(
        &mut self,
        base: Value,
        offset: usize,
        field_ty: &SsaType,
        val: Value,
    ) {
        if let SsaType::Nullable(inner) = field_ty {
            if field_ty.is_tagged_nullable() {
                let target = TargetInfo { ptr_bytes: 8 };
                match self.current_block_data.value_types.get(&val).cloned() {
                    Some(SsaType::Null) => self.store_const_u8(base, offset, 0),
                    Some(SsaType::Nullable(_)) => {
                        // already a tagged nullable (pointer to tag+payload): copy the whole slot
                        let size = ir::layout::sizeof_ssa(field_ty, target)
                            .expect("store_field_value: nullable has no known size");
                        let dst = self.field_addr(base, offset, field_ty);
                        let n = self.current_block_data.fresh_value();
                        self.emit(Instruction::Const {
                            dest: n,
                            ty: SsaType::Usize,
                            value: Operand::ConstInt(size as i64),
                        });
                        self.current_block_data
                            .value_types
                            .insert(n, SsaType::Usize);
                        self.emit_memcpy(dst, val, n);
                    }
                    _ => {
                        // plain `T` into a `T?` slot: write tag = some, then the payload
                        let payload_align = alignof_ssa(inner, target).unwrap_or_else(|e| {
                            panic!("failed to compute alignment for nullable payload: {:?}", e)
                        });
                        let payload_offset = offset + round_up_to_align(1, payload_align);
                        self.store_const_u8(base, offset, 1);
                        self.store_field_value(base, payload_offset, inner, val);
                    }
                }
            } else {
                // pointer-optimized: the bits are the pointer (or 0)
                let val = self.retype_nullable_ptr_bits(val, field_ty);
                self.emit(Instruction::StoreField {
                    base: Operand::Value(base),
                    offset,
                    value: Operand::Value(val),
                });
            }
            return;
        }

        let words = match field_ty {
            SsaType::Slice(_) => 2,
            SsaType::Owned(i) if matches!(i.as_ref(), SsaType::Slice(_)) => 3,
            _ => 0,
        };
        if words > 0 {
            for w in 0..words {
                let word = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: word,
                    base: Operand::Value(val),
                    offset: w * 8,
                });
                let wty = if w == 0 {
                    SsaType::Pointer(Box::new(SsaType::I8))
                } else {
                    SsaType::Usize
                };
                self.current_block_data.value_types.insert(word, wty);
                self.emit(Instruction::StoreField {
                    base: Operand::Value(base),
                    offset: offset + w * 8,
                    value: Operand::Value(word),
                });
            }
            return;
        }

        self.emit(Instruction::StoreField {
            base: Operand::Value(base),
            offset,
            value: Operand::Value(val),
        });
    }

    pub(crate) fn classify_indexed_container(ty: &SsaType) -> Option<IndexedContainer> {
        match ty {
            SsaType::Array(_, len) => Some(IndexedContainer::Array(*len)),
            SsaType::Slice(_) => Some(IndexedContainer::BorrowedSlice),
            SsaType::Owned(inner) if matches!(inner.as_ref(), SsaType::Slice(_)) => {
                Some(IndexedContainer::OwnedSlice)
            }
            SsaType::Pointer(inner) => Self::classify_indexed_container(inner),
            _ => None,
        }
    }

    pub(crate) fn narrowed_field_value(&self, expr: &HirExpr<'a, 'bump>) -> Option<Value> {
        let (root, path) = self.static_field_path_mir(expr)?;
        self.narrowed_fields.get(&(root, path)).copied()
    }

    pub(crate) fn param_types_of(&self, name: &StrId) -> Vec<SsaType> {
        self.funcs
            .get(name)
            .or_else(|| self.global_funcs.get(name))
            .map(|f| f.params.iter().map(|(_, t)| t.clone()).collect())
            .unwrap_or_default()
    }
}
