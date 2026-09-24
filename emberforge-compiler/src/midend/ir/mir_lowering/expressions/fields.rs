use ir::{
    hir::{AssignmentOperator, HirExpr, HirType, StrId},
    ir_conversion::{assign_op_to_bin_op, lower_type_hir},
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::{FunctionLowerer, lowerer::fun_name};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn lower_field_access_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    ) -> Value {
        if let Some((root, mut path)) = self.static_field_path_mir(object) {
            path.push(field);
            if let Some(&narrowed_val) = self.narrowed_fields.get(&(root, path)) {
                return narrowed_val;
            }
        }

        let obj_val = self.lower_expr_as_receiver(object);

        if let Some(obj_ty) = self.current_block_data.value_types.get(&obj_val).cloned() {
            if let Some((offset, field_ty)) = self.resolve_slice_pseudo_field(&obj_ty, field) {
                let dest = self.new_value();
                self.emit(Instruction::LoadField {
                    dest,
                    base: Operand::Value(obj_val),
                    offset,
                });
                self.current_block_data.value_types.insert(dest, field_ty);
                return dest;
            }

            if self.slice_kind(&obj_ty).is_some() {
                panic!(
                    "lower_field_access: `.{}` is not a valid slice field on {:?} \
                 (only `.len`, and `.cap`/`.capacity` on *owned* slices, are supported)",
                    self.context.resolve_string(&field),
                    obj_ty
                );
            }
        }

        let cls_name = match self.current_block_data.value_types.get(&obj_val) {
            Some(SsaType::User(name, _)) => *name,
            Some(SsaType::Pointer(inner)) => fun_name(inner),
            other => panic!(
                "Could not determine object's struct for FieldAccess: {:?} at span {span}",
                other
            ),
        };

        let offsets = self.struct_field_offsets.get(&cls_name).unwrap_or_else(|| {
            panic!(
                "Unknown struct {} in FieldAccess at {span}. \nAll structs: {:?}",
                cls_name,
                self.struct_field_offsets.keys()
            )
        });

        let offset = *offsets
            .get(&field)
            .unwrap_or_else(|| panic!("Unknown field {} on struct {}", field, cls_name));

        let field_type = self
        .structs
        .get(&cls_name)
        .and_then(|hir_struct| hir_struct.fields.iter().find(|f| f.name == field))
        .map(|hir_field| lower_type_hir(&hir_field.field_type, self.enums))
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: lower_field_access could not find field {:?} on struct {:?}, defaulting to I64",
                field, cls_name
            );
            SsaType::I64
        });

        let is_slice_field = matches!(field_type, SsaType::Slice(_))
            || matches!(field_type, SsaType::Owned(ref inner) if matches!(inner.as_ref(), SsaType::Slice(_)));

        if is_slice_field {
            let addr = self.new_value();
            self.emit(Instruction::FieldAddr {
                dest: addr,
                base: Operand::Value(obj_val),
                offset,
            });
            self.current_block_data
                .value_types
                .insert(addr, SsaType::Pointer(Box::new(field_type)));
            return addr;
        }

        let dest = self.new_value();
        self.emit(Instruction::LoadField {
            dest,
            base: Operand::Value(obj_val),
            offset,
        });
        self.current_block_data.value_types.insert(dest, field_type);

        dest
    }

    pub(crate) fn handle_field_access(
        &mut self,
        op: AssignmentOperator,
        rhs: Value,
        object: &'a HirExpr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    ) -> Value {
        // Module-qualified static field: `zeta::io::files.File.DEFAULT`
        if let Some(mangled) = self.try_flatten_module_path(&HirExpr::FieldAccess {
            object,
            field,
            span: Default::default(),
        }) {
            let dest = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest,
                ty: SsaType::I64,
                value: Operand::GlobalRef(mangled),
            });
            self.current_block_data
                .value_types
                .insert(dest, SsaType::I64);
            return dest;
        }

        let obj_val = self.lower_expr_as_receiver(object);

        if let Some(obj_ty) = self.current_block_data.value_types.get(&obj_val).cloned() {
            if let Some((offset, field_ty)) = self.resolve_slice_pseudo_field(&obj_ty, field) {
                let new_value = match op {
                    AssignmentOperator::Assign => rhs,
                    _ => {
                        let current = self.new_value();
                        self.emit(Instruction::LoadField {
                            dest: current,
                            base: Operand::Value(obj_val),
                            offset,
                        });
                        self.current_block_data
                            .value_types
                            .insert(current, field_ty.clone());

                        let bin_op = assign_op_to_bin_op(op);
                        let dest = self.new_value();
                        self.emit(Instruction::Binary {
                            dest,
                            op: bin_op,
                            left: Operand::Value(current),
                            right: Operand::Value(rhs),
                        });
                        self.current_block_data.value_types.insert(dest, field_ty);
                        dest
                    }
                };

                self.emit(Instruction::StoreField {
                    base: Operand::Value(obj_val),
                    offset,
                    value: Operand::Value(new_value),
                });

                return new_value;
            }
        }

        let field_offset = self.get_field_offset(&obj_val, field);

        let this_id = StrId::from_static("this");
        let owner = if matches!(object, HirExpr::This { .. }) {
            Some(this_id)
        } else if let HirExpr::Ident(root, _) = object {
            Some(*root)
        } else {
            None
        };

        if matches!(op, AssignmentOperator::Assign) {
            let cls_name = match self.current_block_data.value_types.get(&obj_val) {
                Some(SsaType::User(name, _)) => Some(*name),
                Some(SsaType::Pointer(inner)) => {
                    if let SsaType::User(name, _) = inner.as_ref() {
                        Some(*name)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(cls_name) = cls_name {
                if let Some(hir_struct) = self.structs.get(&cls_name) {
                    if let Some(f) = hir_struct.fields.iter().find(|f| f.name == field) {
                        let nullable_owned = match f.field_type {
                            HirType::Nullable(inner)
                                if matches!(
                                    inner,
                                    HirType::OwnedPointer {
                                        allocator: Some(_),
                                        ..
                                    }
                                ) =>
                            {
                                Some(*inner)
                            }
                            _ => None,
                        };
                        if let Some(owned_ty) = nullable_owned {
                            let is_uninit =
                                owner.map_or(false, |o| self.drop_state.is_field_moved(o, field));
                            if !is_uninit {
                                let field_addr = self.current_block_data.fresh_value();
                                self.emit(Instruction::FieldAddr {
                                    dest: field_addr,
                                    base: Operand::Value(obj_val),
                                    offset: field_offset,
                                });
                                self.current_block_data.value_types.insert(
                                    field_addr,
                                    SsaType::Pointer(Box::new(lower_type_hir(
                                        &f.field_type,
                                        self.enums,
                                    ))),
                                );
                                self.emit_nullable_owned_field_overwrite_drop(
                                    field_addr, owned_ty, span,
                                );
                            }
                        }
                        self.handle_potential_drop(
                            field,
                            span,
                            obj_val,
                            field_offset,
                            owner,
                            nullable_owned,
                            f,
                        );
                    }
                }
            }
        }

        let new_value = match op {
            AssignmentOperator::Assign => rhs,
            _ => {
                let current = self.new_value();
                self.emit(Instruction::LoadField {
                    dest: current,
                    base: Operand::Value(obj_val),
                    offset: field_offset,
                });

                let _ = self
                    .current_block_data
                    .value_types
                    .entry(current)
                    .or_insert(SsaType::I64);

                let bin_op = assign_op_to_bin_op(op);

                let dest = self.new_value();
                self.emit(Instruction::Binary {
                    dest,
                    op: bin_op,
                    left: Operand::Value(current),
                    right: Operand::Value(rhs),
                });

                let res_ty = self
                    .current_block_data
                    .value_types
                    .get(&current)
                    .cloned()
                    .unwrap_or(SsaType::I64);

                self.current_block_data.value_types.insert(dest, res_ty);

                dest
            }
        };

        let is_slice = matches!(self.current_block_data.value_types.get(&new_value), Some(SsaType::Owned(inner)) if matches!(inner.as_ref(), SsaType::Slice(_)))
            || matches!(
                self.current_block_data.value_types.get(&new_value),
                Some(SsaType::Slice(_))
            );

        if is_slice {
            let ptr_val = self.new_value();
            let len_val = self.new_value();
            let cap_val = self.new_value();

            self.current_block_data
                .value_types
                .insert(ptr_val, SsaType::Pointer(Box::new(SsaType::I8)));
            self.current_block_data
                .value_types
                .insert(len_val, SsaType::Usize);
            self.current_block_data
                .value_types
                .insert(cap_val, SsaType::Usize);

            self.emit(Instruction::LoadField {
                dest: ptr_val,
                base: Operand::Value(new_value),
                offset: 0,
            });
            self.emit(Instruction::LoadField {
                dest: len_val,
                base: Operand::Value(new_value),
                offset: 8,
            });
            self.emit(Instruction::LoadField {
                dest: cap_val,
                base: Operand::Value(new_value),
                offset: 16,
            });

            self.emit(Instruction::StoreField {
                base: Operand::Value(obj_val),
                offset: field_offset + 0,
                value: Operand::Value(ptr_val),
            });
            self.emit(Instruction::StoreField {
                base: Operand::Value(obj_val),
                offset: field_offset + 8,
                value: Operand::Value(len_val),
            });
            self.emit(Instruction::StoreField {
                base: Operand::Value(obj_val),
                offset: field_offset + 16,
                value: Operand::Value(cap_val),
            });
        } else {
            let field_ssa = self.struct_field_ssa_type(obj_val, field);
            match field_ssa {
                Some(ty @ SsaType::Nullable(_)) if matches!(op, AssignmentOperator::Assign) => {
                    self.store_field_value(obj_val, field_offset, &ty, new_value);
                }
                _ => {
                    self.emit(Instruction::StoreField {
                        base: Operand::Value(obj_val),
                        offset: field_offset,
                        value: Operand::Value(new_value),
                    });
                }
            }
        }

        if let Some((root, mut path)) = self.static_field_path_mir(object) {
            path.push(field);
            match self.current_block_data.value_types.get(&new_value).cloned() {
                Some(SsaType::Nullable(_)) | Some(SsaType::Null) | None => {
                    self.narrowed_fields.remove(&(root, path));
                }
                Some(_) => {
                    self.narrowed_fields.insert((root, path), new_value);
                }
            }
        }

        if let Some(o) = owner {
            self.drop_state.mark_field_initialized(o, field);
        }

        new_value
    }

    pub(crate) fn get_field_offset(&mut self, obj: &Value, field: StrId) -> usize {
        let cls_name = match self.current_block_data.value_types.get(obj) {
            Some(SsaType::User(name, _)) => name,
            Some(SsaType::Pointer(inner)) => &fun_name(inner),
            Some(SsaType::Owned(inner)) => &fun_name(inner),
            other => panic!(
                "Could not determine object's struct for FieldAccess: {:?}",
                other
            ),
        };

        self.struct_field_offsets
            .get(&cls_name)
            .unwrap_or_else(|| panic!("Unknown struct {} in FieldAccess", cls_name))
            .get(&field)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "Unknown field {} on struct {}",
                    self.context.resolve_string(&*field),
                    self.context.resolve_string(&*cls_name)
                )
            })
    }
}
