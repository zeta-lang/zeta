use ir::{
    hir::{
        self, AssignmentOperator, DropKind, HirEnum, HirExpr, HirFieldInit, HirType, Operator,
        StrId,
    },
    ir_conversion::{assign_op_to_bin_op, lower_operator_bin, lower_type_hir},
    layout::{TargetInfo, alignof_ssa, round_up_to_align},
    span::SourceSpan,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value, cast_kind},
};
use smallvec::SmallVec;
use zetaruntime::intern_fmt;

use crate::{
    midend::{
        copy_analysis::{
            drop_emitter::{DropEmitter, FnAllocatorResolver},
            drop_tracking::Tri,
        },
        ir::mir_lowering::{
            FunctionLowerer,
            lowerer::{FieldInitVal, IndexedContainer, fun_name},
        },
    },
    optimized_string_buffering,
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn try_lower_bare_enum_variant(
        &mut self,
        enum_name: &StrId,
        variant: &StrId,
    ) -> Option<Value> {
        let hir_enum = self.enums.get(enum_name).or_else(|| {
            let base_str = self.context.resolve_string(enum_name);
            let candidates: Vec<&HirEnum> = self
                .enums
                .values()
                .filter(|e| {
                    e.name.as_str() == base_str && e.variants.iter().any(|v| v.name == *variant)
                })
                .collect();
            if candidates.len() == 1 {
                Some(candidates[0])
            } else {
                candidates.into_iter().find(|e| e.name == *enum_name)
            }
        })?;

        let resolved_enum_name = hir_enum.name;
        hir_enum
            .variants
            .iter()
            .any(|v| v.name == *variant)
            .then(|| self.lower_enum_init(&resolved_enum_name, variant, &[], &Default::default()))
    }

    pub(super) fn lower_enum_init(
        &mut self,
        enum_name: &StrId,
        variant: &StrId,
        args: &[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
    ) -> Value {
        let enums = self.enums;
        let hir_enum = enums
            .get(enum_name)
            .unwrap_or_else(|| panic!("[lower_enum_init] unknown enum `{}` in {span}.", enum_name));
        let resolved_enum_name = hir_enum.name;

        let lowered_variants: Vec<Vec<SsaType>> = hir_enum
            .variants
            .iter()
            .map(|v| {
                v.fields
                    .iter()
                    .map(|f| lower_type_hir(&f.field_type, enums))
                    .collect()
            })
            .collect();

        let tag = hir_enum
            .variants
            .iter()
            .position(|v| v.name == *variant)
            .unwrap_or_else(|| {
                panic!(
                    "[lower_enum_init] enum `{}` has no variant `{}` in {span}",
                    resolved_enum_name, variant
                )
            });

        let field_tys = lowered_variants[tag].clone();
        let (offsets, _) = Self::payload_layout(&field_tys);
        let max_payload = lowered_variants
            .iter()
            .map(|tys| Self::payload_layout(tys).1)
            .max()
            .unwrap_or(0);

        let mut inits = Vec::with_capacity(args.len());
        for (arg, fty) in args.iter().zip(field_tys.iter()) {
            inits.push(self.lower_init_operand(arg, fty));
        }

        let tag_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: tag_v,
            ty: SsaType::I64,
            value: Operand::ConstInt(tag as i64),
        });
        self.current_block_data
            .value_types
            .insert(tag_v, SsaType::I64);

        let obj = self.current_block_data.fresh_value();
        self.emit(Instruction::StackAlloc {
            dest: obj,
            ty: SsaType::Array(Box::new(SsaType::I8), 8 + max_payload),
            count: 1,
        });
        self.current_block_data.value_types.insert(
            obj,
            SsaType::Enum {
                name: resolved_enum_name,
                variants: lowered_variants,
            },
        );

        self.emit(Instruction::StoreField {
            base: Operand::Value(obj),
            offset: 0,
            value: Operand::Value(tag_v),
        });

        for ((init, fty), off) in inits.into_iter().zip(field_tys.iter()).zip(offsets.iter()) {
            self.store_init(obj, 8 + off, fty, init);
        }

        obj
    }

    pub(super) fn lower_range_expr(
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

    pub(super) fn lower_null_comparison(
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

    pub(super) fn lower_cast_expr(
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

    pub(super) fn lower_module_access_expr(
        &mut self,
        hir_module_access: &hir::HirModuleAccess<'a, 'bump>,
    ) -> Value {
        if let Some(v) =
            self.try_lower_bare_enum_variant(&hir_module_access.member, &hir_module_access.member)
        {
            return v;
        }
        for path_seg in hir_module_access.path.iter().rev() {
            if let Some(v) = self.try_lower_bare_enum_variant(path_seg, &hir_module_access.member) {
                return v;
            }
        }
        let mangled = optimized_string_buffering::build_module_scoped_name(
            hir_module_access.path,
            hir_module_access.member,
            None,
            self.context.clone(),
        );

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest,
            ty: SsaType::I64, // TODO: this is a placeholder; refined once type info flows through
            value: Operand::GlobalRef(mangled),
        });
        self.current_block_data
            .value_types
            .insert(dest, SsaType::I64);
        dest
    }

    pub(super) fn lower_deref_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
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

    pub(super) fn lower_this_expr(&mut self) -> Value {
        let this_name = StrId::from_static("this");
        *self.var_map.get(&this_name).unwrap()
    }

    pub(super) fn lower_comparison_expr(
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

    pub(super) fn lower_expr_list_expr(&mut self, list: &[HirExpr<'a, 'bump>]) -> Value {
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

    pub(super) fn lower_interpolated_str(&mut self) -> Value {
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

    pub(super) fn lower_tuple_expr(&mut self, elements: &[HirExpr<'a, 'bump>]) -> Value {
        if elements.is_empty() {
            let v = self.current_block_data.fresh_value();
            self.current_block_data.value_types.insert(v, SsaType::I64);
            v
        } else {
            self.lower_expr(&elements[0])
        }
    }

    pub(super) fn lower_decimal_expr(&mut self, d: &f64) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::F64,
            value: Operand::ConstFloat(*d),
        });
        self.current_block_data.value_types.insert(v, SsaType::F64);
        v
    }

    pub(super) fn lower_bool_expr(&mut self, b: &bool) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I8,
            value: Operand::ConstInt(if *b { 1 } else { 0 }),
        });
        self.current_block_data.value_types.insert(v, SsaType::I8);
        v
    }

    pub(super) fn lower_string_expr(&mut self, s: &StrId) -> Value {
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

    pub(super) fn lower_ident_expr(&mut self, name: &StrId, span: &SourceSpan<'a>) -> Value {
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

    pub(super) fn lower_uninit_value(&mut self, ssa_ty: &SsaType) -> Value {
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

    pub(super) fn lower_field_access_expr(
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

    pub(super) fn lower_index_expr(
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

    pub(super) fn try_flatten_module_path(&self, expr: &HirExpr<'a, 'bump>) -> Option<StrId> {
        match expr {
            HirExpr::ModuleAccess(acc) => self.resolve_module_access_callee(acc.path, acc.member),
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::ModuleAccess(acc) = object {
                    Some(self.resolve_module_qualified_name(acc.path, acc.member, Some(*field)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(super) fn resolve_module_access_callee(
        &self,
        path: &[StrId],
        member: StrId,
    ) -> Option<StrId> {
        let module_target = self
            .dep_graph
            .borrow()
            .resolve_module_path(path)
            .or_else(|| {
                if path.len() == 1 {
                    self.module_import_aliases
                        .get(&self.module_idx)
                        .and_then(|aliases| aliases.get(&path[0]))
                        .copied()
                } else {
                    None
                }
            });
        if let Some(target_idx) = module_target {
            if self.extern_c_names.contains(&member) {
                return Some(member);
            }
            return Some(self.dep_graph.borrow().mangle_free_function(
                target_idx,
                member,
                false,
                &self.context,
            ));
        }

        if path.len() == 1 {
            let type_module_idx = self
                .module_named_imports
                .get(&self.module_idx)
                .and_then(|named| named.get(&path[0]))
                .copied()
                .unwrap_or(self.module_idx);

            let mangled_type =
                self.dep_graph
                    .borrow()
                    .mangle_type_name(type_module_idx, path[0], &self.context);

            if let Some(mangled_method) = self
                .struct_mangled_map
                .get(&mangled_type)
                .and_then(|methods| methods.get(&member))
            {
                return Some(*mangled_method);
            }
        }

        if self.extern_c_names.contains(&member) {
            return Some(member);
        }

        None
    }

    pub(super) fn lower_array_literal(&mut self, elements: &[HirExpr<'a, 'bump>]) -> Value {
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

    pub(super) fn lower_zeroed_value(&mut self, ssa_ty: &SsaType) -> Value {
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

    pub(super) fn resolve_module_qualified_name(
        &self,
        path: &[StrId],
        member: StrId,
        extra: Option<StrId>,
    ) -> StrId {
        let bare_name = extra.unwrap_or(member);

        if self.extern_c_names.contains(&bare_name) {
            return bare_name;
        }

        let target_module_idx = self
            .dep_graph
            .borrow()
            .resolve_module_path(path)
            .or_else(|| {
                if path.len() == 1 {
                    self.module_import_aliases
                        .get(&self.module_idx)
                        .and_then(|aliases| aliases.get(&path[0]))
                        .copied()
                } else {
                    None
                }
            });

        match extra {
            Some(method_name) => {
                let mut segments: Vec<StrId> = Vec::with_capacity(path.len() + 1);
                segments.push(member);
                segments.extend_from_slice(path);
                optimized_string_buffering::build_module_scoped_name(
                    &segments,
                    method_name,
                    None,
                    self.context.clone(),
                )
            }
            None => {
                let Some(target_idx) = target_module_idx else {
                    return optimized_string_buffering::build_module_scoped_name(
                        path,
                        member,
                        None,
                        self.context.clone(),
                    );
                };
                let Some(pkg) = self.dep_graph.borrow().get_module_package(target_idx) else {
                    return member;
                };
                let pkg_str = pkg.to_string();
                let segments: Vec<StrId> = pkg_str
                    .split("::")
                    .map(|seg| StrId(self.context.thread_local().intern(seg)))
                    .collect();
                optimized_string_buffering::build_module_scoped_name(
                    &segments,
                    member,
                    None,
                    self.context.clone(),
                )
            }
        }
    }

    pub(super) fn lower_expr_assignment(
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

    pub(super) fn lower_expr_null(&mut self) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I64,
            value: Operand::ConstInt(0),
        });
        self.current_block_data.value_types.insert(v, SsaType::Null);
        v
    }

    pub(super) fn lower_expr_number(&mut self, n: i64) -> Value {
        self.lower_expr_number_inner(n, SsaType::Usize)
    }

    pub(super) fn lower_expr_number_inner(&mut self, n: i64, ty: SsaType) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: ty.clone(),
            value: Operand::ConstInt(n),
        });
        self.current_block_data.value_types.insert(v, ty);
        v
    }

    pub(super) fn handle_field_access(
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
                        let drop_kind = if nullable_owned.is_some() {
                            DropKind::Undroppable // already handled above
                        } else {
                            f.field_type.drop_kind()
                        };
                        if drop_kind.is_droppable() {
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
                                let mut resolver = FnAllocatorResolver {
                                    var_map: &self.var_map,
                                    context: self.context.clone(),
                                    dep_graph: self.dep_graph,
                                };
                                let mut emitter = DropEmitter::new(
                                    &mut self.current_block_data,
                                    self.context.clone(),
                                    self.struct_mangled_map,
                                    self.struct_field_offsets,
                                    self.structs,
                                    self.enums,
                                    self.allocator_kind,
                                    self.glue_registry,
                                );
                                match &drop_kind {
                                    DropKind::OwnedPointer {
                                        pointee,
                                        pointee_ty,
                                        allocator,
                                    } => {
                                        if matches!(pointee_ty, HirType::Slice(_)) {
                                            emitter.emit_owned_pointer_drop(
                                                None,
                                                pointee,
                                                pointee_ty,
                                                allocator,
                                                field_addr,
                                                false,
                                                None,
                                                &mut resolver,
                                                span,
                                            );
                                        } else {
                                            let old_ptr = emitter.current_block_data.fresh_value();
                                            emitter.emit(Instruction::Load {
                                                dest: old_ptr,
                                                ptr: Operand::Value(field_addr),
                                            });
                                            let pointee_ssa =
                                                lower_type_hir(pointee_ty, emitter.enums);
                                            emitter
                                                .current_block_data
                                                .value_types
                                                .insert(old_ptr, pointee_ssa);
                                            emitter.emit_owned_pointer_drop(
                                                None,
                                                pointee,
                                                pointee_ty,
                                                allocator,
                                                old_ptr,
                                                false,
                                                None,
                                                &mut resolver,
                                                span,
                                            );
                                        }
                                    }
                                    DropKind::Type(_) => {
                                        emitter.emit_element_drop(
                                            &drop_kind,
                                            field_addr,
                                            &mut resolver,
                                            span,
                                        );
                                    }
                                    DropKind::Slice {
                                        element,
                                        element_ty,
                                    } => {
                                        emitter.emit_slice_loop_drop(
                                            element,
                                            element_ty,
                                            field_addr,
                                            &mut resolver,
                                            span,
                                        );
                                    }
                                    DropKind::Undroppable => {}
                                }
                            }
                        }
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

    pub(super) fn lower_expr_as_u32(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
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

    pub(super) fn lower_expr_binary(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: &Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        self.lower_expr_binary_expected(left, op, right, None)
    }

    pub(super) fn lower_expr_binary_expected(
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

    pub(super) fn lower_struct_init(
        &mut self,
        name: &HirExpr,
        args: &[HirFieldInit<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        let struct_name = match name {
            HirExpr::Ident(n, _) => *n,
            other => panic!("StructInit name must be identifier; got {:?}", other),
        };

        let structs = self.structs;
        let hir_struct = structs
            .get(&struct_name)
            .unwrap_or_else(|| panic!("Struct {} not found at {span}", struct_name));
        let field_types: Vec<SsaType> = hir_struct
            .fields
            .iter()
            .map(|f| lower_type_hir(&f.field_type, self.enums))
            .collect();

        let mut inits: Vec<(StrId, SsaType, FieldInitVal)> = Vec::with_capacity(args.len());
        for arg in args {
            let idx = hir_struct
                .fields
                .iter()
                .position(|f| f.name == arg.name)
                .unwrap_or_else(|| {
                    panic!("Struct {} has no field {} at {span}", struct_name, arg.name)
                });
            let field_ty = field_types[idx].clone();
            if Self::is_move_by_value(&field_ty) {
                self.record_arg_move(&arg.value);
            }
            let init = self.lower_init_operand(&arg.value, &field_ty);
            inits.push((arg.name, field_ty, init));
        }

        let alloc_ty = SsaType::User(struct_name, field_types);
        let obj = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: obj,
            ty: alloc_ty.clone(),
            count: 0,
        });
        self.current_block_data.value_types.insert(obj, alloc_ty);

        let offsets_map = self.struct_field_offsets;
        let offsets = offsets_map
            .get(&struct_name)
            .unwrap_or_else(|| panic!("Unknown struct {} when initializing", struct_name));
        for (fname, fty, init) in inits {
            let offset = *offsets
                .get(&fname)
                .unwrap_or_else(|| panic!("Unknown field {} on struct {}", fname, struct_name));
            self.store_init(obj, offset, &fty, init);
        }

        self.store_vtable_if_any(obj, struct_name);
        obj
    }

    pub(super) fn lower_call_args(
        &mut self,
        args: &[HirExpr<'a, 'bump>],
        param_types: &[SsaType],
        param_offset: usize,
    ) -> SmallVec<Operand, 8> {
        let mut ops: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            let pty = param_types.get(i + param_offset).cloned();
            if pty.as_ref().map_or(false, |t| Self::is_move_by_value(t)) {
                self.record_arg_move(a);
            }
            let v = match &pty {
                Some(t) => self.lower_arg_expected(a, t),
                None => self.lower_expr(a),
            };
            ops.push(Operand::Value(v));
        }
        ops
    }

    pub(super) fn lower_arg_expected(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param_ty: &SsaType,
    ) -> Value {
        match param_ty {
            SsaType::Nullable(_) => self.lower_nullable_arg(arg, param_ty),
            _ => self.lower_expr_expected(arg, param_ty),
        }
    }

    pub(super) fn lower_nullable_arg(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param_ty: &SsaType,
    ) -> Value {
        let SsaType::Nullable(inner) = param_ty else {
            unreachable!()
        };

        if matches!(arg, HirExpr::Null(_)) {
            return self.lower_null_ssa(param_ty);
        }

        // Literals/arithmetic take the payload type, not the nullable wrapper.
        let v = match arg {
            HirExpr::Number(..) | HirExpr::Binary { .. } => self.lower_expr_expected(arg, inner),
            _ => self.lower_expr(arg),
        };

        match self.current_block_data.value_types.get(&v).cloned() {
            Some(SsaType::Null) => self.lower_null_ssa(param_ty),
            Some(SsaType::Nullable(_)) => v,
            _ => self.wrap_into_nullable(v, param_ty),
        }
    }

    pub(super) fn lower_call_expr(
        &mut self,
        callee: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        if let Some(mangled) = self.try_flatten_module_path(callee) {
            let param_types = self.param_types_of(&mangled);
            let arg_ops = self.lower_call_args(args, &param_types, 0);

            let dest = self.current_block_data.fresh_value();
            self.emit(Instruction::Call {
                dest: Some(dest),
                func: Operand::FunctionRef(mangled),
                args: arg_ops,
            });

            let ret_ty = self
                .funcs
                .get(&mangled)
                .or_else(|| self.global_funcs.get(&mangled))
                .map(|f| f.ret_type.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "lower_call: unknown non-extern function `{:?}`, not in funcs table or global_funcs",
                        mangled
                    )
                });
            self.current_block_data.value_types.insert(dest, ret_ty);
            return dest;
        }

        match callee {
            HirExpr::Ident(fname, _) => {
                if let Some(&ptr_val) = self.var_map.get(fname) {
                    let (param_types, ret_ty) = match self
                        .current_block_data
                        .value_types
                        .get(&ptr_val)
                    {
                        Some(SsaType::FuncPointer {
                            params,
                            return_type,
                        }) => (params.clone(), (**return_type).clone()),
                        other => panic!(
                            "lower_call: `{}` is called but its value type isn't a function pointer: {:?}",
                            fname, other
                        ),
                    };

                    let arg_ops = self.lower_call_args(args, &param_types, 0);

                    let dest = self.current_block_data.fresh_value();
                    self.emit(Instruction::Call {
                        dest: Some(dest),
                        func: Operand::Value(ptr_val),
                        args: arg_ops,
                    });
                    self.current_block_data.value_types.insert(dest, ret_ty);
                    return dest;
                }

                let param_types = self.param_types_of(fname);
                let arg_ops = self.lower_call_args(args, &param_types, 0);

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(fname.clone()),
                    args: arg_ops,
                });

                let ret_ty = self
                    .funcs
                    .get(fname)
                    .or_else(|| self.global_funcs.get(fname))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or_else(|| {
                        panic!(
                            "lower_call: unknown function `{:?}`, not in funcs table",
                            fname
                        )
                    });
                self.current_block_data.value_types.insert(dest, ret_ty);
                dest
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
            } => self.lower_method_call_expr(object, *field, args, span),

            HirExpr::ModuleAccess(acc) if acc.path.len() == 1 => {
                self.lower_static_call(acc.path[0], acc.member, args)
            }

            other => unimplemented!(
                "Non-identifier callee not yet supported in Call: {:?}",
                other
            ),
        }
    }

    pub(super) fn lower_static_call(
        &mut self,
        type_name: StrId,
        method: StrId,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        let struct_key = self
            .resolve_static_receiver_struct_name(type_name, method)
            .or_else(|| {
                let type_module_idx = self
                    .module_named_imports
                    .get(&self.module_idx)
                    .and_then(|named| named.get(&type_name))
                    .copied()
                    .unwrap_or(self.module_idx);

                let mangled = self.dep_graph.borrow().mangle_type_name(
                    type_module_idx,
                    type_name,
                    &self.context,
                );

                if self.struct_mangled_map.contains_key(&mangled) {
                    Some(mangled)
                } else {
                    self.struct_mangled_map
                        .keys()
                        .find(|k| **k == type_name)
                        .copied()
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] could not resolve type {} for static call {}",
                    type_name, method,
                )
            });

        let direct_name = *self
            .struct_mangled_map
            .get(&struct_key)
            .and_then(|mmap| mmap.get(&method))
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] struct `{}` has no static method `{}` in struct_mangled_map",
                    struct_key, method,
                )
            });

        let param_types: Vec<SsaType> = self
            .funcs
            .get(&direct_name)
            .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
            .unwrap_or_default();

        let arg_ops: SmallVec<Operand, 8> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if Self::is_move_by_value(param_types.get(i).unwrap()) {
                    self.record_arg_move(a);
                }
                Operand::Value(self.lower_expr(a))
            })
            .collect();

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Call {
            dest: Some(dest),
            func: Operand::FunctionRef(direct_name),
            args: arg_ops,
        });

        let ret_ty = self
            .funcs
            .get(&direct_name)
            .or_else(|| self.global_funcs.get(&direct_name))
            .map(|f| f.ret_type.clone())
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] resolved `{}` but it isn't in funcs or global_funcs",
                    direct_name
                )
            });
        self.current_block_data.value_types.insert(dest, ret_ty);
        dest
    }

    pub(super) fn lower_place_addr(
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

    pub(super) fn lower_field_addr(
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

    pub(super) fn resolve_static_receiver_struct_name(
        &self,
        bare_name: StrId,
        field: StrId,
    ) -> Option<StrId> {
        if let Some(type_module_idx) = self
            .module_named_imports
            .get(&self.module_idx)
            .and_then(|named| named.get(&bare_name))
            .copied()
        {
            let mangled =
                self.dep_graph
                    .borrow()
                    .mangle_type_name(type_module_idx, bare_name, &self.context);
            if let Some(mmap) = self.struct_mangled_map.get(&mangled) {
                if mmap.contains_key(&field) {
                    return Some(mangled);
                }
            }
        }

        let pkg = self.dep_graph.borrow().get_module_package(self.module_idx);
        let bare_str = self.context.resolve_string(&bare_name);
        let candidate = pkg.map(|p| {
            let pkg_str = self.context.resolve_string(&p).replace("::", "_");
            StrId(intern_fmt!(self.context, "{}_{}", pkg_str, bare_str))
        });

        if let Some(cand) = candidate {
            if let Some(mmap) = self.struct_mangled_map.get(&cand) {
                if mmap.contains_key(&field) {
                    return Some(cand);
                }
            }
        }

        None
    }

    pub(super) fn lower_method_call_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
        args: &[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
    ) -> Value {
        if let HirExpr::Ident(scope_name, _) = object {
            if !self.var_map.contains_key(scope_name) {
                let mangled_struct_name =
                    self.resolve_static_receiver_struct_name(*scope_name, field);

                let direct_name: Option<StrId> = mangled_struct_name
                    .and_then(|cls| self.struct_mangled_map.get(&cls))
                    .and_then(|mmap| mmap.get(&field))
                    .copied()
                    .or_else(|| self.resolve_module_access_callee(&[*scope_name], field));

                let Some(direct_name) = direct_name else {
                    panic!(
                        "lower_method_call (static): could not resolve `{}.{}` to a mangled \
                         function via struct_mangled_map. resolved struct key: {:?} at span {}",
                        scope_name, field, mangled_struct_name, span,
                    );
                };

                let param_types: Vec<SsaType> = self
                    .funcs
                    .get(&direct_name)
                    .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
                    .unwrap_or_default();

                let mut operands: SmallVec<Operand, 8> = SmallVec::new();
                for (i, a) in args.iter().enumerate() {
                    if Self::is_move_by_value(param_types.get(i).unwrap()) {
                        self.record_arg_move(a);
                    }
                    operands.push(Operand::Value(self.lower_expr(a)));
                }

                let dest: Value = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(direct_name),
                    args: operands,
                });

                let ret_ty = self
                    .funcs
                    .get(&direct_name)
                    .or_else(|| self.global_funcs.get(&direct_name))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or_else(|| {
                        unreachable!(
                            "lower_method_call (static): resolved `{}` via struct_mangled_map but \
                             it isn't in funcs or global_funcs; registry inconsistency",
                            self.context.resolve_string(&direct_name)
                        )
                    });
                self.current_block_data.value_types.insert(dest, ret_ty);
                return dest;
            }
        }

        let obj_val: Value = self.lower_expr_as_receiver(object);
        let mut operands: SmallVec<Operand, 8> = SmallVec::new();

        let maybe_cls_name_ssa: Option<SsaType> =
            self.current_block_data.value_types.get(&obj_val).cloned();

        if let Some(prim) = self.slice_primitive_of(field) {
            if maybe_cls_name_ssa
                .as_ref()
                .and_then(Self::classify_indexed_container)
                .is_some()
            {
                return self.lower_slice_primitive(prim, object, obj_val, args);
            }
        }

        let cls_name_id: Option<StrId> = maybe_cls_name_ssa
            .as_ref()
            .and_then(|ty| self.resolve_receiver_target_key(ty));

        let param_types: Vec<SsaType> = cls_name_id
            .and_then(|cls| self.struct_mangled_map.get(&cls))
            .and_then(|mmap| mmap.get(&field))
            .and_then(|mangled| self.funcs.get(mangled))
            .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
            .unwrap_or_default();

        if let Some(_) = cls_name_id {
            let receiver_is_moved = param_types
                .first()
                .map(|ty| matches!(ty, SsaType::User(_, _)))
                .unwrap_or(false);
            if receiver_is_moved {
                self.record_arg_move(object);
            }
        }

        operands.push(Operand::Value(obj_val));
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i + 1).unwrap_or(&SsaType::I64)) {
                self.record_arg_move(a);
            }
            let av = self.lower_expr(a);
            operands.push(Operand::Value(av));
        }

        if let Some(value) = self.emit_call_expr(field, obj_val, &mut operands, cls_name_id) {
            return value;
        }

        panic!(
            "[lower_method_call] no mangled mapping or vtable slot found for method `{}` on struct `{:?}` at span {}.",
            self.context.resolve_string(&field),
            cls_name_id.map(|id| self.context.resolve_string(&id).to_string()),
            span
        );
    }

    pub(super) fn lower_expr_as_receiver_raw(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
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

    pub(super) fn lower_expr_as_receiver(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
        let v = self.lower_expr_as_receiver_raw(object);
        self.canonicalize_receiver(v)
    }

    pub(super) fn canonicalize_receiver(&mut self, v: Value) -> Value {
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

    pub(super) fn emit_call_expr(
        &mut self,
        field: StrId,
        obj_val: Value,
        operands: &mut SmallVec<Operand, 8>,
        maybe_cls_name: Option<StrId>,
    ) -> Option<Value> {
        let Some(cls_name) = maybe_cls_name else {
            return None;
        };

        let mmap = self.struct_mangled_map.get(&cls_name).or_else(|| {
            self.struct_mangled_map.iter().find_map(
                |(k, v)| {
                    if *k == cls_name { Some(v) } else { None }
                },
            )
        });

        if let Some(mmap) = mmap {
            let field_str = self.context.resolve_string(&field);
            let mangled_name = mmap.get(&field).copied().or_else(|| {
                mmap.iter().find_map(|(k, v)| {
                    if self.context.resolve_string(k) == field_str {
                        Some(*v)
                    } else {
                        None
                    }
                })
            });

            if let Some(mangled_name) = mangled_name {
                let actual_func_name = if self.funcs.contains_key(&mangled_name) {
                    mangled_name
                } else {
                    panic!(
                        "uh oh. failed with {mangled_name} \n\n{:?}",
                        self.funcs.keys()
                    )
                };

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(actual_func_name),
                    args: operands.clone(),
                });

                let ret_ty = self
                    .funcs
                    .get(&actual_func_name)
                    .map(|f| f.ret_type.clone())
                    .unwrap_or(SsaType::I64);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        let struct_slots = self.struct_method_slots.get(&cls_name).or_else(|| {
            self.struct_method_slots
                .iter()
                .find_map(|(k, v)| if *k == cls_name { Some(v) } else { None })
        });

        if let Some(struct_slots) = struct_slots {
            if let Some(slot_idx) = struct_slots.get(&field) {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::InterfaceDispatch {
                    dest: Some(dest),
                    object: obj_val,
                    method_slot: *slot_idx,
                    args: operands.clone(),
                });
                let ret_ty = self
                    .interface_methods
                    .get(&cls_name)
                    .and_then(|methods| methods.iter().find(|(name, _, _)| name == &field))
                    .map(|(_, _, ret)| ret.clone())
                    .unwrap_or(SsaType::I64);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        let iface_slots = self.interface_method_slots.get(&cls_name).or_else(|| {
            self.interface_method_slots
                .iter()
                .find_map(|(k, v)| if *k == cls_name { Some(v) } else { None })
        });

        if let Some(iface_slots) = iface_slots {
            if let Some(slot_idx) = iface_slots.get(&field) {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::InterfaceDispatch {
                    dest: Some(dest),
                    object: obj_val,
                    method_slot: *slot_idx,
                    args: operands.clone(),
                });
                let ret_ty = self
                    .struct_vtable_slots
                    .get(&cls_name)
                    .filter(|slots| *slot_idx < slots.len())
                    .and_then(|slots| self.funcs.get(&slots[*slot_idx]))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or(SsaType::Void);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        None
    }

    pub(super) fn lower_interface_call(
        &mut self,
        callee: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
        interface: StrId,
    ) -> Value {
        let HirExpr::FieldAccess {
            object,
            field,
            span: _,
        } = callee
        else {
            panic!("InterfaceCall callee not FieldAccess; unsupported shape")
        };

        let obj_val = self.lower_expr(object);

        let param_types: Vec<SsaType> = self
            .interface_methods
            .get(&interface)
            .and_then(|methods| methods.iter().find(|(name, _, _)| name == field))
            .map(|(_, params, _)| params.clone())
            .unwrap_or_default();

        if matches!(param_types.first(), Some(SsaType::Dyn)) {
            self.record_arg_move(object);
        }

        let mut operands: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i).unwrap()) {
                self.record_arg_move(a);
            }
            operands.push(Operand::Value(self.lower_expr(a)));
        }

        let iface_id = *self.interface_id_map.get(&interface).unwrap_or_else(|| {
            panic!(
                "Unknown interface {} in InterfaceCall",
                self.context.resolve_string(&*interface)
            )
        });

        let iface_slot_map = self
            .interface_method_slots
            .get(&interface)
            .unwrap_or_else(|| {
                panic!(
                    "Interface {} has no method slots",
                    self.context.resolve_string(&*interface)
                )
            });

        let method_slot_in_iface = iface_slot_map.get(field).unwrap_or_else(|| {
            panic!(
                "Interface {} has no method {}",
                self.context.resolve_string(&*interface),
                self.context.resolve_string(&*field)
            )
        });

        let interface_val = match self.current_block_data.value_types.get(&obj_val).cloned() {
            Some(SsaType::User(ref _name, _args)) => {
                let upcast_dest = self.current_block_data.fresh_value();
                self.emit(Instruction::UpcastToInterface {
                    dest: upcast_dest,
                    object: obj_val,
                    interface_id: iface_id,
                });

                self.current_block_data
                    .value_types
                    .insert(upcast_dest, SsaType::Interface(interface));
                upcast_dest
            }
            _ => obj_val,
        };

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::InterfaceDispatch {
            dest: Some(dest),
            object: interface_val,
            method_slot: *method_slot_in_iface,
            args: operands,
        });

        self.current_block_data
            .value_types
            .insert(dest, SsaType::I64);
        dest
    }

    pub(super) fn get_field_offset(&mut self, obj: &Value, field: StrId) -> usize {
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

    pub(super) fn handle_deref_assign(
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

    pub(super) fn emit_elem_addr(
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

    pub(super) fn payload_layout(field_tys: &[SsaType]) -> (Vec<usize>, usize) {
        let target = TargetInfo { ptr_bytes: 8 };
        let mut cursor = 0usize;
        let mut offsets = Vec::with_capacity(field_tys.len());
        for ty in field_tys {
            let (size, align) = ir::layout::layout_of_ssa(ty, target)
                .map(|l| (l.size, l.align))
                .unwrap_or((8, 8));
            cursor = Self::align_up(cursor, align);
            offsets.push(cursor);
            cursor += size;
        }
        (offsets, cursor)
    }

    pub(super) fn lower_init_operand(
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

    pub(super) fn store_init(
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

    pub(super) fn store_const_u8(&mut self, base: Value, offset: usize, v: i64) {
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

    pub(super) fn field_addr(&mut self, base: Value, offset: usize, ty: &SsaType) -> Value {
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

    pub(super) fn struct_field_ssa_type(&self, obj: Value, field: StrId) -> Option<SsaType> {
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

    pub(super) fn store_field_value(
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

    pub(super) fn classify_indexed_container(ty: &SsaType) -> Option<IndexedContainer> {
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

    pub(super) fn store_vtable_if_any(&mut self, obj: Value, struct_name: StrId) {
        let Some(vslots) = self.struct_vtable_slots.get(&struct_name) else {
            return;
        };
        if vslots.is_empty() {
            return;
        }

        let vtable_name =
            optimized_string_buffering::make_vtable_name(struct_name, self.context.clone());
        self.emit(Instruction::StoreField {
            base: Operand::Value(obj),
            offset: 0usize,
            value: Operand::GlobalRef(vtable_name),
        });
    }

    pub(super) fn narrowed_field_value(&self, expr: &HirExpr<'a, 'bump>) -> Option<Value> {
        let (root, path) = self.static_field_path_mir(expr)?;
        self.narrowed_fields.get(&(root, path)).copied()
    }

    pub(super) fn param_types_of(&self, name: &StrId) -> Vec<SsaType> {
        self.funcs
            .get(name)
            .or_else(|| self.global_funcs.get(name))
            .map(|f| f.params.iter().map(|(_, t)| t.clone()).collect())
            .unwrap_or_default()
    }

    pub(super) fn resolve_receiver_target_key(&self, ty: &SsaType) -> Option<StrId> {
        match ty {
            SsaType::User(name, _) => Some(*name),
            SsaType::Enum { name, .. } => Some(*name),
            SsaType::Interface(name) => Some(*name),
            SsaType::Pointer(inner) | SsaType::Owned(inner) => {
                self.resolve_receiver_target_key(inner)
            }
            other => self.builtin_target_key(other),
        }
    }

    pub(super) fn builtin_target_key(&self, ty: &SsaType) -> Option<StrId> {
        let prim = |s: &str| Some(StrId(self.context.thread_local().intern(s)));

        match ty {
            SsaType::I8 => prim("i8"),
            SsaType::I16 => prim("i16"),
            SsaType::I32 => prim("i32"),
            SsaType::I64 => prim("i64"),
            SsaType::I128 => prim("i128"),
            SsaType::U8 => prim("u8"),
            SsaType::U16 => prim("u16"),
            SsaType::U32 => prim("u32"),
            SsaType::U64 => prim("u64"),
            SsaType::U128 => prim("u128"),
            SsaType::Isize => prim("isize"),
            SsaType::Usize => prim("usize"),
            SsaType::F32 => prim("f32"),
            SsaType::F64 => prim("f64"),
            SsaType::Bool => prim("bool"),
            SsaType::String => prim("str"),
            SsaType::Char => prim("char"),
            SsaType::Slice(elem) | SsaType::Array(elem, _) => {
                let elem_key = match elem.as_ref() {
                    SsaType::User(n, _) => Some(*n),
                    other => self.builtin_target_key(other),
                };
                if let Some(ek) = elem_key {
                    let specialized = StrId(intern_fmt!(self.context, "slice_{}", ek));
                    if self.struct_mangled_map.contains_key(&specialized) {
                        return Some(specialized);
                    }
                }
                prim("slice")
            }
            SsaType::Owned(inner) => self.resolve_receiver_target_key(inner),
            _ => None,
        }
    }
}
