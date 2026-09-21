use ir::{
    hir::{HirEnum, HirExpr, HirMatchArm, HirPattern, HirStmt, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, BlockId, Instruction, Operand, SsaType, Value},
};

use smallvec::{SmallVec, smallvec};

use crate::midend::{
    copy_analysis::drop_tracking::{DropMoveState, DropScope},
    ir::mir_lowering::FunctionLowerer,
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn lower_pattern_test(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) -> Option<Value> {
        if let Some(ty) = scrutinee_ty {
            if matches!(ty, SsaType::Nullable(_)) && Self::pattern_needs_nonnull(pattern) {
                return self.lower_nullable_inner_test(pattern, scrutinee, ty);
            }
        }
        match pattern {
            HirPattern::Wildcard | HirPattern::Ident(_) => None,

            HirPattern::Array(elems) => {
                let SsaType::Array(elem_ty, _) = scrutinee_ty.expect(
                "[lower_pattern_test] array pattern has no scrutinee type; the type checker should have caught this"
            ) else {
                panic!("[lower_pattern_test] array pattern used on a non-array scrutinee");
            };
                let elem_ty = (**elem_ty).clone();
                let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
                    .expect("[lower_pattern_test] array element type has no known size")
                    as i64;

                let mut combined: Option<Value> = None;
                for (i, elem_pat) in elems.iter().enumerate() {
                    let addr = self.current_block_data.fresh_value();
                    self.emit(Instruction::FieldAddr {
                        dest: addr,
                        base: Operand::Value(scrutinee),
                        offset: (i as i64 * elem_size) as usize,
                    });
                    self.current_block_data
                        .value_types
                        .insert(addr, SsaType::Pointer(Box::new(elem_ty.clone())));

                    let elem_val = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest: elem_val,
                        ptr: Operand::Value(addr),
                    });
                    self.current_block_data
                        .value_types
                        .insert(elem_val, elem_ty.clone());

                    if let Some(cond) = self.lower_pattern_test(elem_pat, elem_val, Some(&elem_ty))
                    {
                        combined = Some(self.and_conds(combined, cond));
                    }
                }
                combined
            }

            HirPattern::Struct { name, fields } => {
                if let Some(offsets) = self.struct_field_offsets.get(name) {
                    let hir_struct = self.structs.get(name);
                    let mut combined: Option<Value> = None;
                    for (field_name, field_pat) in fields.iter() {
                        let offset = *offsets.get(field_name).unwrap_or_else(|| {
                            panic!(
                                "lower_pattern_test: unknown field `{}` on struct `{}`",
                                field_name, name
                            )
                        });
                        let field_ty = hir_struct
                            .and_then(|s| s.fields.iter().find(|f| f.name == *field_name))
                            .map(|f| lower_type_hir(&f.field_type, self.enums))
                            .unwrap_or(SsaType::I64);
                        let field_val = self.current_block_data.fresh_value();
                        if Self::is_aggregate_ssa_type(&field_ty) {
                            self.emit(Instruction::FieldAddr {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset,
                            });
                        } else {
                            self.emit(Instruction::LoadField {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset,
                            });
                        }
                        self.current_block_data
                            .value_types
                            .insert(field_val, field_ty.clone());
                        if let Some(cond) =
                            self.lower_pattern_test(field_pat, field_val, Some(&field_ty))
                        {
                            combined = Some(self.and_conds(combined, cond));
                        }
                    }
                    return combined;
                }

                let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
                panic!(
                    "lower_pattern_test: `{}` is neither a known struct nor is the scrutinee ({:?}) \
                     an enum",
                    name, scrutinee_ty
                );
            });
                let hir_enum = self.resolve_enum_for_variant(enum_name, name);
                let (expected_tag, variant_def) = hir_enum
                    .variants
                    .iter()
                    .enumerate()
                    .find(|(_, v)| v.name == *name)
                    .map(|(i, v)| (i as i64, v))
                    .unwrap_or_else(|| {
                        panic!(
                            "lower_pattern_test: enum `{}` has no variant `{}`",
                            enum_name, name
                        )
                    });

                let tag_val = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: tag_val,
                    base: Operand::Value(scrutinee),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(tag_val, SsaType::I64);

                let mut combined = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: combined,
                    op: BinOp::Eq,
                    left: Operand::Value(tag_val),
                    right: Operand::ConstInt(expected_tag),
                });
                self.current_block_data
                    .value_types
                    .insert(combined, SsaType::Bool);
                let mut result = Some(combined);

                let target = TargetInfo { ptr_bytes: 8 };
                let mut cursor = 0usize;
                for vf in variant_def.fields.iter() {
                    let field_ssa_ty = lower_type_hir(&vf.field_type, self.enums);
                    let align = ir::layout::alignof_ssa(&field_ssa_ty, target).unwrap_or(8);
                    cursor = Self::align_up(cursor, align);

                    if let Some((_, field_pat)) = fields.iter().find(|(fname, _)| fname == &vf.name)
                    {
                        let field_val = self.current_block_data.fresh_value();
                        if Self::is_aggregate_ssa_type(&field_ssa_ty) {
                            self.emit(Instruction::FieldAddr {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset: 8 + cursor,
                            });
                        } else {
                            self.emit(Instruction::LoadField {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset: 8 + cursor,
                            });
                        }
                        self.current_block_data
                            .value_types
                            .insert(field_val, field_ssa_ty.clone());
                        if let Some(cond) =
                            self.lower_pattern_test(field_pat, field_val, Some(&field_ssa_ty))
                        {
                            combined = self.and_conds(result, cond);
                            result = Some(combined);
                        }
                    }

                    let size = ir::layout::sizeof_ssa(&field_ssa_ty, target).unwrap_or(8);
                    cursor += size;
                }

                result
            }

            HirPattern::Or(alts) => {
                let mut combined: Option<Value> = None;
                let mut always_matches = false;
                for alt in alts.iter() {
                    match self.lower_pattern_test(alt, scrutinee, scrutinee_ty) {
                        None => always_matches = true,
                        Some(cond) => combined = Some(self.or_conds(combined, cond)),
                    }
                }
                if always_matches { None } else { combined }
            }

            HirPattern::Boolean(b) => {
                let lit = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: lit,
                    ty: SsaType::Bool,
                    value: Operand::ConstBool(*b),
                });
                self.current_block_data
                    .value_types
                    .insert(lit, SsaType::Bool);
                let cmp = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: cmp,
                    op: BinOp::Eq,
                    left: Operand::Value(scrutinee),
                    right: Operand::Value(lit),
                });
                self.current_block_data
                    .value_types
                    .insert(cmp, SsaType::Bool);
                Some(cmp)
            }

            HirPattern::Number(n) => {
                let lit = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: lit,
                    ty: SsaType::I64,
                    value: Operand::ConstInt(*n),
                });
                self.current_block_data
                    .value_types
                    .insert(lit, SsaType::I64);
                let cmp = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: cmp,
                    op: BinOp::Eq,
                    left: Operand::Value(scrutinee),
                    right: Operand::Value(lit),
                });
                self.current_block_data
                    .value_types
                    .insert(cmp, SsaType::Bool);
                Some(cmp)
            }

            HirPattern::String(s) => {
                let lit = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: lit,
                    ty: SsaType::String,
                    value: Operand::ConstString(*s),
                });
                self.current_block_data
                    .value_types
                    .insert(lit, SsaType::String);

                let streq_fn = StrId(self.context.thread_local().intern("__zeta_streq"));
                let cmp = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(cmp),
                    func: Operand::FunctionRef(streq_fn),
                    args: smallvec![Operand::Value(scrutinee), Operand::Value(lit)],
                });
                self.current_block_data
                    .value_types
                    .insert(cmp, SsaType::Bool);
                Some(cmp)
            }

            HirPattern::EnumVariant { variant, .. } => {
                let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
                panic!(
                    "lower_pattern_test: enum pattern `{}(..)` used on a non-enum scrutinee ({:?}); \
                     the type checker should have caught this",
                    variant, scrutinee_ty
                );
            });
                let hir_enum = self.resolve_enum_for_variant(enum_name, variant);

                let expected_tag = hir_enum
                    .variants
                    .iter()
                    .enumerate()
                    .find(|(_, v)| v.name == *variant)
                    .map(|(i, _)| i as i64)
                    .unwrap_or_else(|| {
                        panic!(
                            "lower_pattern_test: enum `{}` has no variant `{}`",
                            enum_name, variant
                        )
                    });

                let tag_val = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: tag_val,
                    base: Operand::Value(scrutinee),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(tag_val, SsaType::I64);

                let cmp = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: cmp,
                    op: BinOp::Eq,
                    left: Operand::Value(tag_val),
                    right: Operand::ConstInt(expected_tag),
                });
                self.current_block_data
                    .value_types
                    .insert(cmp, SsaType::Bool);
                Some(cmp)
            }

            HirPattern::Null => {
                let ty = scrutinee_ty.expect(
                "[lower_pattern_test] `null` pattern has no scrutinee type; the type checker should have caught this"
            );

                let zero_cost_pointee: Option<SsaType> = if let Some(p) = ty.nullable_pointer_repr()
                {
                    Some(p.clone())
                } else if let SsaType::Pointer(inner) = ty {
                    Some((**inner).clone())
                } else {
                    None
                };

                if let Some(pointee_ty) = zero_cost_pointee {
                    let zero = self.current_block_data.fresh_value();
                    self.emit(Instruction::Const {
                        dest: zero,
                        ty: SsaType::Pointer(Box::new(pointee_ty.clone())),
                        value: Operand::ConstInt(0),
                    });
                    self.current_block_data
                        .value_types
                        .insert(zero, SsaType::Pointer(Box::new(pointee_ty)));

                    let cmp = self.current_block_data.fresh_value();
                    self.emit(Instruction::Binary {
                        dest: cmp,
                        op: BinOp::Eq,
                        left: Operand::Value(scrutinee),
                        right: Operand::Value(zero),
                    });
                    self.current_block_data
                        .value_types
                        .insert(cmp, SsaType::Bool);
                    Some(cmp)
                } else if ty.is_tagged_nullable() {
                    let tag_val = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: tag_val,
                        base: Operand::Value(scrutinee),
                        offset: 0,
                    });
                    self.current_block_data
                        .value_types
                        .insert(tag_val, SsaType::I8);

                    let cmp = self.current_block_data.fresh_value();
                    self.emit(Instruction::Binary {
                        dest: cmp,
                        op: BinOp::Eq,
                        left: Operand::Value(tag_val),
                        right: Operand::ConstInt(0),
                    });
                    self.current_block_data
                        .value_types
                        .insert(cmp, SsaType::Bool);
                    Some(cmp)
                } else {
                    panic!(
                        "[lower_pattern_test] `null` pattern used against non-nullable scrutinee \
                     type {:?}; the type checker should have caught this",
                        ty
                    );
                }
            }

            HirPattern::Tuple(_) => {
                todo!("tuple patterns aren't implemented upstream in lower_pattern either")
            }
        }
    }

    pub(super) fn extract_enum_name_from_ty<'b>(
        &'b self,
        ty: Option<&'b SsaType>,
    ) -> Option<&'b StrId> {
        let mut curr = ty?;
        loop {
            match curr {
                SsaType::Enum { name, .. } => return Some(name),
                SsaType::User(name, _) => return Some(name),
                SsaType::Pointer(inner) | SsaType::Owned(inner) | SsaType::Nullable(inner) => {
                    curr = inner.as_ref();
                }
                _ => return None,
            }
        }
    }

    pub(super) fn resolve_enum_for_variant(
        &self,
        enum_name: &StrId,
        variant: &StrId,
    ) -> &HirEnum<'a, 'bump> {
        self.enums.get(enum_name).unwrap_or_else(|| {
            println!("All enums: {:?}", self.enums.keys().collect::<Vec<_>>());
            panic!(
                "[resolve_enum_for_variant] unknown enum {}.{}",
                enum_name, variant
            )
        })
    }

    pub(super) fn bind_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let mut scrutinee = scrutinee;
        let mut scrutinee_ty = scrutinee_ty;
        if let Some(ty) = scrutinee_ty {
            if let SsaType::Nullable(inner) = ty {
                if Self::pattern_needs_nonnull(pattern) {
                    scrutinee = self.unwrap_known_nonnull(scrutinee, ty);
                    scrutinee_ty = Some(&**inner);
                }
            }
        }
        match pattern {
            HirPattern::Ident(name) => {
                let bound = match scrutinee_ty {
                    Some(ty) if matches!(ty, SsaType::Nullable(_)) => {
                        self.unwrap_known_nonnull(scrutinee, ty)
                    }
                    _ => scrutinee,
                };
                self.var_map.insert(*name, bound);
            }
            HirPattern::Array(elems) => {
                let SsaType::Array(elem_ty, _) = scrutinee_ty.expect(
                "bind_pattern: array pattern has no scrutinee type; the type checker should have caught this"
            ) else {
                panic!("bind_pattern: array pattern used on a non-array scrutinee");
            };
                let elem_ty = (**elem_ty).clone();
                let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
                    .expect("bind_pattern: array element type has no known size")
                    as i64;

                for (i, elem_pat) in elems.iter().enumerate() {
                    let addr = self.current_block_data.fresh_value();
                    self.emit(Instruction::FieldAddr {
                        dest: addr,
                        base: Operand::Value(scrutinee),
                        offset: (i as i64 * elem_size) as usize,
                    });
                    self.current_block_data
                        .value_types
                        .insert(addr, SsaType::Pointer(Box::new(elem_ty.clone())));

                    let elem_val = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest: elem_val,
                        ptr: Operand::Value(addr),
                    });
                    self.current_block_data
                        .value_types
                        .insert(elem_val, elem_ty.clone());

                    self.bind_pattern(elem_pat, elem_val, Some(&elem_ty));
                }
            }

            HirPattern::Struct { name, fields } => {
                if let Some(offsets) = self.struct_field_offsets.get(name) {
                    let hir_struct = self.structs.get(name);

                    for (field_name, field_pat) in fields.iter() {
                        let offset = *offsets.get(field_name).unwrap_or_else(|| {
                            panic!(
                                "[bind_pattern] unknown field `{}` on struct `{}`",
                                field_name, name
                            )
                        });
                        let field_ty = hir_struct
                            .and_then(|s| s.fields.iter().find(|f| f.name == *field_name))
                            .map(|f| lower_type_hir(&f.field_type, self.enums))
                            .unwrap_or(SsaType::I64);

                        let field_val = self.current_block_data.fresh_value();
                        if Self::is_aggregate_ssa_type(&field_ty) {
                            self.emit(Instruction::FieldAddr {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset,
                            });
                        } else {
                            self.emit(Instruction::LoadField {
                                dest: field_val,
                                base: Operand::Value(scrutinee),
                                offset,
                            });
                        }
                        self.current_block_data
                            .value_types
                            .insert(field_val, field_ty.clone());

                        self.bind_pattern(field_pat, field_val, Some(&field_ty));
                    }
                } else {
                    let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
                    panic!(
                        "[bind_pattern] `{}` is neither a known struct nor is the scrutinee ({:?}) \
                         an enum",
                        name, scrutinee_ty
                    );
                });

                    let hir_enum = self.resolve_enum_for_variant(enum_name, name);
                    let variant_def = hir_enum
                        .variants
                        .iter()
                        .find(|v| v.name == *name)
                        .unwrap_or_else(|| {
                            panic!(
                                "[bind_pattern] enum `{}` has no variant `{}`",
                                enum_name, name
                            )
                        });

                    // Compute per-field payload offsets (tag occupies the first 8 bytes).
                    let target = TargetInfo { ptr_bytes: 8 };
                    let mut cursor = 0usize;
                    for vf in variant_def.fields.iter() {
                        let field_ssa_ty = lower_type_hir(&vf.field_type, self.enums);
                        let align = ir::layout::alignof_ssa(&field_ssa_ty, target).unwrap_or(8);
                        cursor = Self::align_up(cursor, align);

                        // Only bind fields that appear in the pattern.
                        if let Some((_, field_pat)) =
                            fields.iter().find(|(fname, _)| fname == &vf.name)
                        {
                            let field_val = self.current_block_data.fresh_value();
                            if Self::is_aggregate_ssa_type(&field_ssa_ty) {
                                self.emit(Instruction::FieldAddr {
                                    dest: field_val,
                                    base: Operand::Value(scrutinee),
                                    offset: 8 + cursor,
                                });
                            } else {
                                self.emit(Instruction::LoadField {
                                    dest: field_val,
                                    base: Operand::Value(scrutinee),
                                    offset: 8 + cursor,
                                });
                            }
                            self.current_block_data
                                .value_types
                                .insert(field_val, field_ssa_ty.clone());

                            self.bind_pattern(field_pat, field_val, Some(&field_ssa_ty));
                        }

                        let size = ir::layout::sizeof_ssa(&field_ssa_ty, target).unwrap_or(8);
                        cursor += size;
                    }
                }
            }

            HirPattern::Or(alts) => {
                for alt in alts.iter() {
                    self.bind_pattern(alt, scrutinee, scrutinee_ty);
                }
            }
            HirPattern::EnumVariant {
                variant, bindings, ..
            } if !bindings.is_empty() => {
                let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
                panic!(
                    "bind_pattern: enum pattern `{}(..)` used on a non-enum scrutinee ({:?}); \
                     the type checker should have caught this",
                    variant, scrutinee_ty
                );
            });
                let hir_enum = self.resolve_enum_for_variant(enum_name, variant);
                let variant_def = hir_enum.variants.iter().find(|v| v.name == *variant)
                .unwrap_or_else(|| panic!(
                    "bind_pattern: enum `{}` has no variant `{}`; the type checker should have caught this",
                    enum_name, variant
                ));
                debug_assert_eq!(
                    bindings.len(),
                    variant_def.fields.len(),
                    "bind_pattern: binding count for variant `{}` doesn't match its field count; \
                 the type checker should have caught this",
                    variant
                );

                let target = TargetInfo { ptr_bytes: 8 };
                let mut cursor = 0usize;
                for (&binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
                    let field_ssa_ty = lower_type_hir(&field.field_type, self.enums);
                    let align = ir::layout::alignof_ssa(&field_ssa_ty, target).unwrap_or(8);
                    cursor = Self::align_up(cursor, align);

                    let dest = self.current_block_data.fresh_value();
                    if Self::is_aggregate_ssa_type(&field_ssa_ty) {
                        self.emit(Instruction::FieldAddr {
                            dest,
                            base: Operand::Value(scrutinee),
                            offset: 8 + cursor,
                        });
                    } else {
                        self.emit(Instruction::LoadField {
                            dest,
                            base: Operand::Value(scrutinee),
                            offset: 8 + cursor,
                        });
                    }
                    self.current_block_data
                        .value_types
                        .insert(dest, field_ssa_ty.clone());
                    self.var_map.insert(binding_name, dest); // (or recurse into bind_pattern for the Struct arm)

                    let size = ir::layout::sizeof_ssa(&field_ssa_ty, target).unwrap_or(8);
                    cursor += size;
                }
            }
            _ => {}
        }
    }

    pub(super) fn lower_match_expr(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_match_expr_inner(scrutinee, arms, span, None)
    }

    pub(super) fn lower_match_expr_inner(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
        expected: Option<&SsaType>,
    ) -> Value {
        let scrutinee_val = self.lower_expr(scrutinee);
        let scrutinee_ty = self
            .current_block_data
            .value_types
            .get(&scrutinee_val)
            .cloned();
        let narrowed_before = self.narrowed_fields.clone();

        let drop_before = self.drop_state.clone();
        let mut live_drop: Vec<DropMoveState<'a, 'bump>> = Vec::new();
        let vars_before = self.var_map.clone();
        let merge_bb = self.current_block_data.fresh_block();
        let mut incoming: SmallVec<(BlockId, Value), 4> = SmallVec::new();
        let mut any_reachable = false;

        for (arm_idx, arm) in arms.iter().enumerate() {
            let is_last = arm_idx + 1 == arms.len();
            let body_bb = self.current_block_data.new_block();
            let fail_bb = if is_last {
                None
            } else {
                Some(self.current_block_data.new_block())
            };

            let pattern_cond =
                self.lower_pattern_test(&arm.pattern, scrutinee_val, scrutinee_ty.as_ref());

            let cond = match (pattern_cond, arm.guard) {
                (Some(pc), Some(guard_expr)) => {
                    let guard_bb = self.current_block_data.new_block();
                    self.emit(Instruction::Branch {
                        cond: Operand::Value(pc),
                        then_bb: guard_bb,
                        else_bb: fail_bb.unwrap_or(body_bb),
                    });
                    self.current_block_data.switch_to(guard_bb);
                    Some(self.lower_expr(guard_expr))
                }
                (Some(pc), None) => Some(pc),
                (None, Some(guard_expr)) => Some(self.lower_expr(guard_expr)),
                (None, None) => None,
            };

            match (cond, fail_bb) {
                (Some(c), Some(fb)) => {
                    self.emit(Instruction::Branch {
                        cond: Operand::Value(c),
                        then_bb: body_bb,
                        else_bb: fb,
                    });
                }
                (Some(c), None) => {
                    let trap_bb = self.current_block_data.new_block();
                    self.emit(Instruction::Branch {
                        cond: Operand::Value(c),
                        then_bb: body_bb,
                        else_bb: trap_bb,
                    });
                    self.current_block_data.switch_to(trap_bb);
                    let abort_fn = StrId::from_static("abort");
                    self.emit(Instruction::Call {
                        dest: None,
                        func: Operand::FunctionRef(abort_fn),
                        args: SmallVec::new(),
                    });
                    self.emit(Instruction::Ret { value: None }); // unreachable; abort() doesn't return
                }
                (None, _) => {
                    self.emit(Instruction::Jump { target: body_bb });
                }
            }

            self.drop_state = drop_before.clone();
            self.var_map = vars_before.clone();
            self.current_block_data.switch_to(body_bb);
            self.narrowed_fields = narrowed_before.clone();
            self.scope_stack.push(DropScope { locals: Vec::new() });
            self.bind_pattern(&arm.pattern, scrutinee_val, scrutinee_ty.as_ref());
            let prior_null_arm = arms[..arm_idx]
                .iter()
                .any(|a| a.guard.is_none() && matches!(a.pattern, HirPattern::Null));
            self.adopt_owned_binding(scrutinee, &arm.pattern, prior_null_arm);
            let HirStmt::Block { body, span: _ } = arm.body else {
                panic!("match arm body must be a block")
            };
            let arm_val = self.lower_block_value_inner(body, expected);
            let arm_scope = self.scope_stack.pop().unwrap();
            if !self.block_terminated() {
                self.emit_scope_drops(&arm_scope, span);
                live_drop.push(self.drop_state.clone());
                let arm_end_bb = self.current_block_data.current_block; // read after the drops
                self.emit(Instruction::Jump { target: merge_bb });
                incoming.push((arm_end_bb, arm_val));
                any_reachable = true;
            }

            if let Some(fb) = fail_bb {
                self.current_block_data.switch_to(fb);
            }
        }

        self.narrowed_fields = narrowed_before;
        if !any_reachable {
            self.var_map = vars_before;
            return self.unreachable_value();
        }

        self.current_block_data.push_block(merge_bb);
        self.current_block_data.switch_to(merge_bb);
        if let Some(j) = DropMoveState::join_all(live_drop) {
            self.drop_state = j;
        }

        let ty = if incoming.is_empty() {
            SsaType::Void
        } else {
            self.reconcile_phi_type(&incoming, span)
        };

        if ty == SsaType::Void {
            return self.unit_value();
        }

        let result = self.current_block_data.fresh_value();
        self.emit(Instruction::Phi {
            dest: result,
            incoming,
        });
        self.current_block_data.value_types.insert(result, ty);
        result
    }

    pub(super) fn lower_nullable_inner_test(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        ty: &SsaType,
    ) -> Option<Value> {
        let SsaType::Nullable(inner) = ty else {
            unreachable!()
        };
        let is_null = self
            .lower_pattern_test(&HirPattern::Null, scrutinee, Some(ty))
            .expect("null test always yields a condition");

        let check_bb = self.current_block_data.new_block();
        let null_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(is_null),
            then_bb: null_bb,
            else_bb: check_bb,
        });

        // non-null: only here is it safe to unwrap and look inside
        self.current_block_data.switch_to(check_bb);
        let unwrapped = self.unwrap_known_nonnull(scrutinee, ty);
        let inner_cond = match self.lower_pattern_test(pattern, unwrapped, Some(&**inner)) {
            Some(c) => c,
            None => {
                let t = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: t,
                    ty: SsaType::Bool,
                    value: Operand::ConstBool(true),
                });
                self.current_block_data.value_types.insert(t, SsaType::Bool);
                t
            }
        };
        let check_end = self.current_block_data.current_block; // inner test may have added blocks
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(null_bb);
        let f = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: f,
            ty: SsaType::Bool,
            value: Operand::ConstBool(false),
        });
        self.current_block_data.value_types.insert(f, SsaType::Bool);
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(merge_bb);
        let result = self.current_block_data.fresh_value();
        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(check_end, inner_cond), (null_bb, f)],
        });
        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);
        Some(result)
    }

    pub(super) fn and_conds(&mut self, acc: Option<Value>, cond: Value) -> Value {
        match acc {
            None => cond,
            Some(prev) => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: v,
                    op: BinOp::BitAnd,
                    left: Operand::Value(prev),
                    right: Operand::Value(cond),
                });
                self.current_block_data.value_types.insert(v, SsaType::Bool);
                v
            }
        }
    }

    pub(super) fn or_conds(&mut self, acc: Option<Value>, cond: Value) -> Value {
        match acc {
            None => cond,
            Some(prev) => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: v,
                    op: BinOp::BitOr,
                    left: Operand::Value(prev),
                    right: Operand::Value(cond),
                });
                self.current_block_data.value_types.insert(v, SsaType::Bool);
                v
            }
        }
    }

    pub(super) fn pattern_needs_nonnull(pattern: &HirPattern<'bump>) -> bool {
        !matches!(
            pattern,
            HirPattern::Null | HirPattern::Wildcard | HirPattern::Ident(_) | HirPattern::Or(_)
        )
    }
}
