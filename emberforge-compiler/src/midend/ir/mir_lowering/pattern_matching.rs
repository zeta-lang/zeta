use ir::{
    hir::{HirEnum, HirExpr, HirMatchArm, HirPattern, HirStmt, StrId},
    ir_conversion::lower_type_hir,
    ir_hasher::HashMap,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, BlockId, Instruction, Operand, SsaType, Value},
};

use smallvec::{SmallVec, smallvec};

use crate::midend::{
    copy_analysis::drop_tracking::{DropMoveState, DropScope},
    ir::mir_lowering::FunctionLowerer,
};

/// A snapshot of `var_map` (source variable name -> current SSA value).
type VarSnapshot = HashMap<StrId, Value>;

/// The already-evaluated scrutinee of a `match`, bundled so it can be passed around as one thing.
struct MatchScrutinee<'x, 'a, 'bump> {
    /// The source expression (needed to decide whether a binding can take ownership of it).
    expr: &'x HirExpr<'a, 'bump>,
    /// The SSA value the expression evaluated to.
    val: Value,
    /// The value's recorded SSA type, if any.
    ty: Option<SsaType>,
}

/// What the arms of a `match` that *reach the merge block* collect for the merge step.
struct MatchJoin<'a, 'bump> {
    /// `(arm end block, variables visible after the match)` for each live arm.
    live_vars: Vec<(BlockId, VarSnapshot)>,
    /// The `drop_state` at the end of each live arm.
    live_drop: Vec<DropMoveState<'a, 'bump>>,
    /// `(arm end block, arm value)`: the operands of the result phi.
    incoming: SmallVec<(BlockId, Value), 4>,
}

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
    /// The target description used for every size/alignment query in this module (8-byte pointers).
    fn pattern_target_info() -> TargetInfo {
        TargetInfo { ptr_bytes: 8 }
    }

    /// Emits a constant of type `ty` and returns it (its type is recorded).
    /// Used for literal operands: `true`, `42`, `"text"`, and the `true`/`false` results of
    /// the nullable test.
    fn emit_pattern_const(&mut self, ty: SsaType, value: Operand) -> Value {
        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest,
            ty: ty.clone(),
            value,
        });
        self.current_block_data.value_types.insert(dest, ty);
        dest
    }

    /// Reads the sub-element of `base` that lives `offset` bytes in and has type `ty`.
    ///
    /// * **Aggregate** `ty`: aggregates are handled by address, so emit `FieldAddr`
    ///   (the "value" is the sub-element's address).
    /// * **Scalar** `ty`: emit `LoadField` (the value itself).
    ///
    /// The result's type is recorded as `ty` in both cases, so later code treats it uniformly.
    fn emit_pattern_slot_value(&mut self, base: Value, offset: usize, ty: &SsaType) -> Value {
        let dest = self.current_block_data.fresh_value();
        if Self::is_aggregate_ssa_type(ty) {
            self.emit(Instruction::FieldAddr {
                dest,
                base: Operand::Value(base),
                offset,
            });
        } else {
            self.emit(Instruction::LoadField {
                dest,
                base: Operand::Value(base),
                offset,
            });
        }
        self.current_block_data.value_types.insert(dest, ty.clone());
        dest
    }

    /// Reads element `index` of an array scrutinee: compute its address (`FieldAddr` at
    /// `index * elem_size`), then `Load` through it.
    ///
    /// Unlike [`Self::emit_pattern_slot_value`] this always loads, even for aggregate elements;
    /// that is the behaviour of the original code and is kept as is.
    fn emit_pattern_array_element(
        &mut self,
        base: Value,
        index: usize,
        elem_ty: &SsaType,
        elem_size: i64,
    ) -> Value {
        let addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: addr,
            base: Operand::Value(base),
            offset: (index as i64 * elem_size) as usize,
        });
        self.current_block_data.value_types.insert(
            addr,
            SsaType::Pointer(
                ir::ssa_ir::SsaPointerKind::UnsafeMut,
                Box::new(elem_ty.clone()),
            ),
        );

        let elem_val = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: elem_val,
            ptr: Operand::Value(addr),
        });
        self.current_block_data
            .value_types
            .insert(elem_val, elem_ty.clone());
        elem_val
    }

    /// Extracts `(element type, element size in bytes)` from an array scrutinee type.
    ///
    /// Panics (compiler-internal invariants, not user errors) if the type is missing, is not an
    /// array, or the element has no known size. `who` names the caller in the message.
    fn pattern_array_elem_info(scrutinee_ty: Option<&SsaType>, who: &str) -> (SsaType, i64) {
        let SsaType::Array(elem_ty, _) = scrutinee_ty.unwrap_or_else(|| {
            panic!(
                "{who}: array pattern has no scrutinee type; the type checker should have caught this"
            )
        }) else {
            panic!("{who}: array pattern used on a non-array scrutinee");
        };
        let elem_ty = (**elem_ty).clone();
        let elem_size = ir::layout::sizeof_ssa(&elem_ty, Self::pattern_target_info())
            .unwrap_or_else(|_| panic!("{who}: array element type has no known size"))
            as i64;
        (elem_ty, elem_size)
    }

    /// Looks up a struct field for a struct pattern: returns `(byte offset, field type)`.
    ///
    /// The field's type comes from the struct definition; if the definition cannot be found
    /// it falls back to `i64`. Panics if the field does not exist on the struct.
    fn pattern_struct_field_slot(
        &self,
        struct_name: &StrId,
        field_name: &StrId,
        who: &str,
    ) -> (usize, SsaType) {
        let offsets = self
            .struct_field_offsets
            .get(struct_name)
            .expect("struct pattern on a struct with no recorded field offsets");
        let offset = *offsets.get(field_name).unwrap_or_else(|| {
            panic!(
                "{who}: unknown field `{}` on struct `{}`",
                field_name, struct_name
            )
        });
        let field_ty = self
            .structs
            .get(struct_name)
            .and_then(|s| s.fields.iter().find(|f| f.name == *field_name))
            .map(|f| lower_type_hir(&f.field_type, self.enums, self.structs))
            .unwrap_or(SsaType::I64);
        (offset, field_ty)
    }

    /// Lays out the next field of a packed aggregate (enum payload or tuple).
    ///
    /// Rounds `cursor` up to the field's alignment, returns that offset, and advances `cursor`
    /// past the field. Unknown alignment or size defaults to 8, matching the rest of the code.
    ///
    /// Always call it for **every** field in order, including fields that the pattern does not
    /// mention: skipped fields still occupy space, and the cursor has to move over them.
    fn pattern_next_packed_offset(&self, field_ty: &SsaType, cursor: &mut usize) -> usize {
        let target = Self::pattern_target_info();
        let align = ir::layout::alignof_ssa(field_ty, target).unwrap_or(8);
        *cursor = Self::align_up(*cursor, align);
        let offset = *cursor;
        *cursor += ir::layout::sizeof_ssa(field_ty, target).unwrap_or(8);
        offset
    }

    /// The SSA type of every tuple slot for a tuple pattern of `arity` elements.
    /// Without type information every slot is assumed to be `i64`.
    fn pattern_tuple_field_types(scrutinee_ty: Option<&SsaType>, arity: usize) -> Vec<SsaType> {
        match scrutinee_ty {
            Some(SsaType::Tuple(fs)) => fs.clone(),
            // If we don't have type info yet, fall back to i64 for each slot.
            _ => vec![SsaType::I64; arity],
        }
    }

    /// Byte offset where an enum's payload starts, computed from the enum's real layout
    /// (alignment can push it past the tag).
    fn pattern_enum_payload_offset(&self, hir_enum: &HirEnum<'a, 'bump>) -> usize {
        let variant_types: Vec<Vec<SsaType>> = hir_enum
            .variants
            .iter()
            .map(|v| {
                v.fields
                    .iter()
                    .map(|f| lower_type_hir(&f.field_type, self.enums, self.structs))
                    .collect()
            })
            .collect();

        let enum_layout =
            ir::layout::enum_layout_of_ssa(&variant_types, Self::pattern_target_info())
                .unwrap_or_else(|e| panic!("failed to compute layout for error enum: {:?}", e));
        enum_layout.payload_offset
    }

    /// Index of `variant` inside `hir_enum`'s variant list. This index **is** the variant's tag
    /// value at run time. Panics if the enum has no such variant.
    fn pattern_enum_variant_index(
        hir_enum: &HirEnum<'a, 'bump>,
        enum_name: &StrId,
        variant: &StrId,
        who: &str,
    ) -> usize {
        hir_enum
            .variants
            .iter()
            .position(|v| v.name == *variant)
            .unwrap_or_else(|| {
                panic!(
                    "{who}: enum `{}` has no variant `{}`; the type checker should have caught this",
                    enum_name, variant
                )
            })
    }

    /// Emits `scrutinee.tag == expected_tag` for an enum scrutinee and returns the `Bool`.
    /// The tag is read from offset 0 as an `i64`.
    fn emit_pattern_tag_eq(&mut self, scrutinee: Value, expected_tag: i64) -> Value {
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
        cmp
    }

    /// ANDs `cond` into the running condition `acc`, skipping `None` (an always-true sub-test).
    fn and_pattern_cond(&mut self, acc: &mut Option<Value>, cond: Option<Value>) {
        if let Some(c) = cond {
            *acc = Some(self.and_conds(*acc, c));
        }
    }

    /// If `ty` is an enum, or a pointer / owned box / nullable wrapping one, returns the enum's name.
    ///
    /// Patterns like `Shape::Circle(r)` can be matched against `Shape`, `*Shape`, `Shape?`,
    /// and so on, so this peels those wrappers off one at a time until it hits a nominal type.
    /// Returns `None` for anything else (or when no type is known).
    pub(super) fn extract_enum_name_from_ty<'b>(
        &'b self,
        ty: Option<&'b SsaType>,
    ) -> Option<&'b StrId> {
        let mut curr = ty?;
        loop {
            match curr {
                SsaType::Enum {
                    name,
                    unmangled_name: _,
                    ..
                } => return Some(name),
                SsaType::User(name, _, _) => return Some(name),
                SsaType::Pointer(_, inner) | SsaType::Owned(inner) | SsaType::Nullable(inner) => {
                    curr = inner.as_ref();
                }
                _ => return None,
            }
        }
    }

    /// Finds the HIR definition of the enum called `enum_name`. If it is unknown, prints every known
    /// enum (to help debugging name-mangling mismatches) and panics.
    pub(super) fn resolve_enum_for_variant(&self, enum_name: &StrId) -> &HirEnum<'a, 'bump> {
        self.enums.get(enum_name).unwrap_or_else(|| {
            println!("All enums: {:?}", self.enums.keys().collect::<Vec<_>>());
            panic!("[resolve_enum_for_variant] unknown enum {}", enum_name)
        })
    }

    /// Emits the code that tests whether `scrutinee` matches `pattern`.
    ///
    /// Returns `Some(cond)` with a `Bool` that is true on a match, or `None` if the pattern always
    /// matches (see the module docs).
    ///
    /// **Nullable scrutinees come first.** If the scrutinee is `T?` and the pattern needs to look
    /// *inside* it (anything but `null`, `_`, an identifier, or an or-pattern;
    /// see [`Self::pattern_needs_nonnull`]), the pattern cannot be tested directly: the payload
    /// of a null value must not be inspected. [`Self::lower_nullable_inner_test`] builds a
    /// null check followed by the inner test, guarded by a real branch.
    ///
    /// Otherwise it dispatches on the pattern kind to one of the `test_*_pattern` functions below.
    pub(super) fn lower_pattern_test(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        if let Some(ty) = scrutinee_ty {
            if matches!(ty, SsaType::Nullable(_)) && Self::pattern_needs_nonnull(pattern) {
                return self.lower_nullable_inner_test(pattern, scrutinee, ty, span);
            }
        }
        match pattern {
            HirPattern::Wildcard | HirPattern::Ident(..) => None,
            HirPattern::Array(_) => self.test_array_pattern(pattern, scrutinee, scrutinee_ty, span),
            HirPattern::Struct { .. } => {
                self.test_struct_pattern(pattern, scrutinee, scrutinee_ty, span)
            }
            HirPattern::Or(_) => self.test_or_pattern(pattern, scrutinee, scrutinee_ty, span),
            HirPattern::Boolean(b) => {
                Some(self.test_literal_eq(scrutinee, SsaType::Bool, Operand::ConstBool(*b)))
            }
            HirPattern::Number(n) => {
                Some(self.test_literal_eq(scrutinee, SsaType::I64, Operand::ConstInt(*n)))
            }
            HirPattern::String(s) => Some(self.test_string_eq(scrutinee, Operand::ConstString(*s))),
            HirPattern::EnumVariant { .. } => {
                Some(self.test_enum_variant_pattern(pattern, scrutinee, scrutinee_ty))
            }
            HirPattern::Null => Some(self.test_null_pattern(scrutinee, scrutinee_ty)),
            HirPattern::Tuple(_) => self.test_tuple_pattern(pattern, scrutinee, scrutinee_ty, span),
        }
    }

    /// `[p0, p1, ...]`: every element pattern must match its element.
    ///
    /// Elements are read one by one ([`Self::emit_pattern_array_element`]) and each element
    /// pattern is tested recursively. The element tests are ANDed together.
    fn test_array_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Array(elems) = pattern else {
            unreachable!("test_array_pattern called with a non-array pattern")
        };
        let (elem_ty, elem_size) =
            Self::pattern_array_elem_info(scrutinee_ty, "[lower_pattern_test]");

        let mut combined: Option<Value> = None;
        for (i, elem_pat) in elems.iter().enumerate() {
            let elem_val = self.emit_pattern_array_element(scrutinee, i, &elem_ty, elem_size);
            let cond = self.lower_pattern_test(elem_pat, elem_val, Some(&elem_ty), span);
            self.and_pattern_cond(&mut combined, cond);
        }
        combined
    }

    /// `Name { field: p, ... }`: this syntax is used for two different things, so first decide
    /// which one `Name` is.
    ///
    /// * a real **struct**: [`Self::test_plain_struct_pattern`];
    /// * a **variant of an enum** with named fields: [`Self::test_enum_struct_pattern`]
    ///   (the enum is found from the scrutinee's type).
    fn test_struct_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Struct { name, .. } = pattern else {
            unreachable!("test_struct_pattern called with a non-struct pattern")
        };
        if self.struct_field_offsets.get(name).is_some() {
            self.test_plain_struct_pattern(pattern, scrutinee, span)
        } else {
            self.test_enum_struct_pattern(pattern, scrutinee, scrutinee_ty, span)
        }
    }

    /// Struct pattern on a real struct: read each *mentioned* field at its recorded offset and test
    /// its sub-pattern. Fields the pattern does not mention are never touched.
    fn test_plain_struct_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Struct { name, fields } = pattern else {
            unreachable!("test_plain_struct_pattern called with a non-struct pattern")
        };
        let mut combined: Option<Value> = None;
        for (field_name, field_pat) in fields.iter() {
            let (offset, field_ty) =
                self.pattern_struct_field_slot(name, field_name, "lower_pattern_test");
            let field_val = self.emit_pattern_slot_value(scrutinee, offset, &field_ty);
            let cond = self.lower_pattern_test(field_pat, field_val, Some(&field_ty), span);
            self.and_pattern_cond(&mut combined, cond);
        }
        combined
    }

    /// `Variant { field: p, ... }` on an enum scrutinee.
    ///
    /// The result is `tag == this variant's tag`, ANDed with the sub-pattern tests for every field the
    /// pattern mentions. The payload is walked field by field in declaration order
    /// ([`Self::pattern_next_packed_offset`]) so each mentioned field is read from the right byte.
    ///
    /// The field reads are planned *before* anything is emitted, which keeps the borrow of the enum
    /// definition from overlapping with the emitting calls.
    fn test_enum_struct_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Struct { name, fields } = pattern else {
            unreachable!("test_enum_struct_pattern called with a non-struct pattern")
        };

        let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
            panic!(
                "lower_pattern_test: `{}` is neither a known struct nor is the scrutinee ({:?}) \
                 an enum at {span}",
                name, scrutinee_ty
            );
        });
        let hir_enum = self.resolve_enum_for_variant(enum_name);
        let payload_offset = self.pattern_enum_payload_offset(hir_enum);
        let variant_idx =
            Self::pattern_enum_variant_index(hir_enum, enum_name, name, "lower_pattern_test");
        let variant_def = &hir_enum.variants[variant_idx];

        // Plan: which mentioned fields to read, where, and with which sub-pattern.
        let mut cursor = 0usize;
        let mut plan = Vec::new();
        for vf in variant_def.fields.iter() {
            let field_ty = lower_type_hir(&vf.field_type, self.enums, self.structs);
            let offset = self.pattern_next_packed_offset(&field_ty, &mut cursor);
            if let Some((_, field_pat)) = fields.iter().find(|(fname, _)| fname == &vf.name) {
                plan.push((payload_offset + offset, field_ty, field_pat));
            }
        }

        let tag_cmp = self.emit_pattern_tag_eq(scrutinee, variant_idx as i64);
        let mut result = Some(tag_cmp);
        for (offset, field_ty, field_pat) in plan {
            let field_val = self.emit_pattern_slot_value(scrutinee, offset, &field_ty);
            let cond = self.lower_pattern_test(field_pat, field_val, Some(&field_ty), span);
            self.and_pattern_cond(&mut result, cond);
        }
        result
    }

    /// `p0 | p1 | ...`: matches if *any* alternative matches.
    ///
    /// If any alternative is irrefutable (`None`), the whole pattern is: return `None`.
    /// Otherwise the alternatives' tests are ORed together ([`Self::or_conds`]).
    fn test_or_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Or(alts) = pattern else {
            unreachable!("test_or_pattern called with a non-or pattern")
        };
        let mut combined: Option<Value> = None;
        let mut always_matches = false;
        for alt in alts.iter() {
            match self.lower_pattern_test(alt, scrutinee, scrutinee_ty, span) {
                None => always_matches = true,
                Some(cond) => combined = Some(self.or_conds(combined, cond)),
            }
        }
        if always_matches { None } else { combined }
    }

    /// Literal pattern (`true`, `42`): emit the literal constant and compare with `==`.
    ///
    /// Note that numeric literals are always materialised as `i64`, whatever the scrutinee's width.
    fn test_literal_eq(&mut self, scrutinee: Value, lit_ty: SsaType, lit: Operand) -> Value {
        let lit = self.emit_pattern_const(lit_ty, lit);
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
        cmp
    }

    /// String literal pattern: strings cannot be compared with `==` on raw values, so the runtime's
    /// string-equality function is called with the scrutinee and the literal, and its `Bool`
    /// result is the test.
    fn test_string_eq(&mut self, scrutinee: Value, lit: Operand) -> Value {
        let lit = self.emit_pattern_const(SsaType::String, lit);

        let streq_fn = StrId::from_static("str_zeta_strings_eq");
        let cmp = self.current_block_data.fresh_value();
        self.emit(Instruction::Call {
            dest: Some(cmp),
            func: Operand::FunctionRef(streq_fn),
            args: smallvec![Operand::Value(scrutinee), Operand::Value(lit)],
        });
        self.current_block_data
            .value_types
            .insert(cmp, SsaType::Bool);
        cmp
    }

    /// `Variant` or `Variant(..)` used as a test: only the **tag** is compared.
    ///
    /// The payload of a tuple-like variant is *not* tested here (only `Variant { .. }` struct-style
    /// patterns test fields); tuple-variant bindings are introduced later by [`Self::bind_pattern`].
    fn test_enum_variant_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) -> Value {
        let HirPattern::EnumVariant { variant, .. } = pattern else {
            unreachable!("test_enum_variant_pattern called with a non-variant pattern")
        };
        let enum_name = self.extract_enum_name_from_ty(scrutinee_ty).unwrap_or_else(|| {
            panic!(
                "lower_pattern_test: enum pattern `{}(..)` used on a non-enum scrutinee ({:?}); \
                 the type checker should have caught this",
                variant, scrutinee_ty
            );
        });
        let hir_enum = self.resolve_enum_for_variant(enum_name);
        let variant_idx =
            Self::pattern_enum_variant_index(hir_enum, enum_name, variant, "lower_pattern_test");
        self.emit_pattern_tag_eq(scrutinee, variant_idx as i64)
    }

    /// `null` pattern: test whether a nullable scrutinee is null. Which instruction sequence that is
    /// depends on how the nullable is represented:
    ///
    /// * **Pointer-like** (nullable-pointer representation, or a plain pointer): null is address 0,
    ///   so `scrutinee == 0`.
    /// * **Tagged nullable**: read the tag byte at offset 0; null is tag 0, so `tag == 0`.
    ///
    /// Panics for any other scrutinee type (the type checker should have rejected it).
    fn test_null_pattern(&mut self, scrutinee: Value, scrutinee_ty: Option<&SsaType>) -> Value {
        let ty = scrutinee_ty.expect(
            "[lower_pattern_test] `null` pattern has no scrutinee type; the type checker should have caught this",
        );

        if ty.nullable_pointer_repr().is_some() || matches!(ty, SsaType::Pointer(..)) {
            let cmp = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cmp,
                op: BinOp::Eq,
                left: Operand::Value(scrutinee),
                right: Operand::ConstInt(0),
            });
            self.current_block_data
                .value_types
                .insert(cmp, SsaType::Bool);
            cmp
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
            cmp
        } else {
            panic!(
                "[lower_pattern_test] `null` pattern used against non-nullable scrutinee \
                 type {:?}; the type checker should have caught this",
                ty
            );
        }
    }

    /// `(p0, p1, ...)`: every element pattern must match its tuple slot.
    ///
    /// Slots are laid out packed with per-field alignment, so each slot's offset is found by walking
    /// the field types in order ([`Self::pattern_next_packed_offset`]).
    fn test_tuple_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let HirPattern::Tuple(elems) = pattern else {
            unreachable!("test_tuple_pattern called with a non-tuple pattern")
        };
        let field_types = Self::pattern_tuple_field_types(scrutinee_ty, elems.len());

        let mut combined: Option<Value> = None;
        let mut cursor = 0usize;
        for (i, elem_pat) in elems.iter().enumerate() {
            let fty = field_types.get(i).cloned().unwrap_or(SsaType::I64);
            let offset = self.pattern_next_packed_offset(&fty, &mut cursor);
            let elem_val = self.emit_pattern_slot_value(scrutinee, offset, &fty);
            let cond = self.lower_pattern_test(elem_pat, elem_val, Some(&fty), span);
            self.and_pattern_cond(&mut combined, cond);
        }
        combined
    }

    /// Tests an *inner* pattern against a nullable scrutinee `?T`, without ever looking inside a null.
    ///
    /// Unlike every other test in this module, this one uses real control flow, because unwrapping
    /// and probing a null value is not safe:
    ///
    /// ```text
    ///           is_null = (scrutinee is null)
    ///           /                          \
    ///      null_bb                      check_bb
    ///      f = false                    unwrapped = payload
    ///                                   inner = test(pattern, unwrapped)   (or `true` if irrefutable)
    ///           \                          /
    ///            merge_bb: result = phi [(check_end, inner), (null_bb, f)]
    /// ```
    ///
    /// `check_end` is read after lowering the inner test because that test may have created blocks of
    /// its own (a nested nullable adds a diamond), so the block that jumps to `merge_bb` is not
    /// necessarily `check_bb`. A null scrutinee never matches, hence the constant `false`.
    pub(super) fn lower_nullable_inner_test(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        ty: &SsaType,
        span: SourceSpan<'a>,
    ) -> Option<Value> {
        let SsaType::Nullable(inner) = ty else {
            unreachable!()
        };
        let is_null = self
            .lower_pattern_test(&HirPattern::Null, scrutinee, Some(ty), span)
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
        let inner_cond = match self.lower_pattern_test(pattern, unwrapped, Some(&**inner), span) {
            Some(c) => c,
            None => self.emit_pattern_const(SsaType::Bool, Operand::ConstBool(true)),
        };
        let check_end = self.current_block_data.current_block; // inner test may have added blocks
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(null_bb);
        let f = self.emit_pattern_const(SsaType::Bool, Operand::ConstBool(false));
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

    /// Combines a running condition with another using bitwise AND (`None` means "true so far").
    /// Both operands are evaluated; see the module docs on why that is safe for patterns.
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

    /// Combines a running condition with another using bitwise OR (`None` means "nothing yet").
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

    /// True if `pattern` must look *inside* a value, and so needs a non-null scrutinee.
    ///
    /// `null`, `_` and plain identifiers never inspect the payload. An or-pattern is exempt too:
    /// each alternative is tested separately through [`Self::lower_pattern_test`], and so
    /// applies its own nullability check.
    pub(super) fn pattern_needs_nonnull(pattern: &HirPattern<'bump>) -> bool {
        !matches!(
            pattern,
            HirPattern::Null | HirPattern::Wildcard | HirPattern::Ident(..) | HirPattern::Or(_)
        )
    }

    /// Introduces the variables a pattern binds, by loading each one out of `scrutinee` and
    /// recording it in `var_map`.
    ///
    /// This runs in the arm body, after the test has succeeded, so it can assume the
    /// scrutinee really has the shape the pattern describes. It emits no checks.
    ///
    /// For each pattern kind, it reads the same sub-elements that the test phase read (same offsets,
    /// same layout rules) and recurses into sub-patterns; leaf identifiers write into `var_map`.
    /// Patterns that bind nothing (wildcards, literals, `null`, tuple-less enum variants) do nothing.
    pub(super) fn bind_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let (scrutinee, scrutinee_ty) =
            self.unwrap_scrutinee_for_binding(pattern, scrutinee, scrutinee_ty);
        match pattern {
            HirPattern::Ident(name, _) => {
                let bound = self.bind_ident_value(scrutinee, scrutinee_ty);
                self.var_map.insert(*name, bound);
            }
            HirPattern::Array(_) => self.bind_array_pattern(pattern, scrutinee, scrutinee_ty),
            HirPattern::Struct { .. } => self.bind_struct_pattern(pattern, scrutinee, scrutinee_ty),
            HirPattern::Or(alts) => {
                for alt in alts.iter() {
                    self.bind_pattern(alt, scrutinee, scrutinee_ty);
                }
            }
            HirPattern::EnumVariant { bindings, .. } if !bindings.is_empty() => {
                self.bind_enum_variant_bindings(pattern, scrutinee, scrutinee_ty)
            }
            HirPattern::Tuple(_) => self.bind_tuple_pattern(pattern, scrutinee, scrutinee_ty),
            _ => {}
        }
    }

    /// If the pattern looks inside a nullable scrutinee `T?`, unwraps it first and returns the
    /// unwrapped value together with the inner type `T`; otherwise returns the inputs untouched.
    ///
    /// Unwrapping without a null check is fine here because the test phase already proved the value
    /// non-null before this arm could be chosen ([`Self::lower_nullable_inner_test`]).
    fn unwrap_scrutinee_for_binding<'t>(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&'t SsaType>,
    ) -> (Value, Option<&'t SsaType>) {
        if let Some(ty) = scrutinee_ty {
            if let SsaType::Nullable(inner) = ty {
                if Self::pattern_needs_nonnull(pattern) {
                    let unwrapped = self.unwrap_known_nonnull(scrutinee, ty);
                    return (unwrapped, Some(&**inner));
                }
            }
        }
        (scrutinee, scrutinee_ty)
    }

    /// The value a plain identifier pattern binds to.
    ///
    /// If the scrutinee is nullable, the identifier is bound to the *unwrapped* payload, not the
    /// nullable itself; otherwise to the scrutinee as is.
    fn bind_ident_value(&mut self, scrutinee: Value, scrutinee_ty: Option<&SsaType>) -> Value {
        match scrutinee_ty {
            Some(ty) if matches!(ty, SsaType::Nullable(_)) => {
                self.unwrap_known_nonnull(scrutinee, ty)
            }
            _ => scrutinee,
        }
    }

    /// `[p0, p1, ...]`: read each element and bind its sub-pattern (the mirror of
    /// [`Self::test_array_pattern`]).
    fn bind_array_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let HirPattern::Array(elems) = pattern else {
            unreachable!("bind_array_pattern called with a non-array pattern")
        };
        let (elem_ty, elem_size) = Self::pattern_array_elem_info(scrutinee_ty, "bind_pattern");

        for (i, elem_pat) in elems.iter().enumerate() {
            let elem_val = self.emit_pattern_array_element(scrutinee, i, &elem_ty, elem_size);
            self.bind_pattern(elem_pat, elem_val, Some(&elem_ty));
        }
    }

    /// `Name { field: p, ... }`: bind the sub-patterns of a real struct, or of an enum variant with
    /// named fields (the same split as in the test phase; see [`Self::test_struct_pattern`]).
    fn bind_struct_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let HirPattern::Struct { name, fields } = pattern else {
            unreachable!("bind_struct_pattern called with a non-struct pattern")
        };

        if self.struct_field_offsets.get(name).is_some() {
            for (field_name, field_pat) in fields.iter() {
                let (offset, field_ty) =
                    self.pattern_struct_field_slot(name, field_name, "[bind_pattern]");
                let field_val = self.emit_pattern_slot_value(scrutinee, offset, &field_ty);
                self.bind_pattern(field_pat, field_val, Some(&field_ty));
            }
        } else {
            self.bind_enum_struct_pattern(pattern, scrutinee, scrutinee_ty);
        }
    }

    /// `Variant { field: p, ... }` on an enum: bind the sub-patterns of the mentioned fields, reading each
    /// from its packed position in the payload (the mirror of [`Self::test_enum_struct_pattern`],
    /// without the tag test). Reads are planned before any emission, for the same reason as there.
    fn bind_enum_struct_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let HirPattern::Struct { name, fields } = pattern else {
            unreachable!("bind_enum_struct_pattern called with a non-struct pattern")
        };

        let enum_name = self
            .extract_enum_name_from_ty(scrutinee_ty)
            .unwrap_or_else(|| {
                panic!(
                    "[bind_pattern] `{}` is neither a known struct nor is the scrutinee ({:?}) \
                 an enum",
                    name, scrutinee_ty
                );
            });
        let hir_enum = self.resolve_enum_for_variant(enum_name);
        let variant_idx =
            Self::pattern_enum_variant_index(hir_enum, enum_name, name, "[bind_pattern]");
        let variant_def = &hir_enum.variants[variant_idx];
        let payload_offset = self.pattern_enum_payload_offset(hir_enum);

        // Plan the reads (the tag occupies the start of the enum; the payload follows it).
        let mut cursor = 0usize;
        let mut plan = Vec::new();
        for vf in variant_def.fields.iter() {
            let field_ty = lower_type_hir(&vf.field_type, self.enums, self.structs);
            let offset = self.pattern_next_packed_offset(&field_ty, &mut cursor);
            // Only bind fields that appear in the pattern.
            if let Some((_, field_pat)) = fields.iter().find(|(fname, _)| fname == &vf.name) {
                plan.push((payload_offset + offset, field_ty, field_pat));
            }
        }

        for (offset, field_ty, field_pat) in plan {
            let field_val = self.emit_pattern_slot_value(scrutinee, offset, &field_ty);
            self.bind_pattern(field_pat, field_val, Some(&field_ty));
        }
    }

    /// `Variant(a, b, ...)`: bind each positional name to the matching payload field of the variant.
    ///
    /// Fields are paired with names in declaration order and read from their packed offsets.
    /// This form only supports plain names (no nested sub-patterns); the names go
    /// straight into `var_map`.
    ///
    /// The payload is assumed to start at byte `8` (just after an 8-byte tag), whereas the
    /// struct-style form ([`Self::bind_enum_struct_pattern`]) asks the enum's layout. They agree
    /// as long as the layout puts the payload at 8.
    fn bind_enum_variant_bindings(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let HirPattern::EnumVariant {
            variant, bindings, ..
        } = pattern
        else {
            unreachable!("bind_enum_variant_bindings called with a non-variant pattern")
        };

        let enum_name = self
            .extract_enum_name_from_ty(scrutinee_ty)
            .unwrap_or_else(|| {
                panic!(
                    "bind_pattern: enum pattern `{}(..)` used on a non-enum scrutinee ({:?}); \
                 the type checker should have caught this",
                    variant, scrutinee_ty
                );
            });
        let hir_enum = self.resolve_enum_for_variant(enum_name);
        let variant_idx =
            Self::pattern_enum_variant_index(hir_enum, enum_name, variant, "bind_pattern");
        let variant_def = &hir_enum.variants[variant_idx];
        debug_assert_eq!(
            bindings.len(),
            variant_def.fields.len(),
            "bind_pattern: binding count for variant `{}` doesn't match its field count; \
             the type checker should have caught this",
            variant
        );

        let mut cursor = 0usize;
        let mut plan = Vec::new();
        for (&binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
            let field_ssa_ty = lower_type_hir(&field.field_type, self.enums, self.structs);
            let offset = self.pattern_next_packed_offset(&field_ssa_ty, &mut cursor);
            plan.push((binding_name, 8 + offset, field_ssa_ty));
        }

        for (binding_name, offset, field_ssa_ty) in plan {
            let dest = self.emit_pattern_slot_value(scrutinee, offset, &field_ssa_ty);
            self.var_map.insert(binding_name, dest);
        }
    }

    /// `(p0, p1, ...)`: read each tuple slot at its packed offset and bind its sub-pattern (the mirror of
    /// [`Self::test_tuple_pattern`]).
    fn bind_tuple_pattern(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee: Value,
        scrutinee_ty: Option<&SsaType>,
    ) {
        let HirPattern::Tuple(elems) = pattern else {
            unreachable!("bind_tuple_pattern called with a non-tuple pattern")
        };
        let field_types = Self::pattern_tuple_field_types(scrutinee_ty, elems.len());

        let mut cursor = 0usize;
        for (i, elem_pat) in elems.iter().enumerate() {
            let fty = field_types.get(i).cloned().unwrap_or(SsaType::I64);
            let offset = self.pattern_next_packed_offset(&fty, &mut cursor);
            let elem_val = self.emit_pattern_slot_value(scrutinee, offset, &fty);
            self.bind_pattern(elem_pat, elem_val, Some(&fty));
        }
    }

    /// Lowers a `match` with no expected result type. Wrapper over [`Self::lower_match_expr_inner`].
    pub(super) fn lower_match_expr(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_match_expr_inner(scrutinee, arms, span, None)
    }

    /// Lowers `match scrutinee { pattern [if guard] => { body } ... }`, optionally pushing an
    /// `expected` result type into every arm body. See the module docs for the overall CFG shape.
    ///
    /// The scrutinee is evaluated **once**. Then, per arm:
    /// 1. [`Self::emit_match_arm_test`]: test the pattern (and guard) and branch to the arm's body
    ///    or on to the next arm.
    /// 2. Reset `drop_state`, `var_map` and `narrowed_fields` to their pre-match values: every arm starts
    ///    from the same state, independent of what other arms did.
    /// 3. [`Self::lower_match_arm_body`]: bind the pattern, lower the body, and record the arm for the
    ///    merge if it falls through.
    /// 4. Continue emitting the next arm's test in this arm's failure block.
    ///
    /// Finally [`Self::finish_match`] merges the arms that reach the end.
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
        let vars_before = self.var_map.clone();
        let merge_bb = self.current_block_data.fresh_block();
        let mut join = MatchJoin {
            live_vars: Vec::new(),
            live_drop: Vec::new(),
            incoming: SmallVec::new(),
        };
        let scrutinee = MatchScrutinee {
            expr: scrutinee,
            val: scrutinee_val,
            ty: scrutinee_ty,
        };

        for (arm_idx, arm) in arms.iter().enumerate() {
            let is_last = arm_idx + 1 == arms.len();
            let body_bb = self.current_block_data.new_block();
            let fail_bb = if is_last {
                None
            } else {
                Some(self.current_block_data.new_block())
            };

            self.emit_match_arm_test(arm, &scrutinee, body_bb, fail_bb, span);

            // Every arm starts from the pre-match state.
            self.drop_state = drop_before.clone();
            self.var_map = vars_before.clone();
            self.narrowed_fields = narrowed_before.clone();

            // An earlier unguarded `null` arm means this arm can never see null.
            let prior_null_arm = arms[..arm_idx]
                .iter()
                .any(|a| a.guard.is_none() && matches!(a.pattern, HirPattern::Null));
            self.lower_match_arm_body(
                &scrutinee,
                arm,
                prior_null_arm,
                body_bb,
                merge_bb,
                &vars_before,
                expected,
                span,
                &mut join,
            );

            // The next arm's test is emitted where this arm's test failed.
            if let Some(fb) = fail_bb {
                self.current_block_data.switch_to(fb);
            }
        }

        self.narrowed_fields = narrowed_before;
        self.finish_match(merge_bb, join, vars_before, span)
    }

    /// Emits the test for one arm and the branches that follow from it.
    ///
    /// The arm's *condition* is built from the pattern test and the optional guard:
    ///
    /// | Pattern test | Guard   | Condition                                                      |
    /// |--------------|---------|----------------------------------------------------------------|
    /// | `Some(pc)`   | `Some`  | pattern first (branch to a guard block), then the guard value  |
    /// | `Some(pc)`   | none    | `pc`                                                           |
    /// | `None`       | `Some`  | the guard value                                                |
    /// | `None`       | none    | none (irrefutable: jump straight to the body)                  |
    ///
    /// The guard is evaluated in its own block *after* the pattern succeeded, because a guard may
    /// use variables the pattern binds / assume the pattern matched, so it must not run on a mismatch.
    ///
    /// The final branch depends on whether there is a next arm:
    /// * next arm exists: on failure go to `fail_bb`;
    /// * **last arm** with a condition: on failure the match was non-exhaustive, so go to a panic
    ///   block ([`Self::emit_match_exhaustiveness_branch`]).
    fn emit_match_arm_test(
        &mut self,
        arm: &HirMatchArm<'a, 'bump>,
        scrutinee: &MatchScrutinee<'_, 'a, 'bump>,
        body_bb: BlockId,
        fail_bb: Option<BlockId>,
        span: SourceSpan<'a>,
    ) {
        let pattern_cond =
            self.lower_pattern_test(&arm.pattern, scrutinee.val, scrutinee.ty.as_ref(), span);

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
            (Some(c), None) => self.emit_match_exhaustiveness_branch(c, body_bb),
            (None, _) => {
                self.emit(Instruction::Jump { target: body_bb });
            }
        }
    }

    /// Branches to `body_bb` when `cond` holds, and otherwise to a new block that **panics**.
    ///
    /// This is the end of the arm chain: if the last arm's test fails, no arm matched, which only
    /// happens when the `match` is not exhaustive. The failure block loads a message
    /// string and calls [`Self::emit_debug_panic`], which reports it and terminates the block.
    fn emit_match_exhaustiveness_branch(&mut self, cond: Value, body_bb: BlockId) {
        let trap_bb = self.current_block_data.new_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb: body_bb,
            else_bb: trap_bb,
        });
        self.current_block_data.switch_to(trap_bb);
        let msg = self.emit_pattern_const(
            SsaType::String,
            Operand::ConstString(StrId::from_static(
                "non-exhaustive match: no arm matched the scrutinee",
            )),
        );
        self.emit_debug_panic(msg);
    }

    /// Lowers one arm's body in `body_bb` and, if it falls through, records it for the merge.
    ///
    /// The caller has already reset `drop_state`, `var_map` and `narrowed_fields`.
    ///
    /// * A fresh `DropScope` is pushed so the pattern's bindings and the body's locals are dropped at the
    ///   end of the arm.
    /// * [`Self::bind_pattern`] introduces the pattern's variables, then `adopt_owned_binding`
    ///   decides whether a binding takes over ownership of the scrutinee (and so must not be dropped
    ///   twice); `prior_null_arm` tells it that null was already ruled out by an earlier arm.
    /// * The body must be a block; it is lowered as a value (the arm's result) with `cond_depth`
    ///   raised, which marks the code as conditionally executed.
    /// * If the arm does not diverge: emit its scope drops, then record its end block, the
    ///   variables that existed *before* the match (arm-local names are filtered out), its drop state,
    ///   and its value, and jump to the merge block.
    #[allow(clippy::too_many_arguments)]
    fn lower_match_arm_body(
        &mut self,
        scrutinee: &MatchScrutinee<'_, 'a, 'bump>,
        arm: &HirMatchArm<'a, 'bump>,
        prior_null_arm: bool,
        body_bb: BlockId,
        merge_bb: BlockId,
        vars_before: &VarSnapshot,
        expected: Option<&SsaType>,
        span: SourceSpan<'a>,
        join: &mut MatchJoin<'a, 'bump>,
    ) {
        self.current_block_data.switch_to(body_bb);
        self.scope_stack.push(DropScope::default());
        self.bind_pattern(&arm.pattern, scrutinee.val, scrutinee.ty.as_ref());
        self.adopt_owned_binding(scrutinee.expr, &arm.pattern, prior_null_arm);
        let HirStmt::Block { body, span: _ } = arm.body else {
            panic!("match arm body must be a block")
        };
        self.cond_depth += 1;
        let arm_val = self.lower_block_value_inner(body, expected);
        self.cond_depth -= 1;
        let arm_scope = self.scope_stack.pop().unwrap();
        if !self.block_terminated() {
            self.emit_scope_drops(&arm_scope, span);
            join.live_drop.push(self.drop_state.clone());
            let arm_end_bb = self.current_block_data.current_block;
            let vars: VarSnapshot = self
                .var_map
                .iter()
                .filter(|(k, _)| vars_before.contains_key(*k))
                .map(|(k, v)| (*k, *v))
                .collect();
            join.live_vars.push((arm_end_bb, vars));
            self.emit(Instruction::Jump { target: merge_bb });
            join.incoming.push((arm_end_bb, arm_val));
        }
    }

    /// Final step of a `match`: merge the arms that fall through and build the result value.
    ///
    /// * **No arm reaches the end** (all diverge): restore `var_map` and return a dead `Void` value;
    ///   the merge block is never created.
    /// * Otherwise open `merge_bb`, merge variable maps and drop states of the live arms, and build the
    ///   result phi over their values ([`Self::emit_merge_phi`], which also handles `null`
    ///   adopting the other arms' type and `Void` arms).
    fn finish_match(
        &mut self,
        merge_bb: BlockId,
        join: MatchJoin<'a, 'bump>,
        vars_before: VarSnapshot,
        span: SourceSpan<'a>,
    ) -> Value {
        if join.incoming.is_empty() {
            self.var_map = vars_before;
            return self.unreachable_value();
        }

        self.current_block_data.push_block(merge_bb);
        self.current_block_data.switch_to(merge_bb);
        self.merge_var_maps(join.live_vars);
        if let Some(j) = DropMoveState::join_all(join.live_drop) {
            self.drop_state = j;
        }

        self.emit_merge_phi(join.incoming, span)
    }
}
