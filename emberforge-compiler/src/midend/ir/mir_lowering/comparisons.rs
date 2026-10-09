use ir::{
    hir::{HirExpr, Operator},
    ir_conversion::lower_operator_bin,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value},
};

use crate::midend::ir::mir_lowering::FunctionLowerer;
use smallvec::smallvec;

/// The two shapes an operand of a null comparison can arrive in.
///
/// Field accesses like `obj.field == null` are special: rather than first loading
/// the whole field into a temporary, we may only need its *address* (for example to
/// read just the tag byte of a tagged nullable). Keeping "value" and "address"
/// apart lets later code load the field only when it really has to.
enum NullCheckSource {
    /// The operand has already been evaluated; this is the resulting SSA value.
    Val(Value),
    /// The operand is a field that has *not* been loaded; this is its address.
    Addr(Value),
}

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
    /// Looks up the SSA type recorded for `v` in the current block, defaulting to
    /// `i64` if nothing was recorded.
    ///
    /// The default is a deliberate fallback: untyped intermediates are treated as
    /// word-sized integers, which is what the rest of the pipeline expects.
    fn value_type_or_i64(&self, v: Value) -> SsaType {
        self.current_block_data
            .value_types
            .get(&v)
            .cloned()
            .unwrap_or(SsaType::I64)
    }

    /// Emits `left <op> right` and returns the new [`Value`], tagged as `Bool`.
    fn emit_bool_binary(&mut self, op: BinOp, left: Operand, right: Operand) -> Value {
        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest,
            op,
            left,
            right,
        });
        self.current_block_data
            .value_types
            .insert(dest, SsaType::Bool);
        dest
    }

    /// Emits a constant boolean and returns its [`Value`], tagged as `Bool`.
    fn emit_bool_const(&mut self, value: bool) -> Value {
        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest,
            ty: SsaType::Bool,
            value: Operand::ConstBool(value),
        });
        self.current_block_data
            .value_types
            .insert(dest, SsaType::Bool);
        dest
    }

    /// Emits the logical negation of a boolean `v`.
    ///
    /// The IR has no dedicated `Not` instruction for booleans, so `!v` is encoded
    /// as `v == 0`. Since booleans are `0` or `1`, that flips them.
    fn emit_bool_not(&mut self, v: Value) -> Value {
        self.emit_bool_binary(BinOp::Eq, Operand::Value(v), Operand::ConstInt(0))
    }

    /// Reads the tag byte of a tagged nullable (the byte at offset 0).
    ///
    /// `base` may be either the nullable value itself or the address of the field
    /// holding it; `LoadField` accepts both. The result is typed `U8`.
    ///
    /// Tag semantics: `0` means null, any other value means "has a value".
    fn emit_load_tag(&mut self, base: Value) -> Value {
        let tag = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: tag,
            base: Operand::Value(base),
            offset: 0,
        });
        self.current_block_data.value_types.insert(tag, SsaType::U8);
        tag
    }

    /// Emits `tag != 0` for a tagged nullable, giving a `Bool` that is true when
    /// the nullable currently holds a value.
    fn emit_tag_is_some(&mut self, nullable: Value) -> Value {
        let tag = self.emit_load_tag(nullable);
        self.emit_bool_binary(BinOp::Ne, Operand::Value(tag), Operand::ConstInt(0))
    }

    /// Loads the value stored at `addr` and records `ty` as the loaded value's type.
    fn emit_load_typed(&mut self, addr: Value, ty: &SsaType) -> Value {
        let loaded = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: loaded,
            ptr: Operand::Value(addr),
        });
        self.current_block_data
            .value_types
            .insert(loaded, ty.clone());
        loaded
    }

    /// Turns a [`NullCheckSource`] into a plain SSA value, loading from the
    /// address if the operand was only available as one.
    fn materialize_null_check_source(&mut self, src: NullCheckSource, ty: &SsaType) -> Value {
        match src {
            NullCheckSource::Val(v) => v,
            NullCheckSource::Addr(a) => self.emit_load_typed(a, ty),
        }
    }

    /// Lowers `left && right` with short-circuit evaluation.
    ///
    /// `right` must only run if `left` is true, so we cannot compute both and AND
    /// them. Instead we build a small diamond of basic blocks:
    ///
    /// ```text
    ///            +-----------+
    ///            |  current  |  lhs = <left>
    ///            +-----------+  branch lhs
    ///             /          \
    ///        true            false
    ///         /                \
    ///   +--------+         +----------+
    ///   | rhs_bb |         | false_bb |
    ///   | <right>|         | false    |
    ///   +--------+         +----------+
    ///         \               /
    ///          +-------------+
    ///          |  merge_bb   |  result = phi [(rhs_end, rhs), (false_bb, false)]
    ///          +-------------+
    /// ```
    ///
    /// The `Phi` at the merge picks whichever value the executed path produced.
    ///
    /// Note that `rhs_end` is captured *after* lowering `right`: lowering the right
    /// operand may itself create blocks (it could contain another `&&`, an `if`,
    /// and so on), so the block that finally jumps to `merge_bb` is not necessarily
    /// `rhs_bb`. The `Phi` must name the block that actually jumps.
    pub(super) fn lower_short_circuit_and(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let lhs = self.lower_expr(left);

        let rhs_bb = self.current_block_data.new_block();
        let false_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(lhs),
            then_bb: rhs_bb,
            else_bb: false_bb,
        });

        self.current_block_data.switch_to(rhs_bb);
        let rhs = self.lower_expr_scoped(right);
        let rhs_end = self.current_block_data.current_block;
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(false_bb);
        let false_val = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: false_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(false),
        });
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(merge_bb);
        let result = self.current_block_data.fresh_value();
        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(rhs_end, rhs), (false_bb, false_val)],
        });
        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    /// Lowers `left || right` with short-circuit evaluation.
    ///
    /// This is the mirror image of [`Self::lower_short_circuit_and`]: `right` only
    /// runs if `left` is *false*; if `left` is true the answer is already `true`.
    ///
    /// ```text
    ///            +-----------+
    ///            |  current  |  lhs = <left>
    ///            +-----------+  branch lhs
    ///             /          \
    ///        true            false
    ///         /                \
    ///   +---------+        +--------+
    ///   | true_bb |        | rhs_bb |
    ///   | true    |        | <right>|
    ///   +---------+        +--------+
    ///         \               /
    ///          +-------------+
    ///          |  merge_bb   |  result = phi [(true_bb, true), (rhs_end, rhs)]
    ///          +-------------+
    /// ```
    ///
    /// As with `&&`, the `Phi` names `rhs_end` (the block that finishes evaluating
    /// the right operand), not `rhs_bb`, because lowering `right` can add blocks.
    pub(super) fn lower_short_circuit_or(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let lhs = self.lower_expr(left);

        let true_bb = self.current_block_data.new_block();
        let rhs_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(lhs),
            then_bb: true_bb,
            else_bb: rhs_bb,
        });

        self.current_block_data.switch_to(true_bb);
        let true_val = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: true_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(true),
        });
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(rhs_bb);
        let rhs = self.lower_expr_scoped(right);
        let rhs_end = self.current_block_data.current_block;
        self.emit(Instruction::Jump { target: merge_bb });

        self.current_block_data.switch_to(merge_bb);
        let result = self.current_block_data.fresh_value();
        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(true_bb, true_val), (rhs_end, rhs)],
        });
        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    /// Lowers `operand == null` (when `is_eq`) or `operand != null` (otherwise).
    ///
    /// This is a decision tree on the operand's SSA type:
    /// 1. **`Null`.** The operand is itself the `null` literal's type. The answer is
    ///    known at compile time (`null == null` is `true`), so emit a constant.
    /// 2. **Pointer-like.** A raw pointer or a nullable pointer. Null is address 0, so
    ///    compare the pointer with `0`. See [`Self::lower_pointer_null_check`].
    /// 3. **Tagged nullable.** Compare the tag byte with `0`, never touching the
    ///    payload. See [`Self::lower_tag_null_check`].
    /// 4. **Anything else.** Fall back to comparing against the lowered `null`
    ///    value. See [`Self::lower_fallback_null_check`].
    pub(crate) fn lower_null_comparison(
        &mut self,
        operand: &HirExpr<'a, 'bump>,
        is_eq: bool,
    ) -> Value {
        // `x == null` tests "is the sentinel", `x != null` tests "is not".
        let cmp_op = if is_eq { BinOp::Eq } else { BinOp::Ne };

        let (src, ty) = self.lower_null_check_operand(operand);

        if ty == SsaType::Null {
            return self.emit_bool_const(is_eq);
        }
        if Self::is_pointer_like(&ty) {
            return self.lower_pointer_null_check(src, &ty, cmp_op);
        }
        if ty.is_tagged_nullable() {
            return self.lower_tag_null_check(src, cmp_op);
        }
        self.lower_fallback_null_check(src, &ty, cmp_op)
    }

    /// Evaluates the non-`null` side of a null comparison.
    ///
    /// Returns the operand either as a value or, for plain field accesses, as an
    /// *address* together with the field's type. Returning an address avoids
    /// loading a whole (possibly large) tagged nullable when only its tag byte is
    /// needed.
    ///
    /// The address route is skipped when [`Self::narrowed_field_value`] knows the
    /// field has already been narrowed (for example by an earlier null check in an
    /// enclosing `if`). In that case the ordinary expression path reuses the
    /// narrowed value.
    fn lower_null_check_operand(
        &mut self,
        operand: &HirExpr<'a, 'bump>,
    ) -> (NullCheckSource, SsaType) {
        match operand {
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
                (NullCheckSource::Addr(addr), ty)
            }
            _ => {
                let v = self.lower_expr(operand);
                let ty = self.value_type_or_i64(v);
                (NullCheckSource::Val(v), ty)
            }
        }
    }

    /// True if `ty` is represented as a pointer, so null is the address `0`.
    ///
    /// That covers nullable types whose representation collapses to a pointer, and
    /// ordinary pointer types.
    fn is_pointer_like(ty: &SsaType) -> bool {
        ty.nullable_pointer_repr().is_some() || matches!(ty, SsaType::Pointer(..))
    }

    /// Null check for pointer-represented values: `ptr <cmp_op> 0`.
    ///
    /// If the operand arrived as an address (an un-loaded field), the pointer is
    /// loaded first, since the *contents* of the field are what we compare.
    fn lower_pointer_null_check(
        &mut self,
        src: NullCheckSource,
        ty: &SsaType,
        cmp_op: BinOp,
    ) -> Value {
        let ptr = self.materialize_null_check_source(src, ty);
        self.emit_bool_binary(cmp_op, Operand::Value(ptr), Operand::ConstInt(0))
    }

    /// Null check for tagged nullables: `tag <cmp_op> 0`.
    ///
    /// Only the tag byte is read, so the payload is never loaded. The `Val` and
    /// `Addr` cases are handled identically because `LoadField` works with either a
    /// struct value or its address as the base.
    fn lower_tag_null_check(&mut self, src: NullCheckSource, cmp_op: BinOp) -> Value {
        let base = match src {
            NullCheckSource::Val(v) | NullCheckSource::Addr(v) => v,
        };
        let tag = self.emit_load_tag(base);
        self.emit_bool_binary(cmp_op, Operand::Value(tag), Operand::ConstInt(0))
    }

    /// Fallback null check for types with no special null encoding: compare the
    /// value against whatever [`Self::lower_expr_null`] produces.
    fn lower_fallback_null_check(
        &mut self,
        src: NullCheckSource,
        ty: &SsaType,
        cmp_op: BinOp,
    ) -> Value {
        let val = self.materialize_null_check_source(src, ty);
        let null_val = self.lower_expr_null();
        self.emit_bool_binary(cmp_op, Operand::Value(val), Operand::Value(null_val))
    }

    /// Lowers `nullable == rhs` or `nullable != rhs`, where `nullable` has a
    /// tagged `T?` representation and `rhs` is a plain `T`.
    ///
    /// Semantically, `a == b` holds exactly when `a` is non-null *and* its payload
    /// equals `b`. We compute:
    ///
    /// ```text
    /// is_some = (tag != 0)
    /// peq     = (payload == rhs)
    /// both    = is_some && peq        // result for `==`
    /// result  = both                  // `==`
    ///         | (both == 0)           // `!=`  (negation)
    /// ```
    pub(super) fn lower_tagged_nullable_eq(
        &mut self,
        nullable: Value,
        ty: &SsaType,
        rhs: &HirExpr<'a, 'bump>,
        is_eq: bool,
    ) -> Value {
        let SsaType::Nullable(inner) = ty else {
            unreachable!()
        };

        let rv = self.lower_expr_expected(rhs, inner);

        let is_some = self.emit_tag_is_some(nullable);

        let payload = self.unwrap_known_nonnull(nullable, ty);
        let payload_eq =
            self.emit_bool_binary(BinOp::Eq, Operand::Value(payload), Operand::Value(rv));

        let both = self.and_conds(Some(is_some), payload_eq);
        if is_eq {
            both
        } else {
            self.emit_bool_not(both)
        }
    }

    /// If exactly one side of a comparison is the `null` literal, returns the
    /// *other* side; otherwise `None`.
    ///
    /// | `left`  | `right` | result           |
    /// |---------|---------|------------------|
    /// | `x`     | `null`  | `Some(x)`        |
    /// | `null`  | `x`     | `Some(x)`        |
    /// | `null`  | `null`  | `None`           |
    /// | `x`     | `y`     | `None`           |
    ///
    /// `null == null` is deliberately excluded: it flows through the general path
    /// instead of being treated as a null check on a null operand.
    fn non_null_side<'x>(
        left: &'x HirExpr<'a, 'bump>,
        right: &'x HirExpr<'a, 'bump>,
    ) -> Option<&'x HirExpr<'a, 'bump>> {
        match (left, right) {
            (HirExpr::Null(_), HirExpr::Null(_)) => None,
            (other, HirExpr::Null(_)) | (HirExpr::Null(_), other) => Some(other),
            _ => None,
        }
    }

    /// Lowers an ordinary binary comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) once
    /// null checks and tagged-nullable equality have been ruled out.
    ///
    /// The left operand is already lowered (the dispatcher needed its type), so we
    /// take its value and type. The right operand is lowered with the left's type
    /// as the *expected* type, which lets untyped literals adopt the right width
    /// (`x < 10` makes `10` the same integer type as `x`).
    fn lower_plain_comparison(
        &mut self,
        l: Value,
        l_ty: &SsaType,
        op: &Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let r = self.lower_expr_expected(right, l_ty);
        self.emit_bool_binary(lower_operator_bin(op), Operand::Value(l), Operand::Value(r))
    }

    /// Lowers a comparison expression `left <op> right`; the entry point for all
    /// comparison operators.
    ///
    /// It picks a strategy in this order:
    ///
    /// 1. **Null literal on one side** (`==` / `!=` only): delegate to
    ///    [`Self::lower_null_comparison`] with the non-null side.
    /// 2. **Left side is a tagged nullable** (`==` / `!=` only): delegate to
    ///    [`Self::lower_tagged_nullable_eq`], which compares tag and payload.
    /// 3. **Everything else:** [`Self::lower_plain_comparison`].
    ///
    /// Step 1 runs *before* lowering `left` so that, for example, a field access
    /// can be handled by address rather than loaded first. Step 2 has to lower
    /// `left` first, because only its resulting type reveals a tagged nullable.
    pub(crate) fn lower_comparison_expr(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        let is_equality = matches!(op, Operator::Equals | Operator::NotEquals);
        let is_eq = matches!(op, Operator::Equals);

        if is_equality {
            if let Some(other) = Self::non_null_side(left, right) {
                return self.lower_null_comparison(other, is_eq);
            }
        }

        let l = self.lower_expr(left);
        let l_ty = self.value_type_or_i64(l);

        if is_equality && l_ty.is_tagged_nullable() {
            return self.lower_tagged_nullable_eq(l, &l_ty, right, is_eq);
        }

        self.lower_plain_comparison(l, &l_ty, &op, right)
    }
}
