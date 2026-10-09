use ir::{
    hir::{HirErrorHandlerPattern, HirExpr, HirStmt, HirType, StrId},
    ir_conversion::lower_type_hir,
    ir_hasher::{HashMap, HashSet},
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, BlockId, Instruction, Operand, SsaType, Value},
};
use smallvec::SmallVec;

use crate::midend::{
    copy_analysis::drop_tracking::{DropMoveState, DropScope},
    ir::mir_lowering::FunctionLowerer,
};

/// A snapshot of `var_map` (source variable name -> current SSA value).
type VarSnapshot = HashMap<StrId, Value>;

/// A placeholder phi that is still waiting for more incoming edges:
/// `(variable name, index of the Phi instruction inside its block, the phi's result value)`.
///
/// The index lets [`FunctionLowerer::contribute_join_edge`] find and extend the phi later
/// without searching.
type PendingPhi = (StrId, usize, Value);

/// What one finished arm of an `if` / `else`-unwrap leaves behind for the merge step.
type ArmEdge<'a, 'bump> = (BlockId, VarSnapshot, DropMoveState<'a, 'bump>);

/// Bookkeeping for one enclosing loop, needed so `break`/`continue` (which
/// can appear arbitrarily deep inside nested control flow) can contribute a
/// phi edge to the right join point. `phis` are `(name, instruction-index,
/// phi-dest-value)` triples identifying placeholder `Instruction::Phi`s
/// created by `open_join`, still waiting for more incoming edges.
///
/// One `LoopCtx` is pushed on `loop_stack` when a loop starts and popped when its
/// body is finished, so `loop_stack.last()` is always the innermost loop, which is
/// exactly the loop an unlabeled `break` / `continue` refers to.
pub struct LoopCtx<'a, 'bump> {
    /// Where `continue` jumps to (the loop header for `while` or the
    /// increment block for `for`).
    pub(super) continue_target: BlockId,
    /// The block whose phis a `continue` must contribute an edge to (same
    /// as `continue_target`).
    pub(super) continue_join_bb: BlockId,
    /// The placeholder phis of `continue_join_bb`; a `continue` appends its own
    /// `(current block, current variable values)` edge to each.
    pub(super) continue_join_phis: Vec<PendingPhi>,
    /// Where `break` jumps to (the block right after the loop).
    pub(super) break_target: BlockId,
    /// The block (== `break_target`) whose phis a `break` must contribute
    /// an edge to.
    pub(super) break_join_phis: Vec<PendingPhi>,
    /// `scope_stack.len()` when the loop began. `break` / `continue` must emit drops for
    /// every scope *deeper* than this (the ones they are jumping out of), and none of the
    /// scopes that stay alive.
    pub(super) scope_depth_at_entry: usize,
    /// The `drop_state` at each `break`. The state after the loop is the join of the
    /// loop-header state and all of these, since any of them can be the exit path.
    pub(super) break_states: Vec<DropMoveState<'a, 'bump>>,
}

/// What lowering one arm of an if-*expression* produced.
struct IfArm {
    /// The arm's value (`Void`-typed if the arm diverged or produces no value).
    value: Value,
    /// The block that ends the arm, i.e. the phi predecessor. Not necessarily the block the
    /// arm started in, since lowering the arm may have created more blocks.
    end_bb: BlockId,
    /// True if the arm ended in a terminator (`return`, `break`, ...) and so never reaches the merge.
    terminated: bool,
}

/// Per-arm results collected while lowering an if-expression, consumed by the merge.
/// Only arms that *reach* the merge block add entries.
struct IfJoin<'a, 'bump> {
    /// `(arm end block, variable snapshot)` for each live arm. Feeds [`FunctionLowerer::merge_var_maps`].
    vars: Vec<(BlockId, VarSnapshot)>,
    /// The `drop_state` at the end of each live arm. Feeds `DropMoveState::join_all`.
    drops: Vec<DropMoveState<'a, 'bump>>,
}

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
    /// Lowers `if cond { .. } else { .. }` used as an *expression* with no type hint.
    /// Thin wrapper over [`Self::lower_if_expr_inner`].
    pub(super) fn lower_if_expr(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_if_expr_inner(condition, then_block, else_block, span, None)
    }

    /// Lowers an if-expression, optionally with an `expected` result type that is pushed
    /// down into both arms (so literals and `null` in the arms adopt the right type).
    ///
    /// The CFG produced is the classic diamond:
    ///
    /// ```text
    ///          cond
    ///         /    \
    ///     then_bb  else_bb
    ///         \    /
    ///        merge_bb:  result = phi [(then_end, then_val), (else_end, else_val)]
    /// ```
    ///
    /// Steps:
    /// 1. Snapshot `drop_state` and `var_map` (the else arm must start from the same state as
    ///    the then arm, not from whatever the then arm left behind).
    /// 2. Lower the condition and emit the conditional branch.
    /// 3. Lower each arm ([`Self::lower_if_expr_then_arm`], [`Self::lower_if_expr_else_arm`]).
    /// 4. If *both* arms diverged, nothing reaches the merge: return a dead `Void` value.
    /// 5. Otherwise open the merge block, merge variable maps and drop states, and build
    ///    the result value ([`Self::merge_if_expr_value`]).
    ///
    /// `merge_bb` is created with `fresh_block` (allocated but not yet placed in the
    /// function) and only inserted with `push_block` once we know at least one arm reaches
    /// it. That way a fully-diverging `if` leaves no empty, unreachable merge block behind.
    pub(super) fn lower_if_expr_inner(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
        expected: Option<&SsaType>,
    ) -> Value {
        let drop_before = self.drop_state.clone();
        let vars_before = self.var_map.clone();
        let mut join = IfJoin {
            vars: Vec::new(),
            drops: Vec::new(),
        };
        let cond = self.lower_expr_scoped(condition);

        let narrowed_before = self.narrowed_fields.clone();

        let then_bb = self.current_block_data.new_block();
        let else_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.fresh_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb,
            else_bb,
        });

        self.narrowed_fields = narrowed_before.clone();
        let then_arm = self.lower_if_expr_then_arm(
            condition, then_block, expected, span, then_bb, merge_bb, &mut join,
        );

        self.drop_state = drop_before;
        self.var_map = vars_before;
        self.narrowed_fields = narrowed_before.clone();
        let else_arm = self.lower_if_expr_else_arm(
            condition, else_block, expected, span, else_bb, merge_bb, &mut join,
        );

        if then_arm.terminated && else_arm.terminated {
            return self.unreachable_value();
        }

        self.narrowed_fields = narrowed_before;
        self.current_block_data.push_block(merge_bb);
        self.current_block_data.switch_to(merge_bb);
        self.merge_var_maps(join.vars);
        if let Some(j) = DropMoveState::join_all(join.drops) {
            self.drop_state = j;
        }

        self.merge_if_expr_value(&then_arm, &else_arm, span)
    }

    /// Lowers the `then` arm of an if-expression into `then_bb`.
    ///
    /// What happens, and why:
    /// * A fresh `DropScope` is pushed so locals declared in the arm are dropped when it ends.
    /// * `narrow_nonnull(condition, true)` tells the lowerer "in this arm the condition was
    ///   true", so for `if x != null { .. }` the variable `x` is treated as non-null.
    ///   It returns what it changed so we can *undo* it afterwards: the narrowed binding
    ///   is only valid inside the arm and must not leak into the merge.
    /// * The block is lowered as a value-producing block (last expression is the arm's value).
    /// * Scope drops are emitted only if the arm did **not** terminate (after a `return` the
    ///   drops were already emitted by the `return` itself, and no code may follow a terminator).
    /// * The arm's end block, variables and drop state are recorded in `join`, and a `Jump`
    ///   to the merge block is emitted, but only for arms that actually fall through.
    ///
    /// `end_bb` is read *after* emitting drops because drop emission can itself create blocks.
    #[allow(clippy::too_many_arguments)]
    fn lower_if_expr_then_arm(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        expected: Option<&SsaType>,
        span: SourceSpan<'a>,
        then_bb: BlockId,
        merge_bb: BlockId,
        join: &mut IfJoin<'a, 'bump>,
    ) -> IfArm {
        self.current_block_data.switch_to(then_bb);
        self.scope_stack.push(DropScope::default());
        let then_narrowed = self.narrow_nonnull(condition, true);
        let then_narrowed_path = self.narrow_nonnull_path(condition, true);
        let value = self.lower_block_value_inner(then_block, expected);
        let then_scope = self.scope_stack.pop().unwrap();

        // Undo the narrowing so it does not escape the arm.
        Self::revert_narrow_in_map(&mut self.var_map, then_narrowed);
        if let Some((root, path)) = then_narrowed_path {
            self.narrowed_fields.remove(&(root, path));
        }

        let terminated = self.block_terminated();
        if !terminated {
            self.emit_scope_drops(&then_scope, span);
            join.drops.push(self.drop_state.clone());
        }
        let end_bb = self.current_block_data.current_block;
        if !terminated {
            join.vars.push((end_bb, self.var_map.clone()));
            self.emit(Instruction::Jump { target: merge_bb });
        }

        IfArm {
            value,
            end_bb,
            terminated,
        }
    }

    /// Lowers the `else` arm of an if-expression into `else_bb`.
    ///
    /// An if *expression* must produce a value on every path, so the else arm is mandatory
    /// and must be either:
    /// * a plain block, lowered like the then-arm (own `DropScope`, scope drops at the end), or
    /// * another `if` (an `else if` chain), lowered by recursing into
    ///   [`Self::lower_if_expr_inner`]. The recursion manages its own scopes and merge, so no
    ///   extra scope is pushed here.
    ///
    /// Anything else is a front-end bug and panics with a descriptive message.
    /// Narrowing is applied with `condition` *false* (`if x == null { .. } else { <x non-null> }`)
    /// and reverted at the end, exactly mirroring the then-arm.
    #[allow(clippy::too_many_arguments)]
    fn lower_if_expr_else_arm(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        expected: Option<&SsaType>,
        span: SourceSpan<'a>,
        else_bb: BlockId,
        merge_bb: BlockId,
        join: &mut IfJoin<'a, 'bump>,
    ) -> IfArm {
        self.current_block_data.switch_to(else_bb);
        let else_narrowed = self.narrow_nonnull(condition, false);
        let else_narrowed_path = self.narrow_nonnull_path(condition, false);
        let value = match else_block {
            Some(HirStmt::Block { body, span: _ }) => {
                self.scope_stack.push(DropScope::default());
                let v = self.lower_block_value_inner(body, expected);
                let else_scope = self.scope_stack.pop().unwrap();
                if !self.block_terminated() {
                    self.emit_scope_drops(&else_scope, span);
                }
                v
            }
            Some(HirStmt::If {
                cond: ec,
                then_block: etb,
                else_block: eeb,
                span: espan,
            }) => self.lower_if_expr_inner(ec, etb, *eeb, *espan, expected),
            Some(other) => panic!(
                "if-expression else-arm must be a block or else-if, found {:?}: \
             every path through an if used as an expression needs a value",
                other
            ),
            None => panic!(
                "if-expression used without an else arm at {span}; the type checker should \
             have caught this before MIR lowering"
            ),
        };

        Self::revert_narrow_in_map(&mut self.var_map, else_narrowed);
        if let Some((root, path)) = else_narrowed_path {
            self.narrowed_fields.remove(&(root, path));
        }

        let terminated = self.block_terminated();
        if !terminated {
            join.drops.push(self.drop_state.clone());
        }
        let end_bb = self.current_block_data.current_block;
        if !terminated {
            join.vars.push((end_bb, self.var_map.clone()));
            self.emit(Instruction::Jump { target: merge_bb });
        }

        IfArm {
            value,
            end_bb,
            terminated,
        }
    }

    /// Computes the SSA value of an if-expression once both arms are lowered and the merge
    /// block is current (at least one arm is live).
    ///
    /// * The result type is read from whichever arm is live (the then-arm if it is, else the else-arm).
    /// * `Void` result (e.g. the `if` is used for side effects): no phi, just a unit value.
    /// * Otherwise a phi is built with one operand per *live* arm ([`Self::emit_merge_phi`]).
    fn merge_if_expr_value(
        &mut self,
        then_arm: &IfArm,
        else_arm: &IfArm,
        span: SourceSpan<'a>,
    ) -> Value {
        let reference = if !then_arm.terminated {
            then_arm.value
        } else {
            else_arm.value
        };
        let ty = self.value_type(reference).cloned().unwrap_or(SsaType::Void);

        if ty == SsaType::Void {
            return self.unit_value();
        }

        let mut incoming: SmallVec<(BlockId, Value), 4> = SmallVec::new();
        if !then_arm.terminated {
            incoming.push((then_arm.end_bb, then_arm.value));
        }
        if !else_arm.terminated {
            incoming.push((else_arm.end_bb, else_arm.value));
        }
        self.emit_merge_phi(incoming, span)
    }

    /// Emits the phi that merges several arm values into one result and returns it.
    ///
    /// 1. [`Self::reconcile_phi_type`] picks one common type for all incoming values
    ///    (letting `null` adopt the other side's type, ignoring `Void`).
    /// 2. If the common type is `Void` there is nothing to merge: return a unit value.
    /// 3. [`Self::patch_diverging_edges`] replaces any remaining `Void` operand with a typed dummy,
    ///    because a phi operand must have the phi's type on every edge.
    /// 4. Emit the `Phi` and record its type.
    pub fn emit_merge_phi(
        &mut self,
        mut incoming: SmallVec<(BlockId, Value), 4>,
        span: SourceSpan<'a>,
    ) -> Value {
        let ty = self.reconcile_phi_type(&incoming, span);
        if ty == SsaType::Void {
            return self.unit_value();
        }
        self.patch_diverging_edges(&mut incoming, &ty);
        let result = self.current_block_data.fresh_value();
        self.emit(Instruction::Phi {
            dest: result,
            incoming,
        });
        self.current_block_data.value_types.insert(result, ty);
        result
    }

    /// Lowers `while cond { body }`.
    ///
    /// ```text
    ///   pre_loop ──jump──▶ cond_bb ──true──▶ body_bb ──jump────┐
    ///                       ▲   │                              │  (back-edge)
    ///                       │   └──false──▶ after_bb           │
    ///                       └──────────────────────────────────┘
    /// ```
    ///
    /// * `cond_bb` is the loop header. It is also the `continue` target.
    /// * `after_bb` is the exit. It is also the `break` target.
    /// * Both blocks get placeholder phis ([`Self::open_join`]) because variables can change
    ///   in the body, and the edges that feed those phis are discovered while lowering it.
    ///
    /// Edges into the header phis: the pre-loop edge (added at once), the back-edge from the end
    /// of the body (added after lowering it), and one from each `continue`.
    /// Edges into the exit phis: the condition-false edge (added at once) and one from each `break`.
    pub(super) fn lower_while_loop(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        body: &HirStmt<'a, 'bump>,
    ) {
        let pre_loop_bb = self.current_block_data.current_block;
        let vars_before = self.var_map.clone();
        let narrowed_before = self.narrowed_fields.clone();

        let cond_bb = self.current_block_data.new_block();
        let body_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();

        self.emit(Instruction::Jump { target: cond_bb });

        let (header_phis, header_drop) = self.open_loop_header(cond_bb, pre_loop_bb, &vars_before);

        let cond_val = self.lower_expr_scoped(cond);
        let cond_end = self.current_block_data.current_block;
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond_val),
            then_bb: body_bb,
            else_bb: after_bb,
        });
        let header_vars = self.var_map.clone();

        let exit_phis = self.open_loop_exit_join(after_bb, cond_end, &header_vars);

        self.push_loop_ctx(cond_bb, after_bb, &header_phis, &exit_phis);

        self.var_map = header_vars;
        self.lower_loop_body(body_bb, Some(cond), body);
        self.jump_with_join_edge(cond_bb, &header_phis);
        let ctx = self.loop_stack.pop().unwrap();

        self.narrowed_fields = narrowed_before;
        self.finish_loop(after_bb, exit_phis, header_drop, ctx);
    }

    /// Lowers `for (init; condition; increment) { body }`. Every part of ("...") in "for (...) {}" is optional.
    ///
    /// ```text
    ///   init
    ///     │
    ///     ▼
    ///   cond_bb ──true──▶ body_bb ──▶ incr_bb ──jump──▶ cond_bb   (back-edge)
    ///     │                              ▲
    ///     └──false──▶ after_bb           └── `continue` lands here
    /// ```
    ///
    /// The difference from `while` is the extra `incr_bb`:
    /// * `continue` must run the increment, so its target is `incr_bb`, *not* the header.
    /// * `incr_bb` therefore has its own set of placeholder phis (`incr_phis`): the body's
    ///   normal end and every `continue` feed them, and they in turn feed the header phis.
    /// * With no condition, the header jumps straight to the body (an infinite loop unless `break`).
    pub(super) fn lower_for_loop(
        &mut self,
        init: Option<&'bump HirStmt<'a, 'bump>>,
        condition: Option<&'bump HirExpr<'a, 'bump>>,
        increment: Option<&'bump HirExpr<'a, 'bump>>,
        body: &HirStmt<'a, 'bump>,
    ) {
        if let Some(init_stmt) = init {
            self.lower_stmt(init_stmt);
        }

        let pre_loop_bb = self.current_block_data.current_block;
        let vars_before = self.var_map.clone();
        let narrowed_before = self.narrowed_fields.clone();

        let cond_bb = self.current_block_data.new_block();
        let body_bb = self.current_block_data.new_block();
        let incr_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();

        self.emit(Instruction::Jump { target: cond_bb });

        let (header_phis, header_drop) = self.open_loop_header(cond_bb, pre_loop_bb, &vars_before);

        let cond_end = self.lower_for_condition(condition, cond_bb, body_bb, after_bb);
        let header_vars = self.var_map.clone();

        let exit_phis = self.open_loop_exit_join(after_bb, cond_end, &header_vars);

        // Open the increment block's own phis. It is only entered from the body tail and `continue`s.
        self.current_block_data.switch_to(incr_bb);
        let incr_phis = self.open_join();

        self.push_loop_ctx(incr_bb, after_bb, &incr_phis, &exit_phis);

        self.var_map = header_vars;
        self.lower_loop_body(body_bb, condition, body);
        self.jump_with_join_edge(incr_bb, &incr_phis);
        let ctx = self.loop_stack.pop().unwrap();

        self.lower_for_increment(
            incr_bb,
            cond_bb,
            &incr_phis,
            &header_phis,
            &header_drop,
            increment,
        );

        self.narrowed_fields = narrowed_before;
        self.finish_loop(after_bb, exit_phis, header_drop, ctx);
    }

    /// Starts the loop header block: makes `cond_bb` current, creates its placeholder phis, and
    /// feeds them the pre-loop edge. Returns the header phis and the drop state to use inside the loop.
    ///
    /// The steps and their reasons:
    /// * `narrowed_fields.clear()`: null-narrowing facts from before the loop are not valid on
    ///   the *back-edge* (the body may have reassigned the field), so the header starts with none.
    /// * `open_join` + `contribute_join_edge(pre_loop edge)`: one empty phi per variable; its first
    ///   operand is the variable's value on loop entry.
    /// * `loop_header_state`: the conservative drop state for the header ([`Self::loop_header_state`]).
    ///
    /// The caller must already have emitted `Jump cond_bb` from the pre-loop block.
    fn open_loop_header(
        &mut self,
        cond_bb: BlockId,
        pre_loop_bb: BlockId,
        vars_before: &VarSnapshot,
    ) -> (Vec<PendingPhi>, DropMoveState<'a, 'bump>) {
        self.current_block_data.switch_to(cond_bb);
        self.narrowed_fields.clear();
        let header_phis = self.open_join();
        self.contribute_join_edge(cond_bb, pre_loop_bb, vars_before, &header_phis);

        let header_drop = self.loop_header_state();
        self.drop_state = header_drop.clone();
        (header_phis, header_drop)
    }

    /// Creates the loop's exit block phis and feeds them the "condition was false" edge.
    /// `break` statements add further edges later through `LoopCtx::break_join_phis`.
    ///
    /// `cond_end` is the block that actually ends the condition evaluation, which can differ
    /// from the header block if the condition contains short-circuit operators.
    /// `header_vars` are the variable values at that point, which flow out on that edge.
    fn open_loop_exit_join(
        &mut self,
        after_bb: BlockId,
        cond_end: BlockId,
        header_vars: &VarSnapshot,
    ) -> Vec<PendingPhi> {
        self.current_block_data.switch_to(after_bb);
        let exit_phis = self.open_join();
        self.contribute_join_edge(after_bb, cond_end, header_vars, &exit_phis);
        exit_phis
    }

    /// Registers a new innermost loop on `loop_stack` so nested `break` / `continue` know where
    /// to jump and which phis to extend. `continue_phis` belong to the `continue` target block
    /// (the header for `while`, the increment block for `for`), `break_phis` to the exit block.
    fn push_loop_ctx(
        &mut self,
        continue_bb: BlockId,
        break_bb: BlockId,
        continue_phis: &[PendingPhi],
        break_phis: &[PendingPhi],
    ) {
        self.loop_stack.push(LoopCtx {
            continue_target: continue_bb,
            continue_join_bb: continue_bb,
            continue_join_phis: continue_phis.to_vec(),
            break_target: break_bb,
            break_join_phis: break_phis.to_vec(),
            scope_depth_at_entry: self.scope_stack.len(),
            break_states: Vec::new(),
        });
    }

    /// Lowers the condition part of a `for` header and returns the block that ends it.
    ///
    /// * With a condition: evaluate it and branch to `body_bb` (true) or `after_bb` (false).
    /// * Without one: jump straight to `body_bb`. The loop only exits via `break`/`return`.
    ///
    /// The returned block is the predecessor of `after_bb` on the "condition false" edge. With
    /// no condition there is no such edge in the CFG, so `cond_bb` is returned only as a
    /// placeholder. [`Self::open_loop_exit_join`] still records an exit-phi edge from it
    /// (this matches the original behaviour; the edge never executes because the block
    /// jumps straight to the body).
    fn lower_for_condition(
        &mut self,
        condition: Option<&HirExpr<'a, 'bump>>,
        cond_bb: BlockId,
        body_bb: BlockId,
        after_bb: BlockId,
    ) -> BlockId {
        match condition {
            Some(cond_expr) => {
                let cond_val = self.lower_expr_scoped(cond_expr);
                let cond_end = self.current_block_data.current_block;
                self.emit(Instruction::Branch {
                    cond: Operand::Value(cond_val),
                    then_bb: body_bb,
                    else_bb: after_bb,
                });
                cond_end
            }
            None => {
                self.emit(Instruction::Jump { target: body_bb });
                cond_bb
            }
        }
    }

    /// Lowers a loop body statement inside `body_bb`, applying the loop condition's null-narrowing.
    ///
    /// * `narrowed_fields` is cleared first so the body does not inherit stale facts.
    /// * If there is a condition like `while p != null`, the body is lowered with `p` known
    ///   non-null (`narrow_nonnull(cond, true)`). That narrowing is undone after the body, since
    ///   the *next* iteration re-checks the condition and the fact may no longer hold.
    /// * `narrowed_fields` is cleared again afterwards for the same reason.
    fn lower_loop_body(
        &mut self,
        body_bb: BlockId,
        condition: Option<&HirExpr<'a, 'bump>>,
        body: &HirStmt<'a, 'bump>,
    ) {
        self.current_block_data.switch_to(body_bb);
        self.narrowed_fields.clear();
        let mut loop_narrowed = None;
        if let Some(cond_expr) = condition {
            loop_narrowed = self.narrow_nonnull(cond_expr, true);
            self.narrow_nonnull_path(cond_expr, true);
        }
        self.lower_stmt(body);
        Self::revert_narrow_in_map(&mut self.var_map, loop_narrowed);
        self.narrowed_fields.clear();
    }

    /// If the current block is still open, records the current variables as an incoming edge of
    /// `join_bb`'s placeholder phis and jumps there. A block that already ended in a terminator
    /// (the body ended with `break` / `return`) contributes nothing: it never reaches `join_bb`.
    ///
    /// Used for the loop back-edge (body tail -> header or increment block) and for the end of the
    /// increment block (-> header).
    fn jump_with_join_edge(&mut self, join_bb: BlockId, phis: &[PendingPhi]) {
        if !self.block_terminated() {
            let tail_bb = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.contribute_join_edge(join_bb, tail_bb, &vars, phis);
            self.emit(Instruction::Jump { target: join_bb });
        }
    }

    /// Lowers the increment block of a `for` loop and closes the back-edge to the header.
    ///
    /// On entry the variable map is rebuilt from `incr_phis`, so the increment sees the values
    /// that arrived from the body tail or a `continue`, not whatever the body left in `var_map`.
    /// The drop state is reset to the header's, since both entries to the increment block
    /// (body tail and `continue`) restore it to a consistent per-iteration state.
    fn lower_for_increment(
        &mut self,
        incr_bb: BlockId,
        cond_bb: BlockId,
        incr_phis: &[PendingPhi],
        header_phis: &[PendingPhi],
        header_drop: &DropMoveState<'a, 'bump>,
        increment: Option<&HirExpr<'a, 'bump>>,
    ) {
        self.var_map = incr_phis
            .iter()
            .map(|(name, _, dest)| (*name, *dest))
            .collect();
        self.current_block_data.switch_to(incr_bb);
        self.narrowed_fields.clear();
        self.drop_state = header_drop.clone();
        if let Some(inc_expr) = increment {
            let _ = self.lower_expr_scoped(inc_expr);
        }
        self.jump_with_join_edge(cond_bb, header_phis);
    }

    /// Final step shared by both loops: continue lowering code *after* the loop in `after_bb`.
    ///
    /// * `var_map` becomes the exit phis' results, i.e. each variable's value as seen on
    ///   *every* way out of the loop (condition false or `break`).
    /// * `drop_state` is the join of the header state and each `break`'s state, since the loop
    ///   can be left from any of those points.
    ///
    /// The caller restores `narrowed_fields` to its pre-loop value first (nothing learned
    /// inside the loop survives it).
    fn finish_loop(
        &mut self,
        after_bb: BlockId,
        exit_phis: Vec<PendingPhi>,
        header_drop: DropMoveState<'a, 'bump>,
        ctx: LoopCtx<'a, 'bump>,
    ) {
        self.current_block_data.switch_to(after_bb);
        self.var_map = exit_phis
            .into_iter()
            .map(|(name, _, dest)| (name, dest))
            .collect();
        if let Some(j) =
            DropMoveState::join_all(std::iter::once(header_drop).chain(ctx.break_states))
        {
            self.drop_state = j;
        }
    }

    /// The drop state to use at a loop header: the current state, but with all per-index
    /// knowledge about arrays forgotten.
    ///
    /// Inside a loop the same code runs many times, and an index expression like `a[i]` can
    /// refer to a different element each iteration. Facts of the form "element 3 of `a` was
    /// moved" from before the loop therefore cannot be trusted on later iterations. `havoc_indices`
    /// discards those facts (conservatively) for every array variable that has move flags.
    pub(super) fn loop_header_state(&self) -> DropMoveState<'a, 'bump> {
        let mut s = self.drop_state.clone();
        for name in self.array_flags.keys() {
            s.havoc_indices(*name);
        }
        s
    }

    /// Opens a join point whose incoming edges are not all known yet. Creates one *empty* phi
    /// per variable in `var_map`, rebinds each variable to its phi's result, and returns the
    /// list of placeholders.
    ///
    /// This is how loop headers work: when we reach the header we don't know what the
    /// variables will be on the back-edge, so we create `x_header = phi []` for every `x` now,
    /// use `x_header` throughout the loop, and fill in the operands later with
    /// [`Self::contribute_join_edge`].
    ///
    /// Each phi gets the *type of the variable's current value* (so a variable's type
    /// must not change across the loop; a mismatch is caught later as a "phi type mismatch" panic).
    /// The returned tuples hold `(name, index of the Phi in its block, phi result)`.
    ///
    /// The phis are emitted into the **current** block, which is why it's supposed to be called after
    /// `switch_to(join_block)`.
    pub(super) fn open_join(&mut self) -> Vec<PendingPhi> {
        let names: Vec<StrId> = self.var_map.keys().copied().collect();
        let mut phis = Vec::with_capacity(names.len());
        for name in names {
            let old = self.var_map[&name];

            let ty = self
                .current_block_data
                .value_type(old)
                .expect("phi source should have a type")
                .clone();

            let dest = self.current_block_data.fresh_value();

            self.current_block_data.value_types.insert(dest, ty);

            let idx = self.current_block_data.bb().instructions.len();
            self.emit(Instruction::Phi {
                dest,
                incoming: SmallVec::new(),
            });

            self.var_map.insert(name, dest);
            phis.push((name, idx, dest));
        }
        phis
    }

    /// Adds one incoming edge `(from_bb, value of each variable in vars)` to the placeholder
    /// phis of `join_bb`.
    ///
    /// For every pending phi this: looks up the variable's value on this edge, makes it
    /// type-compatible ([`Self::load_pointee_for_phi_edge`]), validates the type
    /// ([`Self::assert_phi_edge_type`]), and finally appends the operand
    /// ([`Self::append_phi_incoming`]).
    ///
    /// A variable missing from `vars` is skipped for that edge. That happens for variables
    /// declared *inside* the loop body, which do not exist on the paths that never entered it.
    pub(super) fn contribute_join_edge(
        &mut self,
        join_bb: BlockId,
        from_bb: BlockId,
        vars: &HashMap<StrId, Value>,
        phis: &[PendingPhi],
    ) {
        if phis.is_empty() {
            return;
        }

        let mut incoming: Vec<(usize, Value)> = Vec::with_capacity(phis.len());
        for (name, idx, dest) in phis {
            let Some(val) = vars.get(name).copied() else {
                continue;
            };
            let phi_ty = self.current_block_data.value_type(*dest).cloned();
            let val = self.load_pointee_for_phi_edge(val, phi_ty.as_ref(), from_bb);
            self.assert_phi_edge_type(join_bb, from_bb, *name, *dest, val, phi_ty.as_ref());
            incoming.push((*idx, val));
        }

        self.append_phi_incoming(join_bb, from_bb, incoming);
    }

    /// Fixes up the common "variable lives in a stack slot" mismatch on a phi edge.
    ///
    /// Some variables are held by *address* (a pointer to a slot) rather than by value. If the
    /// phi was created for the plain type `T` but the edge supplies `*T` (a pointer to a `T`),
    /// the edge value is read through the pointer (`LoadField` at offset 0) so the phi gets a `T`.
    ///
    /// The load is only emitted when `from_bb` is the **current** block: an instruction can only be
    /// appended to the block we are building, and the edge's source is the current block in
    /// every case where this situation occurs. Otherwise the value is returned unchanged and the
    /// type check afterwards will report any real mismatch.
    fn load_pointee_for_phi_edge(
        &mut self,
        val: Value,
        phi_ty: Option<&SsaType>,
        from_bb: BlockId,
    ) -> Value {
        let val_ty = self.current_block_data.value_type(val).cloned();
        if let (Some(pt), Some(SsaType::Pointer(_, inner))) = (phi_ty, &val_ty) {
            if &**inner == pt && from_bb == self.current_block_data.current_block {
                let loaded = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: loaded,
                    base: Operand::Value(val),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(loaded, pt.clone());
                return loaded;
            }
        }
        val
    }

    /// Panics if `val`'s type differs from the type its phi was created with.
    ///
    /// The usual cause is a front-end/type-checker bug that rebinds a variable to a value of a
    /// different type between the point the phi was opened and the edge being added (for example
    /// narrowing `T?` to `T` and not undoing it before the back-edge). The message names the
    /// variable, both blocks and both types to make that easy to track down.
    fn assert_phi_edge_type(
        &self,
        join_bb: BlockId,
        from_bb: BlockId,
        name: StrId,
        dest: Value,
        val: Value,
        phi_ty: Option<&SsaType>,
    ) {
        let val_ty = self.current_block_data.value_type(val).cloned();
        if let (Some(pt), Some(vt)) = (phi_ty, &val_ty) {
            if pt != vt {
                panic!(
                    "phi type mismatch: joining into block {:?} from block {:?}, \
                     variable `{}` was typed {:?} when this join point was opened \
                     (phi dest {:?}), but the value now bound to it here ({:?}) has \
                     type {:?} instead, `{}` was rebound to a differently-typed \
                     value somewhere between the join point and this edge",
                    join_bb, from_bb, name, pt, dest, val, vt, name
                );
            }
        }
    }

    /// Appends `(from_bb, value)` to the phi instructions of `join_bb` at the given indices.
    ///
    /// `join_bb` is found by id because it is generally *not* the current block (a `break` adds an
    /// edge to the loop exit while lowering code deep inside the body).
    fn append_phi_incoming(
        &mut self,
        join_bb: BlockId,
        from_bb: BlockId,
        edges: Vec<(usize, Value)>,
    ) {
        let block = self
            .current_block_data
            .func
            .blocks
            .iter_mut()
            .find(|b| b.id == join_bb)
            .expect("join block missing");
        for (idx, val) in edges {
            if let Instruction::Phi { incoming, .. } = &mut block.instructions[idx] {
                incoming.push((from_bb, val));
            }
        }
    }

    /// Replaces `var_map` with the merge of the variable maps from each *live* branch that
    /// flows into the current (merge) block, emitting phis where branches disagree.
    ///
    /// * 0 branches: every path diverged, nothing to merge, leave `var_map` alone.
    /// * 1 branch: no merge needed; adopt that branch's map as-is.
    /// * 2+ branches: for each variable seen in any branch, [`Self::merge_var_across_branches`]
    ///   decides between reusing the single common value or emitting a phi.
    ///
    /// Branches are `(end block, variables at that end block)`. The end block is what a phi
    /// operand must name as its predecessor.
    pub(super) fn merge_var_maps(&mut self, branches: Vec<(BlockId, HashMap<StrId, Value>)>) {
        match branches.len() {
            0 => {
                // every incoming branch diverged
            }
            1 => {
                self.var_map = branches.into_iter().next().unwrap().1;
            }
            _ => {
                let mut all_names: HashSet<StrId> = HashSet::default();
                for (_, vars) in &branches {
                    all_names.extend(vars.keys().copied());
                }

                let mut merged = HashMap::default();
                for name in all_names {
                    if let Some(v) = self.merge_var_across_branches(name, &branches) {
                        merged.insert(name, v);
                    }
                }
                self.var_map = merged;
            }
        }
    }

    /// Decides the merged value of a single variable `name` across several branches.
    ///
    /// * **Not present in any branch** -> `None` (the variable simply goes out of scope).
    /// * **Same value in every branch that has it** -> reuse it; no phi (the variable was not
    ///   reassigned, or all branches assigned the *same* SSA value).
    /// * **`Void`-typed** -> reuse the first value; there is nothing meaningful to merge.
    /// * **Otherwise** -> emit `dest = phi [(branch end, value) ...]` in the current (merge)
    ///   block, typed like the first branch's value, and return `dest`.
    ///
    /// Only branches that *contain* the variable contribute operands; a variable declared inside one
    /// arm is simply absent from the other.
    fn merge_var_across_branches(
        &mut self,
        name: StrId,
        branches: &[(BlockId, HashMap<StrId, Value>)],
    ) -> Option<Value> {
        let mut entries: Vec<(BlockId, Value)> = Vec::new();
        for (bb, vars) in branches {
            if let Some(v) = vars.get(&name) {
                entries.push((*bb, *v));
            }
        }
        let (_, first_val) = entries.first().copied()?;
        if entries.iter().all(|(_, v)| *v == first_val) {
            return Some(first_val);
        }

        let first_ty = self
            .current_block_data
            .value_type(first_val)
            .unwrap()
            .clone();

        if first_ty == SsaType::Void {
            return Some(first_val);
        }

        let dest = self.current_block_data.fresh_value();
        self.current_block_data.value_types.insert(dest, first_ty);

        self.emit(Instruction::Phi {
            dest,
            incoming: entries.into_iter().collect(),
        });
        Some(dest)
    }

    /// Undoes a null-narrowing of a local variable made by `narrow_nonnull`.
    ///
    /// `narrowed` is `(variable, value before narrowing, value after narrowing)`. The variable
    /// is rebound to the old value *only if* it is still bound to the narrowed one. If the arm
    /// reassigned the variable in the meantime, that newer binding is the truth and must not
    /// be clobbered.
    pub(super) fn revert_narrow_in_map(
        vars: &mut HashMap<StrId, Value>,
        narrowed: Option<(StrId, Value, Value)>,
    ) {
        if let Some((name, prev, narrowed_val)) = narrowed {
            if vars.get(&name).copied() == Some(narrowed_val) {
                vars.insert(name, prev);
            }
        }
    }

    /// Lowers `if cond { .. } [else ..]` used as a *statement* (no value produced).
    ///
    /// Unlike the expression form, the `else` arm is optional, which creates two shapes:
    ///
    /// ```text
    /// with a real else block:        without one:
    ///       cond                          cond
    ///      /    \                        /    \
    ///   then    else                  then     │
    ///      \    /                        \     │
    ///      merge                          merge   (the false edge goes straight to merge)
    /// ```
    ///
    /// A real else block is created when there is an `else`, **or** when the condition can
    /// narrow nullness ([`Self::cond_may_narrow_null`]): `if x == null { return }` has no `else`,
    /// yet the code *after* it runs with `x` known non-null, and the narrowing is applied in
    /// the (otherwise empty) else block.
    ///
    /// When the false edge skips straight to merge, the "else" state is just the state before the
    /// `if`, with the pre-`if` block as the phi predecessor.
    pub(super) fn lower_if_stmt(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &'bump [HirStmt<'a, 'bump>],
        else_block: &Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        let drop_before = self.drop_state.clone();
        let mut live_drop = Vec::new();
        let narrowed_before = self.narrowed_fields.clone();
        let cond_val = self.lower_expr_scoped(cond);
        let pre_if_bb = self.current_block_data.current_block;
        let vars_before = self.var_map.clone();

        let then_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.fresh_block();
        let needs_real_else_block = else_block.is_some() || Self::cond_may_narrow_null(cond);
        let else_bb = if needs_real_else_block {
            self.current_block_data.new_block()
        } else {
            merge_bb
        };

        self.emit(Instruction::Branch {
            cond: Operand::Value(cond_val),
            then_bb,
            else_bb,
        });

        self.var_map = vars_before.clone();
        self.narrowed_fields = narrowed_before.clone();
        let then_live =
            self.lower_if_stmt_then_arm(cond, then_block, then_bb, merge_bb, span, &mut live_drop);

        let else_live = if needs_real_else_block {
            self.drop_state = drop_before.clone();
            self.var_map = vars_before.clone();
            self.narrowed_fields = narrowed_before.clone();
            self.lower_if_stmt_else_arm(
                cond,
                else_block,
                else_bb,
                merge_bb,
                span,
                then_live.is_some(),
                &mut live_drop,
            )
        } else {
            live_drop.push(drop_before.clone());
            Some((pre_if_bb, vars_before))
        };

        self.narrowed_fields = narrowed_before;
        let live_branches: Vec<_> = [then_live, else_live].into_iter().flatten().collect();
        if !live_branches.is_empty() {
            self.current_block_data.push_block(merge_bb);
            self.current_block_data.switch_to(merge_bb);
            self.merge_var_maps(live_branches);
            match DropMoveState::join_all(live_drop) {
                Some(j) => self.drop_state = j,
                None => self.drop_state = drop_before.clone(),
            }
        }
    }

    /// Lowers the `then` arm of an if-statement. Returns `Some((end block, variables))` if the arm
    /// falls through to the merge, or `None` if it diverged (in which case it is excluded from the merge).
    ///
    /// The caller has already reset `var_map` and `narrowed_fields` for this arm
    fn lower_if_stmt_then_arm(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &'bump [HirStmt<'a, 'bump>],
        then_bb: BlockId,
        merge_bb: BlockId,
        span: SourceSpan<'a>,
        live_drop: &mut Vec<DropMoveState<'a, 'bump>>,
    ) -> Option<(BlockId, VarSnapshot)> {
        self.current_block_data.switch_to(then_bb);
        self.scope_stack.push(DropScope::default());
        let then_narrowed = self.narrow_nonnull(cond, true);
        self.narrow_nonnull_path(cond, true);
        self.lower_stmt_seq(then_block);
        let then_scope = self.scope_stack.pop().unwrap();

        if self.block_terminated() {
            return None;
        }
        self.emit_scope_drops(&then_scope, span);
        live_drop.push(self.drop_state.clone());
        Self::revert_narrow_in_map(&mut self.var_map, then_narrowed);
        let tail = self.current_block_data.current_block;
        let vars = self.var_map.clone();
        self.emit(Instruction::Jump { target: merge_bb });
        Some((tail, vars))
    }

    /// Lowers the (real) `else` block of an if-statement. Same contract as
    /// [`Self::lower_if_stmt_then_arm`]: `Some((end block, variables))` if it falls through, else `None`.
    ///
    /// The caller has already restored `drop_state`, `var_map` and `narrowed_fields` to their
    /// pre-`if` values.
    ///
    /// The block may have *no* source `else` at all (it exists only to carry null-narrowing for
    /// `if x == null { return }`-style code). In that case no `DropScope` is pushed or popped, and the
    /// block just applies the narrowing and jumps to the merge.
    ///
    /// Null-narrowing in this arm, and when it is undone:
    ///
    /// `narrow_nonnull(cond, false)` narrows the variable for `if x == null { .. } else { <x non-null> }`.
    /// What happens to that narrowing at the end of the arm depends on `then_reaches_merge`:
    ///
    /// * **The `then` arm diverged** (`if x == null { return }`, a guard clause). The else block is
    ///   the *only* path to the code after the `if`, so the narrowing is deliberately **kept**:
    ///   after the `if`, `x` is known non-null. This is the reason an else block is created
    ///   even when the source has no `else` (see [`Self::lower_if_stmt`]).
    /// * **Both arms reach the merge.** The `then` arm restored `x` to its nullable binding, so this arm
    ///   must too: otherwise the merge would join an unwrapped `x` (type `T`) with a nullable `x`
    ///   (type `T?`) in one phi. The narrowing is **reverted** (only if the arm did not rebind `x`
    ///   itself, see [`Self::revert_narrow_in_map`]).
    #[allow(clippy::too_many_arguments)]
    fn lower_if_stmt_else_arm(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        else_block: &Option<&'bump HirStmt<'a, 'bump>>,
        else_bb: BlockId,
        merge_bb: BlockId,
        span: SourceSpan<'a>,
        then_reaches_merge: bool,
        live_drop: &mut Vec<DropMoveState<'a, 'bump>>,
    ) -> Option<(BlockId, VarSnapshot)> {
        self.current_block_data.switch_to(else_bb);
        let else_narrowed = self.narrow_nonnull(cond, false);
        self.narrow_nonnull_path(cond, false);

        if let Some(else_stmt) = else_block {
            self.scope_stack.push(DropScope::default());
            self.lower_stmt(else_stmt);
        }

        if self.block_terminated() {
            if else_block.is_some() {
                self.scope_stack.pop();
            }
            None
        } else {
            if else_block.is_some() {
                let else_scope = self.scope_stack.pop().unwrap();
                self.emit_scope_drops(&else_scope, span);
            }
            live_drop.push(self.drop_state.clone());
            if then_reaches_merge {
                Self::revert_narrow_in_map(&mut self.var_map, else_narrowed);
            }
            let tail = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.emit(Instruction::Jump { target: merge_bb });
            Some((tail, vars))
        }
    }

    pub(super) fn stmt_diverges(stmt: &HirStmt) -> bool {
        matches!(
            stmt,
            HirStmt::Return(..) | HirStmt::Break(..) | HirStmt::Continue(_)
        )
    }

    pub(super) fn lower_catch(
        &mut self,
        raw_val: Value,
        pattern: &HirErrorHandlerPattern<'a, 'bump>,
    ) {
        let branches: Vec<(HirType, Option<StrId>, &[HirStmt])> = match pattern {
            HirErrorHandlerPattern::Single {
                error_type,
                binding,
                body,
            } => vec![(*error_type, *binding, *body)],
            HirErrorHandlerPattern::Multiple { branches } => branches
                .iter()
                .map(|b| (b.error_type, b.binding, b.body))
                .collect(),
        };

        let enum_ty = self
            .current_block_data
            .value_type(raw_val)
            .expect("catch target must have a known SsaType")
            .clone();
        let SsaType::Enum {
            variants: variant_types,
            ..
        } = &enum_ty
        else {
            panic!(
                "`catch` used on non-enum SsaType {:?}, thrown values must be SsaType::Enum",
                enum_ty
            );
        };

        let enum_layout =
            ir::layout::enum_layout_of_ssa(variant_types, TargetInfo { ptr_bytes: 8 })
                .unwrap_or_else(|e| panic!("failed to compute layout for error enum: {:?}", e));
        let tag_offset = enum_layout.tag_offset;
        let payload_offset = enum_layout.payload_offset;

        let tag_val = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: tag_val,
            base: Operand::Value(raw_val),
            offset: tag_offset,
        });

        let mut next_check_bb = self.current_block_data.current_block;

        for (i, (error_type, binding, body)) in branches.iter().enumerate() {
            let arm_tag = self.catch_arm_tag(error_type);

            let arm_body_bb = self.current_block_data.new_block();
            let is_last = i == branches.len() - 1;
            let fallthrough_bb = if is_last {
                None
            } else {
                Some(self.current_block_data.new_block())
            };

            self.emit_catch_arm_test(
                next_check_bb,
                tag_val,
                arm_tag,
                arm_body_bb,
                fallthrough_bb.unwrap_or(arm_body_bb),
            );
            if let Some(next) = fallthrough_bb {
                next_check_bb = next;
            }

            self.current_block_data.switch_to(arm_body_bb);
            if let Some(b) = binding {
                let payload = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: payload,
                    base: Operand::Value(raw_val),
                    offset: payload_offset,
                });
                self.var_map.insert(*b, payload);
            }
            self.lower_catch_arm_body(error_type, *body);
        }
    }

    /// Looks up the numeric tag of the error type a `catch` handler names, in the `__throws`
    /// enum's variant-tag table.
    ///
    /// Panics (these are compiler-internal invariants, not user errors) if the handler's type is
    /// not a struct/enum, or is not a member of the `__throws` table.
    fn catch_arm_tag(&self, error_type: &HirType) -> i64 {
        let throws_enum = StrId::from_static("__throws");
        let tags = self
            .enum_variant_tags
            .get(&throws_enum)
            .expect("__throws enum missing from enum_variant_tags");

        let error_name = match error_type {
            HirType::Struct { name, .. } | HirType::Enum { name, .. } => *name,
            other => panic!("catch branch type {:?} is not a nominal error type", other),
        };
        let tag = *tags.get(&error_name).unwrap_or_else(|| {
            panic!(
                "catch branch handles `{:?}`, which is not in the __throws table",
                error_name
            )
        });
        tag as i64
    }

    /// In `check_bb`, emits `tag == arm_tag` and branches to `match_bb` (equal) or `else_bb` (not equal).
    fn emit_catch_arm_test(
        &mut self,
        check_bb: BlockId,
        tag_val: Value,
        arm_tag: i64,
        match_bb: BlockId,
        else_bb: BlockId,
    ) {
        self.current_block_data.switch_to(check_bb);
        let cond = self.emit_eq_const(tag_val, arm_tag);
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb: match_bb,
            else_bb,
        });
    }

    fn lower_catch_arm_body(&mut self, error_type: &HirType, body: &'bump [HirStmt<'a, 'bump>]) {
        let saved = self.drop_state.clone();
        self.lower_stmt_seq(body);
        if !body.last().map_or(false, Self::stmt_diverges) {
            panic!(
                "`catch` arm for `{:?}` must end in return, throw, break, or continue",
                error_type
            );
        }
        self.drop_state = saved;
    }

    /// Brings a value that is *nullable-by-pointer* into the canonical shape expected by the
    /// null-test and unwrap code. Returns `(value, type)`.
    ///
    /// If `ty` is `*?T` (a pointer to a nullable), the "value" we want is the nullable itself:
    /// * **Tagged nullable** (tag + payload aggregate): aggregates are already handled by
    ///   address, so the pointer *is* the nullable; return it unchanged with the inner type.
    /// * **Pointer-represented nullable** (null == address 0): the slot holds the pointer, so load
    ///   it out (`LoadField` offset 0) and return the loaded pointer with the inner type.
    ///
    /// Any other type passes through untouched.
    fn normalize_nullable_operand(&mut self, val: Value, ty: SsaType) -> (Value, SsaType) {
        if let SsaType::Pointer(_, inner) = &ty {
            if let SsaType::Nullable(_) = &**inner {
                let inner = (**inner).clone();
                if inner.is_tagged_nullable() {
                    // aggregates are represented by address: same thing
                    return (val, inner);
                }
                // pointer-repr nullable stored in a slot: load the pointer out
                let loaded = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: loaded,
                    base: Operand::Value(val),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(loaded, inner.clone());
                return (loaded, inner);
            }
        }
        (val, ty)
    }

    /// Emits `left == right` against an integer constant and returns the boolean result.
    /// Used for tag comparisons (`catch` dispatch, null tags) and `ptr == 0` checks.
    /// The result type is the target's boolean type, obtained through `lower_type_hir`
    /// so it always matches what the rest of the pipeline treats as `bool`.
    pub(super) fn emit_eq_const(&mut self, left: Value, right: i64) -> Value {
        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest,
            op: BinOp::Eq,
            left: Operand::Value(left),
            right: Operand::ConstInt(right),
        });
        let bool_ty = lower_type_hir(&HirType::Boolean, self.enums, self.structs);
        self.current_block_data.value_types.insert(dest, bool_ty);
        dest
    }

    /// Emits a "is this value null?" test, true when null. The test depends on the representation:
    ///
    /// * **Pointer representation:** null is address 0, so `val == 0`.
    /// * **Tagged representation:** load the tag byte (offset 0) and compare it with the numeric tag
    ///   of the `null` variant of the internal `__nullable` enum (looked up, not hard-coded).
    ///
    /// Panics for non-nullable types: `else` on a non-nullable value is a type-check error that
    /// should never reach here.
    fn emit_null_test(&mut self, val: Value, ty: &SsaType, span: SourceSpan<'a>) -> Value {
        if ty.nullable_pointer_repr().is_some() {
            return self.emit_eq_const(val, 0);
        }
        if let SsaType::Nullable(_) = ty {
            let null_tag = *self
                .enum_variant_tags
                .get(&StrId::from_static("__nullable"))
                .and_then(|m| m.get(&StrId::from_static("null")))
                .expect("__nullable enum's `null` tag missing from enum_variant_tags");
            let tag = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: tag,
                base: Operand::Value(val),
                offset: 0,
            });
            self.current_block_data.value_types.insert(tag, SsaType::U8);
            return self.emit_eq_const(tag, null_tag as i64);
        }
        panic!("`else` used on non-nullable SsaType {:?} at {span}", ty);
    }

    /// Lowers `value else { else_body }`: yield the unwrapped value if non-null, otherwise run
    /// `else_body` (which either produces a replacement value or diverges).
    ///
    /// ```text
    ///        is_null = test(value)
    ///        /                  \
    ///   none_bb (null)       some_bb (non-null)
    ///   else_body            unwrapped = payload of value
    ///        \                  /
    ///         merge_bb: result = phi [(some_end, unwrapped), (else_end, else_val)]
    /// ```
    ///
    /// Details worth knowing:
    /// * `expected` is the type of the *unwrapped* value, pushed into `else_body`.
    /// * The operand is **moved** by this expression (`record_move_if_any`), and that is recorded
    ///   *before* branching, on both paths: when the value is null there is nothing to drop
    ///   anyway, and a single post-move state keeps the later join trivial.
    /// * If `else_body` diverges (typical: `x else { return err }`) there is only one path out; we
    ///   return the unwrapped value directly with **no merge block and no phi**.
    /// * Divergence is judged from the lowered CFG (`block_terminated`), not the syntax.
    /// * After the merge, variables declared inside `else_body` are dropped from `var_map`
    ///   (`retain`): they are out of scope and exist only on one incoming path.
    pub(super) fn lower_or_else(
        &mut self,
        value: &HirExpr<'a, 'bump>,
        else_body: &[HirStmt<'a, 'bump>],
        span: SourceSpan<'a>,
        expected: Option<&SsaType>, // the *unwrapped* type, if known
    ) -> Value {
        let raw = self.lower_expr(value);
        let raw_ty = self
            .value_type(raw)
            .cloned()
            .unwrap_or_else(|| panic!("`else` operand at {span} has no known SsaType"));
        let (val, ty) = self.normalize_nullable_operand(raw, raw_ty);

        // `x else { .. }` consumes `x`. Do it on both arms: when `x` is null there is
        // nothing to drop, and a single state keeps the join trivial.
        self.record_move_if_any(value);

        let drop_after_move = self.drop_state.clone();
        let vars_before = self.var_map.clone();
        let narrowed_before = self.narrowed_fields.clone();

        let is_null = self.emit_null_test(val, &ty, span);
        let some_bb = self.current_block_data.new_block();
        let none_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.fresh_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(is_null),
            then_bb: none_bb,
            else_bb: some_bb,
        });

        let (else_val, else_edge) =
            self.lower_or_else_none_arm(none_bb, merge_bb, else_body, expected, span);

        // The non-null path starts from the state right after the move, not from the else body's.
        self.current_block_data.switch_to(some_bb);
        self.var_map = vars_before.clone();
        self.drop_state = drop_after_move;
        self.narrowed_fields = narrowed_before.clone();
        let unwrapped = self.unwrap_known_nonnull(val, &ty);

        let Some(else_edge) = else_edge else {
            return unwrapped; // else diverged: one path, no merge needed
        };

        let some_end = self.current_block_data.current_block;
        let some_edge = (some_end, self.var_map.clone(), self.drop_state.clone());
        self.emit(Instruction::Jump { target: merge_bb });

        let else_end = else_edge.0;
        self.narrowed_fields = narrowed_before;
        self.merge_or_else_paths(merge_bb, some_edge, else_edge, &vars_before);

        let mut incoming: SmallVec<(BlockId, Value), 4> = SmallVec::new();
        incoming.push((some_end, unwrapped));
        incoming.push((else_end, else_val));
        self.emit_merge_phi(incoming, span)
    }

    /// Lowers the "value was null" arm of an `else`-unwrap in `none_bb`.
    ///
    /// Returns the arm's value and, if the arm falls through, its [`ArmEdge`]
    /// `(end block, variables, drop state)`; `None` if it diverged.
    ///
    /// Runs in its own `DropScope`. Scope drops are emitted only when the arm did not already terminate;
    /// `end` is read after those drops because they can open blocks.
    fn lower_or_else_none_arm(
        &mut self,
        none_bb: BlockId,
        merge_bb: BlockId,
        else_body: &[HirStmt<'a, 'bump>],
        expected: Option<&SsaType>,
        span: SourceSpan<'a>,
    ) -> (Value, Option<ArmEdge<'a, 'bump>>) {
        self.current_block_data.switch_to(none_bb);
        self.scope_stack.push(DropScope::default());
        let else_val = self.lower_block_value_inner(else_body, expected);
        let else_scope = self.scope_stack.pop().unwrap();

        // Divergence is a property of the lowered CFG, not of the last statement's syntax.
        let edge = if self.block_terminated() {
            None
        } else {
            self.emit_scope_drops(&else_scope, span);
            // emit_scope_drops may have opened blocks: read the tail afterwards
            let end = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            let drops = self.drop_state.clone();
            self.emit(Instruction::Jump { target: merge_bb });
            Some((end, vars, drops))
        };
        (else_val, edge)
    }

    /// Opens the merge block of an `else`-unwrap and merges variable maps and drop states
    /// from its two incoming paths (non-null and null).
    ///
    /// Variables not present before the expression (declared inside `else_body`) are dropped from
    /// `var_map` afterwards, since they are out of scope at the merge.
    fn merge_or_else_paths(
        &mut self,
        merge_bb: BlockId,
        some_edge: ArmEdge<'a, 'bump>,
        else_edge: ArmEdge<'a, 'bump>,
        vars_before: &VarSnapshot,
    ) {
        let (some_end, some_vars, some_drops) = some_edge;
        let (else_end, else_vars, else_drops) = else_edge;

        self.current_block_data.push_block(merge_bb);
        self.current_block_data.switch_to(merge_bb);
        self.merge_var_maps(vec![(some_end, some_vars), (else_end, else_vars)]);
        self.var_map.retain(|k, _| vars_before.contains_key(k)); // drop else-body locals
        if let Some(j) = DropMoveState::join_all(vec![some_drops, else_drops]) {
            self.drop_state = j;
        }
    }

    /// Lowers `continue`: jump to the innermost loop's continue target.
    ///
    /// Order matters:
    /// 1. Capture the variable values *now* (they are what the target's phis must receive).
    /// 2. Emit drops for every scope between here and the loop ([`Self::emit_drops_for_loop_exit`]);
    ///    these run on the way out, before the jump.
    /// 3. Feed the target block's placeholder phis with an edge from the current block.
    /// 4. Emit the `Jump`.
    pub(super) fn handle_continue_stmt(&mut self, span: SourceSpan<'a>) {
        let ctx = self
            .loop_stack
            .last()
            .expect("`continue` outside of a loop (should have been caught by the typechecker)");
        let continue_target = ctx.continue_target;
        let join_bb = ctx.continue_join_bb;
        let phis = ctx.continue_join_phis.clone();
        let depth = ctx.scope_depth_at_entry;
        let vars = self.var_map.clone();
        self.emit_drops_for_loop_exit(depth, span);
        let from_bb = self.current_block_data.current_block;
        self.contribute_join_edge(join_bb, from_bb, &vars, &phis);
        self.emit(Instruction::Jump {
            target: continue_target,
        });
    }

    /// Lowers `break [expr]`: leave the innermost loop.
    ///
    /// Like `continue`, but additionally:
    /// * a `break` operand is *moved* out (`record_move_if_any`), recorded first so the drops
    ///   emitted next do not try to drop it;
    /// * the post-drop `drop_state` is saved in the loop context's `break_states`, because the
    ///   state after the loop is the join of all the ways of leaving it
    ///   ([`Self::finish_loop`]).
    pub(super) fn handle_break_stmt(
        &mut self,
        expr: Option<&HirExpr<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        if let Some(e) = expr {
            self.record_move_if_any(e);
        }
        let ctx = self
            .loop_stack
            .last()
            .expect("`break` outside of a loop (should have been caught by the typechecker)");
        let break_target = ctx.break_target;
        let phis = ctx.break_join_phis.clone();
        let depth = ctx.scope_depth_at_entry;
        let vars = self.var_map.clone();
        self.emit_drops_for_loop_exit(depth, span);
        let st = self.drop_state.clone();
        self.loop_stack.last_mut().unwrap().break_states.push(st);
        let from_bb = self.current_block_data.current_block;
        self.contribute_join_edge(break_target, from_bb, &vars, &phis);
        self.emit(Instruction::Jump {
            target: break_target,
        });
    }

    /// Lowers `return [expr]`.
    ///
    /// 1. Record that the returned expression is moved (so scope-exit drops skip it).
    /// 2. Lower the returned value ([`Self::lower_return_value`]) *before* emitting drops, since it
    ///    may read locals that are about to be dropped.
    /// 3. Emit drops for every live scope ([`Self::emit_drops_for_return`]).
    /// 4. Emit `Ret`.
    pub(super) fn handle_return_stmt(
        &mut self,
        expr: Option<&HirExpr<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        if let Some(e) = expr {
            self.record_move_if_any(e);
        }
        let value = expr.map(|e| Operand::Value(self.lower_return_value(e)));
        self.emit_drops_for_return(span);
        self.emit(Instruction::Ret { value });
    }

    /// Lowers the expression of a `return` against the function's declared return type.
    ///
    /// * A bare `null` is lowered *as the return type* (`lower_null_as`); a `null` has no type of
    ///   its own, and its representation depends on what is being returned (pointer 0 or a tagged
    ///   nullable).
    /// * With a declared return type, the expression is lowered with that as its *expected* type
    ///   (so literals pick the right width), then wrapped into the tagged-nullable shape if the
    ///   return type is nullable and the value is not yet one (`coerce_into_tagged_nullable`).
    /// * With no declared return type, the expression is lowered normally.
    fn lower_return_value(&mut self, e: &HirExpr<'a, 'bump>) -> Value {
        match e {
            HirExpr::Null(_) => self.lower_null_as(self.return_type),
            _ => match self.return_type {
                Some(ref rt) => {
                    let expected = lower_type_hir(rt, self.enums, self.structs);
                    let v = self.lower_expr_expected(e, &expected);
                    self.coerce_into_tagged_nullable(v, &expected)
                }
                None => self.lower_expr(e),
            },
        }
    }

    /// Chooses the single type a phi over `incoming` should have, or panics if the edges
    /// genuinely disagree.
    ///
    /// Rules, in order:
    /// 1. **`null` adopts its neighbour's type.** `if c { p } else { null }` has edge types
    ///    `*T` and `Null`; the phi should be `*T`. Applied only if the real type can represent
    ///    null ([`Self::null_compatible_with`]).
    /// 2. **`Void` edges are ignored.** A diverging arm (e.g. a call to a panic function)
    ///    has no value and types as `Void`; it must neither decide nor conflict with the type.
    /// 3. **Everything else must match exactly**, else a "phi type mismatch" panic reports
    ///    which edge differs.
    ///
    /// The result is the first non-`Void` type, or `Void` if all edges are `Void`.
    pub(super) fn reconcile_phi_type(
        &self,
        incoming: &[(BlockId, Value)],
        span: SourceSpan<'a>,
    ) -> SsaType {
        let mut types = self.phi_incoming_types(incoming);

        // Two passes with different notion of "real type". Both are kept from the original code;
        // see `promote_null_types`.
        Self::promote_null_types(&mut types, false);
        Self::promote_null_types(&mut types, true);

        let first = types
            .iter()
            .find(|t| **t != SsaType::Void)
            .cloned()
            .unwrap_or(SsaType::Void);

        Self::assert_phi_types_agree(incoming, &types, &first, span);
        first
    }

    /// The SSA type of each incoming phi operand, in order. Panics if an operand has no recorded
    /// type, which means some earlier lowering step forgot to record one.
    fn phi_incoming_types(&self, incoming: &[(BlockId, Value)]) -> Vec<SsaType> {
        incoming
            .iter()
            .map(|(_, v)| {
                self.current_block_data
                    .value_type(*v)
                    .unwrap_or_else(|| panic!("phi incoming value {:?} has no known type", v))
                    .clone()
            })
            .collect()
    }

    /// Rewrites every `Null` entry in `types` to the type of a "real" (non-null) entry, if that
    /// type can hold a null.
    ///
    /// The "real" type is the first entry that is not `Null`; with `ignore_void` it must additionally
    /// not be `Void`. The two modes differ only when a `Void` entry precedes the real type:
    /// the first mode would pick `Void` (and, as `Void` cannot hold null, do nothing), while the
    /// second skips past it to find the actual type. The function is called with both modes in
    /// sequence; the second mode subsumes the first in practice, but both calls are kept
    /// to preserve existing behaviour.
    fn promote_null_types(types: &mut [SsaType], ignore_void: bool) {
        let real = types
            .iter()
            .find(|t| {
                if ignore_void {
                    !matches!(t, SsaType::Null | SsaType::Void)
                } else {
                    **t != SsaType::Null
                }
            })
            .cloned();

        if let Some(real_ty) = real {
            if Self::null_compatible_with(&real_ty) {
                for t in types.iter_mut() {
                    if *t == SsaType::Null {
                        *t = real_ty.clone();
                    }
                }
            }
        }
    }

    /// Panics unless every non-`Void` entry equals `first`. A `Void` entry is tolerated whenever
    /// the phi has a real type (`first != Void`); [`Self::patch_diverging_edges`] fills that
    /// edge in later.
    fn assert_phi_types_agree(
        incoming: &[(BlockId, Value)],
        types: &[SsaType],
        first: &SsaType,
        span: SourceSpan<'a>,
    ) {
        for (i, ((bb, v), t)) in incoming.iter().zip(types.iter()).enumerate() {
            if *t == SsaType::Void && *first != SsaType::Void {
                continue;
            }
            if t != first {
                panic!(
                    "phi type mismatch at {span}: incoming edge 0 has type {:?}, but edge {} \
                     (block {:?}, value {:?}) has type {:?}",
                    first, i, bb, v, t
                );
            }
        }
    }

    /// Gives every `Void`-typed phi operand a dummy value of the phi's real type `ty`.
    ///
    /// Why this exists: an arm can end in a diverging call and still fall through to the merge
    /// block syntactically, so it contributes an operand, but that operand is `Void`
    /// and a phi operand must have the phi's type. Control never actually flows along that edge at run
    /// time, so any value works; we just need a *well-typed* one.
    ///
    /// For each such edge the dummy is built by [`Self::placeholder_for_diverged_edge`] and inserted
    /// into that predecessor block just before its terminating `Jump`.
    pub(super) fn patch_diverging_edges(
        &mut self,
        incoming: &mut [(BlockId, Value)],
        ty: &SsaType,
    ) {
        if *ty == SsaType::Void {
            return;
        }
        for (bb, v) in incoming.iter_mut() {
            if self.current_block_data.value_type(*v) != Some(&SsaType::Void) {
                continue;
            }
            let dummy = self.current_block_data.fresh_value();
            let instr = Self::placeholder_for_diverged_edge(dummy, ty);
            self.insert_before_terminator(*bb, instr);
            self.current_block_data
                .value_types
                .insert(dummy, ty.clone());
            *v = dummy;
        }
    }

    /// Builds the dummy instruction defining `dest` for a dead phi edge of type `ty`.
    ///
    /// * **Aggregates** (structs, tagged enums, ...) are passed *by address*, so the dummy is a null
    ///   pointer to `ty` (the same shape the backend already accepts for null pointers).
    /// * **Scalars** use `Undef`, which says "any bit pattern; never read".
    fn placeholder_for_diverged_edge(dest: Value, ty: &SsaType) -> Instruction {
        if Self::is_aggregate_ssa_type(ty) {
            Instruction::Const {
                dest,
                ty: SsaType::Pointer(ir::ssa_ir::SsaPointerKind::UnsafeMut, Box::new(ty.clone())),
                value: Operand::ConstInt(0),
            }
        } else {
            Instruction::Undef {
                dest,
                ty: ty.clone(),
            }
        }
    }

    /// Inserts `instr` into block `bb` immediately before its last instruction (its terminator).
    /// It assumes the block is already terminated; with an empty block it inserts at index 0.
    fn insert_before_terminator(&mut self, bb: BlockId, instr: Instruction) {
        let block = self
            .current_block_data
            .func
            .blocks
            .iter_mut()
            .find(|b| b.id == bb)
            .expect("phi predecessor block missing");
        let at = block.instructions.len().saturating_sub(1); // before the arm's Jump
        block.instructions.insert(at, instr);
    }

    /// Lowers a block of statements whose *last expression is its value*, with no type hint.
    /// Wrapper over [`Self::lower_block_value_inner`].
    pub(super) fn lower_block_value(&mut self, stmts: &[HirStmt<'a, 'bump>]) -> Value {
        self.lower_block_value_inner(stmts, None)
    }

    /// Lowers `stmts` as a block that produces a value (the language is expression-oriented:
    /// `if` arms, `else` bodies and plain blocks all yield their final expression).
    ///
    /// * Empty block: the unit value.
    /// * All statements but the last are lowered for effect. If one of them terminates the
    ///   block (`return`, ...), the rest is dead code: stop and return a dead `Void` value.
    ///   `lower_stmt_inner` is also told the *remaining* statements, so it can look ahead (for
    ///   example to decide whether a variable is used later).
    /// * The last statement is lowered by [`Self::lower_trailing_stmt_value`], which decides whether
    ///   it yields a value.
    ///
    /// `expected` is the type the block's value should have, if the context knows it.
    pub(super) fn lower_block_value_inner(
        &mut self,
        stmts: &[HirStmt<'a, 'bump>],
        expected: Option<&SsaType>,
    ) -> Value {
        if stmts.is_empty() {
            return self.unit_value();
        }
        let (last, rest) = stmts.split_last().unwrap();
        for i in 0..rest.len() {
            if self.block_terminated() {
                return self.unreachable_value();
            }
            self.lower_stmt_inner(&stmts[i], Some(&stmts[i + 1..]));
        }
        self.lower_trailing_stmt_value(last, expected)
    }

    fn lower_trailing_stmt_value(
        &mut self,
        last: &HirStmt<'a, 'bump>,
        expected: Option<&SsaType>,
    ) -> Value {
        match (last, expected) {
            (HirStmt::Expr(e), Some(exp)) => self.lower_expr_expected(e, exp),
            (HirStmt::Expr(e), None) => self.lower_expr(e),

            (HirStmt::Match { expr, arms, span }, _) => match expected {
                Some(exp) => self.lower_match_expr_inner(expr, arms, *span, Some(exp)),
                None => {
                    let match_expr = HirExpr::Match {
                        expr,
                        arms,
                        span: Default::default(),
                    };
                    self.lower_expr(&match_expr)
                }
            },

            (HirStmt::UnsafeBlock { body }, _) => {
                self.unsafe_depth += 1;
                let v = match body {
                    HirStmt::Block { body, .. } => self.lower_block_value_inner(body, expected),
                    other => self.lower_stmt_then_unit(other),
                };
                self.unsafe_depth -= 1;
                v
            }

            (
                HirStmt::If {
                    cond,
                    then_block,
                    else_block: else_block @ Some(_),
                    span,
                },
                _,
            ) => self.lower_if_expr_inner(cond, then_block, *else_block, *span, expected),

            (
                HirStmt::If {
                    else_block: None, ..
                },
                _,
            ) => self.lower_stmt_then_unit(last),

            (HirStmt::Block { body, span: _ }, _) => self.lower_block_value_inner(body, expected),

            (other, _) => self.lower_stmt_then_unit(other),
        }
    }

    /// Lowers `stmt` for its effects only and returns a value for it: an unreachable `Void` if the
    /// statement ended the block (it diverged), otherwise a unit value.
    fn lower_stmt_then_unit(&mut self, stmt: &HirStmt<'a, 'bump>) -> Value {
        self.lower_stmt(stmt);
        if self.block_terminated() {
            self.unreachable_value()
        } else {
            self.unit_value()
        }
    }

    /// True if the current block already ends in a terminator (`Jump`, `Branch`, `Ret`, ...).
    ///
    /// Nothing may be emitted after a terminator, so nearly every lowering routine checks this before
    /// adding drops, jumps, or placeholder values. It is the CFG-level answer to "did this code
    /// diverge?".
    pub(super) fn block_terminated(&mut self) -> bool {
        self.current_block_data
            .bb()
            .instructions
            .last()
            .map_or(false, |i| ir::ssa_ir::inst_is_terminator(i))
    }

    /// Returns a fresh `Void`-typed value standing for "no meaningful value" (a statement used where an
    /// expression is expected, or an `if` with no result).
    ///
    /// An `Undef` of type `Void` is emitted so the value has a defining instruction, but only when
    /// the block is still open; after a terminator nothing can be emitted, and the value is simply
    /// recorded as `Void` (it is dead anyway).
    pub(super) fn unit_value(&mut self) -> Value {
        let v = self.current_block_data.fresh_value();
        if !self.block_terminated() {
            self.emit(Instruction::Undef {
                dest: v,
                ty: SsaType::Void,
            });
        }
        self.current_block_data.value_types.insert(v, SsaType::Void);
        v
    }

    // TODO: should we completely remove this?
    pub(super) fn unreachable_value(&mut self) -> Value {
        self.unit_value()
    }
}
