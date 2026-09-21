use ir::{
    hir::{HirErrorHandlerPattern, HirExpr, HirStmt, HirType, StrId},
    ir_conversion::lower_type_hir,
    ir_hasher::HashMap,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, BlockId, Instruction, Operand, SsaType, Value},
};
use smallvec::SmallVec;

use crate::midend::{
    copy_analysis::drop_tracking::{DropMoveState, DropScope},
    ir::mir_lowering::FunctionLowerer,
};

/// Bookkeeping for one enclosing loop, needed so `break`/`continue` (which
/// can appear arbitrarily deep inside nested control flow) can contribute a
/// phi edge to the right join point. `phis` are `(name, instruction-index,
/// phi-dest-value)` triples identifying placeholder `Instruction::Phi`s
/// created by `open_join`, still waiting for more incoming edges.
pub struct LoopCtx<'a, 'bump> {
    /// Where `continue` jumps to (the loop header for `while` or the
    /// increment block for `for`).
    pub(super) continue_target: BlockId,
    /// The block whose phis a `continue` must contribute an edge to (same
    /// as `continue_target`).
    pub(super) continue_join_bb: BlockId,
    pub(super) continue_join_phis: Vec<(StrId, usize, Value)>,
    /// Where `break` jumps to (the block right after the loop).
    pub(super) break_target: BlockId,
    /// The block (== `break_target`) whose phis a `break` must contribute
    /// an edge to.
    pub(super) break_join_phis: Vec<(StrId, usize, Value)>,
    pub(super) scope_depth_at_entry: usize,
    pub(super) break_states: Vec<DropMoveState<'a, 'bump>>,
}

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn lower_if_expr(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_if_expr_inner(condition, then_block, else_block, span, None)
    }

    pub(super) fn lower_if_expr_inner(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
        expected: Option<&SsaType>,
    ) -> Value {
        let drop_before = self.drop_state.clone();
        let mut live_drop: Vec<DropMoveState<'a, 'bump>> = Vec::new();
        let cond = self.lower_expr(condition);
        let narrowed_before = self.narrowed_fields.clone();

        let then_bb = self.current_block_data.new_block();
        let else_bb = self.current_block_data.new_block();
        let merge_bb = self.current_block_data.new_block();

        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb,
            else_bb,
        });

        // then
        self.current_block_data.switch_to(then_bb);
        self.narrowed_fields = narrowed_before.clone();
        self.scope_stack.push(DropScope { locals: Vec::new() });
        let then_narrowed = self.narrow_nonnull(condition, true);
        let then_narrowed_path = self.narrow_nonnull_path(condition, true);
        let then_val = self.lower_block_value_inner(then_block, expected);
        let then_scope = self.scope_stack.pop().unwrap();
        if let Some((name, prev, narrowed_val)) = then_narrowed {
            if self.var_map.get(&name).copied() == Some(narrowed_val) {
                self.var_map.insert(name, prev);
            }
        }
        if let Some((root, path)) = then_narrowed_path {
            self.narrowed_fields.remove(&(root, path));
        }
        let then_terminated = self.block_terminated();
        if !then_terminated {
            self.emit_scope_drops(&then_scope, span);
            live_drop.push(self.drop_state.clone());
        }
        let then_end = self.current_block_data.current_block;
        if !then_terminated {
            self.emit(Instruction::Jump { target: merge_bb });
        }

        // else
        self.drop_state = drop_before.clone();
        self.current_block_data.switch_to(else_bb);
        self.narrowed_fields = narrowed_before.clone();
        let else_narrowed = self.narrow_nonnull(condition, false);
        let else_narrowed_path = self.narrow_nonnull_path(condition, false);
        let else_val = match else_block {
            Some(HirStmt::Block { body, span: _ }) => {
                self.scope_stack.push(DropScope { locals: Vec::new() });
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
                span,
            }) => self.lower_if_expr(ec, etb, *eeb, *span),
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
        if let Some((name, prev, narrowed_val)) = else_narrowed {
            if self.var_map.get(&name).copied() == Some(narrowed_val) {
                self.var_map.insert(name, prev);
            }
        }
        if let Some((root, path)) = else_narrowed_path {
            self.narrowed_fields.remove(&(root, path));
        }
        let else_terminated = self.block_terminated();
        if !else_terminated {
            live_drop.push(self.drop_state.clone());
        }
        let else_end = self.current_block_data.current_block;
        if !else_terminated {
            self.emit(Instruction::Jump { target: merge_bb });
        }

        if then_terminated && else_terminated {
            return self.unreachable_value();
        }

        self.narrowed_fields = narrowed_before;
        self.current_block_data.switch_to(merge_bb);
        if let Some(j) = DropMoveState::join_all(live_drop) {
            self.drop_state = j;
        }

        let ty = self
            .value_type(if !then_terminated { then_val } else { else_val })
            .cloned()
            .unwrap_or(SsaType::Void);

        let result = if ty == SsaType::Void {
            self.unit_value()
        } else {
            let result = self.current_block_data.fresh_value();
            let mut incoming = SmallVec::new();
            if !then_terminated {
                incoming.push((then_end, then_val));
            }
            if !else_terminated {
                incoming.push((else_end, else_val));
            }
            let ty = self.reconcile_phi_type(&incoming, span);
            self.emit(Instruction::Phi {
                dest: result,
                incoming,
            });
            self.current_block_data
                .value_types
                .insert(result, ty.clone());
            result
        };

        result
    }

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
        self.current_block_data.switch_to(cond_bb);

        self.narrowed_fields.clear();
        let header_phis = self.open_join();
        self.contribute_join_edge(cond_bb, pre_loop_bb, &vars_before, &header_phis);

        let header_drop = self.loop_header_state();
        self.drop_state = header_drop.clone();

        let cond_val = self.lower_expr(cond);
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond_val),
            then_bb: body_bb,
            else_bb: after_bb,
        });
        let header_vars = self.var_map.clone();

        self.current_block_data.switch_to(after_bb);
        let exit_phis = self.open_join();
        self.contribute_join_edge(after_bb, cond_bb, &header_vars, &exit_phis);

        self.loop_stack.push(LoopCtx {
            continue_target: cond_bb,
            continue_join_bb: cond_bb,
            continue_join_phis: header_phis.clone(),
            break_target: after_bb,
            break_join_phis: exit_phis.clone(),
            scope_depth_at_entry: self.scope_stack.len(),
            break_states: Vec::new(),
        });

        self.var_map = header_vars;
        self.current_block_data.switch_to(body_bb);
        self.narrowed_fields.clear();
        let loop_narrowed = self.narrow_nonnull(cond, true);
        self.narrow_nonnull_path(cond, true);
        self.lower_stmt(body);
        Self::revert_narrow_in_map(&mut self.var_map, loop_narrowed);
        self.narrowed_fields.clear();
        if !self.block_terminated() {
            let tail_bb = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.contribute_join_edge(cond_bb, tail_bb, &vars, &header_phis);
            self.emit(Instruction::Jump { target: cond_bb });
        }
        let ctx = self.loop_stack.pop().unwrap();

        self.current_block_data.switch_to(after_bb);
        self.narrowed_fields = narrowed_before;
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

        self.current_block_data.switch_to(cond_bb);
        self.narrowed_fields.clear();
        let header_phis = self.open_join();
        self.contribute_join_edge(cond_bb, pre_loop_bb, &vars_before, &header_phis);

        let header_drop = self.loop_header_state();
        self.drop_state = header_drop.clone();

        match condition {
            Some(cond_expr) => {
                let cond_val = self.lower_expr(cond_expr);
                self.emit(Instruction::Branch {
                    cond: Operand::Value(cond_val),
                    then_bb: body_bb,
                    else_bb: after_bb,
                });
            }
            None => {
                self.emit(Instruction::Jump { target: body_bb });
            }
        }
        let header_vars = self.var_map.clone();

        self.current_block_data.switch_to(after_bb);
        let exit_phis = self.open_join();
        self.contribute_join_edge(after_bb, cond_bb, &header_vars, &exit_phis);

        self.current_block_data.switch_to(incr_bb);
        let incr_phis = self.open_join();

        self.loop_stack.push(LoopCtx {
            continue_target: incr_bb,
            continue_join_bb: incr_bb,
            continue_join_phis: incr_phis.clone(),
            break_target: after_bb,
            break_join_phis: exit_phis.clone(),
            scope_depth_at_entry: self.scope_stack.len(),
            break_states: Vec::new(),
        });

        self.var_map = header_vars;
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
        if !self.block_terminated() {
            let tail_bb = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.contribute_join_edge(incr_bb, tail_bb, &vars, &incr_phis);
            self.emit(Instruction::Jump { target: incr_bb });
        }
        let ctx = self.loop_stack.pop().unwrap();

        self.var_map = incr_phis
            .iter()
            .map(|(name, _, dest)| (*name, *dest))
            .collect();
        self.current_block_data.switch_to(incr_bb);
        self.narrowed_fields.clear();
        self.drop_state = header_drop.clone();
        if let Some(inc_expr) = increment {
            let _ = self.lower_expr(inc_expr);
        }
        if !self.block_terminated() {
            let tail_bb = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.contribute_join_edge(cond_bb, tail_bb, &vars, &header_phis);
            self.emit(Instruction::Jump { target: cond_bb });
        }

        self.current_block_data.switch_to(after_bb);
        self.narrowed_fields = narrowed_before;
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

    pub(super) fn loop_header_state(&self) -> DropMoveState<'a, 'bump> {
        let mut s = self.drop_state.clone();
        for name in self.array_flags.keys() {
            s.havoc_indices(*name);
        }
        s
    }

    pub(super) fn open_join(&mut self) -> Vec<(StrId, usize, Value)> {
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

    pub(super) fn contribute_join_edge(
        &mut self,
        join_bb: BlockId,
        from_bb: BlockId,
        vars: &HashMap<StrId, Value>,
        phis: &[(StrId, usize, Value)],
    ) {
        if phis.is_empty() {
            return;
        }
        for (name, _idx, dest) in phis {
            if let Some(val) = vars.get(name).copied() {
                let phi_ty = self.current_block_data.value_type(*dest).cloned();
                let val_ty = self.current_block_data.value_type(val).cloned();
                if let (Some(pt), Some(vt)) = (&phi_ty, &val_ty) {
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
        }
        let block = self
            .current_block_data
            .func
            .blocks
            .iter_mut()
            .find(|b| b.id == join_bb)
            .expect("join block missing");
        for (name, idx, _dest) in phis {
            if let Some(val) = vars.get(name).copied() {
                if let Instruction::Phi { incoming, .. } = &mut block.instructions[*idx] {
                    incoming.push((from_bb, val));
                }
            }
        }
    }

    pub(super) fn merge_var_maps(&mut self, branches: Vec<(BlockId, HashMap<StrId, Value>)>) {
        match branches.len() {
            0 => {
                // every incoming branch diverged
            }
            1 => {
                self.var_map = branches.into_iter().next().unwrap().1;
            }
            _ => {
                let mut all_names: std::collections::HashSet<StrId> =
                    std::collections::HashSet::new();
                for (_, vars) in &branches {
                    all_names.extend(vars.keys().copied());
                }

                let mut merged = HashMap::default();
                for name in all_names {
                    let mut entries: Vec<(BlockId, Value)> = Vec::new();
                    for (bb, vars) in &branches {
                        if let Some(v) = vars.get(&name) {
                            entries.push((*bb, *v));
                        }
                    }
                    let Some((_, first_val)) = entries.first().copied() else {
                        continue;
                    };
                    if entries.iter().all(|(_, v)| *v == first_val) {
                        merged.insert(name, first_val);
                    } else {
                        let first_ty = self
                            .current_block_data
                            .value_type(first_val)
                            .unwrap()
                            .clone();

                        if first_ty == SsaType::Void {
                            merged.insert(name, first_val);
                            continue;
                        }

                        let dest = self.current_block_data.fresh_value();
                        self.current_block_data.value_types.insert(dest, first_ty);

                        self.emit(Instruction::Phi {
                            dest,
                            incoming: entries.into_iter().collect(),
                        });
                        merged.insert(name, dest);
                    }
                }
                self.var_map = merged;
            }
        }
    }

    pub(super) fn lower_if_stmt(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &'bump [HirStmt<'a, 'bump>],
        else_block: &Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        let drop_before = self.drop_state.clone();
        let mut live_drop: Vec<DropMoveState<'a, 'bump>> = Vec::new();
        let pre_if_bb = self.current_block_data.current_block;
        let vars_before = self.var_map.clone();
        let narrowed_before = self.narrowed_fields.clone();
        let cond_val = self.lower_expr(cond);

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
        self.current_block_data.switch_to(then_bb);
        self.scope_stack.push(DropScope { locals: Vec::new() });
        self.narrow_nonnull(cond, true);
        let then_narrowed = self.narrow_nonnull(cond, true);
        self.narrow_nonnull_path(cond, true);
        self.lower_stmt_seq(then_block);
        let then_scope = self.scope_stack.pop().unwrap();
        let then_terminated = self.block_terminated();
        let then_live = if then_terminated {
            None
        } else {
            self.emit_scope_drops(&then_scope, span);
            live_drop.push(self.drop_state.clone());
            Self::revert_narrow_in_map(&mut self.var_map, then_narrowed);
            let tail = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.emit(Instruction::Jump { target: merge_bb });
            Some((tail, vars))
        };

        let else_live = if needs_real_else_block {
            self.drop_state = drop_before.clone();
            self.var_map = vars_before.clone();
            self.narrowed_fields = narrowed_before.clone();
            self.current_block_data.switch_to(else_bb);
            self.narrow_nonnull(cond, false);
            self.narrow_nonnull_path(cond, false);

            if let Some(else_stmt) = else_block {
                self.scope_stack.push(DropScope { locals: Vec::new() });
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
                let tail = self.current_block_data.current_block;
                let vars = self.var_map.clone();
                self.emit(Instruction::Jump { target: merge_bb });
                Some((tail, vars))
            }
        } else {
            Some((pre_if_bb, vars_before))
        };

        self.narrowed_fields = narrowed_before;
        let live_branches: Vec<_> = [then_live, else_live].into_iter().flatten().collect();
        if !live_branches.is_empty() {
            self.current_block_data.push_block(merge_bb);
            self.current_block_data.switch_to(merge_bb);
            self.merge_var_maps(live_branches);
            if let Some(j) = DropMoveState::join_all(live_drop) {
                self.drop_state = j;
            }
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

        let throws_enum = StrId::from_static("__throws");
        let tags = self.enum_variant_tags.get(&throws_enum).unwrap();

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
            let error_name = match error_type {
                HirType::Struct { name, .. } | HirType::Enum { name, .. } => *name,
                other => panic!("catch branch type {:?} is not a nominal error type", other),
            };
            let arm_tag = *tags.get(&error_name).unwrap_or_else(|| {
                panic!(
                    "catch branch handles `{:?}`, which is not in the __throws table",
                    error_name
                )
            });

            let arm_body_bb = self.current_block_data.new_block();
            let is_last = i == branches.len() - 1;
            let fallthrough_bb = if is_last {
                None
            } else {
                Some(self.current_block_data.new_block())
            };

            self.current_block_data.switch_to(next_check_bb);
            let cond = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cond,
                op: BinOp::Eq,
                left: Operand::Value(tag_val),
                right: Operand::ConstInt(arm_tag as i64),
            });
            match fallthrough_bb {
                Some(next) => {
                    self.emit(Instruction::Branch {
                        cond: Operand::Value(cond),
                        then_bb: arm_body_bb,
                        else_bb: next,
                    });
                    next_check_bb = next;
                }
                None => {
                    self.emit(Instruction::Branch {
                        cond: Operand::Value(cond),
                        then_bb: arm_body_bb,
                        else_bb: arm_body_bb,
                    });
                }
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
            let saved = self.drop_state.clone();
            self.lower_stmt_seq(body);
            if !body.last().map_or(false, Self::stmt_diverges) {
                panic!(
                    "`catch` arm for `{:?}` must end in return, throw, break, or continue",
                    error_name
                );
            }
            self.drop_state = saved;
        }
    }

    pub(super) fn lower_nullable_unwrap(
        &mut self,
        val: Value,
        else_stmts: &HirStmt<'a, 'bump>,
    ) -> Value {
        let then_bb = self.current_block_data.new_block();
        let else_bb = self.current_block_data.new_block();

        let ty = self
            .current_block_data
            .value_type(val)
            .expect("nullable-unwrapped value must have a known SsaType")
            .clone();

        if ty.nullable_pointer_repr().is_some() {
            let cond = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cond,
                op: BinOp::Eq,
                left: Operand::Value(val),
                right: Operand::ConstInt(0),
            });
            self.emit(Instruction::Branch {
                cond: Operand::Value(cond),
                then_bb: else_bb,
                else_bb: then_bb,
            });
        } else if let SsaType::Nullable(_) = &ty {
            let nullable_enum = StrId::from_static("__nullable");
            let null_tag = *self
                .enum_variant_tags
                .get(&nullable_enum)
                .and_then(|m| m.get(&StrId::from_static("null")))
                .expect("__nullable enum's `null` tag missing from enum_variant_tags");

            let tag = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: tag,
                base: Operand::Value(val),
                offset: 0,
            });
            self.current_block_data.value_types.insert(tag, SsaType::U8);

            let cond = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: cond,
                op: BinOp::Eq,
                left: Operand::Value(tag),
                right: Operand::ConstInt(null_tag as i64),
            });
            self.emit(Instruction::Branch {
                cond: Operand::Value(cond),
                then_bb: else_bb,
                else_bb: then_bb,
            });
        } else {
            panic!("`? else` used on non-nullable SsaType {:?}", ty);
        }

        self.current_block_data.switch_to(else_bb);
        let HirStmt::Block { body, span: _ } = else_stmts else {
            unreachable!()
        };
        let saved = self.drop_state.clone();
        self.scope_stack.push(DropScope { locals: Vec::new() });
        self.lower_stmt_seq(body);
        self.scope_stack.pop();
        if !body.last().map_or(false, Self::stmt_diverges) {
            panic!("`stmt? else {{}}` block must end in return, throw, break, or continue");
        }
        self.drop_state = saved;

        self.current_block_data.switch_to(then_bb);
        self.unwrap_known_nonnull(val, &ty)
    }

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

    pub(super) fn handle_return_stmt(
        &mut self,
        expr: Option<&HirExpr<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        if let Some(e) = expr {
            self.record_move_if_any(e);
        }
        let value = expr.as_ref().map(|e| {
            let val = match e {
                HirExpr::Null(_) => self.lower_null_as(self.return_type),
                _ => match self.return_type {
                    Some(ref rt) => {
                        let expected = lower_type_hir(rt, self.enums);
                        let v = self.lower_expr_expected(e, &expected);
                        self.coerce_into_tagged_nullable(v, &expected)
                    }
                    None => self.lower_expr(e),
                },
            };
            Operand::Value(val)
        });
        self.emit_drops_for_return(span);
        self.emit(Instruction::Ret { value });
    }

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

    pub(super) fn reconcile_phi_type(
        &self,
        incoming: &[(BlockId, Value)],
        span: SourceSpan<'a>,
    ) -> SsaType {
        let mut types: Vec<SsaType> = incoming
            .iter()
            .map(|(_, v)| {
                self.current_block_data
                    .value_type(*v)
                    .unwrap_or_else(|| panic!("phi incoming value {:?} has no known type", v))
                    .clone()
            })
            .collect();

        if let Some(real_ty) = types.iter().find(|t| **t != SsaType::Null).cloned() {
            if Self::null_compatible_with(&real_ty) {
                for t in types.iter_mut() {
                    if *t == SsaType::Null {
                        *t = real_ty.clone();
                    }
                }
            }
        }

        let first = types[0].clone();
        for (i, ((bb, v), t)) in incoming.iter().zip(types.iter()).enumerate().skip(1) {
            if *t != first {
                panic!(
                    "phi type mismatch at {span}: incoming edge 0 has type {:?}, but edge {} \
                     (block {:?}, value {:?}) has type {:?}",
                    first, i, bb, v, t
                );
            }
        }
        first
    }

    pub(super) fn lower_block_value(&mut self, stmts: &[HirStmt<'a, 'bump>]) -> Value {
        self.lower_block_value_inner(stmts, None)
    }

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
            ) => {
                self.lower_stmt(last);
                if self.block_terminated() {
                    self.unreachable_value()
                } else {
                    self.unit_value()
                }
            }

            (HirStmt::Block { body, span: _ }, _) => self.lower_block_value_inner(body, expected),

            (other, _) => {
                self.lower_stmt(other);
                if self.block_terminated() {
                    self.unreachable_value()
                } else {
                    self.unit_value()
                }
            }
        }
    }

    pub(super) fn block_terminated(&mut self) -> bool {
        self.current_block_data
            .bb()
            .instructions
            .last()
            .map_or(false, |i| ir::ssa_ir::inst_is_terminator(i))
    }

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

    pub(super) fn unreachable_value(&mut self) -> Value {
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
}
