use ir::{
    hir::{DropKind, HirErrorHandlerPattern, HirExpr, HirStmt, HirType, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};

use crate::midend::{
    copy_analysis::drop_tracking::{DropLocal, DropScope, ScopeAction},
    ir::mir_lowering::FunctionLowerer,
};

impl<'f, 's, 'a, 'bump, 'r> FunctionLowerer<'f, 's, 'a, 'bump, 'r> {
    pub(super) fn handle_block_stmt(&mut self, body: &[HirStmt<'a, 'bump>], span: SourceSpan<'a>) {
        self.scope_stack.push(DropScope::default());
        self.lower_stmt_seq(body);
        let scope = self.scope_stack.pop().unwrap();
        if !self.block_terminated() {
            self.emit_scope_drops(&scope, span);
        }
    }

    pub(super) fn handle_let_stmt(
        &mut self,
        rest: Option<&[HirStmt<'a, 'bump>]>,
        name: &StrId,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        catch_pattern: &Option<HirErrorHandlerPattern<'a, 'bump>>,
        span: SourceSpan<'a>,
        manual: bool,
    ) {
        self.record_move_if_any(value);
        let expected_ssa = lower_type_hir(ty, self.enums, self.structs);
        let mut val = self.lower_expr_expected(value, &expected_ssa);
        val = self.copy_if_place_alias(value, val, &expected_ssa);

        if let Some(pat) = catch_pattern {
            self.lower_catch(val, pat);
        }

        self.var_map.insert(name.clone(), val);
        self.drop_state.reset_local(*name);
        self.array_flags.remove(name);

        if !manual {
            self.add_to_scope_stack_for_drops(name, ty, value, span);
        }

        if matches!(value, HirExpr::Uninit { .. }) {
            self.drop_state.mark_whole_uninit(*name);
            if let HirType::Struct {
                name: struct_name, ..
            } = ty
            {
                if let Some(hir_struct) = self.structs.get(struct_name) {
                    for f in hir_struct.fields.iter() {
                        self.drop_state.mark_field_uninit(*name, f.name);
                    }
                }
            }
        } else if let HirExpr::StructInit { args, .. } = value {
            for fi in args.iter() {
                if matches!(fi.value, HirExpr::Uninit { .. }) {
                    self.drop_state.mark_field_uninit(*name, fi.name);
                }
            }
        }

        if !manual {
            self.register_array_drop_local(
                *name,
                &expected_ssa,
                !matches!(value, HirExpr::Uninit { .. }),
                rest,
            );
        }
    }

    fn copy_if_place_alias(
        &mut self,
        value: &HirExpr<'a, 'bump>,
        val: Value,
        expected: &SsaType,
    ) -> Value {
        if !matches!(
            value,
            HirExpr::FieldAccess { .. }
                | HirExpr::Get { .. }
                | HirExpr::Index { .. }
                | HirExpr::Deref { .. }
        ) {
            return val;
        }
        let inline_agg = match expected {
            SsaType::User(..) | SsaType::Enum { .. } | SsaType::Tuple(_) => true,
            t @ SsaType::Nullable(_) => t.is_tagged_nullable(),
            _ => false,
        };
        if !inline_agg {
            return val;
        }
        // field reads of inline aggregates come back as `Pointer(expected)`, or as `expected` itself
        let aliases = match self.value_type(val) {
            Some(t) if t == expected => true,
            Some(SsaType::Pointer(_, inner)) => inner.as_ref() == expected,
            _ => false,
        };
        let size = ir::layout::sizeof_ssa(expected, TargetInfo { ptr_bytes: 8 }).unwrap_or(0);
        if !aliases || size == 0 {
            return val;
        }
        let tmp = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: tmp,
            ty: expected.clone(),
            count: 1,
        });
        self.current_block_data
            .value_types
            .insert(tmp, expected.clone());
        let n = self.new_value();
        self.emit(Instruction::Const {
            dest: n,
            ty: SsaType::Usize,
            value: Operand::ConstInt(size as i64),
        });
        self.current_block_data
            .value_types
            .insert(n, SsaType::Usize);
        self.emit_memcpy(tmp, val, n);
        tmp
    }

    fn add_to_scope_stack_for_drops(
        &mut self,
        name: &StrId,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        if let HirType::Struct {
            name: struct_name, ..
        } = ty
        {
            if self.glue_registry.is_droppable(*struct_name) || self.struct_owns_chain(*struct_name)
            {
                self.scope_stack
                    .last_mut()
                    .unwrap()
                    .actions
                    .push(ScopeAction::DropLocal(DropLocal {
                        name: *name,
                        kind: DropKind::Type(*struct_name),
                    }));
            }
        } else if let HirType::OwnedPointer { allocator, .. } = ty {
            let drop_kind = if allocator.is_some() {
                ty.drop_kind()
            } else {
                let inferred = self.recover_owned_pointer_drop_kind(ty, value);
                inferred.unwrap_or_else(|| {
                    panic!(
                        "owned-pointer local `{}` has no allocator annotation on its \
                                declared type, and its initializer isn't a `$own(..)` call \
                                the allocator can be recovered from at {span}",
                        name
                    )
                })
            };

            self.scope_stack
                .last_mut()
                .unwrap()
                .actions
                .push(ScopeAction::DropLocal(DropLocal {
                    name: *name,
                    kind: drop_kind,
                }));
        } else if let Some((kind, owned_ty)) = self.nullable_owned_drop_kind(ty, value) {
            self.scope_stack
                .last_mut()
                .unwrap()
                .actions
                .push(ScopeAction::DropLocal(DropLocal { name: *name, kind }));
            self.nullable_owned_locals.insert(*name, owned_ty);
            self.drop_state.mark_whole_initialized(*name);
        }
    }
}
