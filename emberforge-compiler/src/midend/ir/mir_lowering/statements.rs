use ir::{
    hir::{DropKind, HirErrorHandlerPattern, HirExpr, HirStmt, HirType, StrId},
    ir_conversion::lower_type_hir,
    span::SourceSpan,
};

use crate::midend::{
    copy_analysis::drop_tracking::{DropLocal, DropScope},
    ir::mir_lowering::FunctionLowerer,
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn handle_block_stmt(&mut self, body: &[HirStmt<'a, 'bump>], span: SourceSpan<'a>) {
        self.scope_stack.push(DropScope { locals: Vec::new() });
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
        else_block: &Option<&HirStmt<'a, 'bump>>,
    ) {
        self.record_move_if_any(value);
        let expected_ssa = lower_type_hir(ty, self.enums);
        let mut val = self.lower_expr_expected(value, &expected_ssa);

        if let Some(pat) = catch_pattern {
            self.lower_catch(val, pat);
        }
        if let Some(else_stmts) = else_block {
            val = self.lower_nullable_unwrap(val, else_stmts);
        }

        self.var_map.insert(name.clone(), val);
        self.drop_state.reset_local(*name);
        self.array_flags.remove(name);

        if let HirType::Struct {
            name: struct_name, ..
        } = ty
        {
            if self.glue_registry.is_droppable(*struct_name) || self.struct_owns_chain(*struct_name)
            {
                self.scope_stack.last_mut().unwrap().locals.push(DropLocal {
                    name: *name,
                    kind: DropKind::Type(*struct_name),
                });
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
                                the allocator can be recovered from",
                        name
                    )
                })
            };

            self.scope_stack.last_mut().unwrap().locals.push(DropLocal {
                name: *name,
                kind: drop_kind,
            });
        } else if let Some((kind, owned_ty)) = self.nullable_owned_drop_kind(ty, value) {
            self.scope_stack
                .last_mut()
                .unwrap()
                .locals
                .push(DropLocal { name: *name, kind });
            self.nullable_owned_locals.insert(*name, owned_ty);
            self.drop_state.mark_whole_initialized(*name);
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

        self.register_array_drop_local(
            *name,
            &expected_ssa,
            !matches!(value, HirExpr::Uninit { .. }),
            rest,
        );
    }
}
