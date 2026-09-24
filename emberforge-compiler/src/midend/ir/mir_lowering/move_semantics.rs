use ir::{
    hir::{
        self, AssignmentOperator, DropKind, HirExpr, HirPattern, HirStmt, HirType, IntrinsicKind,
        ProvenanceAnnotation, StrId,
    },
    ir_conversion::{assign_op_to_bin_op, lower_type_hir},
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{BinOp, Instruction, Operand, SsaType, Value, cast_kind},
};
use smallvec::SmallVec;

use crate::midend::{
    copy_analysis::{
        drop_emitter::{DropEmitter, FnAllocatorResolver, is_struct_owns_chain},
        drop_tracking::{DropLocal, DropScope, Tri, record_move_if_any},
    },
    ir::mir_lowering::FunctionLowerer,
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn recover_owned_pointer_drop_kind(
        &self,
        declared_ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
    ) -> Option<DropKind<'a, 'bump>> {
        let HirType::OwnedPointer {
            inner,
            allocator: decl_alloc,
        } = declared_ty
        else {
            return None;
        };

        let allocator = if let Some(alloc) = decl_alloc {
            alloc.clone()
        } else if let Some(alloc) = self.infer_allocator_from_expr(value) {
            alloc
        } else {
            return None;
        };

        Some(DropKind::OwnedPointer {
            pointee: Box::new(inner.drop_kind()),
            pointee_ty: **inner,
            allocator,
        })
    }

    pub(super) fn infer_allocator_from_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<hir::ProvenanceAnnotation<'bump>> {
        match expr {
            HirExpr::Intrinsic {
                kind: IntrinsicKind::Own,
                args,
                ..
            } => {
                if args.len() == 3 || args.len() == 2 {
                    if let Some(prov) = self.infer_provenance(&args[1]) {
                        return Some(prov);
                    }
                }
                self.infer_provenance(&HirExpr::This {
                    span: Default::default(),
                })
            }
            HirExpr::Intrinsic {
                kind: IntrinsicKind::Replace,
                args,
                ..
            } => {
                let place = match &args[0] {
                    HirExpr::Ref { expr, .. } => &**expr,
                    other => other,
                };
                match self.hir_field_type_of_place(place)? {
                    HirType::Nullable(inner) => match *inner {
                        HirType::OwnedPointer { allocator, .. } => allocator,
                        _ => None,
                    },
                    HirType::OwnedPointer { allocator, .. } => allocator,
                    _ => None,
                }
            }

            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                if let HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } = callee {
                    if let Some(prov) = self.infer_provenance(object) {
                        return Some(prov);
                    }
                }

                if let Some(first_arg) = args.first() {
                    if let Some(prov) = self.infer_provenance(first_arg) {
                        return Some(prov);
                    }
                    if let Some(prov) = self.infer_allocator_from_expr(first_arg) {
                        return Some(prov);
                    }
                }
                None
            }

            HirExpr::Ident(name, _) => {
                for scope in self.scope_stack.iter().rev() {
                    for local in scope.locals.iter().rev() {
                        if local.name == *name {
                            if let DropKind::OwnedPointer { allocator, .. } = &local.kind {
                                return Some(allocator.clone());
                            }
                        }
                    }
                }
                None
            }

            HirExpr::Cast { expr: inner, .. }
            | HirExpr::Deref { expr: inner, .. }
            | HirExpr::Ref { expr: inner, .. } => self.infer_allocator_from_expr(inner),

            _ => None,
        }
    }

    pub(super) fn infer_provenance(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let mut segments = Vec::new();
        let root = self.infer_provenance_root(expr, &mut segments)?;
        segments.reverse();
        Some(ProvenanceAnnotation {
            root,
            path: self.bump.alloc_slice_copy(&segments),
        })
    }

    pub(super) fn infer_provenance_root(
        &self,
        expr: &HirExpr<'a, 'bump>,
        segments: &mut Vec<hir::ProvenancePathSegment>,
    ) -> Option<hir::ProvenanceRoot> {
        match expr {
            HirExpr::Ident(name, _) => Some(hir::ProvenanceRoot::Var(*name)),
            HirExpr::This { .. } => Some(hir::ProvenanceRoot::ThisRoot),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                segments.push(hir::ProvenancePathSegment::Field(*field));
                self.infer_provenance_root(object, segments)
            }

            HirExpr::Deref { expr: inner, .. } => {
                segments.push(hir::ProvenancePathSegment::Deref);
                self.infer_provenance_root(inner, segments)
            }

            HirExpr::ModuleAccess(access) => {
                let module_idx = self.dep_graph.borrow().resolve_module_path(access.path)?;
                self.dep_graph
                    .borrow()
                    .resolve_global_const(module_idx, access.member)?;
                Some(hir::ProvenanceRoot::Global {
                    module_idx,
                    name: access.member,
                })
            }

            _ => None,
        }
    }

    pub(super) fn hir_field_type_of_place(
        &self,
        place: &HirExpr<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        let (HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. }) =
            place
        else {
            return None;
        };
        let obj_val = match &**object {
            HirExpr::Ident(n, _) => *self.var_map.get(n)?,
            HirExpr::This { .. } => *self.var_map.get(&StrId::from_static("this"))?,
            _ => return None,
        };
        let mut ty = self.current_block_data.value_types.get(&obj_val)?;
        loop {
            match ty {
                SsaType::User(name, _) => {
                    return self
                        .structs
                        .get(name)?
                        .fields
                        .iter()
                        .find(|f| f.name == *field)
                        .map(|f| f.field_type);
                }
                SsaType::Pointer(i) | SsaType::Owned(i) | SsaType::Nullable(i) => ty = i.as_ref(),
                _ => return None,
            }
        }
    }

    pub(super) fn retype_as_owned(&mut self, val: Value, owned_hir: &HirType<'a, 'bump>) -> Value {
        let owned_ssa = lower_type_hir(owned_hir, self.enums);
        let dest = self.new_value();
        self.emit(Instruction::Cast {
            dest,
            value: Operand::Value(val),
            kind: cast_kind(&owned_ssa, &owned_ssa),
        });
        self.current_block_data.value_types.insert(dest, owned_ssa);
        dest
    }

    pub(super) fn drop_kind_for_ssa_type(
        &self,
        elem_ty: &SsaType,
        object: &HirExpr<'a, 'bump>,
    ) -> DropKind<'a, 'bump> {
        match elem_ty {
            SsaType::User(struct_name, _) => {
                if self.glue_registry.is_droppable(*struct_name) {
                    DropKind::Type(*struct_name)
                } else {
                    DropKind::Undroppable
                }
            }
            _ => match object {
                HirExpr::FieldAccess { field, .. } | HirExpr::Get { field, .. } => {
                    let this_id = StrId::from_static("this");
                    let inner_val = *self.var_map.get(&this_id).unwrap_or(&Value(0));
                    let cls_name = match self.current_block_data.value_types.get(&inner_val) {
                        Some(SsaType::User(name, _)) => Some(*name),
                        Some(SsaType::Pointer(inner)) => match inner.as_ref() {
                            SsaType::User(name, _) => Some(*name),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(cls_name) = cls_name {
                        if let Some(st) = self.structs.get(&cls_name) {
                            if let Some(f) = st.fields.iter().find(|f| f.name == *field) {
                                match f.field_type.drop_kind() {
                                    DropKind::Slice { element, .. } => return *element,
                                    DropKind::OwnedPointer { pointee, .. } => {
                                        if let DropKind::Slice { element, .. } = *pointee {
                                            return *element;
                                        }
                                    }
                                    other => return other,
                                }
                            }
                        }
                    }
                    DropKind::Undroppable
                }
                HirExpr::Ident(name, _) => {
                    if let Some(dk) =
                        crate::midend::copy_analysis::drop_tracking::local_is_droppable(
                            &self.scope_stack,
                            *name,
                        )
                    {
                        match dk {
                            DropKind::Slice { element, .. } => return *element,
                            DropKind::OwnedPointer { pointee, .. } => {
                                if let DropKind::Slice { element, .. } = *pointee {
                                    return *element;
                                }
                            }
                            other => return other,
                        }
                    }
                    DropKind::Undroppable
                }
                _ => DropKind::Undroppable,
            },
        }
    }

    pub(super) fn handle_ident(
        &mut self,
        op: AssignmentOperator,
        rhs: Value,
        name: StrId,
        span: SourceSpan<'a>,
    ) -> Value {
        let var_val = *self.var_map.get(&name).unwrap_or_else(|| {
            panic!(
                "handle_ident: variable {:?} referenced before definition",
                name
            )
        });

        if matches!(op, AssignmentOperator::Assign) {
            let droppable = crate::midend::copy_analysis::drop_tracking::local_is_droppable(
                &self.scope_stack,
                name,
            );
            if let Some(drop_kind) = droppable {
                if drop_kind.is_droppable() && !self.drop_state.is_whole_moved(name) {
                    let val_to_drop = if self.promoted_to_stack.contains(&name) {
                        let loaded = self.current_block_data.fresh_value();
                        self.emit(Instruction::Load {
                            dest: loaded,
                            ptr: Operand::Value(var_val),
                        });
                        if let Some(SsaType::Pointer(inner)) =
                            self.current_block_data.value_types.get(&var_val).cloned()
                        {
                            self.current_block_data.value_types.insert(loaded, *inner);
                        }
                        loaded
                    } else {
                        var_val
                    };

                    let mut chain_struct: Option<StrId> = None;
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
                        DropKind::Type(struct_name) => {
                            chain_struct = Some(*struct_name);
                            let partial_move = self.drop_state.has_any_field_moves(name);
                            if !partial_move {
                                if let Some(glue) = self.glue_registry.glue_name_for(*struct_name) {
                                    self.emit(Instruction::Call {
                                        dest: None,
                                        func: Operand::FunctionRef(glue),
                                        args: SmallVec::from_slice_copy(&[Operand::Value(
                                            val_to_drop,
                                        )]),
                                    });
                                }
                            } else {
                                emitter.emit_partial_struct_field_drops(
                                    name,
                                    *struct_name,
                                    val_to_drop,
                                    &self.drop_state,
                                );
                            }
                        }
                        DropKind::OwnedPointer {
                            pointee,
                            pointee_ty,
                            allocator,
                        } => {
                            emitter.emit_owned_pointer_drop(
                                Some(name),
                                pointee,
                                pointee_ty,
                                allocator,
                                val_to_drop,
                                true,
                                Some(&self.drop_state),
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
                                val_to_drop,
                                &mut resolver,
                                span,
                            );
                        }
                        DropKind::Undroppable => {}
                    }
                    if let Some(sn) = chain_struct {
                        self.emit_owned_chain_drops_for_local(sn, val_to_drop, Some(name), span);
                    }
                }
            }

            if matches!(self.local_is_droppable(name), Some(DropKind::Undroppable)) {
                self.emit_array_local_drops(name, var_val, span);
                self.set_array_flags(name, 1); // no-op if `name` has no allocated flags
            }
            self.drop_state.mark_whole_initialized(name);
        }

        if self.promoted_to_stack.contains(&name) {
            let value_to_store = match op {
                AssignmentOperator::Assign => rhs,
                _ => {
                    let pointee_ty = match self.current_block_data.value_types.get(&var_val) {
                        Some(SsaType::Pointer(inner)) => (**inner).clone(),
                        other => {
                            panic!("promoted local `{}` isn't pointer-typed: {:?}", name, other)
                        }
                    };
                    let current = self.new_value();
                    self.emit(Instruction::Load {
                        dest: current,
                        ptr: Operand::Value(var_val),
                    });
                    self.current_block_data
                        .value_types
                        .insert(current, pointee_ty.clone());

                    let dest = self.new_value();
                    self.emit(Instruction::Binary {
                        dest,
                        op: assign_op_to_bin_op(op),
                        left: Operand::Value(current),
                        right: Operand::Value(rhs),
                    });
                    self.current_block_data.value_types.insert(dest, pointee_ty);
                    dest
                }
            };
            self.emit(Instruction::Store {
                ptr: Operand::Value(var_val),
                value: Operand::Value(value_to_store),
            });
            return value_to_store;
        }

        let result = match op {
            AssignmentOperator::Assign => rhs,
            AssignmentOperator::AddAssign
            | AssignmentOperator::SubtractAssign
            | AssignmentOperator::MultiplyAssign
            | AssignmentOperator::DivideAssign
            | AssignmentOperator::ModuloAssign
            | AssignmentOperator::BitAndAssign
            | AssignmentOperator::BitOrAssign
            | AssignmentOperator::BitXorAssign
            | AssignmentOperator::ShiftLeftAssign
            | AssignmentOperator::ShiftRightAssign => {
                let dest = self.new_value();
                let bin_op = assign_op_to_bin_op(op);

                self.emit(Instruction::Binary {
                    dest,
                    op: bin_op,
                    left: Operand::Value(var_val),
                    right: Operand::Value(rhs),
                });

                let result_ty = self
                    .current_block_data
                    .value_types
                    .get(&var_val)
                    .cloned()
                    .or_else(|| self.current_block_data.value_types.get(&rhs).cloned())
                    .unwrap_or(SsaType::I64);

                self.current_block_data.value_types.insert(dest, result_ty);

                dest
            }
        };

        self.var_map.insert(name.clone(), result);
        result
    }

    pub fn record_move_if_any(&mut self, expr: &HirExpr) {
        match expr {
            HirExpr::Ident(name, _) => {
                if self.local_is_droppable(*name).is_some() {
                    self.drop_state.mark_whole_moved(*name);
                }
                self.set_array_flags(*name, 0);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::Ident(root, _) = &**object {
                    if let Some(_) = self.local_is_droppable(*root) {
                        self.drop_state.mark_field_moved(*root, *field);
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn local_is_droppable(&self, name: StrId) -> Option<DropKind<'a, 'bump>> {
        self.scope_stack
            .iter()
            .rev()
            .flat_map(|s| s.locals.iter())
            .find(|l| l.name == name)
            .map(|l| l.kind.clone())
    }

    pub(super) fn emit_scope_drops(&mut self, scope: &DropScope<'a, 'bump>, span: SourceSpan<'a>) {
        for local in scope.locals.iter().rev() {
            let Some(&val) = self.var_map.get(&local.name) else {
                continue;
            };
            if matches!(local.kind, DropKind::Undroppable) {
                self.emit_array_local_drops(local.name, val, span);
                continue;
            }
            if self.drop_state.is_whole_moved(local.name) {
                continue;
            }
            let Some(&val) = self.var_map.get(&local.name) else {
                continue;
            };
            if let Some(owned_ty) = self.nullable_owned_locals.get(&local.name).copied() {
                self.emit_nullable_owned_drop(&local.kind, owned_ty, val, Some(local.name), span);
            } else {
                self.emit_drop_for_kind(&local.kind, val, Some(local.name), span);
            }
        }
    }

    // Performant reimplementation of owned_chains_of that is specialized
    pub(super) fn struct_owns_chain(&self, struct_name: StrId) -> bool {
        is_struct_owns_chain(self.structs, self.struct_field_offsets, struct_name)
    }

    pub(super) fn emit_owned_chain_drops_for_local(
        &mut self,
        struct_name: StrId,
        val: Value,
        local_name: Option<StrId>,
        span: SourceSpan<'a>,
    ) {
        if !self.struct_owns_chain(struct_name) {
            return;
        }
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
        emitter.emit_owned_chain_field_drops(
            struct_name,
            val,
            local_name,
            Some(&self.drop_state),
            &mut resolver,
            span,
        );
    }

    pub(super) fn emit_if_flag(&mut self, flags: Value, idx: usize, body: impl FnOnce(&mut Self)) {
        let f = self.new_value();
        self.emit(Instruction::LoadField {
            dest: f,
            base: Operand::Value(flags),
            offset: idx,
        });
        self.current_block_data.value_types.insert(f, SsaType::U8);
        let set = self.new_value();
        self.emit(Instruction::Binary {
            dest: set,
            op: BinOp::Ne,
            left: Operand::Value(f),
            right: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(set, SsaType::Bool);
        let drop_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(set),
            then_bb: drop_bb,
            else_bb: after_bb,
        });
        self.current_block_data.switch_to(drop_bb);
        body(self);
        self.emit(Instruction::Jump { target: after_bb });
        self.current_block_data.switch_to(after_bb);
    }

    pub(super) fn set_array_flags(&mut self, name: StrId, v: i64) {
        if let Some(&(flags, len)) = self.array_flags.get(&name) {
            self.emit_memset(flags, v, len);
        }
    }

    /// Free-function move recording + runtime flag clearing.
    pub(super) fn record_arg_move(&mut self, expr: &HirExpr) {
        record_move_if_any(&self.scope_stack, &mut self.drop_state, expr);
        if let HirExpr::Ident(n, _) = expr {
            self.set_array_flags(*n, 0);
        }
    }

    pub(super) fn register_array_drop_local(
        &mut self,
        name: StrId,
        ssa: &SsaType,
        initialized: bool,
        rest: Option<&[HirStmt<'a, 'bump>]>,
    ) {
        let SsaType::Array(elem, len) = ssa else {
            return;
        };
        let SsaType::User(elem_struct, _) = elem.as_ref() else {
            return;
        };
        if !(self.glue_registry.is_droppable(*elem_struct) || self.struct_owns_chain(*elem_struct))
        {
            return;
        }
        self.scope_stack.last_mut().unwrap().locals.push(DropLocal {
            name,
            kind: DropKind::Undroppable,
        });

        let needs_flags = match rest {
            Some(r) => self.array_needs_flags(name, r),
            None => true,
        };
        if !needs_flags {
            return;
        }

        let flags = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: flags,
            ty: SsaType::U8,
            count: *len,
        });
        self.current_block_data
            .value_types
            .insert(flags, SsaType::Array(Box::new(SsaType::U8), *len));
        self.emit_memset(flags, initialized as i64, *len);
        self.array_flags.insert(name, (flags, *len));
    }

    pub(super) fn array_needs_flags(&self, name: StrId, rest: &[HirStmt<'a, 'bump>]) -> bool {
        pub(super) fn expr_touches(name: StrId, expr: &HirExpr) -> bool {
            match expr {
                HirExpr::Ident(n, _) => *n == name,
                HirExpr::Assignment { target, value, .. } => {
                    expr_touches(name, target) || expr_touches(name, value)
                }
                HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                    expr_touches(name, object)
                }
                HirExpr::Index { object, index, .. } => {
                    expr_touches(name, object) || expr_touches(name, index)
                }
                HirExpr::Call { callee, args, .. }
                | HirExpr::InterfaceCall { callee, args, .. } => {
                    expr_touches(name, callee) || args.iter().any(|a| expr_touches(name, a))
                }
                HirExpr::StructInit { args, .. } => {
                    args.iter().any(|fi| expr_touches(name, &fi.value))
                }
                HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                    expr_touches(name, left) || expr_touches(name, right)
                }
                HirExpr::Ref { expr, .. }
                | HirExpr::Deref { expr, .. }
                | HirExpr::Cast { expr, .. } => expr_touches(name, expr),
                HirExpr::Slice {
                    object, start, end, ..
                } => {
                    expr_touches(name, object)
                        || expr_touches(name, start)
                        || expr_touches(name, end)
                }
                HirExpr::Range { start, end, .. } => {
                    expr_touches(name, start) || expr_touches(name, end)
                }
                HirExpr::ArrayLiteral { elements, .. } => {
                    elements.iter().any(|e| expr_touches(name, e))
                }
                HirExpr::Tuple(elems, _) | HirExpr::ExprList { list: elems, .. } => {
                    elems.iter().any(|e| expr_touches(name, e))
                }
                HirExpr::EnumInit { args, .. } => args.iter().any(|a| expr_touches(name, a)),
                HirExpr::Intrinsic { args, .. } => args.iter().any(|a| expr_touches(name, a)),
                HirExpr::If { if_stmt, .. } => {
                    let HirStmt::If {
                        cond,
                        then_block,
                        else_block,
                        ..
                    } = *if_stmt
                    else {
                        return false;
                    };
                    expr_touches(name, cond)
                        || stmts_touch(name, then_block)
                        || else_block.map_or(false, |eb| stmt_touches(name, eb))
                }
                HirExpr::Block { body, .. } => stmts_touch(name, body),
                HirExpr::Match { expr, arms, .. } => {
                    expr_touches(name, expr)
                        || arms.iter().any(|arm| {
                            arm.guard.map_or(false, |g| expr_touches(name, g))
                                || stmt_touches(name, arm.body)
                        })
                }
                _ => false,
            }
        }

        pub(super) fn stmt_touches(name: StrId, stmt: &HirStmt) -> bool {
            match stmt {
                HirStmt::Expr(e) => expr_touches(name, e),
                HirStmt::Let { value, .. } => expr_touches(name, value),
                HirStmt::Return(Some(e), _) => expr_touches(name, e),
                HirStmt::Break(Some(e), _) => expr_touches(name, e),
                HirStmt::Block { body, .. } => stmts_touch(name, body),
                HirStmt::If {
                    cond,
                    then_block,
                    else_block,
                    ..
                } => {
                    expr_touches(name, cond)
                        || stmts_touch(name, then_block)
                        || else_block.map_or(false, |eb| stmt_touches(name, eb))
                }
                HirStmt::While { cond, body } => {
                    expr_touches(name, cond) || stmt_touches(name, body)
                }
                HirStmt::For {
                    condition,
                    increment,
                    body,
                    ..
                } => {
                    condition.map_or(false, |c| expr_touches(name, c))
                        || increment.map_or(false, |i| expr_touches(name, i))
                        || stmt_touches(name, body)
                }
                HirStmt::Match { expr, arms, .. } => {
                    expr_touches(name, expr) || arms.iter().any(|arm| stmt_touches(name, arm.body))
                }
                HirStmt::UnsafeBlock { body } => stmt_touches(name, body),
                _ => false,
            }
        }

        pub(super) fn stmts_touch(name: StrId, stmts: &[HirStmt]) -> bool {
            stmts.iter().any(|s| stmt_touches(name, s))
        }

        pub(super) fn walk(name: StrId, stmts: &[HirStmt]) -> bool {
            for stmt in stmts {
                match stmt {
                    HirStmt::If { .. }
                    | HirStmt::While { .. }
                    | HirStmt::For { .. }
                    | HirStmt::Match { .. } => {
                        if stmt_touches(name, stmt) {
                            return true;
                        }
                    }
                    HirStmt::Block { body, .. } => {
                        if walk(name, body) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }

        walk(name, rest)
    }

    /// Unrolled per-element drops, skipping indices still tracked as uninitialized.
    pub(super) fn emit_array_local_drops(&mut self, name: StrId, val: Value, span: SourceSpan<'a>) {
        let Some(SsaType::Array(elem, len)) = self.value_type(val).cloned() else {
            return;
        };
        let SsaType::User(elem_struct, _) = elem.as_ref() else {
            return;
        };
        if !(self.glue_registry.is_droppable(*elem_struct) || self.struct_owns_chain(*elem_struct))
        {
            return;
        }
        let kind = DropKind::Type(*elem_struct);
        let elem_size = ir::layout::sizeof_ssa(&elem, TargetInfo { ptr_bytes: 8 })
            .expect("array element type has no known size");

        for i in 0..len {
            let status = self.drop_state.index_status(name, i as i64);
            if status == Tri::No {
                continue;
            }
            let addr = self.current_block_data.fresh_value();
            self.emit(Instruction::FieldAddr {
                dest: addr,
                base: Operand::Value(val),
                offset: i * elem_size,
            });
            self.current_block_data
                .value_types
                .insert(addr, SsaType::Pointer(Box::new((*elem).clone())));
            match status {
                Tri::Yes => self.emit_indexed_element_drop(&kind, addr, span),
                Tri::Maybe => {
                    let (flags, _) = *self
                        .array_flags
                        .get(&name)
                        .expect("Maybe-initialised array has no drop flags");
                    self.emit_if_flag(flags, i, |s| s.emit_indexed_element_drop(&kind, addr, span));
                }
                Tri::No => unreachable!(),
            }
        }
    }

    pub(super) fn emit_drop_for_kind(
        &mut self,
        kind: &DropKind<'a, 'bump>,
        val: Value,
        local_name: Option<StrId>,
        span: SourceSpan<'a>,
    ) {
        match kind {
            DropKind::Type(struct_name) => {
                match local_name {
                    Some(name) => {
                        let partial_move = self.drop_state.has_any_field_moves(name);
                        match (partial_move, self.glue_registry.glue_name_for(*struct_name)) {
                            (false, Some(glue)) => {
                                self.emit(Instruction::Call {
                                    dest: None,
                                    func: Operand::FunctionRef(glue),
                                    args: SmallVec::from_slice_copy(&[Operand::Value(val)]),
                                });
                            }
                            _ => {
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
                                emitter.emit_partial_struct_field_drops(
                                    name,
                                    *struct_name,
                                    val,
                                    &self.drop_state,
                                );
                            }
                        }
                    }
                    None => match self.glue_registry.glue_name_for(*struct_name) {
                        Some(glue) => {
                            self.emit(Instruction::Call {
                                dest: None,
                                func: Operand::FunctionRef(glue),
                                args: SmallVec::from_slice_copy(&[Operand::Value(val)]),
                            });
                        }
                        None if self.struct_owns_chain(*struct_name) => {}
                        None => panic!(
                            "no drop glue registered for struct `{}`; every droppable \
                             struct should have glue built by DropGlueBuilder::build_all",
                            struct_name
                        ),
                    },
                }
                // Glue doesn't cover `?^Node` chains, so walk them here.
                self.emit_owned_chain_drops_for_local(*struct_name, val, local_name, span);
            }

            DropKind::OwnedPointer {
                pointee,
                pointee_ty,
                allocator,
            } => {
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
                emitter.emit_owned_pointer_drop(
                    local_name,
                    pointee,
                    pointee_ty,
                    allocator,
                    val,
                    true,
                    Some(&self.drop_state),
                    &mut resolver,
                    span,
                );
            }

            DropKind::Undroppable => {}

            DropKind::Slice {
                element,
                element_ty,
            } => {
                self.emit_slice_element_drops((**element).clone(), *element_ty, val, span);
            }
        }
    }

    pub(super) fn emit_slice_element_drops(
        &mut self,
        element_kind: DropKind<'a, 'bump>,
        element_ty: HirType<'a, 'bump>,
        slice_val: Value,
        span: SourceSpan<'a>,
    ) {
        if matches!(element_kind, DropKind::Undroppable) {
            return;
        }

        let elem_ssa_ty = lower_type_hir(&element_ty, self.enums);

        let ptr_v = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: ptr_v,
            base: Operand::Value(slice_val),
            offset: 0,
        });
        self.current_block_data
            .value_types
            .insert(ptr_v, SsaType::Pointer(Box::new(elem_ssa_ty.clone())));

        let len_v = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: len_v,
            base: Operand::Value(slice_val),
            offset: 8,
        });
        self.current_block_data
            .value_types
            .insert(len_v, SsaType::Usize);

        let idx_init = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: idx_init,
            ty: SsaType::Usize,
            value: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(idx_init, SsaType::Usize);

        let pre_bb = self.current_block_data.current_block;
        let cond_bb = self.current_block_data.new_block();
        let body_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();

        self.emit(Instruction::Jump { target: cond_bb });

        self.current_block_data.switch_to(cond_bb);
        let idx_phi = self.current_block_data.fresh_value();
        self.current_block_data
            .value_types
            .insert(idx_phi, SsaType::Usize);
        let phi_idx = self.current_block_data.bb().instructions.len();
        self.emit(Instruction::Phi {
            dest: idx_phi,
            incoming: SmallVec::new(),
        });
        if let Instruction::Phi { incoming, .. } =
            &mut self.current_block_data.bb().instructions[phi_idx]
        {
            incoming.push((pre_bb, idx_init));
        }

        let cont = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: cont,
            op: BinOp::Lt,
            left: Operand::Value(idx_phi),
            right: Operand::Value(len_v),
        });
        self.current_block_data
            .value_types
            .insert(cont, SsaType::Bool);
        self.emit(Instruction::Branch {
            cond: Operand::Value(cont),
            then_bb: body_bb,
            else_bb: after_bb,
        });

        self.current_block_data.switch_to(body_bb);
        let elem_size =
            ir::layout::sizeof_ssa(&elem_ssa_ty, ir::layout::TargetInfo { ptr_bytes: 8 })
                .expect("slice element type has no known size") as i64;
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
            left: Operand::Value(idx_phi),
            right: Operand::Value(size_v),
        });
        self.current_block_data
            .value_types
            .insert(offset_v, SsaType::I64);

        let elem_addr = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: elem_addr,
            op: BinOp::Add,
            left: Operand::Value(ptr_v),
            right: Operand::Value(offset_v),
        });
        self.current_block_data
            .value_types
            .insert(elem_addr, SsaType::Pointer(Box::new(elem_ssa_ty.clone())));

        let elem_val = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: elem_val,
            ptr: Operand::Value(elem_addr),
        });
        self.current_block_data
            .value_types
            .insert(elem_val, elem_ssa_ty.clone());

        self.emit_drop_for_kind(&element_kind, elem_val, None, span);

        let one = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: one,
            ty: SsaType::Usize,
            value: Operand::ConstInt(1),
        });
        self.current_block_data
            .value_types
            .insert(one, SsaType::Usize);
        let next_idx = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: next_idx,
            op: BinOp::Add,
            left: Operand::Value(idx_phi),
            right: Operand::Value(one),
        });
        self.current_block_data
            .value_types
            .insert(next_idx, SsaType::Usize);

        let body_end = self.current_block_data.current_block;
        self.emit(Instruction::Jump { target: cond_bb });
        let cond_block = self
            .current_block_data
            .func
            .blocks
            .iter_mut()
            .find(|b| b.id == cond_bb)
            .expect("slice-drop loop header missing");
        if let Instruction::Phi { incoming, .. } = &mut cond_block.instructions[phi_idx] {
            incoming.push((body_end, next_idx));
        }

        self.current_block_data.switch_to(after_bb);
    }

    pub(super) fn emit_drops_for_return(&mut self, span: SourceSpan<'a>) {
        for scope in self.scope_stack.clone().iter().rev() {
            self.emit_scope_drops(scope, span);
        }
    }

    pub(super) fn emit_drops_for_loop_exit(
        &mut self,
        depth_at_loop_entry: usize,
        span: SourceSpan<'a>,
    ) {
        for i in (depth_at_loop_entry..self.scope_stack.len()).rev() {
            let scope =
                std::mem::replace(&mut self.scope_stack[i], DropScope { locals: Vec::new() });
            self.emit_scope_drops(&scope, span);
            self.scope_stack[i] = scope;
        }
    }

    pub(super) fn is_move_by_value(ty: &SsaType) -> bool {
        matches!(ty, SsaType::User(_, _) | SsaType::Owned(_))
    }

    pub(super) fn nullable_owned_drop_kind(
        &self,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
    ) -> Option<(DropKind<'a, 'bump>, HirType<'a, 'bump>)> {
        let HirType::Nullable(inner) = ty else {
            return None;
        };
        let owned: HirType<'a, 'bump> = **inner;
        if !matches!(owned, HirType::OwnedPointer { .. }) {
            return None;
        }
        // A plain field read doesn't transfer ownership, and the typechecker only tracks
        // moves out of `Ident.field`, so registering here would double-free.
        if matches!(
            value,
            HirExpr::FieldAccess { .. }
                | HirExpr::Get { .. }
                | HirExpr::Deref { .. }
                | HirExpr::Index { .. }
        ) {
            return None;
        }
        let kind = self.recover_owned_pointer_drop_kind(&owned, value)?; // None => leak, as today
        Some((kind, owned))
    }

    pub(super) fn emit_nullable_owned_field_overwrite_drop(
        &mut self,
        field_addr: Value,
        owned_ty: HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        let placeholder = HirExpr::Null(Default::default());
        let Some(kind) = self.recover_owned_pointer_drop_kind(&owned_ty, &placeholder) else {
            return;
        };
        let loaded_ty = match self.value_type(field_addr).cloned() {
            Some(SsaType::Pointer(inner)) => *inner,
            _ => return,
        };
        let old = self.new_value();
        self.emit(Instruction::Load {
            dest: old,
            ptr: Operand::Value(field_addr),
        });
        self.current_block_data.value_types.insert(old, loaded_ty);
        self.emit_nullable_owned_drop(&kind, owned_ty, old, None, span);
    }

    pub(super) fn emit_nullable_owned_drop(
        &mut self,
        kind: &DropKind<'a, 'bump>,
        owned_ty: HirType<'a, 'bump>,
        val: Value,
        name: Option<StrId>,
        span: SourceSpan<'a>,
    ) {
        let ty = self
            .value_type(val)
            .cloned()
            .expect("nullable owned local has no type");
        let pointee = ty
            .nullable_pointer_repr()
            .cloned()
            .expect("`?^T` local isn't pointer-optimized");
        let ptr_ty = SsaType::Pointer(Box::new(pointee));

        let zero = self.new_value();
        self.emit(Instruction::Const {
            dest: zero,
            ty: ptr_ty.clone(),
            value: Operand::ConstInt(0),
        });
        self.current_block_data.value_types.insert(zero, ptr_ty);
        let is_null = self.new_value();
        self.emit(Instruction::Binary {
            dest: is_null,
            op: BinOp::Eq,
            left: Operand::Value(val),
            right: Operand::Value(zero),
        });
        self.current_block_data
            .value_types
            .insert(is_null, SsaType::Bool);

        let drop_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(is_null),
            then_bb: after_bb,
            else_bb: drop_bb,
        });
        self.current_block_data.switch_to(drop_bb);
        let owned = self.retype_as_owned(val, &owned_ty);
        self.emit_drop_for_kind(kind, owned, name, span);
        self.emit(Instruction::Jump { target: after_bb });
        self.current_block_data.switch_to(after_bb);
    }

    /// In a `case x ->` arm that follows a `case null ->` arm, matching a `?^T` local by
    /// value moves ownership from the scrutinee into the binding.
    pub(super) fn adopt_owned_binding(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        pattern: &HirPattern<'bump>,
        prior_null_arm: bool,
    ) {
        if !prior_null_arm {
            return;
        }
        let (HirExpr::Ident(src, _), HirPattern::Ident(binding)) = (scrutinee, pattern) else {
            return;
        };
        let Some(owned_ty) = self.nullable_owned_locals.get(src).copied() else {
            return;
        };
        let Some(kind) = self.local_is_droppable(*src) else {
            return;
        };
        let Some(&bound) = self.var_map.get(binding) else {
            return;
        };

        self.drop_state.mark_whole_moved(*src);
        let owned = self.retype_as_owned(bound, &owned_ty);
        self.var_map.insert(*binding, owned);
        self.scope_stack.last_mut().unwrap().locals.push(DropLocal {
            name: *binding,
            kind,
        });
        self.drop_state.mark_whole_initialized(*binding);
    }

    pub(super) fn emit_indexed_element_drop(
        &mut self,
        drop_kind: &DropKind<'a, 'bump>,
        addr: Value,
        span: SourceSpan<'a>,
    ) {
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
        emitter.emit_element_drop(drop_kind, addr, &mut resolver, span);
    }

    pub(crate) fn handle_potential_drop(
        &mut self,
        field: StrId,
        span: SourceSpan<'a>,
        obj_val: Value,
        field_offset: usize,
        owner: Option<StrId>,
        nullable_owned: Option<HirType<'_, '_>>,
        f: &ir::hir::HirField<'_, '_>,
    ) {
        let drop_kind = if nullable_owned.is_some() {
            DropKind::Undroppable // already handled above
        } else {
            f.field_type.drop_kind()
        };
        if drop_kind.is_droppable() {
            let is_uninit = owner.map_or(false, |o| self.drop_state.is_field_moved(o, field));
            if !is_uninit {
                let field_addr = self.current_block_data.fresh_value();
                self.emit(Instruction::FieldAddr {
                    dest: field_addr,
                    base: Operand::Value(obj_val),
                    offset: field_offset,
                });
                self.current_block_data.value_types.insert(
                    field_addr,
                    SsaType::Pointer(Box::new(lower_type_hir(&f.field_type, self.enums))),
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
                            let pointee_ssa = lower_type_hir(pointee_ty, emitter.enums);
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
                        emitter.emit_element_drop(&drop_kind, field_addr, &mut resolver, span);
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
