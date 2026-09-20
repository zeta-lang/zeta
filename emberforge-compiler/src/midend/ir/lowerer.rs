use crate::midend::copy_analysis::drop_emitter::{DropEmitter, FnAllocatorResolver};
use crate::midend::copy_analysis::drop_glue::DropGlueRegistry;
use crate::midend::copy_analysis::drop_tracking::{
    DropLocal, DropMoveState, DropScope, record_move_if_any,
};
use crate::midend::ir::block_data::CurrentBlockData;
use crate::optimized_string_buffering;
use codex_dependency_graph::DepGraph;
use ir::hir::{
    self, AssignmentOperator, DropKind, HirEnum, HirErrorHandlerPattern, HirExpr, HirFieldInit,
    HirFunc, HirMatchArm, HirParam, HirPattern, HirStmt, HirStruct, HirType, IntrinsicKind,
    Operator, ProvenanceAnnotation, StrId,
};
use ir::ir_conversion::{assign_op_to_bin_op, lower_operator_bin, lower_type_hir};
use ir::ir_hasher::{HashMap, HashSet};
use ir::layout::{TargetInfo, alignof_ssa, round_up_to_align};
use ir::span::SourceSpan;
use ir::ssa_ir::{
    AllocatorKind, BasicBlock, BinOp, BlockId, Function, Instruction, Operand, SsaType, Value,
    cast_kind,
};
use smallvec::SmallVec;
use smallvec::smallvec;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;
use zetaruntime::intern_fmt;
use zetaruntime::string_pool::StringPool;

/// Bookkeeping for one enclosing loop, needed so `break`/`continue` (which
/// can appear arbitrarily deep inside nested control flow) can contribute a
/// phi edge to the right join point. `phis` are `(name, instruction-index,
/// phi-dest-value)` triples identifying placeholder `Instruction::Phi`s
/// created by `open_join`, still waiting for more incoming edges.
struct LoopCtx {
    /// Where `continue` jumps to (the loop header for `while` or the
    /// increment block for `for`).
    continue_target: BlockId,
    /// The block whose phis a `continue` must contribute an edge to (same
    /// as `continue_target`).
    continue_join_bb: BlockId,
    continue_join_phis: Vec<(StrId, usize, Value)>,
    /// Where `break` jumps to (the block right after the loop).
    break_target: BlockId,
    /// The block (== `break_target`) whose phis a `break` must contribute
    /// an edge to.
    break_join_phis: Vec<(StrId, usize, Value)>,
    scope_depth_at_entry: usize,
}

#[derive(Clone, Copy)]
enum IndexedContainer {
    Array(usize),
    BorrowedSlice,
    OwnedSlice,
}

#[derive(Clone, Copy)]
enum SlicePrimitive {
    WriteUninit,
    WriteUninitAll,
    GetUnchecked,
}

#[derive(Clone, Copy)]
enum FieldInitVal {
    Null,
    Uninit,
    Val(Value),
}

pub struct FunctionLowerer<'f, 'a, 'bump> {
    current_block_data: CurrentBlockData<'f>,
    var_map: HashMap<StrId, Value>,
    phantom_data: PhantomData<&'bump ()>,
    loop_stack: Vec<LoopCtx>,
    funcs: &'a HashMap<StrId, Function>,
    struct_field_offsets: &'a HashMap<StrId, HashMap<StrId, usize>>,
    struct_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
    struct_mangled_map: &'a HashMap<StrId, HashMap<StrId, StrId>>,
    struct_vtable_slots: &'a HashMap<StrId, Vec<StrId>>,
    interface_id_map: &'a HashMap<StrId, usize>,
    interface_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
    structs: &'a HashMap<StrId, HirStruct<'a, 'bump>>,
    enum_variant_tags: &'a HashMap<StrId, HashMap<StrId, usize>>,
    enums: &'a HashMap<StrId, HirEnum<'a, 'bump>>,
    context: Arc<StringPool>,
    extern_c_names: &'a HashSet<StrId>,
    pub dep_graph: &'a RefCell<DepGraph>,
    pub module_idx: usize,
    return_type: Option<HirType<'a, 'bump>>,
    global_funcs: &'a HashMap<StrId, Function>,
    scope_stack: Vec<DropScope<'a, 'bump>>,
    drop_state: DropMoveState<'a, 'bump>,
    glue_registry: &'a DropGlueRegistry,
    allocator_kind: &'a HashMap<StrId, AllocatorKind>,
    pub interface_methods: &'a HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
    bump: &'bump GrowableBump<'bump>,
    module_import_aliases: &'a HashMap<usize, HashMap<StrId, usize>>,
    module_named_imports: &'a HashMap<usize, HashMap<StrId, usize>>,
    constants: &'a HashMap<StrId, HirExpr<'a, 'bump>>,
    promoted_to_stack: HashSet<StrId>,
    narrowed_fields: HashMap<(StrId, Vec<StrId>), Value>,
    nullable_owned_locals: HashMap<StrId, HirType<'a, 'bump>>,
}

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump>
where
    'bump: 'a,
{
    pub fn new(
        function: &'f mut Function,
        hir_fn: &HirFunc<'a, 'bump>,
        funcs: &'a HashMap<StrId, Function>,
        global_funcs: &'a HashMap<StrId, Function>,
        struct_field_offsets: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_mangled_map: &'a HashMap<StrId, HashMap<StrId, StrId>>,
        struct_vtable_slots: &'a HashMap<StrId, Vec<StrId>>,
        interface_id_map: &'a HashMap<StrId, usize>,
        interface_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        structs: &'a HashMap<StrId, HirStruct<'a, 'bump>>,
        enum_variant_tags: &'a HashMap<StrId, HashMap<StrId, usize>>,
        context: Arc<StringPool>,
        extern_c_names: &'a HashSet<StrId>,
        dep_graph: &'a RefCell<DepGraph>,
        module_idx: usize,
        glue_registry: &'a DropGlueRegistry,
        allocator_kind: &'a HashMap<StrId, AllocatorKind>,
        interface_methods: &'a HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
        bump: &'bump GrowableBump<'bump>,
        enums: &'a HashMap<StrId, HirEnum<'a, 'bump>>,
        module_import_aliases: &'a HashMap<usize, HashMap<StrId, usize>>,
        module_named_imports: &'a HashMap<usize, HashMap<StrId, usize>>,
        constants: &'a HashMap<StrId, HirExpr<'a, 'bump>>,
    ) -> Result<Self, std::alloc::AllocError> {
        Self::new_internal(
            function,
            hir_fn,
            funcs,
            global_funcs,
            struct_field_offsets,
            struct_method_slots,
            struct_mangled_map,
            struct_vtable_slots,
            interface_id_map,
            interface_method_slots,
            structs,
            enum_variant_tags,
            context,
            extern_c_names,
            dep_graph,
            module_idx,
            glue_registry,
            allocator_kind,
            interface_methods,
            bump,
            enums,
            module_import_aliases,
            module_named_imports,
            constants,
        )
    }

    pub fn new_with_struct(
        function: &'f mut Function,
        hir_fn: &HirFunc<'a, 'bump>,
        funcs: &'a HashMap<StrId, Function>,
        global_funcs: &'a HashMap<StrId, Function>,
        struct_field_offsets: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_mangled_map: &'a HashMap<StrId, HashMap<StrId, StrId>>,
        struct_vtable_slots: &'a HashMap<StrId, Vec<StrId>>,
        interface_id_map: &'a HashMap<StrId, usize>,
        interface_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        structs: &'a HashMap<StrId, HirStruct<'a, 'bump>>,
        enum_variant_tags: &'a HashMap<StrId, HashMap<StrId, usize>>,
        context: Arc<StringPool>,
        extern_c_names: &'a HashSet<StrId>,
        dep_graph: &'a RefCell<DepGraph>,
        module_idx: usize,
        glue_registry: &'a DropGlueRegistry,
        allocator_kind: &'a HashMap<StrId, AllocatorKind>,
        interface_methods: &'a HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
        bump: &'bump GrowableBump<'bump>,
        enums: &'a HashMap<StrId, HirEnum<'a, 'bump>>,
        module_import_aliases: &'a HashMap<usize, HashMap<StrId, usize>>,
        module_named_imports: &'a HashMap<usize, HashMap<StrId, usize>>,
        constants: &'a HashMap<StrId, HirExpr<'a, 'bump>>,
    ) -> Result<Self, std::alloc::AllocError> {
        Self::new_internal(
            function,
            hir_fn,
            funcs,
            global_funcs,
            struct_field_offsets,
            struct_method_slots,
            struct_mangled_map,
            struct_vtable_slots,
            interface_id_map,
            interface_method_slots,
            structs,
            enum_variant_tags,
            context,
            extern_c_names,
            dep_graph,
            module_idx,
            glue_registry,
            allocator_kind,
            interface_methods,
            bump,
            enums,
            module_import_aliases,
            module_named_imports,
            constants,
        )
    }

    fn new_internal(
        function: &'f mut Function,
        hir_fn: &HirFunc<'a, 'bump>,
        funcs: &'a HashMap<StrId, Function>,
        global_funcs: &'a HashMap<StrId, Function>,
        struct_field_offsets: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        struct_mangled_map: &'a HashMap<StrId, HashMap<StrId, StrId>>,
        struct_vtable_slots: &'a HashMap<StrId, Vec<StrId>>,
        interface_id_map: &'a HashMap<StrId, usize>,
        interface_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
        structs: &'a HashMap<StrId, HirStruct<'a, 'bump>>,
        enum_variant_tags: &'a HashMap<StrId, HashMap<StrId, usize>>,
        context: Arc<StringPool>,
        extern_c_names: &'a HashSet<StrId>,
        dep_graph: &'a RefCell<DepGraph>,
        module_idx: usize,
        glue_registry: &'a DropGlueRegistry,
        allocator_kind: &'a HashMap<StrId, AllocatorKind>,
        interface_methods: &'a HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
        bump: &'bump GrowableBump<'bump>,
        enums: &'a HashMap<StrId, HirEnum<'a, 'bump>>,
        module_import_aliases: &'a HashMap<usize, HashMap<StrId, usize>>,
        module_named_imports: &'a HashMap<usize, HashMap<StrId, usize>>,
        constants: &'a HashMap<StrId, HirExpr<'a, 'bump>>,
    ) -> Result<Self, std::alloc::AllocError> {
        let mut var_map = HashMap::default();
        let mut value_types = HashMap::default();

        if let Some(params) = hir_fn.params {
            assert_eq!(
                params.len(),
                function.params.len(),
                "param count mismatch between HIR signature and pre-registered \
                 Function for `{}`, Function::from_signature and this \
                 constructor have gone out of sync on how params are counted",
                hir_fn.name
            );

            for (hir_param, &(value, ref ssa_ty)) in params.iter().zip(function.params.iter()) {
                value_types.insert(value, ssa_ty.clone());
                match hir_param {
                    HirParam::This { .. } => {
                        let name = StrId::from_static("this");
                        var_map.insert(name, value);
                    }
                    HirParam::Normal { name, .. } => {
                        var_map.insert(*name, value);
                    }
                }
            }
        }

        let next_value = function.params.len();
        let mut next_block = 0usize;

        function.blocks.clear();
        let entry_bb = BlockId(next_block);
        next_block += 1;
        function.entry = entry_bb;
        function.blocks.push(BasicBlock {
            id: entry_bb,
            instructions: Vec::new(),
        });

        let current_block_data: CurrentBlockData<'f> =
            CurrentBlockData::new(function, entry_bb, next_value, next_block, value_types);

        Ok(Self {
            current_block_data,
            funcs,
            var_map,
            loop_stack: Vec::new(),
            struct_field_offsets,
            struct_method_slots,
            struct_mangled_map,
            struct_vtable_slots,
            interface_id_map,
            interface_method_slots,
            structs,
            enum_variant_tags,
            context,
            phantom_data: Default::default(),
            extern_c_names,
            dep_graph,
            module_idx,
            return_type: hir_fn.return_type,
            global_funcs,
            scope_stack: vec![DropScope { locals: Vec::new() }],
            drop_state: DropMoveState::default(),
            glue_registry,
            allocator_kind,
            interface_methods,
            bump,
            enums,
            module_import_aliases,
            module_named_imports,
            constants,
            promoted_to_stack: HashSet::default(),
            narrowed_fields: HashMap::default(),
            nullable_owned_locals: HashMap::default(),
        })
    }

    pub(super) fn lower_body(&mut self, body: Option<HirStmt<'a, 'bump>>) {
        if let Some(b) = body {
            match b {
                HirStmt::Block { body, span } => {
                    // if self
                    //     .current_block_data
                    //     .func
                    //     .name
                    //     .as_str()
                    //     .contains("push_back")
                    //     || self
                    //         .current_block_data
                    //         .func
                    //         .name
                    //         .as_str()
                    //         .contains("remove_back")
                    // {
                    //     println!(
                    //         "HIR of {}: \n{:?}",
                    //         self.current_block_data.func.name.clone(),
                    //         body
                    //     );
                    // }

                    self.scope_stack.push(DropScope { locals: Vec::new() });
                    self.lower_stmt_seq(body);

                    let scope = self.scope_stack.pop().unwrap();
                    if !self.block_terminated() {
                        self.emit_scope_drops(&scope, span);
                    }
                }
                _ => panic!(),
            }
        }
    }

    fn lower_stmt_seq(&mut self, stmts: &[HirStmt<'a, 'bump>]) {
        for stmt in stmts {
            if self.block_terminated() {
                break;
            }
            self.lower_stmt(stmt);
        }
    }

    pub(super) fn lower_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        match stmt {
            HirStmt::Expr(expr) => {
                let _ = self.lower_expr(expr);
            }
            HirStmt::UnsafeBlock { body } => {
                self.lower_stmt(body);
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                self.lower_if(cond, then_block, else_block, *span);
            }
            HirStmt::While { cond, body } => {
                self.lower_while(cond, body);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                self.lower_for(*init, *condition, *increment, body);
            }
            HirStmt::Let {
                name,
                ty,
                value,
                catch_pattern,
                else_block,
                ..
            } => {
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

                if let HirType::Struct {
                    name: struct_name, ..
                } = ty
                {
                    if self.glue_registry.is_droppable(*struct_name) {
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
            }

            HirStmt::Return(expr, span) => {
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
                self.emit_drops_for_return(*span);
                self.emit(Instruction::Ret { value });
            }

            HirStmt::Block { body, span } => {
                self.scope_stack.push(DropScope { locals: Vec::new() });
                self.lower_stmt_seq(body);
                let scope = self.scope_stack.pop().unwrap();
                if !self.block_terminated() {
                    self.emit_scope_drops(&scope, *span);
                }
            }

            HirStmt::Break(expr, span) => {
                if let Some(e) = expr {
                    self.record_move_if_any(e);
                }
                let ctx = self.loop_stack.last().expect(
                    "`break` outside of a loop (should have been caught by the typechecker)",
                );
                let break_target = ctx.break_target;
                let phis = ctx.break_join_phis.clone();
                let depth = ctx.scope_depth_at_entry;
                let vars = self.var_map.clone();
                self.emit_drops_for_loop_exit(depth, *span);
                let from_bb = self.current_block_data.current_block;
                self.contribute_join_edge(break_target, from_bb, &vars, &phis);
                self.emit(Instruction::Jump {
                    target: break_target,
                });
            }
            HirStmt::Continue(span) => {
                let ctx = self.loop_stack.last().expect(
                    "`continue` outside of a loop (should have been caught by the typechecker)",
                );
                let continue_target = ctx.continue_target;
                let join_bb = ctx.continue_join_bb;
                let phis = ctx.continue_join_phis.clone();
                let depth = ctx.scope_depth_at_entry;
                let vars = self.var_map.clone();
                self.emit_drops_for_loop_exit(depth, *span);
                let from_bb = self.current_block_data.current_block;
                self.contribute_join_edge(join_bb, from_bb, &vars, &phis);
                self.emit(Instruction::Jump {
                    target: continue_target,
                });
            }
            HirStmt::Match { expr, arms, span } => {
                let match_expr = HirExpr::Match {
                    expr,
                    arms,
                    span: *span,
                };
                let _ = self.lower_expr(&match_expr);
            }
            HirStmt::Defer(_) => {
                // TODO: handle defers
            }
            _ => unimplemented!("Statement {:?} not yet lowered", stmt),
        }
    }

    fn recover_owned_pointer_drop_kind(
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

    fn infer_allocator_from_expr(
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

    fn infer_provenance(&self, expr: &HirExpr<'a, 'bump>) -> Option<ProvenanceAnnotation<'bump>> {
        let mut segments = Vec::new();
        let root = self.infer_provenance_root(expr, &mut segments)?;
        segments.reverse();
        Some(ProvenanceAnnotation {
            root,
            path: self.bump.alloc_slice_copy(&segments),
        })
    }

    fn infer_provenance_root(
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

    fn open_join(&mut self) -> Vec<(StrId, usize, Value)> {
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

    fn contribute_join_edge(
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

    fn merge_var_maps(&mut self, branches: Vec<(BlockId, HashMap<StrId, Value>)>) {
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

    fn cond_may_narrow_null(cond: &HirExpr<'a, 'bump>) -> bool {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return false;
        };
        if !matches!(op, Operator::Equals | Operator::NotEquals) {
            return false;
        }
        let is_narrowable_side = |e: &HirExpr<'a, 'bump>| {
            matches!(
                e,
                HirExpr::Ident(_, _) | HirExpr::FieldAccess { .. } | HirExpr::Get { .. }
            )
        };
        (matches!(right, HirExpr::Null(_)) && is_narrowable_side(left))
            || (matches!(left, HirExpr::Null(_)) && is_narrowable_side(right))
    }

    fn revert_narrow_in_map(
        vars: &mut HashMap<StrId, Value>,
        narrowed: Option<(StrId, Value, Value)>,
    ) {
        if let Some((name, prev, narrowed_val)) = narrowed {
            if vars.get(&name).copied() == Some(narrowed_val) {
                vars.insert(name, prev);
            }
        }
    }

    fn lower_if(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &'bump [HirStmt<'a, 'bump>],
        else_block: &Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
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
            Self::revert_narrow_in_map(&mut self.var_map, then_narrowed);
            let tail = self.current_block_data.current_block;
            let vars = self.var_map.clone();
            self.emit(Instruction::Jump { target: merge_bb });
            Some((tail, vars))
        };

        let else_live = if needs_real_else_block {
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
        }
    }

    fn lower_while(&mut self, cond: &HirExpr<'a, 'bump>, body: &HirStmt<'a, 'bump>) {
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
        self.loop_stack.pop();

        self.current_block_data.switch_to(after_bb);
        self.narrowed_fields = narrowed_before;
        self.var_map = exit_phis
            .into_iter()
            .map(|(name, _, dest)| (name, dest))
            .collect();
    }

    fn lower_for(
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
        self.loop_stack.pop();

        self.var_map = incr_phis
            .iter()
            .map(|(name, _, dest)| (*name, *dest))
            .collect();
        self.current_block_data.switch_to(incr_bb);
        self.narrowed_fields.clear();
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
    }

    fn stmt_diverges(stmt: &HirStmt) -> bool {
        matches!(
            stmt,
            HirStmt::Return(..) | HirStmt::Break(..) | HirStmt::Continue(_)
        )
    }

    fn lower_catch(&mut self, raw_val: Value, pattern: &HirErrorHandlerPattern<'a, 'bump>) {
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
            for s in *body {
                self.lower_stmt(s);
            }
            if !body.last().map_or(false, Self::stmt_diverges) {
                panic!(
                    "`catch` arm for `{:?}` must end in return, throw, break, or continue",
                    error_name
                );
            }
        }
    }

    fn lower_null_as(&mut self, expected: Option<HirType<'a, 'bump>>) -> Value {
        let v = self.current_block_data.fresh_value();

        match expected {
            Some(HirType::Nullable(inner)) => {
                let inner_ssa = lower_type_hir(inner, self.enums);

                if inner_ssa.is_pointer() {
                    // Pointer-optimized nullable: null is just 0.
                    self.emit(Instruction::Const {
                        dest: v,
                        ty: SsaType::I64,
                        value: Operand::ConstInt(0),
                    });
                    self.current_block_data
                        .value_types
                        .insert(v, SsaType::Nullable(Box::new(inner_ssa)));
                } else {
                    let _payload_align =
                        alignof_ssa(&inner_ssa, TargetInfo { ptr_bytes: 8 }).unwrap_or(1);
                    let tag_offset = 0usize; // tag at offset 0 per layout_of_ssa for Nullable
                    let nullable_ty = SsaType::Nullable(Box::new(inner_ssa.clone()));
                    let size = ir::layout::sizeof_ssa(&nullable_ty, TargetInfo { ptr_bytes: 8 })
                        .unwrap_or(16);

                    self.emit(Instruction::StackAlloc {
                        dest: v,
                        ty: nullable_ty.clone(),
                        count: size,
                    });

                    // Write null tag (0) at offset 0.
                    let null_tag_val = self.current_block_data.fresh_value();
                    self.emit(Instruction::Const {
                        dest: null_tag_val,
                        ty: SsaType::U8,
                        value: Operand::ConstInt(0),
                    });
                    self.emit(Instruction::StoreField {
                        base: Operand::Value(v),
                        offset: tag_offset,
                        value: Operand::Value(null_tag_val),
                    });

                    self.current_block_data.value_types.insert(v, nullable_ty);
                }
            }
            _ => {
                self.emit(Instruction::Const {
                    dest: v,
                    ty: SsaType::I64,
                    value: Operand::ConstInt(0),
                });
                self.current_block_data.value_types.insert(v, SsaType::Null);
            }
        }

        v
    }

    fn unwrap_known_nonnull(&mut self, val: Value, ty: &SsaType) -> Value {
        if let Some(pointee) = ty.nullable_pointer_repr() {
            // Pointer-optimized nullable: the bits are already exactly the
            // pointer we want, just re-typed without the Nullable wrapper.
            let pointee = pointee.clone();
            let dest = self.current_block_data.fresh_value();
            let ptr_ty = SsaType::Pointer(Box::new(pointee));
            self.emit(Instruction::Cast {
                dest,
                value: Operand::Value(val),
                kind: cast_kind(&ptr_ty, &ptr_ty),
            });
            self.current_block_data.value_types.insert(dest, ptr_ty);
            dest
        } else if let SsaType::Nullable(inner) = ty {
            // Tag-based nullable: load the payload out from behind the tag byte.
            let payload_align =
                alignof_ssa(inner, TargetInfo { ptr_bytes: 8 }).unwrap_or_else(|e| {
                    panic!("failed to compute alignment for nullable payload: {:?}", e)
                });
            let payload_offset = round_up_to_align(1, payload_align);

            let payload = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: payload,
                base: Operand::Value(val),
                offset: payload_offset,
            });
            self.current_block_data
                .value_types
                .insert(payload, (**inner).clone());
            payload
        } else {
            val
        }
    }

    fn lower_nullable_unwrap(&mut self, val: Value, else_stmts: &HirStmt<'a, 'bump>) -> Value {
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
        self.scope_stack.push(DropScope { locals: Vec::new() });
        for s in *body {
            self.lower_stmt(s);
        }
        self.scope_stack.pop();
        if !body.last().map_or(false, Self::stmt_diverges) {
            panic!("`? else` block must end in return, throw, break, or continue");
        }

        self.current_block_data.switch_to(then_bb);
        self.unwrap_known_nonnull(val, &ty)
    }

    fn narrow_nonnull(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        branch_is_true: bool,
    ) -> Option<(StrId, Value, Value)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };

        let name = match (left, right) {
            // `x == null` / `x != null`: narrows in whichever branch the
            // non-null case holds -- true branch for `!=`, false branch for `==`.
            (HirExpr::Ident(n, _), HirExpr::Null(_)) | (HirExpr::Null(_), HirExpr::Ident(n, _)) => {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                *n
            }
            // `x == <non-null value>` (nullable equality): only the *true*
            // branch implies non-null; `x != value` proves nothing, since
            // `x == null` also satisfies it.
            (HirExpr::Ident(n, _), other) if !matches!(other, HirExpr::Null(_)) => {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                *n
            }
            (other, HirExpr::Ident(n, _)) if !matches!(other, HirExpr::Null(_)) => {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                *n
            }
            _ => return None,
        };

        let cur = *self.var_map.get(&name)?;
        let ty = self.current_block_data.value_type(cur)?.clone();
        if !matches!(ty, SsaType::Nullable(_)) {
            return None;
        }
        let unwrapped = self.unwrap_known_nonnull(cur, &ty);
        self.var_map.insert(name, unwrapped);
        Some((name, cur, unwrapped))
    }

    fn narrow_nonnull_path(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        branch_is_true: bool,
    ) -> Option<(StrId, Vec<StrId>)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };

        let is_field_access =
            |e: &HirExpr<'a, 'bump>| matches!(e, HirExpr::FieldAccess { .. } | HirExpr::Get { .. });

        let target: &HirExpr<'a, 'bump> =
            if matches!(right, HirExpr::Null(_)) && is_field_access(left) {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                left
            } else if matches!(left, HirExpr::Null(_)) && is_field_access(right) {
                let holds_when_nonnull = match op {
                    Operator::NotEquals => branch_is_true,
                    Operator::Equals => !branch_is_true,
                    _ => return None,
                };
                if !holds_when_nonnull {
                    return None;
                }
                right
            } else if is_field_access(left) && !matches!(right, HirExpr::Null(_)) {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                left
            } else if is_field_access(right) && !matches!(left, HirExpr::Null(_)) {
                if !matches!(op, Operator::Equals) || !branch_is_true {
                    return None;
                }
                right
            } else {
                return None;
            };

        let (HirExpr::FieldAccess {
            object,
            field,
            span,
        }
        | HirExpr::Get {
            object,
            field,
            span,
        }) = target
        else {
            unreachable!()
        };

        let (addr, ty) = self.lower_field_addr(object, *field, span);
        let val = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: val,
            ptr: Operand::Value(addr),
        });
        self.current_block_data.value_types.insert(val, ty.clone());
        if !matches!(ty, SsaType::Nullable(_)) {
            return None;
        }

        let unwrapped = self.unwrap_known_nonnull(val, &ty);
        let (root, mut path) = self.static_field_path_mir(object)?;
        path.push(*field);
        self.narrowed_fields.insert((root, path.clone()), unwrapped);
        Some((root, path))
    }

    fn static_field_path_mir(&self, expr: &HirExpr<'a, 'bump>) -> Option<(StrId, Vec<StrId>)> {
        match expr {
            HirExpr::Ident(name, _) => Some((*name, Vec::new())),
            HirExpr::This { .. } => Some((StrId::from_static("this"), Vec::new())),
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let (root, mut path) = self.static_field_path_mir(object)?;
                path.push(*field);
                Some((root, path))
            }
            _ => None,
        }
    }

    fn classify_indexed_container(ty: &SsaType) -> Option<IndexedContainer> {
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

    fn emit_indexed_element_drop(
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

    /// Seed `var_map` and `current_block_data.value_types` with known parameters and locals.
    ///
    /// This should be called before lowering a function body. It associates each parameter
    /// and local with a fresh `Value` and registers its SSA type so later lowering can
    /// rely on lookups without panicking.
    pub fn seed_locals_and_params_from_hir(
        &mut self,
        params: &[(StrId, HirType<'a, 'bump>)],
        locals: &[(StrId, HirType<'a, 'bump>)],
    ) {
        for (name, ty) in params.iter().copied() {
            let v = self.current_block_data.fresh_value();
            self.var_map.insert(name.clone(), v);
            self.current_block_data
                .value_types
                .insert(v, lower_type_hir(&ty, self.enums));
        }

        for (name, ty) in locals.iter().copied() {
            let v = self.current_block_data.fresh_value();
            self.var_map.insert(name.clone(), v);
            self.current_block_data
                .value_types
                .insert(v, lower_type_hir(&ty, self.enums));
        }
    }

    fn lower_expr_expected(&mut self, expr: &HirExpr<'a, 'bump>, expected: &SsaType) -> Value {
        match expr {
            HirExpr::Number(n, _) if expected.is_integer() => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: v,
                    ty: expected.clone(),
                    value: Operand::ConstInt(*n),
                });
                self.current_block_data
                    .value_types
                    .insert(v, expected.clone());
                v
            }

            HirExpr::Binary {
                left, op, right, ..
            } => self.lower_expr_binary_expected(left, op, right, Some(expected)),

            HirExpr::Match {
                expr: scrutinee,
                arms,
                span,
            } => self.lower_match_expr_inner(scrutinee, arms, *span, Some(expected)),

            HirExpr::If { if_stmt, .. } => {
                let HirStmt::If {
                    cond,
                    then_block,
                    else_block,
                    span,
                } = *if_stmt
                else {
                    unreachable!("HirExpr::If must always wrap HirStmt::If")
                };
                self.lower_if_expr_inner(&cond, then_block, *else_block, *span, Some(expected))
            }

            HirExpr::Block { body, .. } => self.lower_block_value_inner(body, Some(expected)),

            _ => self.lower_expr(expr),
        }
    }

    pub fn lower_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
        match expr {
            HirExpr::Null(_) => self.lower_expr_null(),
            HirExpr::Number(n, _) => self.lower_expr_number(*n),

            HirExpr::Binary {
                left,
                op,
                right,
                span: _,
            } => self.lower_expr_binary(left, op, right),

            HirExpr::Ident(name, span) => {
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
                } else if let Some(func) =
                    self.funcs.get(name).or_else(|| self.global_funcs.get(name))
                {
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
            HirExpr::StructInit {
                name,
                args,
                span,
                type_args: _,
            } => self.lower_struct_init(name, args, *span),

            HirExpr::Undefined { span: _, ty } => {
                let ssa_ty = lower_type_hir(ty, self.enums);
                self.lower_zeroed_value(&ssa_ty)
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
            } => self.lower_field_access(object, *field, *span),

            HirExpr::Call {
                callee,
                args,
                span: _,
                type_args: _, // Turns into None after monomorphization
            } => self.lower_call(callee, args),

            HirExpr::InterfaceCall {
                callee,
                args,
                interface,
                span: _,
            } => self.lower_interface_call(callee, args, *interface),

            HirExpr::Assignment {
                target,
                op,
                value,
                span: _,
            } => self.lower_expr_assignment(target, *op, value),

            HirExpr::String(s, _) => {
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

            HirExpr::Boolean(b, _) => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: v,
                    ty: SsaType::I8,
                    value: Operand::ConstInt(if *b { 1 } else { 0 }),
                });
                self.current_block_data.value_types.insert(v, SsaType::I8);
                v
            }

            HirExpr::Decimal(d, _) => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: v,
                    ty: SsaType::F64,
                    value: Operand::ConstFloat(*d),
                });
                self.current_block_data.value_types.insert(v, SsaType::F64);
                v
            }

            HirExpr::Tuple(elements, _) => {
                if elements.is_empty() {
                    let v = self.current_block_data.fresh_value();
                    self.current_block_data.value_types.insert(v, SsaType::I64);
                    v
                } else {
                    self.lower_expr(&elements[0])
                }
            }

            HirExpr::InterpolatedString(_parts) => {
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

            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                span,
                type_args: _,
            } => self.lower_enum_init(enum_name, variant, args, span),

            HirExpr::ExprList { list, span: _ } => {
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

            HirExpr::Comparison {
                left,
                op,
                right,
                span: _,
            } => {
                if matches!(op, Operator::Equals | Operator::NotEquals) {
                    let (l_expr, r_expr): (&HirExpr<'a, 'bump>, &HirExpr<'a, 'bump>) =
                        (left, right);
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
                if matches!(op, Operator::Equals | Operator::NotEquals) && l_ty.is_tagged_nullable()
                {
                    return self.lower_tagged_nullable_eq(
                        l,
                        &l_ty,
                        right,
                        matches!(op, Operator::Equals),
                    );
                }
                let r = self.lower_expr_expected(right, &l_ty);
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: v,
                    op: lower_operator_bin(op),
                    left: Operand::Value(l),
                    right: Operand::Value(r),
                });
                self.current_block_data.value_types.insert(v, SsaType::Bool);
                v
            }

            HirExpr::This { .. } => {
                let this_name = StrId::from_static("this");
                let v = *self.var_map.get(&this_name).unwrap();

                v
            }
            HirExpr::Ref { expr, span, .. } => {
                let (addr, _pointee_ty) = self.lower_place_addr(expr, span);
                addr
            }

            HirExpr::Deref { expr, .. } => {
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
            HirExpr::ModuleAccess(hir_module_access) => {
                if let Some(v) = self.try_lower_bare_enum_variant(
                    &hir_module_access.member,
                    &hir_module_access.member,
                ) {
                    return v;
                }
                for path_seg in hir_module_access.path.iter().rev() {
                    if let Some(v) =
                        self.try_lower_bare_enum_variant(path_seg, &hir_module_access.member)
                    {
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
            HirExpr::Lambda { .. } => {
                unreachable!("There should be no lambdas here")
            }
            HirExpr::Index {
                object,
                index,
                span: _,
            } => self.lower_index(object, index),
            HirExpr::ArrayLiteral { elements, span: _ } => self.lower_array_literal(elements),
            HirExpr::GenericIdent(..) => unreachable!(),
            HirExpr::Cast {
                expr, target_type, ..
            } => {
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
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => {
                use ir::hir::IntrinsicKind;
                use ir::ssa_ir::IntrinsicOp;

                match kind {
                    IntrinsicKind::Replace => {
                        let place_expr: &HirExpr<'a, 'bump> = match &args[0] {
                            HirExpr::Ref { expr, .. } => expr,
                            other => other,
                        };
                        self.lower_replace(place_expr, &args[1], span)
                    }
                    IntrinsicKind::Reinterpret => {
                        let src = self.lower_expr(&args[0]);
                        let src_ty = self
                            .current_block_data
                            .value_types
                            .get(&src)
                            .cloned()
                            .expect("$reinterpret: source value has no known type");
                        let target_ty = lower_type_hir(&type_args[0], self.enums);

                        if src_ty == target_ty {
                            src
                        } else {
                            let kind = cast_kind(&src_ty, &target_ty);
                            let dest = self.current_block_data.fresh_value();
                            self.emit(Instruction::Cast {
                                dest,
                                value: Operand::Value(src),
                                kind,
                            });
                            self.current_block_data.value_types.insert(dest, target_ty);
                            dest
                        }
                    }
                    IntrinsicKind::Unreachable => {
                        let msg = self.current_block_data.fresh_value();
                        let msg_str = self
                            .context
                            .thread_local()
                            .intern("entered unreachable code");
                        self.emit(Instruction::Const {
                            dest: msg,
                            ty: SsaType::String,
                            value: Operand::ConstString(StrId(msg_str)),
                        });
                        self.current_block_data
                            .value_types
                            .insert(msg, SsaType::String);

                        self.emit_debug_panic(msg);

                        let dest = self.current_block_data.fresh_value();
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::Void);
                        dest
                    }
                    IntrinsicKind::SizeOf | IntrinsicKind::AlignOf | IntrinsicKind::TypeName => {
                        let query_ty = lower_type_hir(&type_args[0], self.enums);
                        let op = match kind {
                            IntrinsicKind::SizeOf => IntrinsicOp::SizeOf,
                            IntrinsicKind::AlignOf => IntrinsicOp::AlignOf,
                            IntrinsicKind::TypeName => IntrinsicOp::TypeName,
                            _ => unreachable!(),
                        };

                        let dest = self.current_block_data.fresh_value();
                        self.emit(Instruction::Intrinsic {
                            dest: Some(dest),
                            op,
                            query_ty: Some(query_ty),
                            args: SmallVec::new(),
                        });

                        let result_ty = match kind {
                            IntrinsicKind::SizeOf | IntrinsicKind::AlignOf => SsaType::Usize,
                            IntrinsicKind::TypeName => SsaType::String,
                            _ => unreachable!(),
                        };
                        self.current_block_data.value_types.insert(dest, result_ty);
                        dest
                    }

                    IntrinsicKind::AssertAlign => {
                        let ptr_val = self.lower_expr(&args[0]);
                        let align_val = self.lower_expr(&args[1]);

                        // mask = align - 1; misaligned if (ptr & mask) != 0
                        let one = self.current_block_data.fresh_value();
                        self.emit(Instruction::Const {
                            dest: one,
                            ty: SsaType::Usize,
                            value: Operand::ConstInt(1),
                        });
                        self.current_block_data
                            .value_types
                            .insert(one, SsaType::Usize);

                        let mask = self.current_block_data.fresh_value();
                        self.emit(Instruction::Binary {
                            dest: mask,
                            op: BinOp::Sub,
                            left: Operand::Value(align_val),
                            right: Operand::Value(one),
                        });
                        self.current_block_data
                            .value_types
                            .insert(mask, SsaType::Usize);

                        let masked = self.current_block_data.fresh_value();
                        self.emit(Instruction::Binary {
                            dest: masked,
                            op: BinOp::BitAnd,
                            left: Operand::Value(ptr_val),
                            right: Operand::Value(mask),
                        });
                        self.current_block_data
                            .value_types
                            .insert(masked, SsaType::Usize);

                        let zero = self.current_block_data.fresh_value();
                        self.emit(Instruction::Const {
                            dest: zero,
                            ty: SsaType::Usize,
                            value: Operand::ConstInt(0),
                        });
                        self.current_block_data
                            .value_types
                            .insert(zero, SsaType::Usize);

                        let is_misaligned = self.current_block_data.fresh_value();
                        self.emit(Instruction::Binary {
                            dest: is_misaligned,
                            op: BinOp::Ne,
                            left: Operand::Value(masked),
                            right: Operand::Value(zero),
                        });
                        self.current_block_data
                            .value_types
                            .insert(is_misaligned, SsaType::Bool);

                        let panic_bb = self.current_block_data.new_block();
                        let cont_bb = self.current_block_data.new_block();

                        self.emit(Instruction::Branch {
                            cond: Operand::Value(is_misaligned),
                            then_bb: panic_bb,
                            else_bb: cont_bb,
                        });

                        self.current_block_data.switch_to(panic_bb);
                        let msg = self.current_block_data.fresh_value();
                        let msg_str = self
                            .context
                            .thread_local()
                            .intern("alignment assertion failed");
                        self.emit(Instruction::Const {
                            dest: msg,
                            ty: SsaType::String,
                            value: Operand::ConstString(StrId(msg_str)),
                        });
                        self.current_block_data
                            .value_types
                            .insert(msg, SsaType::String);
                        self.emit_debug_panic(msg);

                        self.current_block_data.switch_to(cont_bb);
                        let dest = self.current_block_data.fresh_value();
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::Void);
                        dest
                    }

                    IntrinsicKind::Own => {
                        let ptr_val = self.lower_expr(&args[0]);
                        let ptr_ty = self
                            .current_block_data
                            .value_types
                            .get(&ptr_val)
                            .cloned()
                            .expect("$own: pointer arg has no known type");
                        let pointee_ty = match &ptr_ty {
                            SsaType::Pointer(inner) => (**inner).clone(),
                            other => {
                                panic!("$own: expected pointer-typed first arg, got {:?}", other)
                            }
                        };

                        let len_cap_exprs = if args.len() == 4 {
                            Some((&args[2], &args[3]))
                        } else {
                            None
                        };

                        match len_cap_exprs {
                            // Owned slice: {ptr, len, cap} fat pointer, 24 bytes.
                            Some((len_expr, cap_expr)) => {
                                let len_val = self.lower_expr(len_expr);
                                let cap_val = self.lower_expr(cap_expr);

                                let fat_ptr = self.current_block_data.fresh_value();
                                let fat_ptr_layout_ty = SsaType::Tuple(vec![
                                    SsaType::Pointer(Box::new(pointee_ty.clone())),
                                    SsaType::Usize, // len
                                    SsaType::Usize, // cap
                                ]);
                                self.emit(Instruction::StackAlloc {
                                    dest: fat_ptr,
                                    ty: fat_ptr_layout_ty,
                                    count: 1,
                                });

                                let slice_ty =
                                    SsaType::Owned(Box::new(SsaType::Slice(Box::new(pointee_ty))));
                                self.current_block_data
                                    .value_types
                                    .insert(fat_ptr, slice_ty);

                                self.emit(Instruction::StoreField {
                                    base: Operand::Value(fat_ptr),
                                    offset: 0,
                                    value: Operand::Value(ptr_val),
                                });
                                self.emit(Instruction::StoreField {
                                    base: Operand::Value(fat_ptr),
                                    offset: 8,
                                    value: Operand::Value(len_val),
                                });
                                self.emit(Instruction::StoreField {
                                    base: Operand::Value(fat_ptr),
                                    offset: 16,
                                    value: Operand::Value(cap_val),
                                });

                                fat_ptr
                            }

                            None => {
                                let owned_ty = SsaType::Owned(Box::new(pointee_ty));
                                self.current_block_data
                                    .value_types
                                    .insert(ptr_val, owned_ty);
                                ptr_val
                            }
                        }
                    }
                    IntrinsicKind::AtomicCasU32 => {
                        let ptr_val = self.lower_expr(&args[0]);
                        let expected_val = self.lower_expr_as_u32(&args[1]);
                        let new_val = self.lower_expr_as_u32(&args[2]);

                        let dest = self.current_block_data.fresh_value();
                        self.emit(Instruction::Intrinsic {
                            dest: Some(dest),
                            op: IntrinsicOp::AtomicCasU32,
                            query_ty: None,
                            args: smallvec![
                                Operand::Value(ptr_val),
                                Operand::Value(expected_val),
                                Operand::Value(new_val)
                            ],
                        });
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::U32);
                        dest
                    }

                    IntrinsicKind::AtomicLoadU32 => {
                        let ptr_val = self.lower_expr(&args[0]);

                        let dest = self.current_block_data.fresh_value();
                        self.emit(Instruction::Intrinsic {
                            dest: Some(dest),
                            op: IntrinsicOp::AtomicLoadU32,
                            query_ty: None,
                            args: smallvec![Operand::Value(ptr_val)],
                        });
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::U32);
                        dest
                    }
                    IntrinsicKind::AtomicStoreU32 => {
                        let ptr_val = self.lower_expr(&args[0]);
                        let val_val = self.lower_expr_as_u32(&args[1]);

                        self.emit(Instruction::Intrinsic {
                            dest: None,
                            op: IntrinsicOp::AtomicStoreU32,
                            query_ty: None,
                            args: smallvec![Operand::Value(ptr_val), Operand::Value(val_val)],
                        });

                        let dest = self.current_block_data.fresh_value();
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::Void);
                        dest
                    }

                    IntrinsicKind::CpuRelax => {
                        self.emit(Instruction::Intrinsic {
                            dest: None,
                            op: IntrinsicOp::CpuRelax,
                            query_ty: None,
                            args: SmallVec::new(),
                        });

                        let dest = self.current_block_data.fresh_value();
                        self.current_block_data
                            .value_types
                            .insert(dest, SsaType::Void);
                        dest
                    }
                }
            }
            HirExpr::Block { body, .. } => self.lower_block_value(body),
            HirExpr::Match { expr, arms, span } => self.lower_match_expr(expr, arms, *span),
            HirExpr::Range { start, end, .. } => self.lower_range_expr(start, end),
            HirExpr::Slice {
                object,
                start,
                end,
                inclusive,
                ..
            } => self.lower_slice_expr(object, start, end, *inclusive),
            HirExpr::UnknownIntrinsic { span, name } => unreachable!("span {span} name {name}"),
            HirExpr::If { if_stmt, span: _ } => {
                let HirStmt::If {
                    cond,
                    then_block,
                    else_block,
                    span,
                } = *if_stmt
                else {
                    unreachable!("HirExpr::If must always wrap HirStmt::If")
                };
                self.lower_if_expr(&cond, then_block, *else_block, *span)
            }
            HirExpr::Char(c, _) => {
                let v = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest: v,
                    ty: SsaType::Char,
                    value: Operand::ConstInt(*c as i64),
                });
                self.current_block_data.value_types.insert(v, SsaType::Char);
                v
            }
            HirExpr::Uninit { span: _, ty } => {
                let ssa_ty = lower_type_hir(ty, self.enums);
                self.lower_uninit_value(&ssa_ty)
            }
        }
    }

    fn lower_uninit_value(&mut self, ssa_ty: &SsaType) -> Value {
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

    fn lower_index_base_len(
        &mut self,
        object: &HirExpr<'a, 'bump>,
    ) -> (Value, SsaType, Option<Operand>) {
        let base = self.lower_expr(object);
        self.split_indexable(base)
    }

    fn split_indexable(&mut self, base: Value) -> (Value, SsaType, Option<Operand>) {
        let base_ty = self
            .current_block_data
            .value_types
            .get(&base)
            .cloned()
            .expect("split_indexable: base value has no known type");

        match &base_ty {
            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Slice(_)) => {
                let elem_inner = match inner.as_ref() {
                    SsaType::Slice(e) => (**e).clone(),
                    _ => unreachable!(),
                };
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new(elem_inner.clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, elem_inner, Some(Operand::Value(len_v)))
            }

            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Owned(_)) => {
                let SsaType::Owned(slice_inner) = inner.as_ref() else {
                    unreachable!()
                };
                let SsaType::Slice(elem_ty) = slice_inner.as_ref() else {
                    panic!(
                        "lower_index_base_len: Pointer(Owned(_)) base whose inner Owned isn't a Slice: {:?}",
                        inner
                    );
                };
                let elem_inner = (**elem_ty).clone();
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new(elem_inner.clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, elem_inner, Some(Operand::Value(len_v)))
            }

            SsaType::Pointer(inner) if matches!(inner.as_ref(), SsaType::Array(_, _)) => {
                let SsaType::Array(elem, len) = inner.as_ref() else {
                    unreachable!()
                };
                (base, (**elem).clone(), Some(Operand::ConstInt(*len as i64)))
            }

            SsaType::Pointer(inner) => (base, (**inner).clone(), None),

            SsaType::Array(inner, len) => (
                base,
                (**inner).clone(),
                Some(Operand::ConstInt(*len as i64)),
            ),

            SsaType::Slice(inner) => {
                let data_ptr = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: data_ptr,
                    base: Operand::Value(base),
                    offset: 0,
                });
                self.current_block_data
                    .value_types
                    .insert(data_ptr, SsaType::Pointer(Box::new((**inner).clone())));
                let len_v = self.current_block_data.fresh_value();
                self.emit(Instruction::LoadField {
                    dest: len_v,
                    base: Operand::Value(base),
                    offset: 8,
                });
                self.current_block_data
                    .value_types
                    .insert(len_v, SsaType::Usize);
                (data_ptr, (**inner).clone(), Some(Operand::Value(len_v)))
            }

            SsaType::Owned(inner) => match inner.as_ref() {
                SsaType::Slice(inner) => {
                    let data_ptr = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: data_ptr,
                        base: Operand::Value(base),
                        offset: 0,
                    });
                    self.current_block_data
                        .value_types
                        .insert(data_ptr, SsaType::Pointer(Box::new((**inner).clone())));
                    let len_v = self.current_block_data.fresh_value();
                    self.emit(Instruction::LoadField {
                        dest: len_v,
                        base: Operand::Value(base),
                        offset: 8,
                    });
                    self.current_block_data
                        .value_types
                        .insert(len_v, SsaType::Usize);
                    (data_ptr, (**inner).clone(), Some(Operand::Value(len_v)))
                }
                _ => panic!("[lower_index_base_len] cannot index into {:?}", inner),
            },

            other => panic!("[lower_index_base_len] cannot index into {:?}", other),
        }
    }

    fn emit_bounds_check(&mut self, idx: Operand, len: Operand, inclusive_upper: bool) {
        let cond = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: cond,
            op: if inclusive_upper {
                BinOp::Le
            } else {
                BinOp::Lt
            },
            left: idx,
            right: len,
        });
        self.current_block_data
            .value_types
            .insert(cond, SsaType::Bool);

        let ok_bb = self.current_block_data.new_block();
        let panic_bb = self.current_block_data.new_block();
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb: ok_bb,
            else_bb: panic_bb,
        });

        self.current_block_data.switch_to(panic_bb);
        let msg = self.current_block_data.fresh_value();
        let msg_str = StrId::from_static("index out of bounds");
        self.emit(Instruction::Const {
            dest: msg,
            ty: SsaType::String,
            value: Operand::ConstString(msg_str),
        });
        self.current_block_data
            .value_types
            .insert(msg, SsaType::String);
        self.emit_debug_panic(msg);

        self.current_block_data.switch_to(ok_bb);
    }

    fn align_up(value: usize, align: usize) -> usize {
        if align == 0 {
            return value;
        }
        (value + align - 1) & !(align - 1)
    }

    fn try_lower_bare_enum_variant(&mut self, enum_name: &StrId, variant: &StrId) -> Option<Value> {
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

    fn lower_enum_init(
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

    fn is_aggregate_ssa_type(ty: &SsaType) -> bool {
        matches!(
            ty,
            SsaType::User(..)
                | SsaType::Enum { .. }
                | SsaType::Tuple(_)
                | SsaType::Array(..)
                | SsaType::Slice(_)
                | SsaType::Owned(_)
        )
    }

    fn lower_range_expr(&mut self, start: &HirExpr<'a, 'bump>, end: &HirExpr<'a, 'bump>) -> Value {
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

    fn lower_null_comparison(&mut self, operand: &HirExpr<'a, 'bump>, is_eq: bool) -> Value {
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

    fn slice_kind(&self, ty: &SsaType) -> Option<bool> {
        match ty {
            SsaType::Pointer(inner) => self.slice_kind(inner),
            SsaType::Owned(inner) => match inner.as_ref() {
                SsaType::Slice(_) => Some(true),
                _ => self.slice_kind(inner).map(|_| true),
            },
            SsaType::Slice(_) => Some(false),
            _ => None,
        }
    }

    fn resolve_slice_pseudo_field(&self, ty: &SsaType, field: StrId) -> Option<(usize, SsaType)> {
        let is_owned = self.slice_kind(ty)?;
        match self.context.resolve_string(&field) {
            "len" => Some((8, SsaType::Usize)),
            "cap" if is_owned => Some((16, SsaType::Usize)),
            _ => None,
        }
    }

    fn lower_short_circuit_and(
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

        // RHS
        self.current_block_data.switch_to(rhs_bb);
        let rhs = self.lower_expr(right);
        self.emit(Instruction::Jump { target: merge_bb });

        // FALSE
        self.current_block_data.switch_to(false_bb);
        let false_val = self.current_block_data.fresh_value();

        self.emit(Instruction::Const {
            dest: false_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(false),
        });

        self.emit(Instruction::Jump { target: merge_bb });

        // MERGE
        self.current_block_data.switch_to(merge_bb);

        let result = self.current_block_data.fresh_value();

        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(rhs_bb, rhs), (false_bb, false_val),],
        });

        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    fn lower_short_circuit_or(
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

        // TRUE
        self.current_block_data.switch_to(true_bb);

        let true_val = self.current_block_data.fresh_value();

        self.emit(Instruction::Const {
            dest: true_val,
            ty: SsaType::Bool,
            value: Operand::ConstBool(true),
        });

        self.emit(Instruction::Jump { target: merge_bb });

        // RHS
        self.current_block_data.switch_to(rhs_bb);

        let rhs = self.lower_expr(right);

        self.emit(Instruction::Jump { target: merge_bb });

        // MERGE
        self.current_block_data.switch_to(merge_bb);

        let result = self.current_block_data.fresh_value();

        self.emit(Instruction::Phi {
            dest: result,
            incoming: smallvec![(true_bb, true_val), (rhs_bb, rhs),],
        });

        self.current_block_data
            .value_types
            .insert(result, SsaType::Bool);

        result
    }

    fn lower_replace(
        &mut self,
        place: &HirExpr<'a, 'bump>,
        new_expr: &HirExpr<'a, 'bump>,
        span: &SourceSpan<'a>,
    ) -> Value {
        self.record_move_if_any(new_expr);

        // Plain local: SSA rebind. Marking it moved first makes handle_ident skip
        // the drop of the old value (it now belongs to the caller) and re-init it.
        if let HirExpr::Ident(name, ident_span) = place {
            let rhs = self.lower_expr(new_expr);
            let old = self.lower_expr(place);
            self.record_move_if_any(place);
            self.handle_ident(AssignmentOperator::Assign, rhs, *name, *ident_span);
            return old;
        }

        let (addr, ty) = self.lower_place_addr(place, span);
        let init = self.lower_init_operand(new_expr, &ty);

        let inline = matches!(
            ty,
            SsaType::User(..)
                | SsaType::Enum { .. }
                | SsaType::Tuple(_)
                | SsaType::Array(..)
                | SsaType::Slice(_)
        ) || matches!(&ty, SsaType::Owned(i) if matches!(i.as_ref(), SsaType::Slice(_)))
            || ty.is_tagged_nullable();

        let old = if inline {
            // Slot is overwritten in place, so snapshot the old bytes first.
            let size = ir::layout::sizeof_ssa(&ty, TargetInfo { ptr_bytes: 8 })
                .expect("$replace: slot type has no known size");
            let (alloc_ty, count) = match &ty {
                SsaType::Array(inner, len) => ((**inner).clone(), *len),
                _ => (ty.clone(), 1),
            };
            let tmp = self.new_value();
            self.emit(Instruction::StackAlloc {
                dest: tmp,
                ty: alloc_ty,
                count,
            });
            self.current_block_data.value_types.insert(tmp, ty.clone());

            let n = self.new_value();
            self.emit(Instruction::Const {
                dest: n,
                ty: SsaType::Usize,
                value: Operand::ConstInt(size as i64),
            });
            self.current_block_data
                .value_types
                .insert(n, SsaType::Usize);
            self.emit_memcpy(tmp, addr, n);
            tmp
        } else {
            let v = self.new_value();
            self.emit(Instruction::Load {
                dest: v,
                ptr: Operand::Value(addr),
            });
            self.current_block_data.value_types.insert(v, ty.clone());
            v
        };

        // Deliberately no drop of the old contents: ownership moved into `old`.
        self.store_init(addr, 0, &ty, init);

        match self.static_field_path_mir(place) {
            Some((root, path)) => self
                .narrowed_fields
                .retain(|(r, p), _| !(*r == root && p.starts_with(&path))),
            None => self.narrowed_fields.clear(),
        }
        if let HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } =
            place
        {
            let owner = match &**object {
                HirExpr::This { .. } => Some(StrId::from_static("this")),
                HirExpr::Ident(root, _) => Some(*root),
                _ => None,
            };
            if let Some(o) = owner {
                self.drop_state.mark_field_initialized(o, *field);
            }
        }

        old
    }

    fn null_compatible_with(ty: &SsaType) -> bool {
        match ty {
            SsaType::Pointer(_) => true,
            SsaType::Nullable(inner) => inner.is_pointer(),
            _ => false,
        }
    }

    fn reconcile_phi_type(&self, incoming: &[(BlockId, Value)], span: SourceSpan<'a>) -> SsaType {
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

    fn lower_if_expr(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_if_expr_inner(condition, then_block, else_block, span, None)
    }

    fn lower_if_expr_inner(
        &mut self,
        condition: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
        expected: Option<&SsaType>,
    ) -> Value {
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
        let then_narrowed = self.narrow_nonnull(condition, true);
        let then_narrowed_path = self.narrow_nonnull_path(condition, true);
        let then_val = self.lower_block_value_inner(then_block, expected);
        if let Some((name, prev, narrowed_val)) = then_narrowed {
            if self.var_map.get(&name).copied() == Some(narrowed_val) {
                self.var_map.insert(name, prev);
            }
        }
        if let Some((root, path)) = then_narrowed_path {
            self.narrowed_fields.remove(&(root, path));
        }
        let then_end = self.current_block_data.current_block;
        let then_terminated = self.block_terminated();
        if !then_terminated {
            self.emit(Instruction::Jump { target: merge_bb });
        }

        // else
        self.current_block_data.switch_to(else_bb);
        self.narrowed_fields = narrowed_before.clone();
        let else_narrowed = self.narrow_nonnull(condition, false);
        let else_narrowed_path = self.narrow_nonnull_path(condition, false);
        let else_val = match else_block {
            Some(HirStmt::Block { body, span: _ }) => self.lower_block_value_inner(body, expected),
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
        let else_end = self.current_block_data.current_block;
        let else_terminated = self.block_terminated();
        if !else_terminated {
            self.emit(Instruction::Jump { target: merge_bb });
        }

        // both arms diverged, nothing reaches merge_bb, so there's no real value
        if then_terminated && else_terminated {
            return self.unreachable_value();
        }

        self.narrowed_fields = narrowed_before;
        self.current_block_data.switch_to(merge_bb);

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

    fn lower_block_value(&mut self, stmts: &[HirStmt<'a, 'bump>]) -> Value {
        self.lower_block_value_inner(stmts, None)
    }

    fn auto_unwrap_receiver(&mut self, val: Value) -> Value {
        let Some(ty) = self.current_block_data.value_types.get(&val).cloned() else {
            return val;
        };
        match &ty {
            SsaType::Nullable(inner)
                if matches!(
                    inner.as_ref(),
                    SsaType::Pointer(_) | SsaType::Owned(_) | SsaType::User(_, _)
                ) =>
            {
                self.unwrap_known_nonnull(val, &ty)
            }
            _ => val,
        }
    }

    fn lower_block_value_inner(
        &mut self,
        stmts: &[HirStmt<'a, 'bump>],
        expected: Option<&SsaType>,
    ) -> Value {
        if stmts.is_empty() {
            return self.unit_value();
        }
        let (last, rest) = stmts.split_last().unwrap();
        for stmt in rest {
            self.lower_stmt(stmt);
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

    fn block_terminated(&mut self) -> bool {
        self.current_block_data
            .bb()
            .instructions
            .last()
            .map_or(false, |i| ir::ssa_ir::inst_is_terminator(i))
    }

    fn unit_value(&mut self) -> Value {
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

    fn unreachable_value(&mut self) -> Value {
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

    fn value_type(&self, v: Value) -> Option<&SsaType> {
        self.current_block_data.value_types.get(&v)
    }

    fn lower_index_addr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        index: &HirExpr<'a, 'bump>,
    ) -> (Value, SsaType) {
        let idx = self.lower_expr(index);
        let (base_ptr, elem_ty, len) = self.lower_index_base_len(object);

        if let Some(len_operand) = len {
            self.emit_bounds_check(Operand::Value(idx), len_operand, false);
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

        (addr_v, elem_ty)
    }

    fn lower_slice_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        start: &HirExpr<'a, 'bump>,
        end: &HirExpr<'a, 'bump>,
        inclusive: bool,
    ) -> Value {
        let (base_addr, elem_ty, len) = self.lower_index_base_len(object);
        let start_v = self.lower_expr(start);
        let end_v = self.lower_expr(end);

        // Fold `inclusive` into `end_v` up front so bound-checking and length
        // computation both just work in exclusive-end terms afterward.
        let end_v = if inclusive {
            let one = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: one,
                ty: SsaType::Usize,
                value: Operand::ConstInt(1),
            });
            self.current_block_data
                .value_types
                .insert(one, SsaType::Usize);
            let bumped = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: bumped,
                op: BinOp::Add,
                left: Operand::Value(end_v),
                right: Operand::Value(one),
            });
            self.current_block_data
                .value_types
                .insert(bumped, SsaType::Usize);
            bumped
        } else {
            end_v
        };

        self.emit_bounds_check(Operand::Value(start_v), Operand::Value(end_v), true); // start <= end
        if let Some(len_operand) = len {
            self.emit_bounds_check(Operand::Value(end_v), len_operand, true); // end <= len
        }

        let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
            .expect("slice element has no known size") as i64;

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
            left: Operand::Value(start_v),
            right: Operand::Value(size_v),
        });
        self.current_block_data
            .value_types
            .insert(offset_v, SsaType::I64);

        let ptr_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: ptr_v,
            op: BinOp::Add,
            left: Operand::Value(base_addr),
            right: Operand::Value(offset_v),
        });
        self.current_block_data
            .value_types
            .insert(ptr_v, SsaType::Pointer(Box::new(elem_ty.clone())));

        let len_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: len_v,
            op: BinOp::Sub,
            left: Operand::Value(end_v),
            right: Operand::Value(start_v),
        });
        self.current_block_data
            .value_types
            .insert(len_v, SsaType::Usize);

        let fat_ptr = self.current_block_data.fresh_value();
        let fat_ty = SsaType::Tuple(vec![
            SsaType::Pointer(Box::new(elem_ty.clone())),
            SsaType::Usize,
        ]);
        self.emit(Instruction::StackAlloc {
            dest: fat_ptr,
            ty: fat_ty,
            count: 1,
        });
        self.current_block_data
            .value_types
            .insert(fat_ptr, SsaType::Slice(Box::new(elem_ty)));

        self.emit(Instruction::StoreField {
            base: Operand::Value(fat_ptr),
            offset: 0,
            value: Operand::Value(ptr_v),
        });
        self.emit(Instruction::StoreField {
            base: Operand::Value(fat_ptr),
            offset: 8,
            value: Operand::Value(len_v),
        });
        fat_ptr
    }

    fn lower_match_expr(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        self.lower_match_expr_inner(scrutinee, arms, span, None)
    }

    fn lower_match_expr_inner(
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

    fn lower_nullable_inner_test(
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

    fn lower_pattern_test(
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

    fn extract_enum_name_from_ty<'b>(&'b self, ty: Option<&'b SsaType>) -> Option<&'b StrId> {
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

    fn resolve_enum_for_variant(&self, enum_name: &StrId, variant: &StrId) -> &HirEnum<'a, 'bump> {
        self.enums.get(enum_name).unwrap_or_else(|| {
            println!("All enums: {:?}", self.enums.keys().collect::<Vec<_>>());
            panic!(
                "[resolve_enum_for_variant] unknown enum {}.{}",
                enum_name, variant
            )
        })
    }

    fn bind_pattern(
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

    fn and_conds(&mut self, acc: Option<Value>, cond: Value) -> Value {
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

    fn or_conds(&mut self, acc: Option<Value>, cond: Value) -> Value {
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

    fn lower_index(&mut self, object: &HirExpr<'a, 'bump>, index: &HirExpr<'a, 'bump>) -> Value {
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

    fn try_flatten_module_path(&self, expr: &HirExpr<'a, 'bump>) -> Option<StrId> {
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

    fn resolve_module_access_callee(&self, path: &[StrId], member: StrId) -> Option<StrId> {
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

    fn resolve_module_qualified_name(
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

    fn lower_array_literal(&mut self, elements: &[HirExpr<'a, 'bump>]) -> Value {
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

    fn lower_zeroed_value(&mut self, ssa_ty: &SsaType) -> Value {
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

    /// Emits a call to the runtime's `__zeta_memset(ptr, value, size)`.
    fn emit_memset(&mut self, ptr: Value, value: i64, size: usize) {
        let memset_fn = StrId(self.context.intern("__zeta_memset"));

        let val_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: val_v,
            ty: SsaType::I32,
            value: Operand::ConstInt(value),
        });
        self.current_block_data
            .value_types
            .insert(val_v, SsaType::I32);

        let size_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: size_v,
            ty: SsaType::Usize,
            value: Operand::ConstInt(size as i64),
        });
        self.current_block_data
            .value_types
            .insert(size_v, SsaType::Usize);

        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(memset_fn),
            args: smallvec![
                Operand::Value(ptr),
                Operand::Value(val_v),
                Operand::Value(size_v),
            ],
        });
    }

    fn emit_debug_panic(&mut self, msg: Value) {
        let panic_fn = StrId::from_static("zeta_debug_debug_panic");
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(panic_fn),
            args: smallvec![Operand::Value(msg)],
        });
        self.emit(Instruction::Ret { value: None });
    }

    fn lower_expr_assignment(
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
                                    let already_init =
                                        if let (HirExpr::Ident(root, _), HirExpr::Number(n, _)) =
                                            (object, index)
                                        {
                                            !self.drop_state.is_index_uninit(*root, *n)
                                        } else {
                                            true
                                        };

                                    if already_init {
                                        self.emit_indexed_element_drop(
                                            &elem_drop_kind,
                                            addr_v,
                                            *span,
                                        );
                                    }

                                    if let (HirExpr::Ident(root, _), HirExpr::Number(n, _)) =
                                        (object, index)
                                    {
                                        self.drop_state.mark_index_initialized(*root, *n);
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
                            op: assign_op_to_bin_op(op), // strips the "Assign" suffix to base op
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

    fn lower_expr_null(&mut self) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I64,
            value: Operand::ConstInt(0),
        });
        self.current_block_data.value_types.insert(v, SsaType::Null);
        v
    }

    fn lower_expr_number(&mut self, n: i64) -> Value {
        let v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: v,
            ty: SsaType::I64,
            value: Operand::ConstInt(n),
        });
        self.current_block_data.value_types.insert(v, SsaType::I64);
        v
    }

    fn handle_ident(
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
                }
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

    fn handle_field_access(
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

    fn drop_kind_for_ssa_type(
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

    fn lower_expr_as_u32(&mut self, expr: &HirExpr<'a, 'bump>) -> Value {
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

    fn lower_expr_binary(
        &mut self,
        left: &HirExpr<'a, 'bump>,
        op: &Operator,
        right: &HirExpr<'a, 'bump>,
    ) -> Value {
        self.lower_expr_binary_expected(left, op, right, None)
    }

    fn lower_expr_binary_expected(
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

    fn lower_struct_init(
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
                record_move_if_any(&self.scope_stack, &mut self.drop_state, &arg.value);
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

    fn store_vtable_if_any(&mut self, obj: Value, struct_name: StrId) {
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

    fn narrowed_field_value(&self, expr: &HirExpr<'a, 'bump>) -> Option<Value> {
        let (root, path) = self.static_field_path_mir(expr)?;
        self.narrowed_fields.get(&(root, path)).copied()
    }

    fn lower_field_access(
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

    fn param_types_of(&self, name: &StrId) -> Vec<SsaType> {
        self.funcs
            .get(name)
            .or_else(|| self.global_funcs.get(name))
            .map(|f| f.params.iter().map(|(_, t)| t.clone()).collect())
            .unwrap_or_default()
    }

    fn lower_call_args(
        &mut self,
        args: &[HirExpr<'a, 'bump>],
        param_types: &[SsaType],
        param_offset: usize,
    ) -> SmallVec<Operand, 8> {
        let mut ops: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            let pty = param_types.get(i + param_offset).cloned();
            if pty.as_ref().map_or(false, |t| Self::is_move_by_value(t)) {
                record_move_if_any(&self.scope_stack, &mut self.drop_state, a);
            }
            let v = match &pty {
                Some(t) => self.lower_arg_expected(a, t),
                None => self.lower_expr(a),
            };
            ops.push(Operand::Value(v));
        }
        ops
    }

    fn lower_arg_expected(&mut self, arg: &HirExpr<'a, 'bump>, param_ty: &SsaType) -> Value {
        match param_ty {
            SsaType::Nullable(_) => self.lower_nullable_arg(arg, param_ty),
            _ => self.lower_expr_expected(arg, param_ty),
        }
    }

    fn lower_nullable_arg(&mut self, arg: &HirExpr<'a, 'bump>, param_ty: &SsaType) -> Value {
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

    /// A `null` of the given nullable type: 0 bits for pointer-optimized,
    /// a stack slot with tag = 0 for tagged nullables.
    fn lower_null_ssa(&mut self, ty: &SsaType) -> Value {
        if ty.nullable_pointer_repr().is_none() && ty.is_tagged_nullable() {
            let slot = self.current_block_data.fresh_value();
            self.emit(Instruction::StackAlloc {
                dest: slot,
                ty: ty.clone(),
                count: 1,
            });
            self.current_block_data.value_types.insert(slot, ty.clone());
            self.store_const_u8(slot, 0, 0);
            slot
        } else {
            let v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: v,
                ty: SsaType::I64,
                value: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(v, ty.clone());
            v
        }
    }

    /// `T` -> `T?`. Pointer-optimized: the bits already are the value.
    /// Tagged: fresh slot, tag = some, payload written after the tag.
    fn wrap_into_nullable(&mut self, val: Value, ty: &SsaType) -> Value {
        if ty.nullable_pointer_repr().is_some() {
            return val;
        }
        let slot = self.current_block_data.fresh_value();
        self.emit(Instruction::StackAlloc {
            dest: slot,
            ty: ty.clone(),
            count: 1,
        });
        self.current_block_data.value_types.insert(slot, ty.clone());
        self.store_field_value(slot, 0, ty, val);
        slot
    }

    fn lower_call(&mut self, callee: &HirExpr<'a, 'bump>, args: &[HirExpr<'a, 'bump>]) -> Value {
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
            } => self.lower_method_call(object, *field, args, span),

            HirExpr::ModuleAccess(acc) if acc.path.len() == 1 => {
                self.lower_static_call(acc.path[0], acc.member, args)
            }

            other => unimplemented!(
                "Non-identifier callee not yet supported in Call: {:?}",
                other
            ),
        }
    }

    fn lower_static_call(
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
                    record_move_if_any(&self.scope_stack, &mut self.drop_state, a);
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

    fn lower_place_addr(
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

    fn lower_field_addr(
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

    fn resolve_static_receiver_struct_name(&self, bare_name: StrId, field: StrId) -> Option<StrId> {
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

    fn lower_method_call(
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
                        record_move_if_any(&self.scope_stack, &mut self.drop_state, a);
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
                record_move_if_any(&self.scope_stack, &mut self.drop_state, object);
            }
        }

        operands.push(Operand::Value(obj_val));
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i + 1).unwrap_or(&SsaType::I64)) {
                record_move_if_any(&self.scope_stack, &mut self.drop_state, a);
            }
            let av = self.lower_expr(a);
            operands.push(Operand::Value(av));
        }

        if let Some(value) = self.emit_call(field, obj_val, &mut operands, cls_name_id) {
            return value;
        }

        panic!(
            "[lower_method_call] no mangled mapping or vtable slot found for method `{}` on struct `{:?}` at span {}.",
            self.context.resolve_string(&field),
            cls_name_id.map(|id| self.context.resolve_string(&id).to_string()),
            span
        );
    }

    fn resolve_receiver_target_key(&self, ty: &SsaType) -> Option<StrId> {
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

    fn builtin_target_key(&self, ty: &SsaType) -> Option<StrId> {
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

    fn lower_expr_as_receiver_raw(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
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

    fn lower_expr_as_receiver(&mut self, object: &HirExpr<'a, 'bump>) -> Value {
        let v = self.lower_expr_as_receiver_raw(object);
        self.canonicalize_receiver(v)
    }

    fn canonicalize_receiver(&mut self, v: Value) -> Value {
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

    fn coerce_into_tagged_nullable(&mut self, val: Value, expected: &SsaType) -> Value {
        if !expected.is_tagged_nullable() {
            return val; // pointer-optimized nullables share bits with the pointer
        }
        match self.value_type(val) {
            Some(SsaType::Null) | Some(SsaType::Nullable(_)) | Some(SsaType::Void) | None => {
                return val;
            }
            _ => {}
        }
        let slot = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: slot,
            ty: expected.clone(),
            count: 1,
        });
        self.current_block_data
            .value_types
            .insert(slot, expected.clone());
        self.store_field_value(slot, 0, expected, val); // writes tag=1, then the payload
        slot
    }

    fn emit_call(
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

    fn lower_interface_call(
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
            record_move_if_any(&self.scope_stack, &mut self.drop_state, object);
        }

        let mut operands: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i).unwrap()) {
                record_move_if_any(&self.scope_stack, &mut self.drop_state, a);
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

    fn get_field_offset(&mut self, obj: &Value, field: StrId) -> usize {
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

    #[inline(always)]
    fn emit(&mut self, instr: Instruction) {
        self.current_block_data.bb().instructions.push(instr);
    }

    #[inline(always)]
    fn new_value(&mut self) -> Value {
        self.current_block_data.fresh_value()
    }

    fn handle_deref_assign(&mut self, ptr: Value, rhs: Value, op: AssignmentOperator) -> Value {
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

    pub(super) fn finish(self) {
        // if self
        //     .current_block_data
        //     .func
        //     .name
        //     .as_str()
        //     .contains("push_back")
        //     || self
        //         .current_block_data
        //         .func
        //         .name
        //         .as_str()
        //         .contains("remove_back")
        // {
        //     for b in &self.current_block_data.func.blocks {
        //         println!("bb{}:", b.id.0);
        //         for i in &b.instructions {
        //             println!("    {:?}", i);
        //         }
        //     }
        // }
        self.current_block_data.finish()
    }

    pub fn record_move_if_any(&mut self, expr: &HirExpr) {
        match expr {
            HirExpr::Ident(name, _) => {
                if self.local_is_droppable(*name).is_some() {
                    self.drop_state.mark_whole_moved(*name);
                }
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

    fn emit_scope_drops(&mut self, scope: &DropScope<'a, 'bump>, span: SourceSpan<'a>) {
        for local in scope.locals.iter().rev() {
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

    fn emit_drop_for_kind(
        &mut self,
        kind: &DropKind<'a, 'bump>,
        val: Value,
        local_name: Option<StrId>,
        span: SourceSpan<'a>,
    ) {
        match kind {
            DropKind::Type(struct_name) => match local_name {
                Some(name) => {
                    let partial_move = self.drop_state.has_any_field_moves(name);
                    if !partial_move {
                        if let Some(glue) = self.glue_registry.glue_name_for(*struct_name) {
                            self.emit(Instruction::Call {
                                dest: None,
                                func: Operand::FunctionRef(glue),
                                args: SmallVec::from_slice_copy(&[Operand::Value(val)]),
                            });
                            return;
                        }
                    }
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
                None => {
                    let glue = self
                        .glue_registry
                        .glue_name_for(*struct_name)
                        .unwrap_or_else(|| {
                            panic!(
                                "no drop glue registered for struct `{}`; every droppable \
                             struct should have glue built by DropGlueBuilder::build_all",
                                struct_name
                            )
                        });
                    self.emit(Instruction::Call {
                        dest: None,
                        func: Operand::FunctionRef(glue),
                        args: SmallVec::from_slice_copy(&[Operand::Value(val)]),
                    });
                }
            },

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

    fn emit_slice_element_drops(
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

    fn emit_drops_for_return(&mut self, span: SourceSpan<'a>) {
        for scope in self.scope_stack.clone().iter().rev() {
            self.emit_scope_drops(scope, span);
        }
    }

    fn emit_drops_for_loop_exit(&mut self, depth_at_loop_entry: usize, span: SourceSpan<'a>) {
        for i in (depth_at_loop_entry..self.scope_stack.len()).rev() {
            let scope =
                std::mem::replace(&mut self.scope_stack[i], DropScope { locals: Vec::new() });
            self.emit_scope_drops(&scope, span);
            self.scope_stack[i] = scope;
        }
    }

    fn is_move_by_value(ty: &SsaType) -> bool {
        matches!(ty, SsaType::User(_, _) | SsaType::Owned(_))
    }

    fn slice_primitive_of(&self, field: StrId) -> Option<SlicePrimitive> {
        match self.context.resolve_string(&field) {
            "write_uninit" => Some(SlicePrimitive::WriteUninit),
            "write_uninit_all" => Some(SlicePrimitive::WriteUninitAll),
            "get_unchecked" => Some(SlicePrimitive::GetUnchecked),
            _ => None,
        }
    }

    fn emit_elem_addr(&mut self, base_ptr: Value, idx: Value, elem_ty: &SsaType) -> Value {
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

    /// `__zeta_memcpy(dst, src, size_bytes)`
    fn emit_memcpy(&mut self, dst: Value, src: Value, size: Value) {
        let f = StrId(self.context.intern("__zeta_memcpy"));
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(f),
            args: smallvec![
                Operand::Value(dst),
                Operand::Value(src),
                Operand::Value(size)
            ],
        });
    }

    fn lower_slice_primitive(
        &mut self,
        prim: SlicePrimitive,
        object: &HirExpr<'a, 'bump>,
        obj_val: Value,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        let (base_ptr, elem_ty, _len) = self.split_indexable(obj_val);

        match prim {
            SlicePrimitive::GetUnchecked => {
                let idx = self.lower_expr_expected(&args[0], &SsaType::Usize);
                let addr = self.emit_elem_addr(base_ptr, idx, &elem_ty);
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest,
                    ptr: Operand::Value(addr),
                });
                self.current_block_data.value_types.insert(dest, elem_ty);
                dest
            }

            SlicePrimitive::WriteUninit => {
                if Self::is_move_by_value(&elem_ty) {
                    self.record_move_if_any(&args[1]);
                }
                let idx = self.lower_expr_expected(&args[0], &SsaType::Usize);
                let val = self.lower_expr_expected(&args[1], &elem_ty);
                let addr = self.emit_elem_addr(base_ptr, idx, &elem_ty);
                self.emit(Instruction::Store {
                    ptr: Operand::Value(addr),
                    value: Operand::Value(val),
                });
                if let (HirExpr::Ident(root, _), HirExpr::Number(n, _)) = (object, &args[0]) {
                    self.drop_state.mark_index_initialized(*root, *n);
                }
                self.unit_value()
            }

            SlicePrimitive::WriteUninitAll => {
                let src_val = self.lower_expr(&args[0]);
                let (src_ptr, _src_elem, src_len) = self.split_indexable(src_val);
                let len_op = src_len.expect("write_uninit_all: source has no length");

                let elem_size = ir::layout::sizeof_ssa(&elem_ty, TargetInfo { ptr_bytes: 8 })
                    .expect("write_uninit_all: element type has no known size")
                    as i64;

                let bytes = self.current_block_data.fresh_value();
                self.emit(Instruction::Binary {
                    dest: bytes,
                    op: BinOp::Mul,
                    left: len_op,
                    right: Operand::ConstInt(elem_size),
                });
                self.current_block_data
                    .value_types
                    .insert(bytes, SsaType::Usize);

                self.emit_memcpy(base_ptr, src_ptr, bytes);
                self.unit_value()
            }
        }
    }

    fn pattern_needs_nonnull(pattern: &HirPattern<'bump>) -> bool {
        !matches!(
            pattern,
            HirPattern::Null | HirPattern::Wildcard | HirPattern::Ident(_) | HirPattern::Or(_)
        )
    }

    fn payload_layout(field_tys: &[SsaType]) -> (Vec<usize>, usize) {
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

    fn lower_init_operand(
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

    fn store_init(&mut self, base: Value, offset: usize, field_ty: &SsaType, init: FieldInitVal) {
        match init {
            FieldInitVal::Uninit => {}
            FieldInitVal::Null => self.store_null_field(base, offset, field_ty),
            FieldInitVal::Val(v) => self.store_field_value(base, offset, field_ty, v),
        }
    }

    fn store_const_u8(&mut self, base: Value, offset: usize, v: i64) {
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

    fn nullable_owned_drop_kind(
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

    fn hir_field_type_of_place(&self, place: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
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

    fn retype_as_owned(&mut self, val: Value, owned_hir: &HirType<'a, 'bump>) -> Value {
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

    fn lower_tagged_nullable_eq(
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

        let tag = self.new_value();
        self.emit(Instruction::LoadField {
            dest: tag,
            base: Operand::Value(nullable),
            offset: 0,
        });
        self.current_block_data.value_types.insert(tag, SsaType::U8);
        let is_some = self.new_value();
        self.emit(Instruction::Binary {
            dest: is_some,
            op: BinOp::Ne,
            left: Operand::Value(tag),
            right: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(is_some, SsaType::Bool);

        // payload load is harmless when null: the slot is valid memory
        let payload = self.unwrap_known_nonnull(nullable, ty);
        let peq = self.new_value();
        self.emit(Instruction::Binary {
            dest: peq,
            op: BinOp::Eq,
            left: Operand::Value(payload),
            right: Operand::Value(rv),
        });
        self.current_block_data
            .value_types
            .insert(peq, SsaType::Bool);

        let both = self.and_conds(Some(is_some), peq);
        if is_eq {
            both
        } else {
            let v = self.new_value();
            self.emit(Instruction::Binary {
                dest: v,
                op: BinOp::Eq,
                left: Operand::Value(both),
                right: Operand::ConstInt(0),
            });
            self.current_block_data.value_types.insert(v, SsaType::Bool);
            v
        }
    }

    fn emit_nullable_owned_field_overwrite_drop(
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

    fn emit_nullable_owned_drop(
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
    fn adopt_owned_binding(
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

    fn field_addr(&mut self, base: Value, offset: usize, ty: &SsaType) -> Value {
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

    fn store_null_field(&mut self, base: Value, offset: usize, field_ty: &SsaType) {
        if field_ty.is_tagged_nullable() {
            self.store_const_u8(base, offset, 0); // null tag
            return;
        }
        // pointer-optimized nullable: null is all-zero bits
        let zero = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: zero,
            ty: SsaType::I64,
            value: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(zero, SsaType::I64);
        self.emit(Instruction::StoreField {
            base: Operand::Value(base),
            offset,
            value: Operand::Value(zero),
        });
    }

    fn retype_nullable_ptr_bits(&mut self, val: Value, field_ty: &SsaType) -> Value {
        let Some(pointee) = field_ty.nullable_pointer_repr() else {
            return val;
        };
        let ptr_ty = SsaType::Pointer(Box::new(pointee.clone()));
        match self.value_type(val).cloned() {
            Some(SsaType::Nullable(_)) | Some(SsaType::Null) => {
                let dest = self.new_value();
                self.emit(Instruction::Cast {
                    dest,
                    value: Operand::Value(val),
                    kind: cast_kind(&ptr_ty, &ptr_ty),
                });
                self.current_block_data.value_types.insert(dest, ptr_ty);
                dest
            }
            _ => val,
        }
    }

    fn struct_field_ssa_type(&self, obj: Value, field: StrId) -> Option<SsaType> {
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

    fn store_field_value(&mut self, base: Value, offset: usize, field_ty: &SsaType, val: Value) {
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
}

fn fun_name(inner: &Box<SsaType>) -> StrId {
    if let SsaType::User(name, _) = inner.as_ref() {
        *name
    } else if let SsaType::Pointer(ptr_inner) = inner.as_ref() {
        fun_name(ptr_inner)
    } else if let SsaType::Owned(ptr_inner) = inner.as_ref() {
        fun_name(ptr_inner)
    } else {
        panic!("FieldAccess through pointer to non-User type: {:?}", inner)
    }
}
