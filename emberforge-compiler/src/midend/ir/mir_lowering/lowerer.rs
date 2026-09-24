use crate::midend::copy_analysis::drop_glue::DropGlueRegistry;
use crate::midend::copy_analysis::drop_tracking::{DropMoveState, DropScope};
use crate::midend::ir::block_data::CurrentBlockData;
use crate::midend::ir::mir_lowering::LoopCtx;
use codex_dependency_graph::DepGraph;
use ir::hir::{HirEnum, HirExpr, HirFunc, HirParam, HirStmt, HirStruct, HirType, StrId};
use ir::ir_conversion::lower_type_hir;
use ir::ir_hasher::{HashMap, HashSet};
use ir::ssa_ir::{
    AllocatorKind, BasicBlock, BlockId, Function, Instruction, Operand, SsaType, Value,
};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

#[derive(Clone, Copy)]
pub(crate) enum IndexedContainer {
    Array(usize),
    BorrowedSlice,
    OwnedSlice,
}

#[derive(Clone, Copy)]
pub(crate) enum SlicePrimitive {
    WriteUninit,
    WriteUninitAll,
    GetUnchecked,
}

#[derive(Clone, Copy)]
pub(crate) enum FieldInitVal {
    Null,
    Uninit,
    Val(Value),
}

pub struct FunctionLowerer<'f, 'a, 'bump> {
    pub(super) current_block_data: CurrentBlockData<'f>,
    pub(super) var_map: HashMap<StrId, Value>,
    pub(super) phantom_data: PhantomData<&'bump ()>,
    pub(super) loop_stack: Vec<LoopCtx<'a, 'bump>>,
    pub(super) funcs: &'a HashMap<StrId, Function>,
    pub(super) struct_field_offsets: &'a HashMap<StrId, HashMap<StrId, usize>>,
    pub(super) struct_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
    pub(super) struct_mangled_map: &'a HashMap<StrId, HashMap<StrId, StrId>>,
    pub(super) struct_vtable_slots: &'a HashMap<StrId, Vec<StrId>>,
    pub(super) interface_id_map: &'a HashMap<StrId, usize>,
    pub(super) interface_method_slots: &'a HashMap<StrId, HashMap<StrId, usize>>,
    pub(super) structs: &'a HashMap<StrId, HirStruct<'a, 'bump>>,
    pub(super) enum_variant_tags: &'a HashMap<StrId, HashMap<StrId, usize>>,
    pub(super) enums: &'a HashMap<StrId, HirEnum<'a, 'bump>>,
    pub(super) context: Arc<StringPool>,
    pub(super) extern_c_names: &'a HashSet<StrId>,
    pub(super) dep_graph: &'a RefCell<DepGraph>,
    pub(super) module_idx: usize,
    pub(super) return_type: Option<HirType<'a, 'bump>>,
    pub(super) global_funcs: &'a HashMap<StrId, Function>,
    pub(super) scope_stack: Vec<DropScope<'a, 'bump>>,
    pub(super) drop_state: DropMoveState<'a, 'bump>,
    pub(super) glue_registry: &'a DropGlueRegistry,
    pub(super) allocator_kind: &'a HashMap<StrId, AllocatorKind>,
    pub(super) interface_methods: &'a HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
    pub(super) bump: &'bump GrowableBump<'bump>,
    pub(super) module_import_aliases: &'a HashMap<usize, HashMap<StrId, usize>>,
    pub(super) module_named_imports: &'a HashMap<usize, HashMap<StrId, usize>>,
    pub(super) constants: &'a HashMap<StrId, HirExpr<'a, 'bump>>,
    pub(super) promoted_to_stack: HashSet<StrId>,
    pub(super) narrowed_fields: HashMap<(StrId, Vec<StrId>), Value>,
    pub(super) nullable_owned_locals: HashMap<StrId, HirType<'a, 'bump>>,
    pub(super) array_flags: HashMap<StrId, (Value, usize)>,
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

    pub(super) fn new_internal(
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
            array_flags: HashMap::default(),
            nullable_owned_locals: HashMap::default(),
        })
    }

    pub(crate) fn lower_body(&mut self, body: Option<HirStmt<'a, 'bump>>) {
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
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn lower_stmt_seq(&mut self, stmts: &[HirStmt<'a, 'bump>]) {
        let mut remaining = stmts;
        while let Some((stmt, rest)) = remaining.split_first() {
            if self.block_terminated() {
                break;
            }
            self.lower_stmt_inner(stmt, Some(rest));
            remaining = rest;
        }
    }

    pub(super) fn lower_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        self.lower_stmt_inner(stmt, None);
    }

    pub(super) fn lower_stmt_inner(
        &mut self,
        stmt: &HirStmt<'a, 'bump>,
        rest: Option<&[HirStmt<'a, 'bump>]>,
    ) {
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
                self.lower_if_stmt(cond, then_block, else_block, *span);
            }
            HirStmt::While { cond, body } => {
                self.lower_while_loop(cond, body);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                self.lower_for_loop(*init, *condition, *increment, body);
            }
            HirStmt::Let {
                name,
                ty,
                value,
                catch_pattern,
                else_block,
                ..
            } => {
                self.handle_let_stmt(rest, name, ty, value, catch_pattern, else_block);
            }

            HirStmt::Return(expr, span) => {
                self.handle_return_stmt(*expr, *span);
            }

            HirStmt::Block { body, span } => {
                self.handle_block_stmt(body, *span);
            }

            HirStmt::Break(expr, span) => {
                self.handle_break_stmt(*expr, *span);
            }
            HirStmt::Continue(span) => {
                self.handle_continue_stmt(*span);
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

    pub(super) fn static_field_path_mir(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<StrId>)> {
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

    pub(super) fn lower_expr_expected(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        expected: &SsaType,
    ) -> Value {
        match expr {
            HirExpr::Number(n, _) if expected.is_integer() => {
                self.lower_expr_number_inner(*n, expected.clone())
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

            HirExpr::Ident(name, span) => self.lower_ident_expr(name, span),
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
            } => self.lower_field_access_expr(object, *field, *span),

            HirExpr::Call {
                callee,
                args,
                span: _,
                type_args: _, // Turns into None after monomorphization
            } => self.lower_call_expr(callee, args),

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

            HirExpr::String(s, _) => self.lower_string_expr(s),

            HirExpr::Boolean(b, _) => self.lower_bool_expr(b),

            HirExpr::Decimal(d, _) => self.lower_decimal_expr(d),

            HirExpr::Tuple(elements, _) => self.lower_tuple_expr(elements),

            HirExpr::InterpolatedString(_parts) => self.lower_interpolated_str(),

            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                span,
                type_args: _,
            } => self.lower_enum_init(enum_name, variant, args, span),

            HirExpr::ExprList { list, span: _ } => self.lower_expr_list_expr(list),

            HirExpr::Comparison {
                left,
                op,
                right,
                span: _,
            } => self.lower_comparison_expr(left, *op, right),

            HirExpr::This { .. } => self.lower_this_expr(),
            HirExpr::Ref { expr, span, .. } => self.lower_place_addr(expr, span).0,

            HirExpr::Deref { expr, .. } => self.lower_deref_expr(expr),
            HirExpr::ModuleAccess(hir_module_access) => {
                self.lower_module_access_expr(hir_module_access)
            }
            HirExpr::Lambda { .. } => {
                unreachable!("There should be no lambdas here")
            }
            HirExpr::Index {
                object,
                index,
                span: _,
            } => self.lower_index_expr(object, index),
            HirExpr::ArrayLiteral { elements, span: _ } => self.lower_array_literal(elements),
            HirExpr::GenericIdent(..) => unreachable!(),
            HirExpr::Cast {
                expr, target_type, ..
            } => self.lower_cast_expr(expr, target_type),
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => self.lower_intrinsic_expr(*kind, type_args, args, *span),
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

    pub(super) fn is_aggregate_ssa_type(ty: &SsaType) -> bool {
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

    pub(super) fn value_type(&self, v: Value) -> Option<&SsaType> {
        self.current_block_data.value_types.get(&v)
    }

    #[inline(always)]
    pub(super) fn emit(&mut self, instr: Instruction) {
        self.current_block_data.bb().instructions.push(instr);
    }

    #[inline(always)]
    pub(super) fn new_value(&mut self) -> Value {
        self.current_block_data.fresh_value()
    }

    pub(crate) fn finish(self) {
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
}

pub fn fun_name(inner: &Box<SsaType>) -> StrId {
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
