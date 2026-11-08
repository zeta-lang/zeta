#![allow(unused)]

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::ControlFlow,
    path::PathBuf,
    sync::Arc,
};

use ir::{
    ast::{
        self, AstExecutionMode, Block, ConstStmt, DeferAction, ElseBranch, EnumDecl,
        ErrorHandlerPattern, Expr, ExternModifier, ForKind, FuncDecl, IfStmt, ImplDecl,
        InterfaceDecl, LetStmt, MatchStmt, Pattern, PatternKind, Stmt, StructDecl, Type,
        TypeAliasDecl, TypeKind,
    },
    hir::{StrId, Visibility},
    ir_hasher::{FpHasher, HashCtx, StableHash},
    span::SourceSpan,
    tokens::Tokens,
};
use zetaruntime::string_pool::StringPool;

use crate::{
    dep_graph::AstModule,
    nodes::{
        BodyRule, EdgeKind, ExecutionMode, Fingerprint, ImplFacts, ImportFacts, ItemFacts, ItemIdx,
        ItemKey, ItemRole, ModuleFacts, ModuleId, ModuleIdx, PackageFacts, ParamHow, ParamUse,
        RefFacts, RefTarget, ResolvedItem, ScanDiag, ScanKey, SymbolKind, SymbolName, ViaArg,
    },
};

const TAG_PACKAGE: u8 = 1;
const TAG_IMPORTS: u8 = 2;
const TAG_ITEMS: u8 = 3;
const TAG_COMPTIME_BODY: u8 = 4;

const VALUE_KINDS: [SymbolKind; 4] = [
    SymbolKind::Const,
    SymbolKind::Static,
    SymbolKind::Function,
    SymbolKind::Type,
];

pub struct Scanner<'p, 'a, 'bump> {
    pool: &'p StringPool,
    cx: HashCtx<'p>,
    items: Vec<ItemFacts<'a>>,
    refs: Vec<RefFacts<'a>>,
    impls: Vec<ImplFacts<'a>>,
    imports: Vec<ImportFacts<'a, 'bump>>,
    package: Option<PackageFacts<'a, 'bump>>,
    diags: Vec<ScanDiag<'a>>,
    iface_digests: Vec<u128>,
    body_digests: Vec<u128>,
    cur: Option<ScanKey>,
    /// `(cur, param_uses)` of the enclosing items,
    /// so items nested in an `impl`/`interface` don't clobber the outer item's state.
    frames: Vec<(Option<ScanKey>, Vec<ParamUse>)>,
    scopes: Vec<BTreeSet<StrId>>,
    self_type: Option<StrId>,
    next_item: usize,
    module_idx: ModuleIdx,
    import_digests: Vec<u128>,
    package_digest: Option<u128>,

    /// How many items with a given `(name, role)` have been seen so far
    ordinals: BTreeMap<(StrId, ItemRole), u64>,
    /// Declared generic parameter names in scope, outermost first
    type_params: Vec<StrId>,
    /// Generic-argument positions we are currently inside, by value
    via_stack: Vec<ViaArg>,
    /// `ParamUse`s found for the item currently being scanned
    param_uses: Vec<ParamUse>,
}

impl<'p, 'a, 'bump> Scanner<'p, 'a, 'bump> {
    pub fn new(pool: &'p StringPool, module_idx: ModuleIdx) -> Self {
        Self {
            pool,
            cx: HashCtx { pool },
            module_idx,
            next_item: 0,
            items: Vec::new(),
            refs: Vec::new(),
            impls: Vec::new(),
            imports: Vec::new(),
            package: None,
            diags: Vec::new(),
            import_digests: Vec::new(),
            package_digest: None,
            iface_digests: Vec::new(),
            body_digests: Vec::new(),
            ordinals: BTreeMap::new(),
            cur: None,
            frames: Vec::new(),
            scopes: Vec::new(),
            self_type: None,
            via_stack: Vec::new(),
            param_uses: Vec::new(),
            type_params: Vec::new(),
        }
    }

    fn begin_item(&mut self, name: StrId, role: ItemRole) -> ScanKey {
        let ord = self.ordinals.entry((name, role)).or_insert(0);
        let ordinal = *ord;
        *ord += 1;

        let item = ItemIdx(self.next_item);
        self.next_item += 1;

        let key = ScanKey {
            name,
            item_key: ItemKey {
                module: self.module_idx,
                item,
                role,
            },
            ordinal,
        };
        self.frames
            .push((self.cur, std::mem::take(&mut self.param_uses)));
        self.cur = Some(key);
        key
    }

    /// Undoes `begin_item` for an item that is dropped without being recorded.
    fn abort_item(&mut self, key: ScanKey) {
        debug_assert_eq!(self.cur, Some(key), "abort_item out of order");
        debug_assert_eq!(
            key.item_key.item.0 + 1,
            self.next_item,
            "can only abort the most recently started item"
        );
        self.next_item -= 1;
        if let Some(n) = self.ordinals.get_mut(&(key.name, key.item_key.role)) {
            *n = n.saturating_sub(1);
        }
        let (prev_cur, prev_uses) = self.frames.pop().expect("abort_item without begin_item");
        self.cur = prev_cur;
        self.param_uses = prev_uses;
        self.via_stack.clear();
    }

    fn emit(&mut self, target: RefTarget, kind: EdgeKind, span: SourceSpan<'a>) {
        self.emit_via(target, kind, Vec::new(), span);
    }

    fn emit_via(
        &mut self,
        target: RefTarget,
        kind: EdgeKind,
        via: Vec<ViaArg>,
        span: SourceSpan<'a>,
    ) {
        let from = self.cur.expect("emit outside an item");
        self.refs.push(RefFacts {
            from,
            target,
            edge_kind: kind,
            via,
            span,
        });
    }

    fn enter_generics(&mut self, names: impl Iterator<Item = StrId>) -> usize {
        let depth = self.type_params.len();
        self.type_params.extend(names);
        depth
    }

    fn exit_generics(&mut self, depth: usize) {
        self.type_params.truncate(depth);
    }

    fn param_index(&self, name: StrId, path: &[StrId], generics_len: usize) -> Option<u32> {
        if !path.is_empty() || generics_len != 0 {
            return None;
        }
        self.type_params
            .iter()
            .rposition(|p| *p == name)
            .map(|i| i as u32)
    }

    fn record_param_use(&mut self, param: u32) {
        let how = if self.via_stack.is_empty() {
            ParamHow::Direct
        } else {
            ParamHow::Through(self.via_stack.clone())
        };
        let u = ParamUse { param, how };
        if !self.param_uses.contains(&u) {
            self.param_uses.push(u);
        }
    }

    fn finish_item(
        &mut self,
        key: ScanKey,
        vis: Visibility,
        mode: ExecutionMode,
        span: SourceSpan<'a>,
        iface: FpHasher,
        body: FpHasher,
    ) {
        let (i, b) = (Fingerprint(iface.finish()), Fingerprint(body.finish()));
        self.iface_digests.push(self.digest(key, i));
        self.body_digests.push(self.digest(key, b));

        debug_assert!(
            self.via_stack.is_empty(),
            "unbalanced generic-argument stack"
        );
        let (prev_cur, prev_uses) = self.frames.pop().expect("finish_item without begin_item");
        let param_uses = std::mem::replace(&mut self.param_uses, prev_uses);

        self.items.push(ItemFacts {
            key,
            visibility: vis,
            execution_mode: mode,
            span,
            iface_fp: i,
            body_fp: b,
            param_uses,
        });

        debug_assert_eq!(self.cur, Some(key), "finish_item out of order");
        self.cur = prev_cur;
    }

    fn digest(&self, key: ScanKey, fp: Fingerprint) -> u128 {
        let mut h = FpHasher::new();
        key.name.stable_hash(&self.cx, &mut h);
        h.u8(key.item_key.role as u8);
        h.u64(key.ordinal);
        h.u128(fp.0);
        h.finish()
    }

    pub fn scan_top(&mut self, stmts: &[Stmt<'a, 'bump>]) {
        for stmt in stmts {
            self.scan_top_stmt(stmt);
        }
    }

    fn scan_top_stmt(&mut self, stmt: &Stmt<'a, 'bump>) {
        self.scan_stmt(stmt, BodyRule::Required);
    }

    fn scan_stmt(&mut self, stmt: &Stmt<'a, 'bump>, body_rule: BodyRule) {
        match stmt {
            Stmt::Return(_)
            | Stmt::If(_)
            | Stmt::While(_)
            | Stmt::For(_)
            | Stmt::Match(_)
            | Stmt::UnsafeBlock(_)
            | Stmt::Block(_)
            | Stmt::Defer(_)
            | Stmt::ExprStmt(_)
            | Stmt::Break(..)
            | Stmt::Continue(_) => {
                self.diags.push(ScanDiag::StrayTopLevel {
                    span: stmt.get_span(),
                });
            }

            Stmt::Let(l) if !l.is_static => {
                self.diags.push(ScanDiag::StrayTopLevel {
                    span: stmt.get_span(),
                });
            }

            Stmt::Const(c) => self.scan_const_decl(c),
            Stmt::FuncDecl(f) => self.scan_func_decl(f, body_rule),
            Stmt::StructDecl(s) => self.scan_struct_decl(s),
            Stmt::InterfaceDecl(i) => self.scan_interface_decl(i),
            Stmt::ImplDecl(i) => self.scan_impl_decl(i),
            Stmt::EnumDecl(e) => self.scan_enum_decl(e),
            Stmt::Module(m) => self.scan_top(m.body),
            Stmt::TypeAliasDecl(a) => self.scan_type_alias_decl(a),
            Stmt::Attributed(stmt) => self.scan_stmt(&stmt.inner, body_rule),
            Stmt::Let(l) => self.scan_static_decl(l),
            Stmt::Import(s) => self.scan_import_stmt(s),
            Stmt::Package(s) => self.scan_package_stmt(s),
        }
    }

    fn scan_const_decl(&mut self, c: &ConstStmt<'a, 'bump>) {
        let key = self.begin_item(c.ident, ItemRole::Const);
        let depth = if let Some(generics) = c.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        c.stable_hash(&self.cx, &mut ih);

        self.walk_type(&c.type_annotation);

        // `A :: B * 2` gives A -> B
        self.walk_expr(c.value, EdgeKind::ConstEval);

        self.exit_generics(depth);
        self.finish_item(
            key,
            c.visibility,
            ExecutionMode::Comptime,
            c.span,
            ih,
            FpHasher::new(),
        );
    }

    fn scan_type_alias_decl(&mut self, a: &TypeAliasDecl<'a, 'bump>) {
        let key = self.begin_item(a.name, ItemRole::Type);
        let depth = if let Some(generics) = a.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        a.stable_hash(&self.cx, &mut ih);

        // The alias *is* its target, so the target is referenced by value.
        self.walk_type(&a.ty);

        self.exit_generics(depth);
        self.finish_item(
            key,
            a.visibility,
            ExecutionMode::Runtime,
            a.span,
            ih,
            FpHasher::new(),
        );
    }

    fn scan_static_decl(&mut self, l: &LetStmt<'a, 'bump>) {
        let key = self.begin_item(l.ident, ItemRole::Static);

        let mut ih = FpHasher::new();
        let mut bh = FpHasher::new();
        l.stable_hash_signature(&self.cx, &mut ih);
        l.stable_hash_body(&self.cx, &mut bh);

        self.walk_type(&l.type_annotation);
        self.walk_expr(&l.value, EdgeKind::StaticInit);

        self.finish_item(key, l.visibility, ExecutionMode::Runtime, l.span, ih, bh);
    }

    fn impl_item_name(&self, target: &Type<'a, 'bump>) -> StrId {
        match target.kind {
            TypeKind::Struct { name, .. } => name,
            _ => StrId(self.pool.intern("<impl>")),
        }
    }

    fn scan_impl_decl(&mut self, i: &ImplDecl<'a, 'bump>) {
        let name = self.impl_item_name(&i.target);
        let key = self.begin_item(name, ItemRole::Impl);
        let depth = if let Some(generics) = i.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        i.stable_hash(&self.cx, &mut ih);

        self.walk_indirectly_referenced_type(&i.target);
        if let Some(interface) = &i.interface {
            self.walk_indirectly_referenced_type(interface);
        }

        if let Some(methods) = i.methods {
            for method in methods {
                self.scan_func_decl(method, BodyRule::Optional);
            }
        }

        if let Some(constants) = i.constants {
            for constant in constants {
                self.scan_const_decl(constant);
            }
        }

        self.exit_generics(depth);

        let bh = FpHasher::new();
        self.finish_item(
            key,
            Visibility::Public,
            ExecutionMode::Runtime,
            i.span,
            ih,
            bh,
        );
    }

    fn scan_interface_decl(&mut self, i: &InterfaceDecl<'a, 'bump>) {
        let key = self.begin_item(i.name, ItemRole::Interface);
        let depth = if let Some(generics) = i.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        i.stable_hash(&self.cx, &mut ih);

        if let Some(permits) = i.permits {
            for permitted in permits.types.iter() {
                self.walk_type(permitted);
            }
        }

        if let Some(methods) = i.methods {
            for method in methods {
                self.scan_func_decl(method, BodyRule::Optional);
            }
        }

        self.exit_generics(depth);

        let bh = FpHasher::new();
        self.finish_item(key, i.visibility, ExecutionMode::Runtime, i.span, ih, bh);
    }

    fn scan_enum_decl(&mut self, e: &EnumDecl<'a, 'bump>) {
        let key = self.begin_item(e.name, ItemRole::Type);
        let depth = if let Some(generics) = e.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        e.stable_hash(&self.cx, &mut ih);

        for variant in e.variants.iter() {
            for field in variant.fields.iter() {
                self.walk_type(&field.field_type);
            }
        }

        self.exit_generics(depth);

        let bh = FpHasher::new();
        self.finish_item(key, e.visibility, ExecutionMode::Runtime, e.span, ih, bh);
    }

    fn scan_import_stmt(&mut self, s: &ast::ImportStmt<'a, 'bump>) {
        self.imports.push(ImportFacts {
            path: s.path,
            alias: None,
            span: s.span,
        });
        let mut eh = FpHasher::new();
        s.stable_hash(&self.cx, &mut eh);
        self.import_digests.push(eh.finish());
    }

    fn scan_package_stmt(&mut self, s: &ast::PackageStmt<'a, 'bump>) {
        if let Some(first) = &self.package {
            self.diags.push(ScanDiag::MultiplePackages {
                first: first.span,
                extra: s.span,
            });
            return;
        }
        self.package = Some(PackageFacts {
            path: s.path,
            span: s.span,
        });
        let mut eh = FpHasher::new();
        s.stable_hash(&self.cx, &mut eh);
        self.package_digest = Some(eh.finish());
    }

    fn scan_struct_decl(&mut self, s: &StructDecl<'a, 'bump>) {
        let key = self.begin_item(s.name, ItemRole::Type);
        let depth = if let Some(generics) = s.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        s.stable_hash(&self.cx, &mut ih);

        if let Some(fields) = s.params {
            for field in fields.iter() {
                match *field {
                    ast::Param::Normal(field) => {
                        self.walk_type(&field.type_annotation);
                    }
                    ast::Param::This(this_param) => {
                        self.diags.push(ScanDiag::StructFieldsContainsThis {
                            span: this_param.span,
                        })
                    }
                }
            }
        }

        self.exit_generics(depth);

        let bh = FpHasher::new();
        self.finish_item(key, s.visibility, ExecutionMode::Runtime, s.span, ih, bh);
    }

    fn scan_func_decl(&mut self, f: &FuncDecl<'a, 'bump>, body_rule: BodyRule) {
        let meta = f.function_metadata;
        let exec_mode = lower_execution_mode(meta.execution_mode);

        let key = self.begin_item(f.name, ItemRole::FuncSig);
        let depth = if let Some(generics) = f.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };
        // a "defer self.exit_generics(depth);" would be nice here :)
        // If only I was making my own language that fixes this... wait

        let mut ih = FpHasher::new();
        f.stable_hash_signature(&self.cx, &mut ih);

        let is_extern = !matches!(meta.extern_modifier, ExternModifier::None);

        let bh = match f.body {
            Some(body) => {
                if exec_mode == ExecutionMode::Comptime {
                    ih.u8(TAG_COMPTIME_BODY);
                    f.stable_hash_body(&self.cx, &mut ih);
                }

                let mut bh = FpHasher::new();
                f.stable_hash_body(&self.cx, &mut bh);
                bh
            }
            None if matches!(body_rule, BodyRule::Required) && !is_extern => {
                self.diags
                    .push(ScanDiag::RequiredBodyAbsent { span: f.name_span });
                self.exit_generics(depth);
                self.abort_item(key);
                return;
            }
            None => FpHasher::new(),
        };

        if let Some(params) = f.params {
            for param in params.iter() {
                // `this` receivers name the enclosing type, there is nothing to walk there
                if let ast::Param::Normal(param) = *param {
                    self.walk_type(&param.type_annotation);
                }
            }
        }

        if let Some(ret) = &f.return_type {
            self.walk_type(ret);
        }

        self.exit_generics(depth);
        self.finish_item(key, meta.visibility, exec_mode, f.span, ih, bh);
    }

    fn walk_type(&mut self, ty: &Type<'a, 'bump>) {
        match &ty.kind {
            TypeKind::U8
            | TypeKind::I8
            | TypeKind::U16
            | TypeKind::I16
            | TypeKind::U32
            | TypeKind::I32
            | TypeKind::U64
            | TypeKind::I64
            | TypeKind::U128
            | TypeKind::I128
            | TypeKind::F32
            | TypeKind::F64
            | TypeKind::UF32
            | TypeKind::UF64
            | TypeKind::Usize
            | TypeKind::Isize
            | TypeKind::Boolean
            | TypeKind::Char
            | TypeKind::String
            | TypeKind::Void
            | TypeKind::Never
            | TypeKind::Infer
            | TypeKind::AnySlice => {}

            TypeKind::SafePointer { inner, .. }
            | TypeKind::UnsafePointer { inner, .. }
            | TypeKind::Ref { inner, .. } => self.walk_indirectly_referenced_type(inner),

            TypeKind::OwnedPointer { inner, allocator } => {
                self.walk_indirectly_referenced_type(inner);
            }

            TypeKind::Array { inner, length } => {
                self.walk_type(inner);
                self.walk_expr(length, EdgeKind::ConstEval); // `f :: func(): [N]u8` gives f -> N
            }

            TypeKind::Tuple { values } => {
                for v in values.iter() {
                    self.walk_type(v);
                }
            }

            TypeKind::Slice { inner } => self.walk_indirectly_referenced_type(inner),

            TypeKind::Lambda {
                params,
                return_type,
            } => {
                for p in params.iter() {
                    self.walk_indirectly_referenced_type(p);
                }
                self.walk_indirectly_referenced_type(return_type);
            }

            TypeKind::Struct {
                name,
                path,
                generics,
            } => {
                if let Some(p) = self.param_index(*name, path, generics.len()) {
                    self.record_param_use(p);
                    return;
                }

                let span = ty.span;
                let target = self.check_type_target(*name, path);

                let via = self.via_stack.clone();
                self.emit_via(target.clone(), EdgeKind::TypeByValue, via, span);

                for (i, g) in generics.iter().enumerate() {
                    self.via_stack.push(ViaArg {
                        generic: target.clone(),
                        index: i as u32,
                    });
                    self.walk_type(g);
                    self.via_stack.pop();
                }
            }

            TypeKind::This => {
                if let Some(name) = self.this_name() {
                    let span = ty.span;
                    self.emit(
                        RefTarget::Name {
                            name,
                            kind: SymbolKind::Type,
                        },
                        EdgeKind::TypeByValue,
                        span,
                    );
                }
            }

            TypeKind::Dyn { .. } => {
                let span = ty.span;
                self.diags.push(ScanDiag::ByValueDyn { span });
            }

            // An anonymous struct/enum holds its fields by value
            TypeKind::AnonStruct { fields } => {
                for field in fields.iter() {
                    self.walk_type(&field.field_type);
                }
            }

            TypeKind::AnonEnum { variants } => {
                for variant in variants.iter() {
                    for field in variant.fields.iter() {
                        self.walk_type(&field.field_type);
                    }
                }
            }

            TypeKind::Type => {}
        }
    }

    fn walk_indirectly_referenced_type(&mut self, ty: &Type<'a, 'bump>) {
        match &ty.kind {
            TypeKind::U8
            | TypeKind::I8
            | TypeKind::U16
            | TypeKind::I16
            | TypeKind::U32
            | TypeKind::I32
            | TypeKind::U64
            | TypeKind::I64
            | TypeKind::U128
            | TypeKind::I128
            | TypeKind::F32
            | TypeKind::F64
            | TypeKind::UF32
            | TypeKind::UF64
            | TypeKind::Usize
            | TypeKind::Isize
            | TypeKind::Boolean
            | TypeKind::Char
            | TypeKind::String
            | TypeKind::Void
            | TypeKind::Never
            | TypeKind::Infer
            | TypeKind::AnySlice => {}

            TypeKind::SafePointer { inner, .. }
            | TypeKind::UnsafePointer { inner, .. }
            | TypeKind::Ref { inner, .. }
            | TypeKind::Slice { inner } => self.walk_indirectly_referenced_type(inner),

            TypeKind::OwnedPointer { inner, allocator } => {
                self.walk_indirectly_referenced_type(inner);
            }

            TypeKind::Array { inner, .. } => self.walk_indirectly_referenced_type(inner),
            TypeKind::Tuple { values } => {
                for v in values.iter() {
                    self.walk_indirectly_referenced_type(v);
                }
            }
            TypeKind::Lambda {
                params,
                return_type,
            } => {
                for p in params.iter() {
                    self.walk_indirectly_referenced_type(p);
                }
                self.walk_indirectly_referenced_type(return_type);
            }

            TypeKind::Struct {
                name,
                path,
                generics,
            } => {
                if self.param_index(*name, path, generics.len()).is_some() {
                    return;
                }

                let span = ty.span;
                let target = self.check_type_target(*name, path);

                self.emit(target, EdgeKind::TypeIndirect, span);
                for g in generics.iter() {
                    self.walk_indirectly_referenced_type(g);
                }
            }

            TypeKind::This => {
                if let Some(name) = self.this_name() {
                    let span = ty.span;
                    self.emit(
                        RefTarget::Name {
                            name,
                            kind: SymbolKind::Type,
                        },
                        EdgeKind::TypeIndirect,
                        span,
                    );
                }
            }

            TypeKind::Dyn { bounds } => {
                for b in bounds.iter() {
                    self.walk_indirectly_referenced_type(b);
                }
            }
            TypeKind::AnonStruct { fields } => {
                for field in fields.iter() {
                    self.walk_indirectly_referenced_type(&field.field_type);
                }
            }
            TypeKind::AnonEnum { variants } => {
                for variant in variants.iter() {
                    for field in variant.fields.iter() {
                        self.walk_indirectly_referenced_type(&field.field_type);
                    }
                }
            }
            TypeKind::Type => {}
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeSet::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_local(&mut self, name: StrId) {
        if self.scopes.is_empty() {
            self.push_scope();
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name);
        }
    }

    fn is_local(&self, name: StrId) -> bool {
        self.scopes.iter().any(|s| s.contains(&name))
    }

    /// The target of an unqualified value name (`B` in `A :: B * 2`).
    fn value_target(&self, name: StrId) -> RefTarget {
        RefTarget::ValueName(name)
    }

    fn walk_name(&mut self, name: StrId, span: SourceSpan<'a>, edge_kind: EdgeKind) {
        if self.is_local(name) {
            return;
        }
        // a const generic parameter, `N` in `Buf<N> :: [N]u8`
        if let Some(p) = self.param_index(name, &[], 0) {
            self.record_param_use(p);
            return;
        }
        let target = self.value_target(name);
        self.emit(target, edge_kind, span);
    }

    fn walk_expr(&mut self, expr: &Expr<'a, 'bump>, edge_kind: EdgeKind) {
        match expr {
            Expr::Null { .. }
            | Expr::Number { .. }
            | Expr::Decimal { .. }
            | Expr::String { .. }
            | Expr::Char { .. }
            | Expr::Boolean { .. }
            | Expr::This { .. }
            | Expr::Undefined { .. }
            | Expr::Uninit { .. } => {}

            Expr::Ident { name, span } => self.walk_name(*name, *span, edge_kind),

            // `Y<Z>`: a generic function or a type, either way `Y` is referenced and `Z` is a type.
            Expr::GenericIdent {
                name,
                generic_args,
                span,
            } => {
                self.walk_name(*name, *span, edge_kind);
                for g in generic_args.iter() {
                    self.walk_type(g);
                }
            }

            Expr::ModulePath { segments, span } => {
                self.emit(RefTarget::Path(segments.to_vec()), edge_kind, *span);
            }

            Expr::ModuleAccess {
                segments,
                member,
                span,
            } => {
                let mut path = segments.to_vec();
                path.push(*member);
                self.emit(RefTarget::Path(path), edge_kind, *span);
            }

            Expr::Intrinsic {
                generic_args,
                arguments,
                ..
            } => {
                for g in generic_args.iter() {
                    self.walk_type(g);
                }
                for a in arguments.iter() {
                    self.walk_expr(a, edge_kind);
                }
            }

            Expr::Cast {
                expr: inner,
                target_type,
                ..
            } => {
                self.walk_expr(inner, edge_kind);
                self.walk_type(target_type);
            }

            Expr::Comparison { lhs, rhs, .. } | Expr::Assignment { lhs, rhs, .. } => {
                self.walk_expr(lhs, edge_kind);
                self.walk_expr(rhs, edge_kind);
            }
            Expr::Binary { left, right, .. } => {
                self.walk_expr(left, edge_kind);
                self.walk_expr(right, edge_kind);
            }
            Expr::Unary { operand, .. } => self.walk_expr(operand, edge_kind),
            Expr::Deref { expr: inner, .. }
            | Expr::Ref { expr: inner, .. }
            | Expr::Attributed { expr: inner, .. } => self.walk_expr(inner, edge_kind),
            Expr::FieldAccess { object, .. } | Expr::Get { object, .. } => {
                self.walk_expr(object, edge_kind)
            }
            Expr::ArrayIndex {
                expr: inner, index, ..
            } => {
                self.walk_expr(inner, edge_kind);
                self.walk_expr(index, edge_kind);
            }
            Expr::Tuple { values, .. } => {
                for v in values.iter() {
                    self.walk_expr(v, edge_kind);
                }
            }
            Expr::ArrayLiteral { elements, .. } => {
                for e in elements.iter() {
                    self.walk_expr(e, edge_kind);
                }
            }
            Expr::ExprList { expressions, .. } => {
                for e in expressions.iter() {
                    self.walk_expr(e, edge_kind);
                }
            }

            Expr::Call {
                callee,
                generic_args,
                arguments,
                ..
            } => {
                self.walk_expr(callee, edge_kind);
                for g in generic_args.iter() {
                    self.walk_type(g);
                }
                for a in arguments.iter() {
                    self.walk_expr(a, edge_kind);
                }
            }

            Expr::StructInit {
                callee,
                arguments,
                type_args,
                ..
            } => {
                self.walk_init_target(callee, edge_kind);
                for t in type_args.iter() {
                    self.walk_type(t);
                }
                for field in arguments.iter() {
                    self.walk_expr(&field.value, edge_kind);
                }
            }

            Expr::OrElse {
                value, else_block, ..
            } => {
                self.walk_expr(value, edge_kind);
                self.walk_block(else_block, edge_kind);
            }

            Expr::If { if_stmt, .. } => self.walk_if(if_stmt, edge_kind),
            Expr::Match { match_stmt, .. } => self.walk_match(match_stmt, edge_kind),

            Expr::Lambda {
                params,
                return_type,
                ensures,
                body,
                ..
            } => {
                for p in params.iter() {
                    if let Some(t) = &p.type_annotation {
                        self.walk_indirectly_referenced_type(t);
                    }
                }
                if let Some(t) = return_type {
                    self.walk_indirectly_referenced_type(t);
                }
                self.push_scope();
                for p in params.iter() {
                    self.declare_local(p.name);
                }
                for e in ensures.iter() {
                    self.walk_expr(e.condition, edge_kind);
                }
                self.walk_block(body, edge_kind);
                self.pop_scope();
            }

            Expr::Block(block) | Expr::ComptimeBlock(block) => self.walk_block(block, edge_kind),
            Expr::UnsafeBlock(block) => self.walk_block(block.block, edge_kind),
        }
    }

    fn walk_init_target(&mut self, callee: &Expr<'a, 'bump>, edge_kind: EdgeKind) {
        match callee {
            Expr::Ident { name, span } => {
                if self.param_index(*name, &[], 0).is_none() {
                    let target = self.check_type_target(*name, &[]);
                    self.emit(target, EdgeKind::TypeByValue, *span);
                }
            }
            Expr::GenericIdent {
                name,
                generic_args,
                span,
            } => {
                let target = self.check_type_target(*name, &[]);
                self.emit(target, EdgeKind::TypeByValue, *span);
                for g in generic_args.iter() {
                    self.walk_type(g);
                }
            }
            Expr::ModulePath { segments, span } => {
                self.emit(
                    RefTarget::Path(segments.to_vec()),
                    EdgeKind::TypeByValue,
                    *span,
                );
            }
            Expr::ModuleAccess {
                segments,
                member,
                span,
            } => {
                let target = self.check_type_target(*member, segments);
                self.emit(target, EdgeKind::TypeByValue, *span);
            }
            other => self.walk_expr(other, edge_kind),
        }
    }

    fn walk_block(&mut self, block: &Block<'a, 'bump>, edge_kind: EdgeKind) {
        self.push_scope();
        for stmt in block.block.iter() {
            self.walk_body_stmt(stmt, edge_kind);
        }
        self.pop_scope();
    }

    fn walk_if(&mut self, stmt: &IfStmt<'a, 'bump>, edge_kind: EdgeKind) {
        self.walk_expr(stmt.condition, edge_kind);
        self.walk_block(stmt.then_branch, edge_kind);
        match stmt.else_branch {
            Some(ElseBranch::If(nested)) => self.walk_if(nested, edge_kind),
            Some(ElseBranch::Else(block)) => self.walk_block(block, edge_kind),
            None => {}
        }
    }

    fn walk_match(&mut self, stmt: &MatchStmt<'a, 'bump>, edge_kind: EdgeKind) {
        self.walk_expr(stmt.expr, edge_kind);
        for arm in stmt.arms.iter() {
            self.push_scope();
            let mut bound = Vec::new();
            Self::pattern_bindings(&arm.pattern, &mut bound);
            for name in bound {
                self.declare_local(name);
            }
            if let Some(guard) = arm.guard {
                self.walk_expr(guard, edge_kind);
            }
            self.walk_block(arm.block, edge_kind);
            self.pop_scope();
        }
    }

    fn pattern_bindings(pattern: &Pattern<'a, 'bump>, out: &mut Vec<StrId>) {
        match &pattern.kind {
            PatternKind::Ident(name, _) => out.push(*name),
            PatternKind::Tuple(ps) | PatternKind::Array(ps) | PatternKind::Or(ps) => {
                for p in ps.iter() {
                    Self::pattern_bindings(p, out);
                }
            }
            PatternKind::Struct { fields, .. } => {
                for (_, p) in fields.iter() {
                    Self::pattern_bindings(p, out);
                }
            }
            PatternKind::EnumVariant { bindings, .. } => {
                for p in bindings.iter() {
                    Self::pattern_bindings(p, out);
                }
            }
            PatternKind::Number(_)
            | PatternKind::String(_)
            | PatternKind::Boolean(_)
            | PatternKind::Wildcard
            | PatternKind::Null => {}
        }
    }

    fn walk_let(&mut self, l: &LetStmt<'a, 'bump>, edge_kind: EdgeKind) {
        self.walk_type(&l.type_annotation);
        self.walk_expr(l.value, edge_kind);
        match &l.catch_pattern {
            Some(ErrorHandlerPattern::Single {
                error_type,
                binding,
                body,
            }) => {
                self.walk_type(error_type);
                self.push_scope();
                if let Some(b) = binding {
                    self.declare_local(*b);
                }
                self.walk_block(body, edge_kind);
                self.pop_scope();
            }
            Some(ErrorHandlerPattern::Multiple { branches }) => {
                for branch in branches.iter() {
                    self.walk_type(&branch.error_type);
                    self.push_scope();
                    if let Some(b) = branch.binding {
                        self.declare_local(b);
                    }
                    self.walk_block(branch.body, edge_kind);
                    self.pop_scope();
                }
            }
            None => {}
        }
        self.declare_local(l.ident);
    }

    /// A statement inside a `comptime { .. }` block or a lambda body.
    fn walk_body_stmt(&mut self, stmt: &Stmt<'a, 'bump>, edge_kind: EdgeKind) {
        match stmt {
            Stmt::Let(l) => self.walk_let(l, edge_kind),
            Stmt::Const(c) => {
                self.walk_type(&c.type_annotation);

                self.walk_expr(c.value, edge_kind);
                self.declare_local(c.ident);
            }
            Stmt::Return(r) => {
                if let Some(value) = r.value {
                    self.walk_expr(value, edge_kind);
                }
            }
            Stmt::If(i) => self.walk_if(i, edge_kind),
            Stmt::While(w) => {
                self.walk_expr(w.condition, edge_kind);
                self.walk_block(w.block, edge_kind);
            }
            Stmt::For(f) => {
                self.push_scope();
                match &f.kind {
                    ForKind::CStyle {
                        let_stmt,
                        condition,
                        increment,
                    } => {
                        if let Some(l) = let_stmt {
                            self.walk_let(l, edge_kind);
                        }
                        if let Some(c) = condition {
                            self.walk_expr(c, edge_kind);
                        }
                        if let Some(i) = increment {
                            self.walk_expr(i, edge_kind);
                        }
                    }
                    ForKind::RangeBased { variable, iterable } => {
                        self.walk_expr(iterable, edge_kind);
                        self.declare_local(*variable);
                    }
                }
                self.walk_block(f.block, edge_kind);
                self.pop_scope();
            }
            Stmt::Match(m) => self.walk_match(m, edge_kind),
            Stmt::UnsafeBlock(u) => self.walk_block(u.block, edge_kind),
            Stmt::Block(b) => self.walk_block(b, edge_kind),
            Stmt::Defer(d) => match d.action {
                DeferAction::Block(b) => self.walk_block(b, edge_kind),
                DeferAction::Stmt(s) => self.walk_body_stmt(s, edge_kind),
            },
            Stmt::ExprStmt(e) => self.walk_expr(e.expr, edge_kind),
            Stmt::Break(value, _) => {
                if let Some(value) = value {
                    self.walk_expr(value, edge_kind);
                }
            }
            Stmt::Continue(_) => {}
            Stmt::Attributed(a) => self.walk_body_stmt(&a.inner, edge_kind),

            Stmt::Import(_)
            | Stmt::Package(_)
            | Stmt::Module(_)
            | Stmt::FuncDecl(_)
            | Stmt::StructDecl(_)
            | Stmt::InterfaceDecl(_)
            | Stmt::ImplDecl(_)
            | Stmt::EnumDecl(_)
            | Stmt::TypeAliasDecl(_) => {
                self.unsupported(stmt, "declaration inside an expression body");
            }
        }
    }

    fn check_type_target(&self, name: StrId, path: &[StrId]) -> RefTarget {
        if path.is_empty() {
            RefTarget::Name {
                name,
                kind: SymbolKind::Type,
            }
        } else {
            let mut p = path.to_vec();
            p.push(name);
            RefTarget::Path(p)
        }
    }

    fn this_name(&self) -> Option<StrId> {
        self.self_type.or_else(|| self.cur.map(|k| k.name))
    }

    fn unsupported(&mut self, stmt: &Stmt<'a, 'bump>, what: &'static str) {
        self.diags.push(ScanDiag::Unsupported {
            span: stmt.get_span(),
            what,
        });
    }

    pub fn finish(mut self, path: PathBuf) -> ModuleFacts<'a, 'bump> {
        debug_assert!(self.cur.is_none(), "item left open at end of scan");

        self.items.sort_by_key(|i| i.key.item_key.item);
        debug_assert!(
            self.items
                .iter()
                .enumerate()
                .all(|(n, i)| i.key.item_key.item.0 == n),
            "ItemIdx values are not dense"
        );

        let mut ih = FpHasher::new();
        ih.u8(TAG_PACKAGE);
        match self.package_digest {
            Some(d) => {
                ih.u8(1);
                ih.u128(d);
            }
            None => ih.u8(0),
        }
        ih.u8(TAG_IMPORTS);
        fold_unordered(self.import_digests, &mut ih);
        ih.u8(TAG_ITEMS);
        fold_unordered(self.iface_digests, &mut ih);

        let mut bh = FpHasher::new();
        fold_unordered(self.body_digests, &mut bh);

        ModuleFacts {
            path,
            package_facts: self.package,
            refs: self.refs,
            imports: self.imports,
            items: self.items,
            impls: self.impls,
            diags: self.diags,
            iface_fp: Fingerprint(ih.finish()),
            body_fp: Fingerprint(bh.finish()),
        }
    }
}

impl<'a, 'bump> ModuleFacts<'a, 'bump> {
    pub fn scan(
        ast: &AstModule<'a, 'bump>,
        tokens: &Tokens<'a>,
        pool: Arc<StringPool>,
        module_idx: ModuleIdx,
    ) -> ModuleFacts<'a, 'bump> {
        let mut scanner = Scanner::new(&pool, module_idx);
        scanner.scan_top(ast.stmts);
        scanner.finish(ast.path.clone())
    }
}

fn fold_unordered(mut digests: Vec<u128>, h: &mut FpHasher) {
    digests.sort_unstable();
    h.u64(digests.len() as u64);
    for d in digests {
        h.u128(d);
    }
}

pub fn stable_hash_unordered<'i, T, I>(items: I, cx: &HashCtx<'_>, h: &mut FpHasher)
where
    T: StableHash + 'i,
    I: IntoIterator<Item = &'i T>,
{
    let digests: Vec<u128> = items
        .into_iter()
        .map(|it| {
            let mut eh = FpHasher::new();
            it.stable_hash(cx, &mut eh);
            eh.finish()
        })
        .collect();
    fold_unordered(digests, h);
}

const fn lower_execution_mode(e: AstExecutionMode) -> ExecutionMode {
    match e {
        AstExecutionMode::Runtime => ExecutionMode::Runtime,
        AstExecutionMode::Comptime => ExecutionMode::Comptime,
    }
}
