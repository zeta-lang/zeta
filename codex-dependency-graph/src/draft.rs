#![allow(unused)]

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::ControlFlow,
    path::PathBuf,
    sync::Arc,
};

use ir::{
    ast::{
        self, AstExecutionMode, ConstStmt, EnumDecl, Expr, FuncDecl, ImplDecl, InterfaceDecl,
        LetStmt, Stmt, StructDecl, Type, TypeKind,
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
        ItemKey, ItemRole, ModuleFacts, ModuleIdx, PackageFacts, ParamHow, ParamUse, RefFacts,
        RefTarget, ScanDiag, ScanKey, SymbolKind, ViaArg,
    },
};

const TAG_PACKAGE: u8 = 1;
const TAG_IMPORTS: u8 = 2;
const TAG_ITEMS: u8 = 3;
const TAG_COMPTIME_BODY: u8 = 4;

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
        self.cur = Some(key);
        key
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
        let param_uses = std::mem::take(&mut self.param_uses);

        self.items.push(ItemFacts {
            key,
            visibility: vis,
            execution_mode: mode,
            span,
            iface_fp: i,
            body_fp: b,
            param_uses,
        });

        let popped = self.cur.take();
        debug_assert_eq!(popped, Some(key), "finish_item out of order");
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
            Stmt::Import(s) => {
                self.scan_import_stmt(s);
            }

            Stmt::Package(s) => {
                self.scan_package_stmt(s);
            }

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
            Stmt::TypeAliasDecl(_) => self.unsupported(stmt, "type alias declaration"),
            Stmt::Attributed(stmt) => self.scan_stmt(&stmt.inner, body_rule),
            Stmt::Let(l) => self.scan_static_decl(l),
        }
    }

    fn scan_const_decl(&mut self, c: &ConstStmt<'a, 'bump>) {
        let key = self.begin_item(c.ident, ItemRole::Const);

        let mut ih = FpHasher::new();
        c.stable_hash(&self.cx, &mut ih);

        if let Some(ty) = &c.type_annotation {
            self.walk_type(ty);
        }

        // `const A: usize = B * 2` gives A -> B
        self.walk_expr(&c.value, EdgeKind::ConstEval);

        self.finish_item(
            key,
            c.visibility,
            ExecutionMode::Comptime,
            c.span,
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

    fn scan_impl_decl(&mut self, i: &ImplDecl<'a, 'bump>) {
        let key = self.begin_item(i.target, ItemRole::Type);
        let depth = if let Some(generics) = i.generics {
            self.enter_generics(generics.iter().map(|g| g.type_name))
        } else {
            0
        };

        let mut ih = FpHasher::new();
        i.stable_hash(&self.cx, &mut ih);

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
        self.finish_item(key, i.visibility, ExecutionMode::Runtime, i.span, ih, bh);
    }

    fn scan_interface_decl(&mut self, i: &InterfaceDecl<'a, 'bump>) {
        let key = self.begin_item(i.name, ItemRole::Type);
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
                let ast::Param::Normal(field) = *field else {
                    unreachable!()
                };

                self.walk_type(&field.type_annotation);
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

        let mut bh = if let Some(body) = f.body
            && !body.block.is_empty()
        {
            if exec_mode == ExecutionMode::Comptime {
                ih.u8(TAG_COMPTIME_BODY);
                f.stable_hash_body(&self.cx, &mut ih);
            }

            let mut bh = FpHasher::new();
            f.stable_hash_body(&self.cx, &mut bh);
            bh
        } else if let BodyRule::Required = body_rule {
            self.diags
                .push(ScanDiag::RequiredBodyAbsent { span: f.name_span });
            self.exit_generics(depth);
            return;
        } else {
            FpHasher::new()
        };

        if let Some(params) = f.params {
            for param in params.iter() {
                let ast::Param::Normal(param) = *param else {
                    unreachable!()
                };
                self.walk_type(&param.type_annotation);
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
                self.walk_expr(length, EdgeKind::ConstEval); // fn f() -> [u8; N]  gives  f -> N
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

            TypeKind::AnonStruct { fields } => {
                for field in fields.iter() {
                    self.walk_indirectly_referenced_type(&field.field_type);
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
            TypeKind::Type => {}
        }
    }

    fn walk_expr(&self, expr: &Expr<'a, 'bump>, edge_kind: EdgeKind) {
        match expr {
            Expr::OrElse {
                value,
                else_block,
                span,
            } => todo!(),
            Expr::Attributed { attrs, expr, span } => todo!(),
            Expr::Intrinsic {
                name,
                generic_args,
                arguments,
                span,
            } => todo!(),
            Expr::Null { span } => todo!(),
            Expr::Cast {
                expr,
                target_type,
                span,
            } => todo!(),
            Expr::Number { value, span } => todo!(),
            Expr::Decimal { value, span } => todo!(),
            Expr::String { value, span } => todo!(),
            Expr::Ident { name, span } => todo!(),
            Expr::GenericIdent {
                name,
                generic_args,
                span,
            } => todo!(),
            Expr::Boolean { value, span } => todo!(),
            Expr::Comparison { lhs, op, rhs, span } => todo!(),
            Expr::StructInit {
                callee,
                arguments,
                type_args,
                span,
            } => todo!(),
            Expr::FieldAccess {
                object,
                field,
                span,
            } => todo!(),
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => todo!(),
            Expr::Call {
                callee,
                generic_args,
                arguments,
                span,
            } => todo!(),
            Expr::Get {
                object,
                field,
                span,
            } => todo!(),
            Expr::If { if_stmt, span } => todo!(),
            Expr::Match { match_stmt, span } => todo!(),
            Expr::Assignment { lhs, op, rhs, span } => todo!(),
            Expr::ExprList { expressions, span } => todo!(),
            Expr::Char { value, span } => todo!(),
            Expr::Unary { op, operand, span } => todo!(),
            Expr::ArrayIndex { expr, index, span } => todo!(),
            Expr::Deref { expr, span } => todo!(),
            Expr::This { span } => todo!(),
            Expr::Ref {
                expr,
                ref_kind,
                span,
            } => todo!(),
            Expr::Lambda {
                is_move,
                modifiers,
                params,
                return_type,
                ensures,
                body,
                span,
            } => todo!(),
            Expr::Tuple { values, span } => todo!(),
            Expr::ModulePath { segments, span } => todo!(),
            Expr::ModuleAccess {
                segments,
                member,
                span,
            } => todo!(),
            Expr::ArrayLiteral { elements, span } => todo!(),
            Expr::Undefined { span } => todo!(),
            Expr::Uninit { span } => todo!(),
            Expr::Block(block) => todo!(),
            Expr::UnsafeBlock(unsafe_block) => todo!(),
            Expr::ComptimeBlock(block) => todo!(),
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
