use ir::ast::ExternModifier;
use ir::hir::{
    ConstStmt, Hir, HirErrorHandlerPattern, HirExpr, HirFunc, HirGeneric, HirImpl, HirMatchArm,
    HirModule, HirParam, HirPattern, HirStmt, HirType, InterpolationPart, StrId,
};
use ir::ir_hasher::FxHashMap;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub struct DceConfig {
    /// Free functions that are always live (default: `main`).
    pub entry_points: Vec<StrId>,
    /// Names that are always live as free function, method name and type name.
    pub root_names: Vec<StrId>,
    /// Method names the compiler calls implicitly on live type.
    pub implicit_methods: Vec<StrId>,
    /// Interfaces that mark a type as an allocator (default: `Allocator`).
    /// The drop emitter inserts `free` / `free_raw` calls on allocators
    /// automatically, so a live allocator type keeps EVERY method of EVERY impl.
    pub allocator_interfaces: Vec<StrId>,
    /// Method names that also mark a type as an allocator, whatever interface
    /// it implements.
    pub allocator_marker_methods: Vec<StrId>,
    /// Modules that are kept completely (library mode, `zeta::lang`, ...).
    pub keep_module: Option<fn(&HirModule<'_, '_>) -> bool>,
}

impl DceConfig {
    pub fn new(pool: &StringPool) -> Self {
        let intern = |s: &str| StrId(pool.intern_bytes(s.as_bytes()));
        Self {
            entry_points: vec![intern("main")],
            root_names: vec![
                intern("zeta_sys_args_init"),
                intern("zeta_debug_debug_panic"),
            ],
            implicit_methods: vec![intern("drop")],
            allocator_interfaces: vec![intern("Allocator"), intern("RawAllocator")],
            allocator_marker_methods: vec![intern("free_raw"), intern("free")],
            keep_module: None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DceStats {
    pub funcs_removed: usize,
    pub types_removed: usize,
    pub impls_removed: usize,
    pub impl_methods_removed: usize,
}

struct FuncEntry<'a, 'bump> {
    func: &'bump HirFunc<'a, 'bump>,
    /// Index of the owning impl, `None` for top-level functions.
    owner: Option<usize>,
    module: usize,
    live: bool,
    /// Its name was mentioned as a method (`x.name`) somewhere in live code.
    called: bool,
}

struct TypeEntry<'a, 'bump> {
    item: Hir<'a, 'bump>,
    module: usize,
    live: bool,
}

struct ImplEntry<'a, 'bump> {
    imp: &'bump HirImpl<'a, 'bump>,
    methods: Vec<usize>,
    /// Type entries (struct/enum) this impl targets. Empty = builtin target.
    target_types: Vec<usize>,
    /// Impl of an allocator interface, or has an allocator marker method.
    allocator: bool,
    module: usize,
    live: bool,
}

#[derive(Clone, Copy)]
enum Slot {
    Keep,
    Func(usize),
    Type(usize),
    Impl(usize),
}

#[derive(Clone, Copy)]
enum Work {
    Func(usize),
    Type(usize),
    Impl(usize),
}

pub struct DeadCodeEliminator<'a, 'bump, 'cfg> {
    cfg: &'cfg DceConfig,
    bump: &'bump GrowableBump<'bump>,

    funcs: Vec<FuncEntry<'a, 'bump>>,
    types: Vec<TypeEntry<'a, 'bump>>,
    impls: Vec<ImplEntry<'a, 'bump>>,

    free_by_name: FxHashMap<StrId, Vec<usize>>,
    method_by_name: FxHashMap<StrId, Vec<usize>>,
    type_by_name: FxHashMap<StrId, Vec<usize>>,
    impls_by_type: FxHashMap<usize, Vec<usize>>,
    impls_by_iface: FxHashMap<StrId, Vec<usize>>,

    work: Vec<Work>,
    pub stats: DceStats,
    pub removed: Vec<StrId>,
}

const BUILTIN_TARGETS: &[&str] = &[
    "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize", "f32",
    "f64", "bool", "char", "str", "slice",
];

/// Impl targets the compiler special-cases (primitives, slices): `str.from_utf8`,
/// slice primitives like `len`, ... MIR lowering resolves these by key, not by
/// a call it can see in the HIR, so these impls are always kept whole.
fn is_builtin_target(name: StrId) -> bool {
    let s = name.as_str();
    BUILTIN_TARGETS.contains(&s) || s.starts_with("slice_")
}

fn is_named(f: &HirFunc<'_, '_>, set: &[StrId]) -> bool {
    set.iter().any(|n| *n == f.name || *n == f.unmangled_name)
}

fn register(map: &mut FxHashMap<StrId, Vec<usize>>, a: StrId, b: StrId, idx: usize) {
    map.entry(a).or_default().push(idx);
    if a != b {
        map.entry(b).or_default().push(idx);
    }
}

impl<'a, 'bump, 'cfg> DeadCodeEliminator<'a, 'bump, 'cfg>
where
    'bump: 'a,
{
    pub fn new(cfg: &'cfg DceConfig, bump: &'bump GrowableBump<'bump>) -> Self {
        Self {
            cfg,
            bump,
            funcs: Vec::new(),
            types: Vec::new(),
            impls: Vec::new(),
            free_by_name: FxHashMap::default(),
            method_by_name: FxHashMap::default(),
            type_by_name: FxHashMap::default(),
            impls_by_type: FxHashMap::default(),
            impls_by_iface: FxHashMap::default(),
            work: Vec::new(),
            stats: DceStats::default(),
            removed: Vec::new(),
        }
    }

    pub fn run(&mut self, modules: &[HirModule<'a, 'bump>]) -> Vec<HirModule<'a, 'bump>> {
        let mut root_consts: Vec<&'bump ConstStmt<'a, 'bump>> = Vec::new();
        let mut root_stmts: Vec<&'bump HirStmt<'a, 'bump>> = Vec::new();
        let mut root_exprs: Vec<&'bump HirExpr<'a, 'bump>> = Vec::new();
        let mut slots: Vec<Vec<Slot>> = Vec::with_capacity(modules.len());

        for (mi, module) in modules.iter().enumerate() {
            let mut mslots = Vec::with_capacity(module.items.len());
            for item in module.items.iter() {
                let slot = match *item {
                    Hir::Func(f) => {
                        let idx = self.funcs.len();
                        self.funcs.push(FuncEntry {
                            func: f,
                            owner: None,
                            module: mi,
                            live: false,
                            called: false,
                        });
                        register(&mut self.free_by_name, f.name, f.unmangled_name, idx);
                        if f.impl_target.is_some() {
                            register(&mut self.method_by_name, f.name, f.unmangled_name, idx);
                        }
                        Slot::Func(idx)
                    }
                    Hir::Struct(s) => {
                        Slot::Type(self.push_type(*item, mi, s.name, s.unmangled_name))
                    }
                    Hir::Enum(e) => Slot::Type(self.push_type(*item, mi, e.name, e.unmangled_name)),
                    Hir::Interface(i) => {
                        Slot::Type(self.push_type(*item, mi, i.name, i.unmangled_name))
                    }
                    Hir::Impl(imp) => {
                        let impl_idx = self.impls.len();
                        let mut methods = Vec::new();
                        for m in imp.methods.into_iter().flatten() {
                            let idx = self.funcs.len();
                            self.funcs.push(FuncEntry {
                                func: m,
                                owner: Some(impl_idx),
                                module: mi,
                                live: false,
                                called: false,
                            });
                            register(&mut self.method_by_name, m.name, m.unmangled_name, idx);
                            methods.push(idx);
                        }
                        self.impls.push(ImplEntry {
                            imp,
                            methods,
                            target_types: Vec::new(),
                            allocator: false,
                            module: mi,
                            live: false,
                        });
                        Slot::Impl(impl_idx)
                    }
                    Hir::Const(c) => {
                        root_consts.push(c);
                        Slot::Keep
                    }
                    Hir::Stmt(s) => {
                        root_stmts.push(s);
                        Slot::Keep
                    }
                    Hir::Expr(e) => {
                        root_exprs.push(e);
                        Slot::Keep
                    }
                    Hir::Module(_) => Slot::Keep,
                };
                mslots.push(slot);
            }
            slots.push(mslots);
        }

        for i in 0..self.impls.len() {
            let imp = self.impls[i].imp;
            let targets = self
                .type_by_name
                .get(&imp.target)
                .cloned()
                .unwrap_or_default();
            for &t in &targets {
                self.impls_by_type.entry(t).or_default().push(i);
            }
            self.impls[i].target_types = targets;
            if let Some(iface) = imp.interface {
                self.impls_by_iface.entry(iface).or_default().push(i);
            }

            let cfg = self.cfg;
            let iface_is_alloc = imp.interface.map_or(false, |iface| {
                cfg.allocator_interfaces.contains(&iface)
                    || self.type_by_name.get(&iface).map_or(false, |ids| {
                        ids.iter().any(|&t| {
                            let (a, b) = self.type_names(t);
                            cfg.allocator_interfaces.contains(&a)
                                || cfg.allocator_interfaces.contains(&b)
                        })
                    })
            });
            let has_marker = self.impls[i]
                .methods
                .iter()
                .any(|&m| is_named(self.funcs[m].func, &cfg.allocator_marker_methods));
            self.impls[i].allocator = iface_is_alloc || has_marker;
        }

        let keep: Vec<bool> = modules
            .iter()
            .map(|m| self.cfg.keep_module.map_or(false, |f| f(m)))
            .collect();

        for i in 0..self.funcs.len() {
            let f = self.funcs[i].func;
            let is_entry = self.funcs[i].owner.is_none()
                && self
                    .cfg
                    .entry_points
                    .iter()
                    .any(|e| *e == f.name || *e == f.unmangled_name);
            let is_extern = self.funcs[i].owner.is_none()
                && matches!(f.function_metadata.extern_modifier, ExternModifier::Abi(_));
            if is_entry
                || is_extern
                || (keep[self.funcs[i].module] && self.funcs[i].owner.is_none())
            {
                self.mark_func_idx(i);
            }
        }
        for i in 0..self.types.len() {
            if keep[self.types[i].module] {
                self.mark_type_idx(i);
            }
        }
        for i in 0..self.impls.len() {
            if keep[self.impls[i].module] {
                self.mark_impl(i);
                for m in self.impls[i].methods.clone() {
                    self.mark_func_idx(m);
                }
            }
        }
        for i in 0..self.impls.len() {
            if self.impls[i].target_types.is_empty() || is_builtin_target(self.impls[i].imp.target)
            {
                self.mark_impl(i);
                for m in self.impls[i].methods.clone() {
                    self.mark_func_idx(m);
                }
            }
        }
        let cfg = self.cfg;
        for &n in &cfg.root_names {
            self.mark_free(n);
            self.mark_type(n);
            self.mark_method_forced(n);
        }
        for c in root_consts {
            self.visit_const(c);
        }
        for s in root_stmts {
            self.visit_stmt(s);
        }
        for e in root_exprs {
            self.visit_expr(e);
        }

        while let Some(w) = self.work.pop() {
            match w {
                Work::Func(i) => self.activate_func(i),
                Work::Type(i) => self.activate_type(i),
                Work::Impl(i) => self.activate_impl(i),
            }
        }

        let mut out = Vec::with_capacity(modules.len());
        for (mi, module) in modules.iter().enumerate() {
            let mut items: Vec<Hir<'a, 'bump>> = Vec::with_capacity(module.items.len());
            for (item, slot) in module.items.iter().zip(slots[mi].iter()) {
                match *slot {
                    Slot::Keep => items.push(*item),
                    Slot::Func(i) => {
                        if self.funcs[i].live {
                            items.push(*item);
                        } else {
                            self.stats.funcs_removed += 1;
                            self.removed.push(self.funcs[i].func.name);
                        }
                    }
                    Slot::Type(i) => {
                        if self.types[i].live {
                            items.push(*item);
                        } else {
                            self.stats.types_removed += 1;
                        }
                    }
                    Slot::Impl(i) => {
                        if !self.impls[i].live {
                            self.stats.impls_removed += 1;
                            for &m in &self.impls[i].methods {
                                self.removed.push(self.funcs[m].func.name);
                            }
                            continue;
                        }
                        let imp = self.impls[i].imp;
                        let Some(all) = imp.methods else {
                            items.push(*item);
                            continue;
                        };
                        let kept: Vec<HirFunc<'a, 'bump>> = self.impls[i]
                            .methods
                            .iter()
                            .filter(|&&m| self.funcs[m].live)
                            .map(|&m| *self.funcs[m].func)
                            .collect();
                        if kept.len() == all.len() {
                            items.push(*item);
                        } else {
                            self.stats.impl_methods_removed += all.len() - kept.len();
                            for &m in &self.impls[i].methods {
                                if !self.funcs[m].live {
                                    self.removed.push(self.funcs[m].func.name);
                                }
                            }
                            let mut new_impl = (*imp).clone();
                            new_impl.methods = Some(self.bump.alloc_slice(&kept));
                            items.push(Hir::Impl(self.bump.alloc_value_immutable(new_impl)));
                        }
                    }
                }
            }
            out.push(HirModule {
                items: self.bump.alloc_slice(&items),
                ..*module
            });
        }
        out
    }

    fn push_type(&mut self, item: Hir<'a, 'bump>, module: usize, a: StrId, b: StrId) -> usize {
        let idx = self.types.len();
        self.types.push(TypeEntry {
            item,
            module,
            live: false,
        });
        register(&mut self.type_by_name, a, b, idx);
        idx
    }

    fn type_names(&self, i: usize) -> (StrId, StrId) {
        match self.types[i].item {
            Hir::Struct(s) => (s.name, s.unmangled_name),
            Hir::Enum(e) => (e.name, e.unmangled_name),
            Hir::Interface(it) => (it.name, it.unmangled_name),
            _ => unreachable!("type entries are struct/enum/interface"),
        }
    }

    fn mark_func_idx(&mut self, i: usize) {
        if !self.funcs[i].live {
            self.funcs[i].live = true;
            self.work.push(Work::Func(i));
        }
    }

    fn mark_type_idx(&mut self, i: usize) {
        if !self.types[i].live {
            self.types[i].live = true;
            self.work.push(Work::Type(i));
        }
    }

    fn mark_impl(&mut self, i: usize) {
        if !self.impls[i].live {
            self.impls[i].live = true;
            self.work.push(Work::Impl(i));
        }
    }

    fn mark_free(&mut self, name: StrId) {
        let n = match self.free_by_name.get(&name) {
            Some(v) => v.len(),
            None => return,
        };
        for k in 0..n {
            let i = self.free_by_name[&name][k];
            if self.funcs[i].owner.is_none() {
                self.mark_func_idx(i);
            }
        }
    }

    fn mark_type(&mut self, name: StrId) {
        let n = match self.type_by_name.get(&name) {
            Some(v) => v.len(),
            None => return,
        };
        for k in 0..n {
            let i = self.type_by_name[&name][k];
            self.mark_type_idx(i);
        }
    }

    fn mark_method(&mut self, name: StrId) {
        let n = match self.method_by_name.get(&name) {
            Some(v) => v.len(),
            None => return,
        };
        for k in 0..n {
            let i = self.method_by_name[&name][k];
            self.funcs[i].called = true;
            self.try_activate_method(i);
        }
    }

    fn mark_method_forced(&mut self, name: StrId) {
        let n = match self.method_by_name.get(&name) {
            Some(v) => v.len(),
            None => return,
        };
        for k in 0..n {
            let i = self.method_by_name[&name][k];
            self.funcs[i].called = true;
            self.mark_func_idx(i);
        }
    }

    fn owner_live(&self, imp: usize) -> bool {
        let e = &self.impls[imp];
        e.target_types.is_empty() || e.target_types.iter().any(|&t| self.types[t].live)
    }

    fn try_activate_method(&mut self, i: usize) {
        let e = &self.funcs[i];
        if e.live || !e.called {
            return;
        }
        if let Some(o) = e.owner {
            if !self.owner_live(o) {
                return;
            }
        }
        self.mark_func_idx(i);
    }

    fn activate_func(&mut self, i: usize) {
        let f = self.funcs[i].func;
        let owner = self.funcs[i].owner;
        self.visit_func(f);
        if let Some(o) = owner {
            self.mark_impl(o);
        }
    }

    fn activate_impl(&mut self, i: usize) {
        let imp = self.impls[i].imp;
        self.mark_type(imp.target);
        self.visit_generics(imp.generics);
        for t in imp.interface_generics.into_iter().flatten() {
            self.visit_type(t);
        }
        for t in imp.target_generics.into_iter().flatten() {
            self.visit_type(t);
        }
        if let Some(iface) = imp.interface {
            self.mark_type(iface);
            for m in self.impls[i].methods.clone() {
                self.mark_func_idx(m);
            }
        }
    }

    fn activate_type(&mut self, i: usize) {
        let item = self.types[i].item;
        match item {
            Hir::Struct(s) => {
                self.visit_generics(s.generics);
                for f in s.fields.iter() {
                    self.visit_type(&f.field_type);
                }
            }
            Hir::Enum(e) => {
                self.visit_generics(e.generics);
                for v in e.variants.iter() {
                    for f in v.fields.iter() {
                        self.visit_type(&f.field_type);
                    }
                }
            }
            Hir::Interface(it) => {
                self.visit_generics(it.generics);
                for m in it.methods.into_iter().flatten() {
                    self.visit_func(m);
                }
                for key in [it.name, it.unmangled_name] {
                    let ids = self.impls_by_iface.get(&key).cloned().unwrap_or_default();
                    for imp in ids {
                        if self.owner_live(imp) {
                            self.mark_impl(imp);
                        }
                    }
                }
            }
            _ => {}
        }

        let impls = self.impls_by_type.get(&i).cloned().unwrap_or_default();
        let is_allocator = impls.iter().any(|&imp| self.impls[imp].allocator);
        for imp in impls {
            self.mark_impl(imp);
            for m in self.impls[imp].methods.clone() {
                if is_allocator || is_named(self.funcs[m].func, &self.cfg.implicit_methods) {
                    self.mark_func_idx(m);
                } else {
                    self.try_activate_method(m);
                }
            }
        }
    }

    fn visit_func(&mut self, f: &HirFunc<'a, 'bump>) {
        self.visit_generics(f.generics);
        for p in f.params.into_iter().flatten() {
            if let HirParam::Normal { param_type, .. } = p {
                self.visit_type(param_type);
            }
        }
        if let Some(rt) = &f.return_type {
            self.visit_type(rt);
        }
        if let Some(body) = &f.body {
            self.visit_stmt(body);
        }
    }

    fn visit_generics(&mut self, generics: Option<&[HirGeneric<'a, 'bump>]>) {
        for g in generics.into_iter().flatten() {
            self.visit_types(g.constraints);
            if let Some(d) = &g.default_type {
                self.visit_type(d);
            }
        }
    }

    fn visit_types(&mut self, ts: &[HirType<'a, 'bump>]) {
        for t in ts {
            self.visit_type(t);
        }
    }

    fn visit_type(&mut self, ty: &HirType<'a, 'bump>) {
        match ty {
            HirType::Struct {
                name, type_args, ..
            }
            | HirType::Enum {
                name, type_args, ..
            } => {
                self.mark_type(*name);
                self.visit_types(type_args);
            }
            HirType::DynInterface(name, args) => {
                self.mark_type(*name);
                self.visit_types(args);
            }
            HirType::SafePointer { inner, .. }
            | HirType::UnsafePointer { inner, .. }
            | HirType::Ref { inner, .. } => self.visit_type(inner),
            HirType::OwnedPointer { inner, .. } => self.visit_type(inner),
            HirType::Nullable(inner) | HirType::Array(inner, _) | HirType::Slice(inner) => {
                self.visit_type(inner)
            }
            HirType::Range { elem, .. } => self.visit_type(elem),
            HirType::Lambda {
                params,
                return_type,
            } => {
                self.visit_types(params);
                self.visit_type(return_type);
            }
            HirType::Dyn { bounds } => self.visit_types(bounds),
            HirType::Tuple(ts) => self.visit_types(ts),
            _ => {}
        }
    }

    fn visit_const(&mut self, c: &ConstStmt<'a, 'bump>) {
        self.visit_type(&c.ty);
        self.visit_expr(&c.value);
    }

    fn visit_pattern(&mut self, p: &HirPattern<'bump>) {
        match p {
            HirPattern::Struct { name, fields } => {
                self.mark_type(*name);
                for (_, fp) in fields.iter() {
                    self.visit_pattern(fp);
                }
            }
            HirPattern::Tuple(ps) | HirPattern::Array(ps) | HirPattern::Or(ps) => {
                for sp in ps.iter() {
                    self.visit_pattern(sp);
                }
            }
            HirPattern::Ident(..)
            | HirPattern::Number(_)
            | HirPattern::String(_)
            | HirPattern::Boolean(_)
            | HirPattern::EnumVariant { .. }
            | HirPattern::Wildcard
            | HirPattern::Null => {}
        }
    }

    fn visit_arm(&mut self, arm: &HirMatchArm<'a, 'bump>) {
        self.visit_pattern(&arm.pattern);
        if let Some(g) = arm.guard {
            self.visit_expr(g);
        }
        self.visit_stmt(arm.body);
    }

    fn visit_stmts(&mut self, ss: &[HirStmt<'a, 'bump>]) {
        for s in ss {
            self.visit_stmt(s);
        }
    }

    fn visit_exprs(&mut self, es: &[HirExpr<'a, 'bump>]) {
        for e in es {
            self.visit_expr(e);
        }
    }

    fn visit_stmt(&mut self, s: &HirStmt<'a, 'bump>) {
        match s {
            HirStmt::Let {
                ty,
                value,
                catch_pattern,
                ..
            } => {
                self.visit_type(ty);
                self.visit_expr(value);
                match catch_pattern {
                    Some(HirErrorHandlerPattern::Single {
                        error_type, body, ..
                    }) => {
                        self.visit_type(error_type);
                        self.visit_stmts(body);
                    }
                    Some(HirErrorHandlerPattern::Multiple { branches }) => {
                        for b in branches.iter() {
                            self.visit_type(&b.error_type);
                            self.visit_stmts(b.body);
                        }
                    }
                    None => {}
                }
            }
            HirStmt::Const(c) => self.visit_const(c),
            HirStmt::Return(e, _) | HirStmt::Break(e, _) => {
                if let Some(e) = e {
                    self.visit_expr(e);
                }
            }
            HirStmt::Expr(e) => self.visit_expr(e),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                self.visit_expr(cond);
                self.visit_stmts(then_block);
                if let Some(e) = else_block {
                    self.visit_stmt(e);
                }
            }
            HirStmt::While { cond, body } => {
                self.visit_expr(cond);
                self.visit_stmt(body);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.visit_stmt(i);
                }
                if let Some(c) = condition {
                    self.visit_expr(c);
                }
                if let Some(i) = increment {
                    self.visit_expr(i);
                }
                self.visit_stmt(body);
            }
            HirStmt::Match { expr, arms, .. } => {
                self.visit_expr(expr);
                for a in arms.iter() {
                    self.visit_arm(a);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => self.visit_stmt(body),
            HirStmt::Block { body, .. } => self.visit_stmts(body),
            HirStmt::Import(path, _) => {
                if let Some(m) = path.member {
                    self.mark_free(m);
                    self.mark_type(m);
                }
            }
            HirStmt::Continue(_) | HirStmt::Package(..) => {}
        }
    }

    fn visit_expr(&mut self, e: &HirExpr<'a, 'bump>) {
        match e {
            HirExpr::OrElse {
                value, else_body, ..
            } => {
                self.visit_expr(value);
                self.visit_stmts(else_body);
            }
            HirExpr::Intrinsic {
                type_args, args, ..
            } => {
                self.visit_types(type_args);
                self.visit_exprs(args);
            }
            HirExpr::Cast {
                expr, target_type, ..
            } => {
                self.visit_expr(expr);
                self.visit_type(target_type);
            }
            HirExpr::Uninit { ty, .. } | HirExpr::Undefined { ty, .. } => self.visit_type(ty),
            HirExpr::Ident(name, _) => {
                self.mark_free(*name);
                self.mark_type(*name);
            }
            HirExpr::GenericIdent(name, targs, _) => {
                self.mark_free(*name);
                self.mark_type(*name);
                self.visit_types(targs);
            }
            HirExpr::Tuple(es, _) => self.visit_exprs(es),
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            HirExpr::Call {
                callee,
                args,
                type_args,
                ..
            } => {
                self.visit_expr(callee);
                self.visit_exprs(args);
                if let Some(ta) = type_args {
                    self.visit_types(ta);
                }
            }
            HirExpr::InterfaceCall {
                callee,
                args,
                interface,
                ..
            } => {
                self.mark_type(*interface);
                self.visit_expr(callee);
                self.visit_exprs(args);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                self.mark_method(*field);
                if matches!(**object, HirExpr::Ident(..) | HirExpr::ModuleAccess(_)) {
                    self.mark_free(*field);
                    self.mark_type(*field);
                }
                self.visit_expr(object);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.visit_expr(target);
                self.visit_expr(value);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.visit_expr(e);
                    }
                }
            }
            HirExpr::EnumInit {
                enum_name,
                args,
                type_args,
                ..
            } => {
                self.mark_type(*enum_name);
                self.visit_exprs(args);
                if let Some(ta) = type_args {
                    self.visit_types(ta);
                }
            }
            HirExpr::ExprList { list, .. } => self.visit_exprs(list),
            HirExpr::StructInit {
                name,
                args,
                type_args,
                ..
            } => {
                self.visit_expr(name);
                for a in args.iter() {
                    self.visit_expr(&a.value);
                }
                if let Some(ta) = type_args {
                    self.visit_types(ta);
                }
            }
            HirExpr::Deref { expr, .. } | HirExpr::Ref { expr, .. } => self.visit_expr(expr),
            // Three shapes share this node:
            //   `std::io.println`   path = module segments, member = free fn / type
            //   `IOError.from_errno` path = [Type],        member = static method
            //   `IOError.WriteZero`  path = [Enum],        member = variant
            HirExpr::ModuleAccess(m) => {
                for seg in m.path.iter() {
                    self.mark_type(*seg);
                }
                self.mark_free(m.member);
                self.mark_type(m.member);
                self.mark_method(m.member);
            }
            HirExpr::Lambda {
                params,
                return_type,
                body,
                ..
            } => {
                for p in params.iter() {
                    if let Some(t) = &p.param_type {
                        self.visit_type(t);
                    }
                }
                self.visit_type(return_type);
                self.visit_stmt(body);
            }
            HirExpr::Index { object, index, .. } => {
                self.visit_expr(object);
                self.visit_expr(index);
            }
            HirExpr::ArrayLiteral { elements, .. } => self.visit_exprs(elements),
            HirExpr::If { if_stmt, .. } => self.visit_stmt(if_stmt),
            HirExpr::Match { expr, arms, .. } => {
                self.visit_expr(expr);
                for a in arms.iter() {
                    self.visit_arm(a);
                }
            }
            HirExpr::Block { body, .. } => self.visit_stmts(body),
            HirExpr::Range { start, end, .. } => {
                self.visit_expr(start);
                self.visit_expr(end);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.visit_expr(object);
                self.visit_expr(start);
                self.visit_expr(end);
            }
            HirExpr::Null(_)
            | HirExpr::Number(..)
            | HirExpr::Char(..)
            | HirExpr::String(..)
            | HirExpr::Boolean(..)
            | HirExpr::Decimal(..)
            | HirExpr::This { .. }
            | HirExpr::UnknownIntrinsic { .. } => {}
        }
    }
}
