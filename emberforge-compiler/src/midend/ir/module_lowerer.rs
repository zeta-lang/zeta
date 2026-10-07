use crate::midend::copy_analysis::drop_glue::{DropGlueBuilder, DropGlueRegistry};
use crate::midend::ir::mir_lowering::FunctionLowerer;
use codex_dependency_graph::DepGraph;
use ir::hir::{
    Hir, HirEnum, HirExpr, HirFunc, HirInterface, HirModule, HirParam, HirStmt, HirStruct, HirType,
    Operator, StrId, ThisPassingKind,
};
use ir::hir_utils::hir_contains_this;
use ir::ir_conversion::lower_type_hir;
use ir::ir_hasher::{FxHashMap, HashMap, HashSet};
use ir::registry::global_registry::{
    GlobalRegistry, StaticDef, StaticInit, StaticReloc, static_flag_name,
};
use ir::span::SourceSpan;
use ir::ssa_ir::{AllocatorKind, Function, Module, SsaType};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;
use zetaruntime::intern_fmt;
use zetaruntime::string_pool::StringPool;

pub struct MirModuleLowerer<'a, 'cx, 'r, 'bump, 'g>
where
    'bump: 'a,
    'bump: 'cx,
    'cx: 'a,
    'r: 'a,
{
    pub module: Module<'a, 'bump>,

    interface_methods: HashMap<StrId, Vec<(StrId, Vec<SsaType>, SsaType)>>,
    interface_id_map: HashMap<StrId, usize>,
    interface_method_slots: HashMap<StrId, HashMap<StrId, usize>>,

    struct_mangled_map: HashMap<StrId, HashMap<StrId, StrId>>,
    struct_vtable_slots: HashMap<StrId, Vec<StrId>>,
    struct_method_slots: HashMap<StrId, HashMap<StrId, usize>>,
    struct_field_offsets: HashMap<StrId, HashMap<StrId, usize>>,

    phantom_data: PhantomData<&'cx ()>,

    context: Arc<StringPool>,
    extern_c_names: Rc<HashSet<StrId>>,
    enums: HashMap<StrId, HirEnum<'a, 'bump>>,

    enum_variant_tags: HashMap<StrId, HashMap<StrId, usize>>,
    struct_interfaces: HashMap<StrId, Vec<StrId>>,
    pub dep_graph: &'a RefCell<DepGraph>,
    pub module_idx: usize,
    g_phantom_data: PhantomData<&'g ()>,
    glue_registry: DropGlueRegistry,
    allocator_kind: HashMap<StrId, AllocatorKind>,
    bump: GrowableBump<'bump>,
    interface_default_methods: HashMap<StrId, HashMap<StrId, HirFunc<'a, 'bump>>>,
    module_import_aliases: HashMap<usize, HashMap<StrId, usize>>,
    module_named_imports: HashMap<usize, HashMap<StrId, usize>>,
    constants: HashMap<StrId, HirExpr<'a, 'bump>>,
    instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_struct_methods: Rc<RefCell<FxHashMap<StrId, FxHashMap<StrId, StrId>>>>,
    registry: GlobalRegistry<'r, 'bump>,
    dynamic_static_inits: Vec<(StrId, HirExpr<'a, 'bump>)>,
}

impl<'a, 'cx, 'r, 'bump, 'g> MirModuleLowerer<'a, 'cx, 'r, 'bump, 'g>
where
    'bump: 'a,
    'bump: 'cx,
    'cx: 'a,
    'r: 'a,
{
    pub fn new(
        context: Arc<StringPool>,
        extern_c_names: Rc<HashSet<StrId>>,
        dep_graph: &'a RefCell<DepGraph>,
        module_idx: usize,
        glue_registry: DropGlueRegistry,
        instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_struct_methods: Rc<RefCell<FxHashMap<StrId, FxHashMap<StrId, StrId>>>>,
        registry: GlobalRegistry<'r, 'bump>,
    ) -> Self {
        let mut enum_variant_tags: HashMap<StrId, HashMap<StrId, usize>> = HashMap::default();

        let nullable_enum_name = StrId::from_static("__nullable");
        let mut nullable_tags = HashMap::default();
        nullable_tags.insert(StrId::from_static("null"), 0usize);
        nullable_tags.insert(StrId::from_static("some"), 1usize);
        enum_variant_tags.insert(nullable_enum_name, nullable_tags);

        let throws_enum_name = StrId::from_static("__throws");
        let mut throws_tags = HashMap::default();
        throws_tags.insert(StrId::from_static("__success"), 0usize);
        enum_variant_tags.insert(throws_enum_name, throws_tags);

        Self {
            module: Module::new(),
            interface_methods: HashMap::default(),
            interface_id_map: HashMap::default(),
            interface_method_slots: HashMap::default(),
            struct_mangled_map: HashMap::default(),
            struct_vtable_slots: HashMap::default(),
            struct_method_slots: HashMap::default(),
            struct_field_offsets: HashMap::default(),
            interface_default_methods: HashMap::default(),
            module_import_aliases: HashMap::default(),
            module_named_imports: HashMap::default(),
            constants: HashMap::default(),
            enum_variant_tags,
            struct_interfaces: HashMap::default(),
            enums: HashMap::default(),
            phantom_data: PhantomData,
            context,
            extern_c_names,

            dep_graph,
            module_idx,
            g_phantom_data: PhantomData,
            allocator_kind: HashMap::default(),
            glue_registry,
            bump: GrowableBump::new(4096, 8),
            instantiated_functions,
            instantiated_struct_methods,
            registry,
            dynamic_static_inits: Vec::new(),
        }
    }

    fn register_enum(&mut self, hir_enum: &ir::hir::HirEnum<'a, 'bump>) {
        self.enums.insert(hir_enum.name, *hir_enum);
        self.module.enums.insert(hir_enum.name, hir_enum.clone());
        let mut tags = HashMap::default();
        for (i, variant) in hir_enum.variants.iter().enumerate() {
            tags.insert(variant.name, i);
        }
        self.enum_variant_tags.insert(hir_enum.name, tags);
    }

    pub fn lower_all_modules(
        mut self,
        hir_modules: &[HirModule<'a, 'bump>],
        compilation_order: &[usize],
    ) -> Module<'a, 'bump> {
        for hir_module in hir_modules {
            for item in hir_module.items {
                if let Hir::Enum(hir_enum) = item {
                    self.register_enum(*hir_enum);
                }
            }
        }

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                match item {
                    Hir::Interface(iface) => self.lower_interface(*iface),
                    Hir::Impl(impl_block) => {
                        if let Some(iface) = impl_block.interface {
                            self.struct_interfaces
                                .entry(impl_block.target)
                                .or_insert_with(Vec::new)
                                .push(iface);
                        }
                    }
                    _ => {}
                }
            }
        }

        self.compute_allocator_kinds();

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                if let Hir::Struct(s) = item {
                    self.register_struct(*s); // only insert + mangled_map entry
                }
            }
        }
        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                if let Hir::Struct(s) = item {
                    if s.generics.is_none() {
                        self.compute_field_offsets(s);
                    }
                }
            }
        }

        for &idx in compilation_order {
            let mut aliases = HashMap::default();
            let mut named = HashMap::default();
            for import_path in hir_modules[idx].imports {
                let Some(target_idx) = self
                    .dep_graph
                    .borrow()
                    .resolve_module_path(import_path.path)
                else {
                    continue;
                };
                match import_path.member {
                    None => {
                        if let Some(&last) = import_path.path.last() {
                            aliases.insert(last, target_idx);
                        }
                    }
                    Some(member) => {
                        named.insert(member, target_idx);
                    }
                }
            }
            self.module_import_aliases.insert(idx, aliases);
            self.module_named_imports.insert(idx, named);
        }

        let mut ambiguous: HashSet<(StrId, StrId)> = HashSet::default();

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                match item {
                    Hir::Func(func) => match func.impl_target {
                        Some(struct_name) => {
                            if func.generics.is_none() {
                                let map = self
                                    .struct_mangled_map
                                    .entry(struct_name)
                                    .or_insert_with(|| HashMap::default());
                                map.insert(func.name, func.name); // exact lookup always works
                                match map.get(&func.unmangled_name).copied() {
                                    Some(prev) if prev != func.name => {
                                        ambiguous.insert((struct_name, func.unmangled_name));
                                    }
                                    _ => {
                                        map.insert(func.unmangled_name, func.name);
                                    }
                                }
                                self.register_method_signature(func);
                            }
                        }
                        None => {
                            if func.generics.is_none() {
                                let f = Function::from_signature(
                                    func,
                                    &self.module.structs,
                                    &self.module.enums,
                                    &self.module.interfaces,
                                    &self.context,
                                );
                                self.module.functions.insert(f.name, f);

                                // Closure fns are callable through their env: `env.__call(args)`.
                                if let Some(env) = Self::closure_env_of(func) {
                                    let call = StrId::from_static("__call");
                                    self.struct_mangled_map
                                        .entry(env)
                                        .or_insert_with(|| HashMap::default())
                                        .insert(call, func.name);
                                }
                            }
                        }
                    },
                    Hir::Impl(impl_block) => {
                        if let Some(methods) = impl_block.methods {
                            for method in methods {
                                if method.generics.is_none() && impl_block.generics.is_none() {
                                    let map = self
                                        .struct_mangled_map
                                        .entry(impl_block.target)
                                        .or_insert_with(|| HashMap::default());
                                    map.insert(method.unmangled_name, method.name);
                                    self.register_method_signature(method);
                                }
                            }
                        }
                    }
                    Hir::Const(c) => {
                        self.constants.insert(c.name, c.value.clone());
                    }
                    _ => {}
                }
            }
        }

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                if let Hir::Stmt(HirStmt::Let {
                    name,
                    ty,
                    value,
                    is_static: true,
                    span,
                    ..
                }) = item
                {
                    self.register_static_def(*name, ty, value, *span);
                }
            }
        }

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                if let Hir::Struct(ty_struct) = item {
                    self.build_struct_vtable(ty_struct);
                }
            }
        }

        let owned_structs: Vec<StrId> = compilation_order
            .iter()
            .flat_map(|&idx| {
                hir_modules[idx].items.iter().filter_map(|item| match item {
                    Hir::Struct(s) => Some(s.name),
                    _ => None,
                })
            })
            .collect();
        for (name, func) in DropGlueBuilder::build_all(
            &self.glue_registry,
            &self.module.structs,
            &self.module.enums,
            &self.struct_mangled_map,
            &self.struct_field_offsets,
            &self.allocator_kind,
            self.context.clone(),
            &owned_structs,
            self.instantiated_functions.clone(),
            self.instantiated_struct_methods.clone(),
            &self.bump,
        ) {
            self.module.functions.insert(name, func);
        }

        for &idx in compilation_order {
            self.module_idx = idx;
            for item in hir_modules[idx].items {
                match item {
                    Hir::Func(func) => {
                        if func.generics.is_none() {
                            self.lower_function_body(func);
                        }
                    }
                    Hir::Interface(iface) => {
                        if let Some(methods) = iface.methods {
                            for m in methods.iter() {
                                if m.generics.is_none() && m.body.is_some() {
                                    self.lower_function_body(m);
                                }
                            }
                        }
                    }
                    Hir::Impl(impl_block) => {
                        if impl_block.generics.is_none() {
                            if let Some(methods) = impl_block.methods {
                                for method in methods {
                                    if method.generics.is_none() {
                                        self.lower_function_body(method);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for (s, m) in &ambiguous {
            if let Some(map) = self.struct_mangled_map.get_mut(s) {
                map.remove(m);
            }
        }

        self.module
    }

    fn compute_allocator_kinds(&mut self) {
        let allocator_iface = StrId::from_static("Allocator");
        let raw_allocator_iface = StrId::from_static("RawAllocator");

        for (&struct_name, ifaces) in self.struct_interfaces.iter() {
            if ifaces.contains(&allocator_iface) {
                self.allocator_kind
                    .insert(struct_name, AllocatorKind::Owning);
            } else if ifaces.contains(&raw_allocator_iface) {
                self.allocator_kind
                    .insert(struct_name, AllocatorKind::RawOnly);
            }
        }
    }

    fn register_method_signature(&mut self, hir_method: &HirFunc<'a, 'bump>) {
        let func = Function::from_signature(
            hir_method,
            &self.module.structs,
            &self.module.enums,
            &self.module.interfaces,
            &self.context,
        );
        self.module.functions.insert(func.name, func);
    }

    fn register_struct(&mut self, hir_struct: &HirStruct<'a, 'bump>) {
        if hir_struct.generics.is_some() {
            return;
        }

        self.struct_mangled_map
            .entry(hir_struct.name)
            .or_insert_with(|| HashMap::default());

        self.module
            .structs
            .insert(hir_struct.name, hir_struct.clone());
    }

    fn lower_iface_ty(&self, t: &HirType<'a, 'bump>) -> SsaType {
        if hir_contains_this(t) {
            SsaType::Dyn // implementer-specific; erased in the vtable signature
        } else {
            lower_type_hir(t, &self.enums, &self.module.structs)
        }
    }

    fn lower_interface(&mut self, hir_iface: &HirInterface<'a, 'bump>) {
        let iface_id = self.interface_id_map.len();
        self.interface_id_map.insert(hir_iface.name, iface_id);

        self.module
            .interfaces
            .insert(hir_iface.name, hir_iface.clone());

        let mut methods = Vec::new();
        let mut slot_map = HashMap::default();
        let mut defaults = HashMap::default();
        if let Some(hir_methods) = hir_iface.methods {
            for m in hir_methods.iter() {
                if m.generics.is_some() {
                    continue;
                }
                let slot = methods.len();
                let param_types: Vec<SsaType> = m
                    .params
                    .unwrap_or_default()
                    .iter()
                    .map(|p| match p {
                        HirParam::Normal {
                            name: _,
                            param_type,
                            span: _,
                            multi_place: _,
                        } => self.lower_iface_ty(&param_type),
                        HirParam::This {
                            kind,
                            span: _,
                            multi_place: _,
                        } => match kind {
                            ThisPassingKind::Move | ThisPassingKind::MoveMut => SsaType::Dyn,
                            _ => SsaType::Pointer(
                                ir::ssa_ir::SsaPointerKind::UnsafeMut,
                                Box::new(SsaType::Dyn),
                            ),
                        },
                    })
                    .collect::<Vec<_>>();
                let ret = m
                    .return_type
                    .as_ref()
                    .map(|t| self.lower_iface_ty(t))
                    .unwrap_or(SsaType::Void);
                methods.push((m.unmangled_name.clone(), param_types, ret));
                slot_map.insert(m.unmangled_name.clone(), slot);
                slot_map.insert(m.name.clone(), slot);

                if m.body.is_some() {
                    let func = Function::from_signature(
                        m,
                        &self.module.structs,
                        &self.module.enums,
                        &self.module.interfaces,
                        &self.context,
                    );
                    self.module.functions.insert(func.name, func);
                    defaults.insert(m.unmangled_name, *m);
                }
            }
        }

        self.interface_methods
            .insert(hir_iface.name.clone(), methods);
        self.interface_method_slots.insert(hir_iface.name, slot_map);
        self.interface_default_methods
            .insert(hir_iface.name, defaults);
    }

    fn register_static_def(
        &mut self,
        name: StrId,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        let target = ir::layout::TargetInfo { ptr_bytes: 8 };
        let ssa_ty = lower_type_hir(ty, &self.enums, &self.module.structs);
        let size = ir::layout::sizeof_ssa(&ssa_ty, target)
            .unwrap_or_else(|e| panic!("static `{name}`: unknown size at {span}: {e:?}"));

        let init = match value {
            HirExpr::Uninit { .. } | HirExpr::Undefined { .. } => StaticInit::Zero,
            _ => {
                let mut bytes = vec![0u8; size];
                let mut relocs: Vec<StaticReloc> = Vec::new();
                if self.write_static_init(value, &ssa_ty, &mut bytes, 0, &mut relocs, 0) {
                    if relocs.is_empty() {
                        StaticInit::Bytes(bytes)
                    } else {
                        StaticInit::Relocated { bytes, relocs }
                    }
                } else {
                    // calls, enums (for now), unary minus, generic structs...: init at top of `main`
                    self.dynamic_static_inits.push((name, value.clone()));
                    StaticInit::Zero
                }
            }
        };
        // `undefined` is zeroed but initialized; only `uninit` starts uninitialized.
        let initialized = !matches!(value, HirExpr::Uninit { .. });

        let def = StaticDef {
            name,
            ty: ssa_ty,
            init,
            symbol: format!("__zeta_static_{}", name),
        };
        let flag_name = static_flag_name(&self.context, name);
        let flag = StaticDef {
            name: flag_name,
            ty: SsaType::U8,
            init: if initialized {
                StaticInit::Bytes(vec![1])
            } else {
                StaticInit::Zero
            },
            symbol: format!("__zeta_static_{}", flag_name),
        };

        for d in [def, flag] {
            self.registry.statics.borrow_mut().insert(d.name, d.clone());
            self.module.statics.insert(d.name, d);
        }
    }

    fn const_int(&self, e: &HirExpr<'a, 'bump>, depth: u32) -> Option<i64> {
        if depth > 16 {
            return None;
        }
        match e {
            HirExpr::Number(n, _) => Some(*n as i64),
            HirExpr::Cast { expr, .. } => self.const_int(expr, depth + 1),
            HirExpr::Ident(n, _) => self
                .constants
                .get(n)
                .and_then(|c| self.const_int(c, depth + 1)),
            HirExpr::Binary {
                left, op, right, ..
            } => {
                let l = self.const_int(left, depth + 1)?;
                let r = self.const_int(right, depth + 1)?;
                match op {
                    Operator::Add => Some(l.wrapping_add(r)),
                    Operator::Subtract => Some(l.wrapping_sub(r)),
                    Operator::Multiply => Some(l.wrapping_mul(r)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn write_static_init(
        &self,
        e: &HirExpr<'a, 'bump>,
        ty: &SsaType,
        out: &mut [u8],
        base: usize,
        relocs: &mut Vec<StaticReloc>,
        depth: u32,
    ) -> bool {
        if depth > 32 {
            return false;
        }
        let target = ir::layout::TargetInfo { ptr_bytes: 8 };
        match (ty, e) {
            (_, HirExpr::Undefined { .. }) | (_, HirExpr::Uninit { .. }) => true, // zeroed
            (SsaType::Nullable(_) | SsaType::Pointer(..), HirExpr::Null(_)) => true, // 0 / tag 0

            // str = { ptr, len }; ptr is a relocation to the raw bytes
            (SsaType::String, HirExpr::String(s, _)) if out.len() == 16 => {
                let len = self.context.resolve_bytes(s).len();
                out[8..16].copy_from_slice(&(len as u64).to_le_bytes());
                relocs.push(StaticReloc {
                    offset: base,
                    string: *s,
                });
                true
            }

            (SsaType::F32, HirExpr::Decimal(d, _)) if out.len() == 4 => {
                out.copy_from_slice(&(*d as f32).to_le_bytes());
                true
            }
            (SsaType::F64, HirExpr::Decimal(d, _)) if out.len() == 8 => {
                out.copy_from_slice(&(*d as f64).to_le_bytes());
                true
            }
            (SsaType::F32, _) if out.len() == 4 => match self.const_int(e, 0) {
                Some(i) => {
                    out.copy_from_slice(&(i as f32).to_le_bytes());
                    true
                }
                None => false,
            },
            (SsaType::F64, _) if out.len() == 8 => match self.const_int(e, 0) {
                Some(i) => {
                    out.copy_from_slice(&(i as f64).to_le_bytes());
                    true
                }
                None => false,
            },
            (SsaType::Bool, HirExpr::Boolean(b, _)) if !out.is_empty() => {
                out[0] = *b as u8;
                true
            }
            (SsaType::Char, HirExpr::Char(c, _)) if out.len() == 4 => {
                out.copy_from_slice(&(*c as u32).to_le_bytes());
                true
            }
            (t, _) if t.is_integer() && t != &SsaType::Char => match self.const_int(e, 0) {
                Some(i) if out.len() <= 16 => {
                    let b = (i as i128).to_le_bytes();
                    out.copy_from_slice(&b[..out.len()]);
                    true
                }
                _ => false,
            },
            (SsaType::Array(inner, n), HirExpr::ArrayLiteral { elements, .. }) => {
                let Ok(stride) = ir::layout::sizeof_ssa(inner, target) else {
                    return false;
                };
                if elements.len() > *n || stride * elements.len() > out.len() {
                    return false;
                }
                elements.iter().enumerate().all(|(i, el)| {
                    self.write_static_init(
                        el,
                        inner,
                        &mut out[i * stride..(i + 1) * stride],
                        base + i * stride,
                        relocs,
                        depth + 1,
                    )
                })
            }
            (SsaType::User(sname, _, ftys), HirExpr::StructInit { args, .. }) => {
                let Some(def) = self.module.structs.get(sname) else {
                    return false;
                };
                let Some(offs) = self.struct_field_offsets.get(sname) else {
                    return false;
                };
                args.iter().all(|a| {
                    let Some(idx) = def.fields.iter().position(|f| f.name == a.name) else {
                        return false;
                    };
                    let Some(&off) = offs.get(&a.name) else {
                        return false;
                    };
                    let fty = &ftys[idx];
                    let Ok(sz) = ir::layout::sizeof_ssa(fty, target) else {
                        return false;
                    };
                    if off + sz > out.len() {
                        return false;
                    }
                    self.write_static_init(
                        &a.value,
                        fty,
                        &mut out[off..off + sz],
                        base + off,
                        relocs,
                        depth + 1,
                    )
                })
            }
            (SsaType::Tuple(tys), HirExpr::Tuple(es, _)) if tys.len() == es.len() => {
                let mut cursor = 0usize;
                for (t, el) in tys.iter().zip(es.iter()) {
                    let Ok(l) = ir::layout::layout_of_ssa(t, target) else {
                        return false;
                    };
                    cursor = ir::layout::round_up_to_align(cursor, l.align);
                    if cursor + l.size > out.len() {
                        return false;
                    }
                    if !self.write_static_init(
                        el,
                        t,
                        &mut out[cursor..cursor + l.size],
                        base + cursor,
                        relocs,
                        depth + 1,
                    ) {
                        return false;
                    }
                    cursor += l.size;
                }
                true
            }
            (_, HirExpr::Ident(n, _)) => self.constants.get(n).map_or(false, |c| {
                self.write_static_init(c, ty, out, base, relocs, depth + 1)
            }),
            _ => false,
        }
    }

    fn compute_field_offsets(&mut self, hir_struct: &HirStruct<'a, 'bump>) {
        let mut offsets = HashMap::default();
        let mut current_offset = 0usize;

        for f in hir_struct.fields.iter() {
            let field_ssa_ty = lower_type_hir(&f.field_type, &self.enums, &self.module.structs);
            if let SsaType::User(n, _, fs) = &field_ssa_ty {
                if fs.is_empty() {
                    if let Some(def) = self.module.structs.get(n) {
                        assert!(
                            def.fields.is_empty(),
                            "struct `{n}` lowered as ZST but has fields"
                        );
                    }
                }
            }

            let layout =
                ir::layout::layout_of_ssa(&field_ssa_ty, ir::layout::TargetInfo { ptr_bytes: 8 })
                    .unwrap_or_else(|e| {
                        panic!("failed to compute alignment for field {}: {:?}", f.name, e)
                    });

            debug_assert!(
                layout.align > 0,
                "layout_of_ssa returned align=0 for field `{}` of struct `{}` (type {:?}); \
                 this would corrupt every subsequent field's offset",
                f.name,
                hir_struct.name,
                field_ssa_ty
            );

            current_offset = ir::layout::round_up_to_align(current_offset, layout.align);
            offsets.insert(f.name, current_offset);
            current_offset += layout.size;
        }

        self.struct_field_offsets.insert(hir_struct.name, offsets);
    }

    fn build_struct_vtable(&mut self, hir_struct: &HirStruct<'a, 'bump>) {
        self.build_vtable_for_target(hir_struct.name);
    }

    fn build_vtable_for_target(&mut self, target_name: StrId) {
        let Some(interfaces) = self.struct_interfaces.get(&target_name).cloned() else {
            return;
        };

        let target_methods = self
            .struct_mangled_map
            .get(&target_name)
            .unwrap_or_else(|| {
                panic!("Missing mangled method map for impl target {}", target_name)
            });

        for iface_name in &interfaces {
            let Some(iface_methods) = self.interface_methods.get(iface_name) else {
                continue;
            };
            let hir_iface = self.module.interfaces.get(iface_name).unwrap_or_else(|| {
                panic!(
                    "Target {} implements unknown interface {}",
                    target_name, iface_name
                )
            });
            let unmangled_names: Vec<StrId> = hir_iface
                .methods
                .unwrap_or(&[])
                .iter()
                .map(|m| m.unmangled_name)
                .collect();

            let mut vtable_slots = Vec::new();
            let mut slot_map = HashMap::default();
            for (slot, (method_name, _, _)) in iface_methods.iter().enumerate() {
                let unmangled_name = unmangled_names.get(slot).copied().unwrap_or(*method_name);
                let mangled = target_methods
                    .get(&unmangled_name)
                    .copied()
                    .or_else(|| {
                        self.interface_default_methods
                            .get(iface_name)
                            .and_then(|defaults| defaults.get(&unmangled_name))
                            .map(|f| f.name)
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "Target {} implements interface {} but does not provide method {}, \
                             and the interface has no default body for it",
                            target_name, iface_name, unmangled_name
                        )
                    });
                vtable_slots.push(mangled);
                slot_map.insert(unmangled_name, slot);
            }

            self.struct_vtable_slots.insert(
                StrId(intern_fmt!(self.context, "{}_{}", target_name, iface_name)),
                vtable_slots,
            );
            self.struct_method_slots.insert(target_name, slot_map);
        }
    }

    fn lower_function_body(&mut self, hir_fn: &HirFunc<'a, 'bump>) {
        let mut function: Function = self
            .module
            .functions
            .remove(&hir_fn.name)
            .expect("function signature should already be registered");

        let static_inits: Vec<(StrId, HirExpr<'a, 'bump>)> = if hir_fn.name.as_str() == "main" {
            self.dynamic_static_inits.clone()
        } else {
            Vec::new()
        };

        let mut fl = FunctionLowerer::new(
            &mut function,
            hir_fn,
            &self.module.functions,
            &self.module.functions,
            &self.struct_field_offsets,
            &self.struct_method_slots,
            &self.struct_mangled_map,
            &self.struct_vtable_slots,
            &self.interface_id_map,
            &self.interface_method_slots,
            &self.module.structs,
            &self.enum_variant_tags,
            self.context.clone(),
            &self.extern_c_names,
            self.dep_graph,
            self.module_idx,
            &self.glue_registry,
            &self.allocator_kind,
            &self.interface_methods,
            &self.bump,
            &self.enums,
            &self.module_import_aliases,
            &self.module_named_imports,
            &self.constants,
            self.instantiated_functions.clone(),
            self.instantiated_struct_methods.clone(),
            self.registry.clone(), // here
        )
        .unwrap();
        for (name, value) in &static_inits {
            fl.lower_static_dynamic_init(*name, value);
        }

        fl.lower_body(hir_fn.body);
        fl.finish();

        self.module.functions.insert(hir_fn.name, function);
    }

    /// A hoisted closure fn is named `__closure_fn_N` and takes its env first:
    /// `&__closure_env_N` (Fn/FnMut) or `__closure_env_N` (FnOnce).
    fn closure_env_of(func: &HirFunc<'a, 'bump>) -> Option<StrId> {
        if !func.name.as_str().starts_with("__closure_fn_") {
            return None;
        }
        let Some(HirParam::Normal { param_type, .. }) = func.params?.first() else {
            return None;
        };
        let ty = match param_type {
            HirType::Ref { inner, .. } => *inner,
            other => other,
        };
        match ty {
            HirType::Struct { name, .. } if name.as_str().starts_with("__closure_env_") => {
                Some(*name)
            }
            _ => None,
        }
    }
}
