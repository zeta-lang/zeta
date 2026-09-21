use crate::midend::copy_analysis::drop_glue::{DropGlueBuilder, DropGlueRegistry};
use crate::midend::ir::mir_lowering::FunctionLowerer;
use codex_dependency_graph::DepGraph;
use ir::hir::{
    Hir, HirEnum, HirExpr, HirFunc, HirInterface, HirModule, HirParam, HirStruct, StrId,
    ThisPassingKind,
};
use ir::ir_conversion::lower_type_hir;
use ir::ir_hasher::{HashMap, HashSet};
use ir::ssa_ir::{AllocatorKind, Function, Module, SsaType};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;
use zetaruntime::intern_fmt;
use zetaruntime::string_pool::StringPool;

pub struct MirModuleLowerer<'a, 'cx, 'bump, 'g>
where
    'bump: 'a,
    'bump: 'cx,
    'cx: 'a,
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
    glue_registry: &'a DropGlueRegistry,
    allocator_kind: HashMap<StrId, AllocatorKind>,
    bump: GrowableBump<'bump>,
    interface_default_methods: HashMap<StrId, HashMap<StrId, HirFunc<'a, 'bump>>>,
    module_import_aliases: HashMap<usize, HashMap<StrId, usize>>,
    module_named_imports: HashMap<usize, HashMap<StrId, usize>>,
    constants: HashMap<StrId, HirExpr<'a, 'bump>>,
}

impl<'a, 'cx, 'bump, 'g> MirModuleLowerer<'a, 'cx, 'bump, 'g>
where
    'bump: 'a,
    'bump: 'cx,
    'cx: 'a,
{
    pub fn new(
        context: Arc<StringPool>,
        extern_c_names: Rc<HashSet<StrId>>,
        dep_graph: &'a RefCell<DepGraph>,
        module_idx: usize,
        glue_registry: &'a DropGlueRegistry,
    ) -> Self {
        let mut enum_variant_tags: HashMap<StrId, HashMap<StrId, usize>> = HashMap::default();

        let nullable_enum_name = StrId(context.thread_local().intern("__nullable"));
        let mut nullable_tags = HashMap::default();
        nullable_tags.insert(StrId(context.thread_local().intern("null")), 0usize);
        nullable_tags.insert(StrId(context.thread_local().intern("some")), 1usize);
        enum_variant_tags.insert(nullable_enum_name, nullable_tags);

        let throws_enum_name = StrId(context.thread_local().intern("__throws"));
        let mut throws_tags = HashMap::default();
        throws_tags.insert(StrId(context.thread_local().intern("__success")), 0usize);
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
                if let Hir::Struct(ty_struct) = item {
                    self.register_struct(*ty_struct);
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

        for &idx in compilation_order {
            for item in hir_modules[idx].items {
                match item {
                    Hir::Func(func) => match func.impl_target {
                        Some(struct_name) => {
                            self.struct_mangled_map
                                .entry(struct_name)
                                .or_insert_with(|| HashMap::default())
                                .insert(func.unmangled_name, func.name);
                            if func.generics.is_none() {
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
                            }
                        }
                    },
                    Hir::Impl(impl_block) => {
                        if let Some(methods) = impl_block.methods {
                            for method in methods {
                                self.struct_mangled_map
                                    .entry(impl_block.target)
                                    .or_insert_with(|| HashMap::default())
                                    .insert(method.unmangled_name, method.name);
                                if method.generics.is_none() && impl_block.generics.is_none() {
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
            self.glue_registry,
            &self.module.structs,
            &self.module.enums,
            &self.struct_mangled_map,
            &self.struct_field_offsets,
            &self.allocator_kind,
            self.context.clone(),
            &owned_structs,
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

        self.module
    }

    fn compute_allocator_kinds(&mut self) {
        let allocator_iface = StrId(self.context.thread_local().intern("Allocator"));
        let raw_allocator_iface = StrId(self.context.thread_local().intern("RawAllocator"));

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

        self.compute_field_offsets(hir_struct);

        self.struct_mangled_map
            .entry(hir_struct.name)
            .or_insert_with(|| HashMap::default());

        self.module
            .structs
            .insert(hir_struct.name, hir_struct.clone());
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
                        } => lower_type_hir(&param_type, &self.enums),
                        HirParam::This {
                            kind,
                            span: _,
                            multi_place: _,
                        } => match kind {
                            ThisPassingKind::Move | ThisPassingKind::MoveMut => SsaType::Dyn,
                            _ => SsaType::Pointer(Box::new(SsaType::Dyn)),
                        },
                    })
                    .collect::<Vec<_>>();
                let ret = m
                    .return_type
                    .as_ref()
                    .map(|t| lower_type_hir(t, &self.enums))
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

    fn compute_field_offsets(&mut self, hir_struct: &HirStruct<'a, 'bump>) {
        let mut offsets = HashMap::default();
        let mut current_offset = 0usize;

        for f in hir_struct.fields.iter() {
            let field_ssa_ty = lower_type_hir(&f.field_type, &self.enums);
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
        let mut function = self
            .module
            .functions
            .remove(&hir_fn.name)
            .expect("function signature should already be registered");

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
            self.glue_registry,
            &self.allocator_kind,
            &self.interface_methods,
            &self.bump,
            &self.enums,
            &self.module_import_aliases,
            &self.module_named_imports,
            &self.constants,
        )
        .unwrap();
        fl.lower_body(hir_fn.body);
        fl.finish();

        self.module.functions.insert(hir_fn.name, function);
    }
}
