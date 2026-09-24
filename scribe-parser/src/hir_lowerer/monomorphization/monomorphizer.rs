use smallvec::SmallVec;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;

use crate::hir_lowerer::LoweringCtx;
use crate::hir_lowerer::monomorphization::assertions::{
    contains_unresolved_generic, peel_to_struct,
};

use super::naming::suffix_for_subs;
use super::type_substitution::substitute_type;
use ir::hir::{Hir, HirFunc, HirModule, HirParam, HirType, StrId};
use ir::ir_hasher::FxHashMap;
use zetaruntime::string_pool::StringPool;

pub struct Monomorphizer<'a, 'bump, 'ctx> {
    pub(crate) instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub(crate) instantiated_structs: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub(crate) instantiated_struct_origins:
        Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    pub(crate) instantiated_enums: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub(crate) instantiated_enum_origins:
        Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    pub(crate) current_return_type: RefCell<Option<HirType<'a, 'bump>>>,
    pub(crate) functions: Rc<RefCell<FxHashMap<StrId, HirFunc<'a, 'bump>>>>,
    pub(crate) bump: &'bump GrowableBump<'bump>,
    pub(crate) context: Arc<StringPool>,
    pub(crate) ctx: &'ctx mut LoweringCtx<'a, 'bump>,
    pub(crate) current_this: RefCell<Option<HirType<'a, 'bump>>>,
    pub(crate) current_params: RefCell<FxHashMap<StrId, HirType<'a, 'bump>>>,
    pub(crate) env_structs: FxHashMap<StrId, StrId>,
}

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub fn new(
        context: Arc<StringPool>,
        bump: &'bump GrowableBump<'bump>,
        functions: Rc<RefCell<FxHashMap<StrId, HirFunc<'a, 'bump>>>>,
        ctx: &'ctx mut LoweringCtx<'a, 'bump>,
        instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_structs: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_struct_origins: Rc<
            RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>,
        >,
        instantiated_enums: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_enum_origins: Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
        env_structs: FxHashMap<StrId, StrId>,
    ) -> Self {
        Self {
            instantiated_functions,
            instantiated_structs,
            instantiated_enums,
            instantiated_enum_origins,
            instantiated_struct_origins,
            functions,
            bump,
            context,
            ctx,
            current_this: RefCell::new(None),
            current_return_type: RefCell::new(None),
            current_params: RefCell::new(FxHashMap::default()),
            env_structs,
        }
    }

    pub fn run(&mut self, module: HirModule<'a, 'bump>) -> HirModule<'a, 'bump> {
        self.ctx.named_imports.borrow_mut().clear();
        self.ctx.imported_modules.borrow_mut().clear();
        for import_path in module.imports {
            let Some(target_idx) = self
                .ctx
                .dep_graph
                .borrow()
                .resolve_module_path(import_path.path)
            else {
                continue;
            };
            match import_path.member {
                None => {
                    if let Some(&last) = import_path.path.last() {
                        self.ctx
                            .imported_modules
                            .borrow_mut()
                            .insert(last, target_idx);
                    }
                }
                Some(member) => {
                    self.ctx
                        .named_imports
                        .borrow_mut()
                        .insert(member, target_idx);
                }
            }
        }

        let empty_subs: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
        let mut new_items: Vec<Hir<'a, 'bump>> = Vec::with_capacity(module.items.len());

        for item in module.items {
            match item {
                Hir::Func(f) => {
                    let mut new_func = (*f).clone();
                    if new_func.generics.is_none() {
                        if let Some(body) = new_func.body {
                            let prev_module_idx = self.ctx.module_idx;
                            self.ctx.module_idx = f.declaring_module_idx;

                            let prev_return_type =
                                self.current_return_type.replace(new_func.return_type);
                            let new_body = self.monomorphize_stmt(&body, &empty_subs);
                            self.current_return_type.replace(prev_return_type);
                            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));

                            self.ctx.module_idx = prev_module_idx;
                        }
                    }
                    new_items.push(Hir::Func(self.bump.alloc_value_immutable(new_func)));
                }
                Hir::Interface(iface) => {
                    let mut new_iface = (*iface).clone();
                    if new_iface.generics.is_none() {
                        if let Some(methods) = new_iface.methods {
                            let mut new_methods = Vec::with_capacity(methods.len());
                            for m in methods.iter() {
                                let mut nm = m.clone();
                                if nm.generics.is_none() {
                                    if let Some(params) = nm.params {
                                        let new_params: Vec<HirParam> = params
                                            .iter()
                                            .map(|p| match p {
                                                HirParam::Normal {
                                                    name,
                                                    param_type,
                                                    multi_place,
                                                    span,
                                                } => HirParam::Normal {
                                                    name: *name,
                                                    param_type: self.instantiate_type_recursively(
                                                        *param_type,
                                                        *span,
                                                    ),
                                                    multi_place: *multi_place,
                                                    span: *span,
                                                },
                                                HirParam::This {
                                                    kind,
                                                    span,
                                                    multi_place,
                                                } => HirParam::This {
                                                    kind: *kind,
                                                    multi_place: *multi_place,
                                                    span: *span,
                                                },
                                            })
                                            .collect();
                                        nm.params = Some(self.bump.alloc_slice(&new_params));
                                    }
                                    nm.return_type = nm
                                        .return_type
                                        .map(|rt| self.instantiate_type_recursively(rt, nm.span));
                                }
                                new_methods.push(nm);
                            }
                            new_iface.methods = Some(self.bump.alloc_slice(&new_methods));
                        }
                    }
                    new_items.push(Hir::Interface(self.bump.alloc_value_immutable(new_iface)));
                }
                Hir::Impl(i) => {
                    let mut new_impl = (*i).clone();
                    if new_impl.generics.is_none() {
                        if let Some(methods) = new_impl.methods {
                            let mut new_methods = Vec::new();
                            for m in methods.iter() {
                                if m.generics.is_some() {
                                    continue;
                                }

                                let mut nm = m.clone();

                                if nm.generics.is_none() {
                                    if let Some(params) = nm.params {
                                        let new_params: Vec<HirParam> = params
                                            .iter()
                                            .map(|p| match p {
                                                HirParam::Normal {
                                                    name,
                                                    param_type,
                                                    multi_place,
                                                    span,
                                                } => {
                                                    let substituted = substitute_type(
                                                        param_type,
                                                        &empty_subs,
                                                        &&self.bump,
                                                    );
                                                    HirParam::Normal {
                                                        name: *name,
                                                        param_type: self
                                                            .instantiate_type_recursively(
                                                                substituted,
                                                                *span,
                                                            ),
                                                        multi_place: *multi_place,
                                                        span: *span,
                                                    }
                                                }
                                                HirParam::This {
                                                    kind,
                                                    span,
                                                    multi_place,
                                                } => HirParam::This {
                                                    kind: *kind,
                                                    span: *span,
                                                    multi_place: *multi_place,
                                                },
                                            })
                                            .collect();
                                        nm.params = Some(self.bump.alloc_slice(&new_params));
                                    }

                                    if let Some(body) = nm.body {
                                        let prev_module_idx = self.ctx.module_idx;
                                        self.ctx.module_idx = m.declaring_module_idx;

                                        let return_type_for_body = nm.return_type.map(|ret_ty| {
                                            substitute_type(&ret_ty, &empty_subs, &self.bump)
                                        });

                                        let prev_return_type =
                                            self.current_return_type.replace(return_type_for_body);
                                        let new_body = self.monomorphize_stmt(&body, &empty_subs);
                                        self.current_return_type.replace(prev_return_type);

                                        nm.body = Some(*self.bump.alloc_value_immutable(new_body));
                                        nm.return_type = return_type_for_body.map(|ty| {
                                            self.instantiate_type_recursively(ty, nm.span)
                                        });

                                        self.ctx.module_idx = prev_module_idx;
                                    }
                                }
                                new_methods.push(nm);
                            }
                            new_impl.methods = Some(self.bump.alloc_slice(&new_methods));
                        }
                    }
                    new_items.push(Hir::Impl(self.bump.alloc_value_immutable(new_impl)));
                }
                other => new_items.push(*other),
            }
        }

        new_items.retain(|hir| match hir {
            Hir::Func(hir_func) => hir_func.generics.is_none(),
            Hir::Struct(hir_struct) => hir_struct.generics.is_none(),
            Hir::Interface(hir_interface) => hir_interface.generics.is_none(),
            Hir::Impl(hir_impl) => hir_impl.generics.is_none(),
            Hir::Enum(hir_enum) => hir_enum.generics.is_none(),
            _ => true,
        });

        #[cfg(debug_assertions)]
        {
            self.assert_fully_monomorphized(&new_items);
            self.assert_closures_fully_concrete(&new_items);
        }

        HirModule {
            name: module.name,
            imports: module.imports,
            items: self.bump.alloc_slice(&new_items),
        }
    }

    pub fn monomorphize_function<'subs>(
        &self,
        func: &HirFunc<'a, 'bump>,
        substitutions: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<StrId> {
        if let Some(type_params) = func.generics {
            if let Some(missing) = type_params
                .iter()
                .find(|p| !substitutions.contains_key(&p.name))
            {
                panic!(
                    "cannot monomorphize `{}`: type parameter `{}` has no substitution (have: {:?})",
                    func.name,
                    missing.name,
                    substitutions.keys().collect::<Vec<_>>()
                );
            }
        }

        if let Some((bad_param, bad_ty)) = substitutions
            .iter()
            .find(|(_, ty)| contains_unresolved_generic(ty))
        {
            panic!(
                "cannot monomorphize `{}`: substitution for type parameter `{}` is still generic ({:?}). \
                 This means an unresolved generic leaked into `substitutions` from an outer scope \
                 without being resolved via `substitute_type(_, outer_subs, ...)` first.",
                func.name, bad_param, bad_ty
            );
        }

        let mut new_func = func.clone();
        self.apply_substitutions_to_func(&mut new_func, substitutions, func.span);

        if new_func.impl_target.is_some() {
            let cur_self = *self.current_this.borrow();
            if let Some(ref self_ty) = cur_self {
                if let HirType::Struct { name, .. } = peel_to_struct(self_ty) {
                    new_func.impl_target = Some(*name);
                }
            }
        }

        let suffix = suffix_for_subs(self.context.clone(), substitutions);
        let orig_name = new_func.name.clone();

        let key = (orig_name.clone(), suffix.clone());
        {
            let instantiated_functions = self.instantiated_functions.borrow();
            if let Some(existing) = instantiated_functions.get(&key) {
                return Some(*existing);
            }
        }

        const UNDERSCORE_LEN: usize = 1;
        let mut small_vec: SmallVec<u8, 32> =
            SmallVec::with_capacity(orig_name.len() + UNDERSCORE_LEN + suffix.len());

        small_vec.extend_from_slice(self.context.resolve_bytes(&*orig_name));
        small_vec.push(b'_');
        small_vec.extend_from_slice(self.context.resolve_bytes(&*suffix));

        let new_name = StrId(self.context.intern_bytes(small_vec.as_slice()));

        new_func.name = new_name;

        let mut param_map: FxHashMap<StrId, HirType> = FxHashMap::default();
        if let Some(params) = new_func.params {
            for p in params.iter() {
                if let HirParam::Normal {
                    name, param_type, ..
                } = p
                {
                    param_map.insert(*name, *param_type);
                }
            }
        }
        let prev_params = self.current_params.replace(param_map);
        let prev_return_type = self.current_return_type.replace(new_func.return_type);

        if let Some(body) = new_func.body {
            let new_body = self.monomorphize_stmt(&body, substitutions);
            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));
        }

        self.current_params.replace(prev_params);
        self.current_return_type.replace(prev_return_type);

        self.functions
            .borrow_mut()
            .insert(new_func.name.clone(), new_func);

        self.instantiated_functions
            .borrow_mut()
            .insert(key, new_name);

        Some(new_name)
    }
}
