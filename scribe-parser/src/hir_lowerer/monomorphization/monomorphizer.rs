use ir::hir_utils::hir_contains_this;
use ir::span::SourceSpan;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;

use crate::hir_lowerer::LoweringCtx;
use crate::hir_lowerer::monomorphization::assertions::peel_to_struct;

#[cfg(debug_assertions)]
use crate::hir_lowerer::monomorphization::assertions::contains_unresolved_generic;

use super::naming::suffix_for_subs;
use super::type_substitution::substitute_type;
use ir::hir::{Hir, HirField, HirFunc, HirModule, HirParam, HirType, HirTypeKind, StrId};
use ir::ir_hasher::FxHashMap;
use zetaruntime::string_pool::StringPool;

pub(crate) struct FnScope<'a, 'bump> {
    variables: FxHashMap<StrId, HirType<'a, 'bump>>,
    params: FxHashMap<StrId, HirType<'a, 'bump>>,
    ret: Option<HirType<'a, 'bump>>,
}

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
                    let mut functions = self.functions.borrow_mut();
                    if !functions.contains_key(&f.name) {
                        functions.insert(f.name, (**f).clone());
                    }
                }
                _ => {}
            }
        }

        for item in module.items {
            match item {
                Hir::Func(f) => {
                    let prev_module_idx =
                        std::mem::replace(&mut self.ctx.module_idx, f.declaring_module_idx);

                    let mut new_func = (*f).clone();
                    if new_func.generics.is_none() {
                        if let Some(params) = new_func.params {
                            let new_params: Vec<HirParam> = params
                                .iter()
                                .map(|p| match p {
                                    HirParam::Normal {
                                        name,
                                        param_type,
                                        multi_place,
                                        span,
                                    } => {
                                        let s =
                                            substitute_type(param_type, &empty_subs, &self.bump);
                                        HirParam::Normal {
                                            name: *name,
                                            param_type: self.instantiate_type_recursively(s, *span),
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
                            new_func.params = Some(self.bump.alloc_slice(&new_params));
                        }
                        new_func.return_type = new_func.return_type.map(|rt| {
                            let s = substitute_type(&rt, &empty_subs, &self.bump);
                            self.instantiate_type_recursively(s, new_func.span)
                        });

                        if let Some(body) = new_func.body {
                            let prev_module_idx = self.ctx.module_idx;
                            self.ctx.module_idx = f.declaring_module_idx;

                            // TODO: this should give field types, and unmangled names. This is a bug that affects `sizeof` and `type_name`
                            let this_ty = f.impl_target.map(|name| HirType {
                                kind: HirTypeKind::Struct {
                                    name,
                                    unmangled_name: name,
                                    field_types: &[],
                                    type_args: &[],
                                },
                                span: new_func
                                    .params
                                    .map(|params| params.first())
                                    .flatten()
                                    .map(|p| p.get_span())
                                    .unwrap_or(SourceSpan::default()),
                            });
                            let prev_this = self.current_this.replace(this_ty);
                            let scope = self.enter_fn_scope(new_func.params, new_func.return_type);

                            let new_body = self.monomorphize_stmt(&body, &empty_subs);

                            self.exit_fn_scope(scope);
                            self.current_this.replace(prev_this);
                            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));
                            self.ctx.module_idx = prev_module_idx;
                        }
                    }
                    new_items.push(Hir::Func(self.bump.alloc_value_immutable(new_func)));
                    self.ctx.module_idx = prev_module_idx;
                }
                Hir::Interface(iface) => {
                    let prev_module_idx =
                        std::mem::replace(&mut self.ctx.module_idx, iface.declaring_module_idx);

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
                                    if let Some(body) = nm.body {
                                        let prev_module_idx = self.ctx.module_idx;
                                        self.ctx.module_idx = m.declaring_module_idx;

                                        let return_type_for_body = nm.return_type.map(|ret_ty| {
                                            substitute_type(&ret_ty, &empty_subs, &self.bump)
                                        });

                                        let this_ty = HirType {
                                            kind: HirTypeKind::Struct {
                                                name: new_iface.name,
                                                unmangled_name: new_iface.unmangled_name,
                                                field_types: &[],
                                                type_args: &[],
                                            },
                                            span: nm
                                                .params
                                                .map(|params| params.first())
                                                .flatten()
                                                .map(|p| p.get_span())
                                                .unwrap_or(SourceSpan::default()),
                                        };
                                        let prev_this = self.current_this.replace(Some(this_ty));
                                        let scope =
                                            self.enter_fn_scope(nm.params, return_type_for_body);
                                        let new_body = self.monomorphize_stmt(&body, &empty_subs);
                                        self.exit_fn_scope(scope);
                                        self.current_this.replace(prev_this);

                                        nm.body = Some(*self.bump.alloc_value_immutable(new_body));
                                        nm.return_type = return_type_for_body.map(|ty| {
                                            self.instantiate_type_recursively(ty, nm.span)
                                        });

                                        self.ctx.module_idx = prev_module_idx;
                                    } else {
                                        nm.return_type = nm.return_type.map(|rt| {
                                            self.instantiate_type_recursively(rt, nm.span)
                                        });
                                    }
                                }
                                new_methods.push(nm);
                            }
                            new_iface.methods = Some(self.bump.alloc_slice(&new_methods));
                        }
                    }
                    new_items.push(Hir::Interface(self.bump.alloc_value_immutable(new_iface)));

                    self.ctx.module_idx = prev_module_idx;
                }

                Hir::Struct(s) if self.env_structs.contains_key(&s.name) => {
                    let new_fields: Vec<HirField<'a, 'bump>> = s
                        .fields
                        .iter()
                        .map(|f| {
                            let sub = substitute_type(&f.field_type, &empty_subs, &self.bump);
                            HirField {
                                name: f.name,
                                field_type: self
                                    .instantiate_type_recursively(sub, Default::default()),
                                visibility: f.visibility,
                                manual: f.manual,
                                span: f.span,
                            }
                        })
                        .collect();
                    let mut ns = (**s).clone();
                    ns.fields = self.bump.alloc_slice(&new_fields);
                    self.ctx.structs.borrow_mut().insert(ns.name, ns.clone());
                    new_items.push(Hir::Struct(self.bump.alloc_value_immutable(ns)));
                }

                Hir::Impl(i) => {
                    let prev_module_idx =
                        std::mem::replace(&mut self.ctx.module_idx, i.declaring_module_idx);

                    let mut new_impl = (*i).clone();
                    if new_impl.generics.is_none() {
                        if let Some(methods) = new_impl.methods {
                            let mut new_methods = Vec::with_capacity(methods.len());
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

                                        // TODO: this should give field types, and unmangled names. This is a bug that affects `sizeof` and `type_name`
                                        let this_ty = HirType {
                                            kind: HirTypeKind::Struct {
                                                name: new_impl.target,
                                                unmangled_name: new_impl.target,
                                                field_types: &[],
                                                type_args: &[],
                                            },
                                            span: nm
                                                .params
                                                .map(|params| params.first())
                                                .flatten()
                                                .map(|p| p.get_span())
                                                .unwrap_or(SourceSpan::default()),
                                        };
                                        let prev_this = self.current_this.replace(Some(this_ty));
                                        let scope =
                                            self.enter_fn_scope(nm.params, return_type_for_body);
                                        let new_body = self.monomorphize_stmt(&body, &empty_subs);
                                        self.exit_fn_scope(scope);
                                        self.current_this.replace(prev_this);

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

                    self.ctx.module_idx = prev_module_idx;
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
        &mut self,
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

        #[cfg(debug_assertions)]
        if let Some((bad_param, bad_ty)) = substitutions
            .iter()
            .find(|(_, ty)| contains_unresolved_generic(ty))
        {
            panic!(
                "cannot monomorphize `{}`: substitution for type parameter `{}` is still generic ({:?}). \
                 This means an unresolved generic leaked into `substitutions` from an outer scope \
                 without being resolved via `substitute_type(_, outer_subs, ...)` first at function {}",
                func.name, bad_param, bad_ty, func.span
            );
        }

        let normalized: FxHashMap<StrId, HirType<'a, 'bump>> = substitutions
            .iter()
            .map(|(k, v)| (*k, self.instantiate_type_recursively(*v, func.span)))
            .collect();
        let substitutions = &normalized;

        let is_method = func.impl_target.is_some()
            || func
                .params
                .is_some_and(|ps| matches!(ps.first(), Some(HirParam::This { .. })));
        // Free functions never see a `this`, even when instantiated from inside a method.
        let this_for_key: Option<HirType<'a, 'bump>> = if is_method {
            *self.current_this.borrow()
        } else {
            None
        };

        let orig_name = func.name.clone();
        let self_is_redundant = this_for_key.map_or(false, |t| match peel_to_struct(&t).kind {
            HirTypeKind::Struct {
                name, type_args, ..
            }
            | HirTypeKind::Enum {
                name, type_args, ..
            } => {
                type_args.is_empty()
                    && !self
                        .instantiated_struct_origins
                        .borrow()
                        .contains_key(&name)
                    && !self.instantiated_enum_origins.borrow().contains_key(&name)
            }
            _ => false,
        });

        let suffix = match this_for_key {
            Some(this_ty) if !self_is_redundant => {
                let mut keyed = substitutions.clone();
                keyed.insert(StrId::from_static("Self"), this_ty);
                suffix_for_subs(self.context.clone(), &keyed)
            }
            _ => suffix_for_subs(self.context.clone(), substitutions),
        };
        let key = (orig_name.clone(), suffix.clone());

        let prev_module_idx =
            std::mem::replace(&mut self.ctx.module_idx, func.declaring_module_idx);

        let mut new_func = func.clone();
        self.apply_substitutions_to_func(&mut new_func, substitutions, func.span);

        if let Some(self_ty) = this_for_key {
            if let Some(params) = new_func.params {
                let p: Vec<HirParam> = params
                    .iter()
                    .map(|p| match p {
                        HirParam::Normal {
                            name,
                            param_type,
                            multi_place,
                            span,
                        } => HirParam::Normal {
                            name: *name,
                            param_type: replace_this(param_type, &self_ty, self.bump, *span),
                            multi_place: *multi_place,
                            span: *span,
                        },
                        other => *other,
                    })
                    .collect();
                new_func.params = Some(self.bump.alloc_slice(&p));
            }
            new_func.return_type = new_func
                .return_type
                .map(|t| replace_this(&t, &self_ty, self.bump, t.span));
        }

        if new_func.impl_target.is_some() {
            if let Some(ref self_ty) = this_for_key {
                if let HirTypeKind::Struct { name, .. } | HirTypeKind::Enum { name, .. } =
                    peel_to_struct(self_ty).kind
                {
                    new_func.impl_target = Some(name);
                }
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

        self.instantiated_functions
            .borrow_mut()
            .insert(key, new_name);
        self.functions
            .borrow_mut()
            .insert(new_name, new_func.clone());

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
        let prev_this = self.current_this.replace(this_for_key);
        let prev_return_type = self.current_return_type.replace(new_func.return_type);
        let prev_variable_types = std::mem::take(&mut *self.ctx.variable_types.borrow_mut());

        if let Some(body) = new_func.body {
            let new_body = self.monomorphize_stmt(&body, substitutions);
            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));
        }

        self.current_params.replace(prev_params);
        self.current_this.replace(prev_this);
        self.current_return_type.replace(prev_return_type);
        *self.ctx.variable_types.borrow_mut() = prev_variable_types;

        self.functions.borrow_mut().insert(new_name, new_func);
        self.ctx.module_idx = prev_module_idx;

        Some(new_name)
    }

    fn params_map(params: Option<&[HirParam<'a, 'bump>]>) -> FxHashMap<StrId, HirType<'a, 'bump>> {
        let mut m = FxHashMap::default();
        for p in params.into_iter().flatten() {
            if let HirParam::Normal {
                name, param_type, ..
            } = p
            {
                m.insert(*name, *param_type);
            }
        }
        m
    }

    pub(crate) fn enter_fn_scope(
        &self,
        params: Option<&[HirParam<'a, 'bump>]>,
        ret: Option<HirType<'a, 'bump>>,
    ) -> FnScope<'a, 'bump> {
        FnScope {
            variables: std::mem::take(&mut *self.ctx.variable_types.borrow_mut()),
            params: self.current_params.replace(Self::params_map(params)),
            ret: self.current_return_type.replace(ret),
        }
    }

    pub(crate) fn exit_fn_scope(&self, s: FnScope<'a, 'bump>) {
        *self.ctx.variable_types.borrow_mut() = s.variables;
        self.current_params.replace(s.params);
        self.current_return_type.replace(s.ret);
    }
}

pub fn replace_this<'a, 'bump>(
    ty: &HirType<'a, 'bump>,
    self_ty: &HirType<'a, 'bump>,
    bump: &'bump GrowableBump<'bump>,
    span: SourceSpan<'a>,
) -> HirType<'a, 'bump> {
    if !hir_contains_this(ty) {
        return *ty;
    }
    let r = |t: &HirType<'a, 'bump>| replace_this(t, self_ty, bump, span);
    match ty.kind {
        HirTypeKind::This => *self_ty,
        HirTypeKind::Nullable(i) => HirType::nullable(bump.alloc_value(r(i)), span),
        HirTypeKind::Slice(i) => HirType::slice(bump.alloc_value(r(i)), span),
        HirTypeKind::Array(i, n) => HirType::array(bump.alloc_value(r(i)), n, span),
        HirTypeKind::Ref {
            inner,
            ref_kind,
            provenance,
        } => HirType {
            kind: HirTypeKind::Ref {
                inner: bump.alloc_value(r(inner)),
                ref_kind,
                provenance,
            },
            span,
        },
        HirTypeKind::SafePointer {
            inner,
            mutability_state,
        } => HirType {
            kind: HirTypeKind::SafePointer {
                inner: bump.alloc_value(r(inner)),
                mutability_state,
            },
            span,
        },
        HirTypeKind::UnsafePointer {
            inner,
            mutability_state,
        } => HirType {
            kind: HirTypeKind::UnsafePointer {
                inner: bump.alloc_value(r(inner)),
                mutability_state,
            },
            span,
        },
        HirTypeKind::OwnedPointer { inner, allocator } => HirType {
            kind: HirTypeKind::OwnedPointer {
                inner: bump.alloc_value(r(inner)),
                allocator,
            },
            span,
        },
        HirTypeKind::Tuple(ts) => {
            let v: Vec<_> = ts.iter().map(r).collect();
            HirType {
                kind: HirTypeKind::Tuple(bump.alloc_slice(&v)),
                span,
            }
        }
        HirTypeKind::Struct {
            name,
            field_types,
            unmangled_name,
            type_args,
        } => {
            let a: Vec<_> = type_args.iter().map(r).collect();
            HirType {
                kind: HirTypeKind::Struct {
                    name,
                    unmangled_name,
                    field_types,
                    type_args: bump.alloc_slice(&a),
                },
                span,
            }
        }
        HirTypeKind::Enum {
            name,
            unmangled_name,
            variants,
            type_args,
        } => {
            let a: Vec<_> = type_args.iter().map(r).collect();
            HirType {
                kind: HirTypeKind::Enum {
                    name,
                    unmangled_name,
                    variants,
                    type_args: bump.alloc_slice(&a),
                },
                span,
            }
        }
        HirTypeKind::DynInterface {
            name,
            unmangled_name,
            type_args,
        } => {
            let a: Vec<_> = type_args.iter().map(r).collect();
            HirType {
                kind: HirTypeKind::DynInterface {
                    name,
                    unmangled_name,
                    type_args: bump.alloc_slice(&a),
                },
                span,
            }
        }
        HirTypeKind::Dyn { bounds } => {
            let b: Vec<_> = bounds.iter().map(r).collect();
            HirType {
                kind: HirTypeKind::Dyn {
                    bounds: bump.alloc_slice(&b),
                },
                span,
            }
        }
        HirTypeKind::Lambda {
            params,
            return_type,
        } => {
            let p: Vec<_> = params.iter().map(r).collect();
            HirType::lambda(bump.alloc_slice(&p), bump.alloc_value(r(return_type)), span)
        }
        _ => *ty,
    }
}
