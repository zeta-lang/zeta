use codex_dependency_graph::DepGraph;
use ir::hir::JoinStr;
use ir::{
    hir::{HirEnum, HirFunc, HirInterface, HirStruct, HirType, StrId},
    ir_hasher::FxHashMap,
};
use smallvec::SmallVec;
use std::{cell::RefCell, rc::Rc, sync::Arc};
use zetaruntime::{bump::GrowableBump, string_pool::StringPool};

use crate::type_checker::{ImplCondition, SymbolId};

pub trait AsStrId {
    fn to_str_id(&self, pool: &StringPool) -> StrId;
    fn as_str_slice(&self) -> &str;
}

impl AsStrId for StrId {
    #[inline(always)]
    fn to_str_id(&self, _pool: &StringPool) -> StrId {
        *self
    }
    #[inline(always)]
    fn as_str_slice(&self) -> &str {
        self.as_str()
    }
}

impl AsStrId for &StrId {
    #[inline(always)]
    fn to_str_id(&self, _pool: &StringPool) -> StrId {
        **self
    }
    #[inline(always)]
    fn as_str_slice(&self) -> &str {
        self.as_str()
    }
}

impl AsStrId for &str {
    #[inline(always)]
    fn to_str_id(&self, pool: &StringPool) -> StrId {
        StrId(pool.intern(self))
    }
    #[inline(always)]
    fn as_str_slice(&self) -> &str {
        self
    }
}

impl AsStrId for String {
    #[inline(always)]
    fn to_str_id(&self, pool: &StringPool) -> StrId {
        StrId(pool.intern(self))
    }
    #[inline(always)]
    fn as_str_slice(&self) -> &str {
        self.as_str()
    }
}

impl AsStrId for &String {
    #[inline(always)]
    fn to_str_id(&self, pool: &StringPool) -> StrId {
        StrId(pool.intern(self.as_str()))
    }
    #[inline(always)]
    fn as_str_slice(&self) -> &str {
        self.as_str()
    }
}

pub trait ItemKey {
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T>;
}

impl ItemKey for StrId {
    #[inline(always)]
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T> {
        table.by_id.get(self)
    }
}

impl ItemKey for &StrId {
    #[inline(always)]
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T> {
        table.by_id.get(*self)
    }
}

impl ItemKey for &str {
    #[inline]
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T> {
        table
            .by_id
            .iter()
            .find(|(k, _)| k.as_str() == *self)
            .map(|(_, v)| v)
    }
}

impl ItemKey for String {
    #[inline]
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T> {
        table
            .by_id
            .iter()
            .find(|(k, _)| k.as_str() == self.as_str())
            .map(|(_, v)| v)
    }
}

impl ItemKey for &String {
    #[inline]
    fn lookup<'m, T>(&self, table: &'m ItemTable<T>) -> Option<&'m T> {
        table
            .by_id
            .iter()
            .find(|(k, _)| k.as_str() == self.as_str())
            .map(|(_, v)| v)
    }
}

#[derive(Debug, Clone)]
pub struct ItemTable<T> {
    pub by_id: FxHashMap<StrId, T>,
}

impl<T> Default for ItemTable<T> {
    fn default() -> Self {
        Self {
            by_id: FxHashMap::default(),
        }
    }
}

impl<T> std::ops::Deref for ItemTable<T> {
    type Target = FxHashMap<StrId, T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.by_id
    }
}

impl<T> std::ops::DerefMut for ItemTable<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.by_id
    }
}

impl<T> ItemTable<T> {
    #[inline(always)]
    pub fn get<K: ItemKey>(&self, key: K) -> Option<&T> {
        key.lookup(self)
    }

    #[inline(always)]
    pub fn get_by_id(&self, key: StrId) -> Option<&T> {
        self.by_id.get(&key)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructInterfaceList(pub SmallVec<StrId, 4>);

impl std::ops::Deref for StructInterfaceList {
    type Target = SmallVec<StrId, 4>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for StructInterfaceList {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for StructInterfaceList {
    type Item = StrId;
    type IntoIter = smallvec::IntoIter<StrId, 4>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'s> IntoIterator for &'s StructInterfaceList {
    type Item = &'s StrId;
    type IntoIter = std::slice::Iter<'s, StrId>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub trait InterfaceNameKey {
    fn is_in(&self, list: &StructInterfaceList) -> bool;
}

impl InterfaceNameKey for StrId {
    #[inline(always)]
    fn is_in(&self, list: &StructInterfaceList) -> bool {
        list.0.contains(self)
    }
}

impl InterfaceNameKey for &StrId {
    #[inline(always)]
    fn is_in(&self, list: &StructInterfaceList) -> bool {
        list.0.contains(*self)
    }
}

impl InterfaceNameKey for &str {
    #[inline]
    fn is_in(&self, list: &StructInterfaceList) -> bool {
        list.0.iter().any(|s| s.as_str() == *self)
    }
}

impl InterfaceNameKey for String {
    #[inline]
    fn is_in(&self, list: &StructInterfaceList) -> bool {
        list.0.iter().any(|s| s.as_str() == self.as_str())
    }
}

impl InterfaceNameKey for &String {
    #[inline]
    fn is_in(&self, list: &StructInterfaceList) -> bool {
        list.0.iter().any(|s| s.as_str() == self.as_str())
    }
}

impl StructInterfaceList {
    #[inline]
    pub fn contains<K: InterfaceNameKey>(&self, key: K) -> bool {
        key.is_in(self)
    }
}

pub trait StructInterfaceKey {
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList>;
}

impl StructInterfaceKey for StrId {
    #[inline(always)]
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList> {
        map.by_id.get(self)
    }
}

impl StructInterfaceKey for &StrId {
    #[inline(always)]
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList> {
        map.by_id.get(*self)
    }
}

impl StructInterfaceKey for &str {
    #[inline]
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList> {
        map.by_id
            .iter()
            .find(|(k, _)| k.as_str() == *self)
            .map(|(_, v)| v)
    }
}

impl StructInterfaceKey for String {
    #[inline]
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList> {
        map.by_id
            .iter()
            .find(|(k, _)| k.as_str() == self.as_str())
            .map(|(_, v)| v)
    }
}

impl StructInterfaceKey for &String {
    #[inline]
    fn lookup<'m>(&self, map: &'m StructInterfacesMap) -> Option<&'m StructInterfaceList> {
        map.by_id
            .iter()
            .find(|(k, _)| k.as_str() == self.as_str())
            .map(|(_, v)| v)
    }
}

#[derive(Debug, Clone, Default)]
pub struct StructInterfacesMap {
    pub by_id: FxHashMap<StrId, StructInterfaceList>,
}

impl std::ops::Deref for StructInterfacesMap {
    type Target = FxHashMap<StrId, StructInterfaceList>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.by_id
    }
}

impl std::ops::DerefMut for StructInterfacesMap {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.by_id
    }
}

impl StructInterfacesMap {
    #[inline]
    pub fn get<K: StructInterfaceKey>(&self, key: K) -> Option<&StructInterfaceList> {
        key.lookup(self)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeMethodTable<'a, 'bump> {
    pub methods: FxHashMap<StrId, HirFunc<'a, 'bump>>,
}

impl<'a, 'bump> TypeMethodTable<'a, 'bump> {
    #[inline]
    pub fn insert(&mut self, method_name: StrId, func: HirFunc<'a, 'bump>) {
        self.methods.insert(method_name, func);
    }

    #[inline]
    pub fn get(&self, method_name: &str) -> Option<&HirFunc<'a, 'bump>> {
        self.methods
            .iter()
            .find(|(k, _)| k.as_str() == method_name)
            .map(|(_, v)| v)
    }

    #[inline]
    pub fn get_id(&self, method_name: StrId) -> Option<&HirFunc<'a, 'bump>> {
        self.methods.get(&method_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binding<'a, 'bump> {
    pub symbol: SymbolId,
    pub ty: HirType<'a, 'bump>,
    pub mutable: bool,
    pub dangling: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Scope<'a, 'bump>(pub(crate) Option<Rc<Frame<'a, 'bump>>>);

#[derive(Debug)]
pub struct Frame<'a, 'bump> {
    pub parent: Scope<'a, 'bump>,
    pub name: StrId,
    pub binding: Binding<'a, 'bump>,
}

impl<'a, 'bump> Scope<'a, 'bump> {
    #[inline]
    pub fn empty() -> Self {
        Self(None)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    #[inline]
    pub fn bind(&self, name: StrId, binding: Binding<'a, 'bump>) -> Self {
        Self(Some(Rc::new(Frame {
            parent: self.clone(),
            name,
            binding,
        })))
    }

    #[inline]
    pub fn get(&self, name: StrId) -> Option<&Binding<'a, 'bump>> {
        let mut curr = self.0.as_deref();
        while let Some(frame) = curr {
            if frame.name == name {
                return Some(&frame.binding);
            }
            curr = frame.parent.0.as_deref();
        }
        None
    }

    #[inline]
    pub fn get_by_str(&self, name: &str) -> Option<&Binding<'a, 'bump>> {
        let mut curr = self.0.as_deref();
        while let Some(frame) = curr {
            if frame.name.as_str() == name {
                return Some(&frame.binding);
            }
            curr = frame.parent.0.as_deref();
        }
        None
    }

    #[inline]
    pub fn contains(&self, name: StrId) -> bool {
        self.get(name).is_some()
    }

    #[inline]
    pub fn mark_dangling(&self, name: StrId) -> Self {
        if let Some(&b) = self.get(name) {
            let mut updated = b;
            updated.dangling = true;
            self.bind(name, updated)
        } else {
            self.clone()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScopeShim<'a, 'bump> {
    pub(crate) scope: Scope<'a, 'bump>,
    pub(crate) pool: Arc<StringPool>,
}

impl<'a, 'bump> ScopeShim<'a, 'bump> {
    #[inline]
    pub fn contains_key<K: AsStrId>(&self, name: K) -> bool {
        let id = name.to_str_id(&self.pool);
        self.scope.contains(id)
    }

    #[inline]
    pub fn get<K: AsStrId>(&self, name: K) -> Option<(SymbolId, HirType<'a, 'bump>)> {
        let id = name.to_str_id(&self.pool);
        self.scope.get(id).map(|b| (b.symbol, b.ty))
    }
}

#[derive(Debug, Clone)]
pub struct Globals<'a, 'bump> {
    pub structs: ItemTable<HirStruct<'a, 'bump>>,
    pub struct_owner_module: ItemTable<usize>,
    pub enums: ItemTable<HirEnum<'a, 'bump>>,
    pub enum_owner_module: ItemTable<usize>,
    pub interfaces: ItemTable<HirInterface<'a, 'bump>>,
    pub interface_owner_module: ItemTable<usize>,
    pub module_functions: FxHashMap<usize, ItemTable<HirFunc<'a, 'bump>>>,
    pub type_methods: FxHashMap<StrId, TypeMethodTable<'a, 'bump>>,
    pub struct_interfaces: StructInterfacesMap,
    pub conditional_impls: FxHashMap<(StrId, StrId), Vec<ImplCondition<'a, 'bump>>>,
    pub generic_struct_instantiations:
        RefCell<FxHashMap<(StrId, SmallVec<HirType<'a, 'bump>, 4>), &'bump [HirType<'a, 'bump>]>>,
    pub generic_enum_instantiations: RefCell<
        Vec<(
            (StrId, Vec<HirType<'a, 'bump>>),
            &'bump [(StrId, &'bump [HirType<'a, 'bump>])],
        )>,
    >,
    pub dep_graph: &'a RefCell<DepGraph>,
    pub bump: &'bump GrowableBump<'bump>,
    pub string_pool: Arc<StringPool>,
}

#[derive(Debug, Clone)]
pub struct GlobalsBuilder<'a, 'bump> {
    pub structs: ItemTable<HirStruct<'a, 'bump>>,
    pub struct_owner_module: ItemTable<usize>,
    pub enums: ItemTable<HirEnum<'a, 'bump>>,
    pub enum_owner_module: ItemTable<usize>,
    pub interfaces: ItemTable<HirInterface<'a, 'bump>>,
    pub interface_owner_module: ItemTable<usize>,
    pub module_functions: FxHashMap<usize, ItemTable<HirFunc<'a, 'bump>>>,
    pub type_methods: FxHashMap<StrId, TypeMethodTable<'a, 'bump>>,
    pub struct_interfaces: StructInterfacesMap,
    pub conditional_impls: FxHashMap<(StrId, StrId), Vec<ImplCondition<'a, 'bump>>>,
    pub generic_struct_instantiations:
        RefCell<FxHashMap<(StrId, SmallVec<HirType<'a, 'bump>, 4>), &'bump [HirType<'a, 'bump>]>>,
    pub generic_enum_instantiations: RefCell<
        Vec<(
            (StrId, Vec<HirType<'a, 'bump>>),
            &'bump [(StrId, &'bump [HirType<'a, 'bump>])],
        )>,
    >,
    pub dep_graph: &'a RefCell<DepGraph>,
    pub bump: &'bump GrowableBump<'bump>,
    pub string_pool: Arc<StringPool>,
}

impl<'a, 'bump> GlobalsBuilder<'a, 'bump> {
    pub fn new(
        dep_graph: &'a RefCell<DepGraph>,
        bump: &'bump GrowableBump<'bump>,
        string_pool: Arc<StringPool>,
    ) -> Self {
        Self {
            structs: ItemTable::default(),
            struct_owner_module: ItemTable::default(),
            enums: ItemTable::default(),
            enum_owner_module: ItemTable::default(),
            interfaces: ItemTable::default(),
            interface_owner_module: ItemTable::default(),
            module_functions: FxHashMap::default(),
            type_methods: FxHashMap::default(),
            struct_interfaces: StructInterfacesMap::default(),
            conditional_impls: FxHashMap::default(),
            generic_struct_instantiations: RefCell::new(FxHashMap::default()),
            generic_enum_instantiations: RefCell::new(Vec::new()),
            dep_graph,
            bump,
            string_pool,
        }
    }

    pub fn from_globals(globals: &Globals<'a, 'bump>) -> Self {
        Self {
            structs: globals.structs.clone(),
            struct_owner_module: globals.struct_owner_module.clone(),
            enums: globals.enums.clone(),
            enum_owner_module: globals.enum_owner_module.clone(),
            interfaces: globals.interfaces.clone(),
            interface_owner_module: globals.interface_owner_module.clone(),
            module_functions: globals.module_functions.clone(),
            type_methods: globals.type_methods.clone(),
            struct_interfaces: globals.struct_interfaces.clone(),
            conditional_impls: globals.conditional_impls.clone(),
            generic_struct_instantiations: globals.generic_struct_instantiations.clone(),
            generic_enum_instantiations: globals.generic_enum_instantiations.clone(),
            dep_graph: globals.dep_graph,
            bump: globals.bump,
            string_pool: globals.string_pool.clone(),
        }
    }

    pub fn freeze(self) -> Arc<Globals<'a, 'bump>> {
        Arc::new(Globals {
            structs: self.structs,
            struct_owner_module: self.struct_owner_module,
            enums: self.enums,
            enum_owner_module: self.enum_owner_module,
            interfaces: self.interfaces,
            interface_owner_module: self.interface_owner_module,
            module_functions: self.module_functions,
            type_methods: self.type_methods,
            struct_interfaces: self.struct_interfaces,
            conditional_impls: self.conditional_impls,
            generic_struct_instantiations: self.generic_struct_instantiations,
            generic_enum_instantiations: self.generic_enum_instantiations,
            dep_graph: self.dep_graph,
            bump: self.bump,
            string_pool: self.string_pool,
        })
    }

    pub fn add_struct(&mut self, module_idx: usize, name: StrId, struct_def: HirStruct<'a, 'bump>) {
        self.struct_owner_module.insert(name, module_idx);
        self.structs.insert(name, struct_def);
    }

    pub fn add_enum(&mut self, module_idx: usize, name: StrId, enum_def: HirEnum<'a, 'bump>) {
        self.enum_owner_module.insert(name, module_idx);
        self.enums.insert(name, enum_def);
    }

    pub fn add_interface(
        &mut self,
        module_idx: usize,
        name: StrId,
        iface: HirInterface<'a, 'bump>,
    ) {
        self.interface_owner_module.insert(name, module_idx);
        self.interfaces.insert(name, iface);
    }

    pub fn add_function(&mut self, module_idx: usize, name: StrId, func: HirFunc<'a, 'bump>) {
        self.module_functions
            .entry(module_idx)
            .or_default()
            .insert(name, func);
    }

    pub fn add_struct_interface(&mut self, struct_name: StrId, interface_name: StrId) {
        let v: &mut StructInterfaceList = self.struct_interfaces.entry(struct_name).or_default();
        if !v.contains(interface_name) {
            v.push(interface_name);
        }
    }

    pub fn add_conditional_impl(
        &mut self,
        target: StrId,
        interface: StrId,
        cond: ImplCondition<'a, 'bump>,
    ) {
        let v = self
            .conditional_impls
            .entry((target, interface))
            .or_default();
        if !v.contains(&cond) {
            v.push(cond);
        }
    }
}

impl<'a, 'bump> Globals<'a, 'bump> {
    pub fn new(
        dep_graph: &'a RefCell<DepGraph>,
        bump: &'bump GrowableBump<'bump>,
        string_pool: Arc<StringPool>,
    ) -> Self {
        GlobalsBuilder::new(dep_graph, bump, string_pool).freeze_direct()
    }

    #[inline]
    pub fn get_struct(&self, name: impl AsStrId) -> Option<HirStruct<'a, 'bump>> {
        let id = name.to_str_id(&self.string_pool);
        self.structs.get(id).copied()
    }

    #[inline]
    pub fn struct_owner(&self, name: impl AsStrId) -> Option<usize> {
        let id = name.to_str_id(&self.string_pool);
        self.struct_owner_module.get(id).copied()
    }

    #[inline]
    pub fn get_enum(&self, name: impl AsStrId) -> Option<HirEnum<'a, 'bump>> {
        let id = name.to_str_id(&self.string_pool);
        self.enums.get(id).copied()
    }

    #[inline]
    pub fn enum_owner(&self, name: impl AsStrId) -> Option<usize> {
        let id = name.to_str_id(&self.string_pool);
        self.enum_owner_module.get(id).copied()
    }

    #[inline]
    pub fn get_interface(&self, name: impl AsStrId) -> Option<HirInterface<'a, 'bump>> {
        let id = name.to_str_id(&self.string_pool);
        self.interfaces.get(id).copied()
    }

    #[inline]
    pub fn interface_owner(&self, name: impl AsStrId) -> Option<usize> {
        let id = name.to_str_id(&self.string_pool);
        self.interface_owner_module.get(id).copied()
    }

    #[inline]
    pub fn get_module_function(
        &self,
        module_idx: usize,
        name: impl AsStrId,
    ) -> Option<HirFunc<'a, 'bump>> {
        let id = name.to_str_id(&self.string_pool);
        self.module_functions.get(&module_idx)?.get(id).copied()
    }

    #[inline]
    pub fn get_method(
        &self,
        type_name: impl AsStrId,
        method_name: impl AsStrId,
    ) -> Option<&HirFunc<'a, 'bump>> {
        let type_id = type_name.to_str_id(&self.string_pool);
        let method_id = method_name.to_str_id(&self.string_pool);
        self.type_methods
            .get(&type_id)
            .and_then(|t| t.methods.get(&method_id))
    }

    #[inline]
    pub fn struct_implements(
        &self,
        struct_name: impl AsStrId,
        interface_name: impl AsStrId,
    ) -> bool {
        let s_id = struct_name.to_str_id(&self.string_pool);
        let i_id = interface_name.to_str_id(&self.string_pool);
        self.struct_interfaces
            .get(s_id)
            .is_some_and(|list| list.contains(i_id))
    }

    #[inline]
    pub fn get_struct_interfaces(&self, struct_name: impl AsStrId) -> Option<&[StrId]> {
        let s_id = struct_name.to_str_id(&self.string_pool);
        self.struct_interfaces.get(s_id).map(|v| v.as_slice())
    }

    #[inline]
    pub fn conditional_impls_for(
        &self,
        target: impl AsStrId,
        interface: impl AsStrId,
    ) -> Vec<ImplCondition<'a, 'bump>> {
        let t_id = target.to_str_id(&self.string_pool);
        let i_id = interface.to_str_id(&self.string_pool);
        self.conditional_impls
            .get(&(t_id, i_id))
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_enum_instantiation(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [(StrId, &'bump [HirType<'a, 'bump>])]> {
        self.generic_enum_instantiations
            .borrow()
            .iter()
            .find(|((n, a), _)| *n == name && a.as_slice() == args)
            .map(|(_, variants)| *variants)
    }

    pub fn cache_enum_instantiation(
        &self,
        name: StrId,
        args: Vec<HirType<'a, 'bump>>,
        variants: &'bump [(StrId, &'bump [HirType<'a, 'bump>])],
    ) {
        self.generic_enum_instantiations
            .borrow_mut()
            .push(((name, args), variants));
    }

    pub fn get_struct_instantiation(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [HirType<'a, 'bump>]> {
        let key = (name, SmallVec::from_slice_copy(args));
        self.generic_struct_instantiations
            .borrow()
            .get(&key)
            .copied()
    }

    pub fn cache_struct_instantiation(
        &self,
        name: StrId,
        args: SmallVec<HirType<'a, 'bump>, 4>,
        fields: &'bump [HirType<'a, 'bump>],
    ) {
        self.generic_struct_instantiations
            .borrow_mut()
            .insert((name, args), fields);
    }
}

impl<'a, 'bump> GlobalsBuilder<'a, 'bump> {
    pub fn freeze_direct(self) -> Globals<'a, 'bump> {
        Globals {
            structs: self.structs,
            struct_owner_module: self.struct_owner_module,
            enums: self.enums,
            enum_owner_module: self.enum_owner_module,
            interfaces: self.interfaces,
            interface_owner_module: self.interface_owner_module,
            module_functions: self.module_functions,
            type_methods: self.type_methods,
            struct_interfaces: self.struct_interfaces,
            conditional_impls: self.conditional_impls,
            generic_struct_instantiations: self.generic_struct_instantiations,
            generic_enum_instantiations: self.generic_enum_instantiations,
            dep_graph: self.dep_graph,
            bump: self.bump,
            string_pool: self.string_pool,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeContext<'a, 'bump> {
    pub globals: Arc<Globals<'a, 'bump>>,
    pub scope: Scope<'a, 'bump>,
    pub variables: ScopeShim<'a, 'bump>,
    pub current_return_type: Option<HirType<'a, 'bump>>,
    pub in_loop: bool,
    pub current_module_idx: usize,
}

impl<'a, 'bump> std::ops::Deref for TypeContext<'a, 'bump> {
    type Target = Globals<'a, 'bump>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.globals
    }
}

impl<'a, 'bump> TypeContext<'a, 'bump> {
    pub fn new(
        dep_graph: &'a RefCell<DepGraph>,
        bump: &'bump GrowableBump<'bump>,
        string_pool: Arc<StringPool>,
    ) -> Self {
        let globals = Arc::new(Globals::new(dep_graph, bump, string_pool.clone()));
        Self {
            globals,
            scope: Scope::empty(),
            variables: ScopeShim {
                scope: Scope::empty(),
                pool: string_pool,
            },
            current_return_type: None,
            in_loop: false,
            current_module_idx: usize::MAX,
        }
    }

    pub fn with_globals(globals: Arc<Globals<'a, 'bump>>) -> Self {
        let pool = globals.string_pool.clone();
        Self {
            globals,
            scope: Scope::empty(),
            variables: ScopeShim {
                scope: Scope::empty(),
                pool,
            },
            current_return_type: None,
            in_loop: false,
            current_module_idx: usize::MAX,
        }
    }

    #[inline]
    pub fn create_child_scope(&self) -> Self {
        self.clone()
    }

    #[inline]
    pub fn add_struct_interface(
        &mut self,
        struct_name: impl AsStrId,
        interface_name: impl AsStrId,
    ) {
        let s_id = struct_name.to_str_id(&self.string_pool);
        let i_id: StrId = interface_name.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        let list = globals.struct_interfaces.entry(s_id).or_default();
        if !list.contains(i_id) {
            list.push(i_id);
        }
    }

    #[inline]
    pub fn add_conditional_impl(
        &mut self,
        target: impl AsStrId,
        interface: impl AsStrId,
        cond: ImplCondition<'a, 'bump>,
    ) {
        let t_id = target.to_str_id(&self.string_pool);
        let i_id = interface.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        let v = globals.conditional_impls.entry((t_id, i_id)).or_default();
        if !v.contains(&cond) {
            v.push(cond);
        }
    }

    #[inline]
    pub fn add_function(
        &mut self,
        module_idx: usize,
        name: impl AsStrId,
        func: HirFunc<'a, 'bump>,
    ) {
        let id = name.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        globals
            .module_functions
            .entry(module_idx)
            .or_default()
            .insert(id, func);
    }

    #[inline]
    pub fn add_struct(
        &mut self,
        module_idx: usize,
        name: impl AsStrId,
        struct_def: HirStruct<'a, 'bump>,
    ) {
        let id = name.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        globals.struct_owner_module.insert(id, module_idx);
        globals.structs.insert(id, struct_def);
    }

    #[inline]
    pub fn add_enum(
        &mut self,
        module_idx: usize,
        name: impl AsStrId,
        enum_def: HirEnum<'a, 'bump>,
    ) {
        let id = name.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        globals.enum_owner_module.insert(id, module_idx);
        globals.enums.insert(id, enum_def);
    }

    #[inline]
    pub fn add_interface(
        &mut self,
        module_idx: usize,
        name: impl AsStrId,
        iface: HirInterface<'a, 'bump>,
    ) {
        let id = name.to_str_id(&self.string_pool);
        let globals = Arc::make_mut(&mut self.globals);
        globals.interface_owner_module.insert(id, module_idx);
        globals.interfaces.insert(id, iface);
    }

    #[inline]
    pub fn add_variable(
        &mut self,
        name: impl AsStrId,
        ty: HirType<'a, 'bump>,
        symbol_id: SymbolId,
    ) {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope = self.scope.bind(
            id,
            Binding {
                symbol: symbol_id,
                ty,
                mutable: false,
                dangling: false,
            },
        );
        self.variables.scope = self.scope.clone();
    }

    #[inline]
    pub fn add_variable_with_mutability(
        &mut self,
        name: impl AsStrId,
        ty: HirType<'a, 'bump>,
        mutable: bool,
        symbol_id: SymbolId,
    ) {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope = self.scope.bind(
            id,
            Binding {
                symbol: symbol_id,
                ty,
                mutable,
                dangling: false,
            },
        );
        self.variables.scope = self.scope.clone();
    }

    #[inline]
    pub fn get_variable(&self, name: impl AsStrId) -> Option<(SymbolId, HirType<'a, 'bump>)> {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope.get(id).map(|b| (b.symbol, b.ty))
    }

    #[inline]
    pub fn is_mutable(&self, name: impl AsStrId) -> bool {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope.get(id).map_or(false, |b| b.mutable)
    }

    #[inline]
    pub fn is_dangling(&self, name: impl AsStrId) -> bool {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope.get(id).map_or(false, |b| b.dangling)
    }

    #[inline]
    pub fn mark_dangling(&mut self, name: impl AsStrId) {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope = self.scope.mark_dangling(id);
        self.variables.scope = self.scope.clone();
    }

    #[inline]
    pub fn is_local_binding(&self, name: impl AsStrId) -> bool {
        let id = name.to_str_id(&self.globals.string_pool);
        self.scope.contains(id)
    }

    #[inline]
    pub fn get_function(&self, name: impl AsStrId) -> Option<HirFunc<'a, 'bump>> {
        self.get_module_function(self.current_module_idx, name)
    }

    #[inline]
    pub fn enter_loop(&mut self) {
        self.in_loop = true;
    }

    #[inline]
    pub fn exit_loop(&mut self) {
        self.in_loop = false;
    }

    #[inline]
    pub fn with_return_type(mut self, return_type: HirType<'a, 'bump>) -> Self {
        self.current_return_type = Some(return_type);
        self
    }

    pub fn mangle_function_name(&self, name: StrId) -> StrId {
        let Some(pkg) = self
            .globals
            .dep_graph
            .borrow()
            .get_module_package(self.current_module_idx)
        else {
            return name;
        };

        let mut segments: Vec<StrId> = pkg.split("::").collect();
        segments.push(name);

        let joined = segments.join("_");

        StrId(self.globals.string_pool.intern(&joined))
    }
}
