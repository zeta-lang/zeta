use crate::attributes::{AttrTarget, KnownKind};
use crate::hir::{HirEnum, HirExpr, HirFunc, HirInterface, HirStruct, HirType, StrId};
use crate::ir_hasher::FxHashMap;
use crate::ssa_ir::SsaType;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct StaticDef {
    pub name: StrId,
    pub ty: SsaType,
    pub init: StaticInit,
    pub symbol: String,
}

#[derive(Debug, Clone)]
pub struct StaticReloc {
    /// byte offset inside the static where an 8-byte pointer is written
    pub offset: usize,
    /// the pointer targets this string's raw bytes
    pub string: StrId,
}

#[derive(Debug, Clone)]
pub enum StaticInit {
    Zero,
    Bytes(Vec<u8>),
    Relocated {
        bytes: Vec<u8>,
        relocs: Vec<StaticReloc>,
    },
}

#[derive(Clone, Default)]
pub struct ModuleSymbols {
    pub structs: Vec<StrId>,
    pub enums: Vec<StrId>,
    pub interfaces: Vec<StrId>,
    pub functions: Vec<StrId>,
    pub struct_interfaces: Vec<StrId>,
    pub struct_methods: Vec<(StrId, StrId)>, // (struct_name, method_name)
}

/// Cloning this is very cheap as every field is wrapped in Rc
#[derive(Clone)]
pub struct GlobalRegistry<'a, 'bump> {
    pub structs: Rc<RefCell<FxHashMap<StrId, HirStruct<'a, 'bump>>>>,
    pub consts: Rc<RefCell<FxHashMap<StrId, HirExpr<'a, 'bump>>>>,
    pub enums: Rc<RefCell<FxHashMap<StrId, HirEnum<'a, 'bump>>>>,
    pub interfaces: Rc<RefCell<FxHashMap<StrId, HirInterface<'a, 'bump>>>>,
    pub functions: Rc<RefCell<FxHashMap<StrId, HirFunc<'a, 'bump>>>>,
    pub struct_interfaces: Rc<RefCell<FxHashMap<StrId, Vec<StrId>>>>,
    pub struct_methods: Rc<RefCell<FxHashMap<StrId, FxHashMap<StrId, StrId>>>>,
    /// (original_name, type_suffix) -> instantiated function name. Shared
    /// across every module's Monomorphizer so `push` for `ArrayList<i32>`
    /// is specialized exactly once for the whole compilation, not once per
    /// module that happens to call it.
    pub instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    /// (original_name, type_suffix) -> instantiated struct name.
    pub instantiated_structs: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    /// instantiated struct name -> (base struct name, concrete args it was
    /// built from). This is the reverse of instantiated_structs
    /// needed so a receiver's concrete type (`ArrayList_..._i32`) can
    /// be traced back to "ArrayList instantiated with [i32]" for method
    /// resolution.
    pub instantiated_struct_origins:
        Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    pub instantiated_enums: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub instantiated_enum_origins: Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    pub struct_owner_module: Rc<RefCell<FxHashMap<StrId, usize>>>,
    pub enum_owner_module: Rc<RefCell<FxHashMap<StrId, usize>>>,
    owned_by_module: Rc<RefCell<FxHashMap<StrId, ModuleSymbols>>>,
    pub known: Rc<RefCell<FxHashMap<(KnownKind, StrId), AttrTarget>>>,
    pub statics: Rc<RefCell<FxHashMap<StrId, StaticDef>>>,
}

impl<'a, 'bump> GlobalRegistry<'a, 'bump> {
    pub fn new() -> Self {
        Self {
            structs: Rc::new(RefCell::new(FxHashMap::default())),
            consts: Rc::new(RefCell::new(FxHashMap::default())),
            enums: Rc::new(RefCell::new(FxHashMap::default())),
            interfaces: Rc::new(RefCell::new(FxHashMap::default())),
            functions: Rc::new(RefCell::new(FxHashMap::default())),
            struct_interfaces: Rc::new(RefCell::new(FxHashMap::default())),
            struct_methods: Rc::new(RefCell::new(FxHashMap::default())),
            owned_by_module: Rc::new(RefCell::new(FxHashMap::default())),
            instantiated_struct_origins: Rc::new(RefCell::new(FxHashMap::default())),
            instantiated_structs: Rc::new(RefCell::new(FxHashMap::default())),
            instantiated_enums: Rc::new(RefCell::new(FxHashMap::default())),
            instantiated_enum_origins: Rc::new(RefCell::new(FxHashMap::default())),
            struct_owner_module: Rc::new(RefCell::new(FxHashMap::default())),
            enum_owner_module: Rc::new(RefCell::new(FxHashMap::default())),
            instantiated_functions: Rc::new(RefCell::new(FxHashMap::default())),
            known: Rc::new(RefCell::new(FxHashMap::default())),
            statics: Rc::new(RefCell::new(FxHashMap::default())),
        }
    }

    pub fn unregister_module(&self, module: StrId) {
        let Some(owned) = self.owned_by_module.borrow_mut().remove(&module) else {
            return;
        };
        let mut structs = self.structs.borrow_mut();
        for k in owned.structs {
            structs.remove(&k);
        }
        let mut interfaces = self.interfaces.borrow_mut();
        for k in owned.interfaces {
            interfaces.remove(&k);
        }
        let mut functions = self.functions.borrow_mut();
        for k in owned.functions {
            functions.remove(&k);
        }
        let mut si = self.struct_interfaces.borrow_mut();
        for k in owned.struct_interfaces {
            si.remove(&k);
        }
        let mut sm = self.struct_methods.borrow_mut();
        for (struct_name, method_name) in owned.struct_methods {
            if let Some(methods) = sm.get_mut(&struct_name) {
                methods.remove(&method_name);
            }
        }

        self.instantiated_functions.borrow_mut().clear();
        self.instantiated_structs.borrow_mut().clear();
        self.instantiated_struct_origins.borrow_mut().clear();
    }

    pub fn record_owned(&self, module: StrId, symbols: ModuleSymbols) {
        self.owned_by_module.borrow_mut().insert(module, symbols);
    }
}

pub fn static_flag_name(pool: &zetaruntime::string_pool::StringPool, name: StrId) -> StrId {
    StrId(pool.intern(&format!("{}__init", name)))
}
