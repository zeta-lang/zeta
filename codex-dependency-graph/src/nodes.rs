use std::{
    fmt,
    ops::{Index, IndexMut},
};

use ir::{hir::StrId, span::SourceSpan};

macro_rules! define_idx {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
            pub struct $name(pub usize);

            impl<T> Index<$name> for [T] {
                type Output = T;
                #[inline]
                fn index(&self, idx: $name) -> &T { &self[idx.0] }
            }

            impl<T> IndexMut<$name> for [T] {
                #[inline]
                fn index_mut(&mut self, idx: $name) -> &mut T { &mut self[idx.0] }
            }

            impl fmt::Display for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.0)
                }
            }
        )+
    };
}

define_idx!(ModuleIdx, ItemIdx, MethodIdx, NodeIdx);

#[derive(Clone, Debug)]
pub enum NodeKind {
    Module {
        module_idx: ModuleIdx,
    },
    TypeDecl {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    FuncSig {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    FuncBody {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    ConstDecl {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    TraitDecl {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    TraitImpl {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
    },
    Method {
        module_idx: ModuleIdx,
        item_idx: ItemIdx,
        method_idx: MethodIdx,
    },
}

impl NodeKind {
    pub fn is_func_body(&self) -> bool {
        matches!(self, NodeKind::FuncBody { .. })
    }
    pub fn is_method(&self) -> bool {
        matches!(self, NodeKind::Method { .. })
    }
    pub fn is_type_decl(&self) -> bool {
        matches!(self, NodeKind::TypeDecl { .. })
    }
    pub fn module_idx(&self) -> ModuleIdx {
        match self {
            NodeKind::Module { module_idx }
            | NodeKind::TypeDecl { module_idx, .. }
            | NodeKind::FuncSig { module_idx, .. }
            | NodeKind::FuncBody { module_idx, .. }
            | NodeKind::ConstDecl { module_idx, .. }
            | NodeKind::TraitDecl { module_idx, .. }
            | NodeKind::TraitImpl { module_idx, .. }
            | NodeKind::Method { module_idx, .. } => *module_idx,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Import,
    TypeByValue,
    TypeIndirect,
    Use,
    Call,
    MethodCall,
    Signature,
    Structural,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge<'a> {
    pub to: NodeIdx,
    pub kind: EdgeKind,
    pub span: SourceSpan<'a>,
}

#[derive(Debug)]
pub struct DepNode<'a> {
    pub idx: NodeIdx,
    pub kind: NodeKind,
    pub hint: Option<StrId>,
    /// Outgoing edges: this node depends on these.
    pub deps: Vec<Edge<'a>>,
    /// Reverse edges: these nodes depend on this one.
    pub rev_deps: Vec<NodeIdx>,
}

impl<'a> DepNode<'a> {
    pub fn new(idx: NodeIdx, kind: NodeKind, hint: Option<StrId>) -> Self {
        DepNode {
            idx,
            kind,
            hint,
            deps: Vec::new(),
            rev_deps: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ItemKey {
    pub module: ModuleIdx,
    pub item: ItemIdx,
    pub role: ItemRole,
}

#[derive(Clone, Copy)]
pub struct Ctx {
    pub from: NodeIdx,
    pub module: ModuleIdx,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ItemRole {
    Module,
    Static,
    Const,
    Type,
    Trait,
    Impl,
    FuncSig,
    FuncBody,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MethodKey {
    pub module: ModuleIdx,
    pub item: ItemIdx,
    pub method: MethodIdx,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SymbolKey {
    pub module: ModuleIdx,
    pub name: StrId,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MethodLookupKey {
    pub owner: ItemKey,
    pub name: StrId,
}

impl ItemRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Static => "static",
            Self::FuncSig => "func_sig",
            Self::FuncBody => "func_body",
            Self::Type => "type",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Const => "const",
        }
    }

    pub fn from_tag(s: &str) -> Option<Self> {
        match s {
            "module" => Some(Self::Module),
            "static" => Some(Self::Static),
            "func_sig" => Some(Self::FuncSig),
            "func_body" => Some(Self::FuncBody),
            "type" => Some(Self::Type),
            "trait" => Some(Self::Trait),
            "impl" => Some(Self::Impl),
            "const" => Some(Self::Const),
            _ => None,
        }
    }

    pub const fn symbol_priority(self) -> u8 {
        match self {
            Self::Module => 0,
            Self::FuncBody => 2,
            _ => 1,
        }
    }

    pub const fn is_function(self) -> bool {
        matches!(self, Self::FuncSig | Self::FuncBody)
    }
}

impl PartialEq<&str> for ItemRole {
    fn eq(&self, o: &&str) -> bool {
        self.as_str() == *o
    }
}

impl fmt::Display for ItemRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
