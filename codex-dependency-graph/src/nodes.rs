use ir::hir::StrId;

use crate::NodeIdx;

#[derive(Clone, Debug)]
pub enum NodeKind {
    Module {
        module_idx: usize,
    },
    TypeDecl {
        module_idx: usize,
        item_idx: usize,
    },
    FuncSig {
        module_idx: usize,
        item_idx: usize,
    },
    FuncBody {
        module_idx: usize,
        item_idx: usize,
    },
    ConstDecl {
        module_idx: usize,
        item_idx: usize,
    },
    TraitDecl {
        module_idx: usize,
        item_idx: usize,
    },
    TraitImpl {
        module_idx: usize,
        item_idx: usize,
    },
    Method {
        module_idx: usize,
        item_idx: usize,
        method_idx: usize,
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
    pub fn module_idx(&self) -> Option<usize> {
        match self {
            NodeKind::Module { module_idx }
            | NodeKind::TypeDecl { module_idx, .. }
            | NodeKind::FuncSig { module_idx, .. }
            | NodeKind::FuncBody { module_idx, .. }
            | NodeKind::ConstDecl { module_idx, .. }
            | NodeKind::TraitDecl { module_idx, .. }
            | NodeKind::TraitImpl { module_idx, .. }
            | NodeKind::Method { module_idx, .. } => Some(*module_idx),
        }
    }
}

#[derive(Debug)]
pub struct DepNode {
    pub idx: NodeIdx,
    pub kind: NodeKind,
    pub hint: Option<StrId>,
    /// Outgoing edges: this node depends on these.
    pub deps: Vec<NodeIdx>,
    /// Reverse edges: these nodes depend on this one.
    pub rev_deps: Vec<NodeIdx>,
}

impl DepNode {
    pub(super) fn new(idx: NodeIdx, kind: NodeKind, hint: Option<StrId>) -> Self {
        DepNode {
            idx,
            kind,
            hint,
            deps: Vec::new(),
            rev_deps: Vec::new(),
        }
    }
}
