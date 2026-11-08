use std::{
    fmt,
    ops::{Index, IndexMut},
    path::PathBuf,
    sync::Arc,
};

use ir::{hir::Visibility, ir_hasher::HashMap};
use smallvec::SmallVec;

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
    ConstEval,
    StaticInit,
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

/// Stable item ID assigned by the graph.
///
/// an Item can be a type, function, constant, static, trait, implementation declaration, or another graph-relevant item
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ItemKey {
    /// The owner of this Item.
    pub module: ModuleIdx,
    /// Where the item resides in that module.
    pub item: ItemIdx,
    /// The kind of item it is (is it a static? const? type? function? etc)
    pub role: ItemRole,
}

#[derive(Clone, Copy)]
pub struct Ctx {
    pub from: NodeIdx,
    pub module: ModuleIdx,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
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

/// Stable module ID (file) assigned by the graph.
///
/// This should be a stable ID that doesn't change if its dependency order does not change.
#[derive(Eq, Copy, Hash, Clone, PartialEq, Debug)]
pub struct ModuleId(pub u32);

/// An identity for a build/versioning unit. A uniit refers to, the application being compiled, the standard library, or one external library dependency.
///
/// A unit may contain one or more source roots and declares which other units it depends on.
#[derive(Eq, Copy, Hash, Clone, PartialEq)]
pub struct UnitId(pub u32);

/// A version counter of how many times the graph itself (not an individual file, item, ...) has been updated.
///
/// Comments and whitespaces will not increment this.
#[derive(Eq, Copy, Hash, Clone, PartialEq)]
pub struct Revision(pub u64);

/// A stable hash of the contents of a file which is based on a normalized representation (not the hash of direct raw source bytes).
///
/// This will exclude whitespace and comments.
#[derive(Eq, Copy, Hash, Clone, PartialEq, Debug)]
pub struct Fingerprint(pub u128);

#[derive(Eq, Copy, Hash, Clone, PartialEq)]
pub enum ModuleImportMode {
    Local,
    ReExport,
}

#[derive(Eq, Copy, Hash, Clone, PartialEq)]
pub enum ExecutionMode {
    /// the item is evaluated/executed in ordinary program execution.
    Runtime,
    /// the item participates in compile-time evaluation/metaprogramming.
    Comptime,
}

#[derive(Eq, Copy, Hash, Clone, PartialEq, Debug)]
pub enum GraphRefKind {
    ByValue,
    ByPointer,
}

/// Facts extracted from one source module which is the graph's input representation for that file.
pub struct ModuleFacts<'a, 'bump> {
    /// The source file's path as discovered
    pub path: PathBuf,
    /// The package/namespace declaration provided by this module, if any.
    pub package_facts: Option<PackageFacts<'a, 'bump>>,
    /// Other items that the module depends on.
    pub refs: Vec<RefFacts<'a>>,
    /// What the file imports.
    pub imports: Vec<ImportFacts<'a, 'bump>>,
    /// The items declared in this file
    pub items: Vec<ItemFacts<'a>>,
    pub impls: Vec<ImplFacts<'a>>,
    pub diags: Vec<ScanDiag<'a>>,

    /// Package declarations, item keys, visibilities, signatures, fields, variant types, generic parameters, imports and re-exports of a module.
    pub iface_fp: Fingerprint,
    pub body_fp: Fingerprint,
}

pub struct ItemFacts<'a> {
    pub key: ScanKey,
    pub visibility: Visibility,
    pub execution_mode: ExecutionMode,
    pub span: SourceSpan<'a>,
    pub iface_fp: Fingerprint,
    pub body_fp: Fingerprint,
    pub param_uses: Vec<ParamUse>,
}

pub struct RefFacts<'a> {
    /// The declaration that contains the reference.
    pub from: ScanKey,
    /// The reference target, whether it points to a symbol, local, method, etc.
    pub target: RefTarget,
    pub via: Vec<ViaArg>,
    /// The kind of edge
    pub edge_kind: EdgeKind,
    pub span: SourceSpan<'a>,
}

/// Facts extracted from a package declaration.
///
/// For example, `package chess.engine` is represented as:
/// `[chess, engine]`.
#[derive(Clone)]
pub struct PackageFacts<'a, 'bump> {
    pub path: &'bump ir::hir::Path<'a, 'bump>,

    /// Location of the package declaration.
    pub span: SourceSpan<'a>,
}

/// Facts extracted from an import declaration.
///
/// For example:
/// `import chess.engine as engine`
pub struct ImportFacts<'a, 'bump> {
    /// The imported module or package path.
    pub path: &'bump ir::hir::Path<'a, 'bump>,

    /// Optional local name introduced by the import.
    ///
    /// For `import chess.engine as engine`, this is `engine`.
    pub alias: Option<StrId>,

    /// Location of the import declaration.
    pub span: SourceSpan<'a>,
}

/// Facts extracted from an implementation declaration.
///
/// Supports both inherent implementations and trait implementations.
///
/// Inherent:
/// `impl ChessBoard { ... }`
///
/// Trait implementation:
/// `impl Display for ChessBoard { ... }`
pub struct ImplFacts<'a> {
    /// Identity of the implementation declaration itself.
    pub item: ItemKey,

    /// The type being implemented for.
    pub self_type: RefTarget,

    /// The implemented trait, if this is a trait implementation.
    ///
    /// `None` means this is an inherent implementation.
    pub trait_type: Option<RefTarget>,

    /// Methods declared inside this implementation.
    pub methods: Vec<ItemKey>,
    pub span: SourceSpan<'a>,
}

/// The position of an item in its containing module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ItemPos {
    /// Index of the containing top-level item in the module.
    pub item_index: ItemKey,

    /// Index of the method within the implementation, if applicable.
    ///
    /// `None` means it's a module-level item and not a function of a type.
    pub method_index: Option<usize>,
}

impl ItemPos {
    /// Creates a position for a module-level item.
    pub const fn module_item(item_index: ItemKey) -> Self {
        Self {
            item_index,
            method_index: None,
        }
    }

    /// Creates a position for a method inside an implementation.
    pub const fn impl_method(item_index: ItemKey, method_index: usize) -> Self {
        Self {
            item_index,
            method_index: Some(method_index),
        }
    }
}

/// A unit is a set of modules that are versioned and depended upon together, which can be an application, a library or an external dependency.
pub struct UnitDecl {
    /// The unit's unique name within this graph, such as the application name like `my_chess_engine` or `std`.
    pub name: StrId,
    /// Canonical directory roots owned by this unit.
    pub roots: Vec<CanonicalPath>,
    /// The units this unit is allowed to depend on.
    pub deps: Vec<UnitDep>,
    /// Readable unit provenance such as `path:./app` or `dependency:llvm_bindings@1.2.3`
    pub origin: UnitOrigin,
    /// Optional prefix for symbols emitted from this unit.
    pub mangle_prefix: Option<StrId>,
}

// An opaque origin that could be `path:./app` or `dependency:llvm_bindings@1.2.3`
pub type UnitOrigin = Arc<str>;

/// A normalized filesystem path used to identify a module in the dependency graph.
///
/// Different path representations that refer to the same lexical location should
/// produce equal `CanonicalPath` values for reliable keying in hash maps and prevents the same module from being registered more
/// than once under different path representations.
///
/// Normalization defines how relative paths, absolute paths, `.` and `..`
/// components, path separators, and platform-specific case sensitivity are handled and represents it into the "common" path without requiring the path to *actually* exist.
///
/// This means that `./src/..` is the same as `.` or `src/./main.zeta` is the same as `src/main.zeta`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalPath(pub PathBuf);

impl CanonicalPath {
    /// Creates a normalized path.
    ///
    /// Examples:
    /// - `src/./main.zeta` -> `src/main.zeta`
    /// - `src/lib/../main.zeta` -> `src/main.zeta`
    /// - `./` -> `.`
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self(normalize_lexically(path.as_ref()))
    }

    /// Wraps a path without normalizing it.
    ///
    /// The caller must guarantee that the path already follows
    /// the normalization policy used by this type.
    ///
    /// Violating the invariant cannot cause memory unsafety and therefore this unchecked API is not `unsafe`, but it can cause duplicate graph entries if you're not careful.
    pub fn new_unchecked(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// Returns the underlying path.
    pub fn as_path(&self) -> &std::path::Path {
        &self.0
    }

    /// Consumes this value and returns its underlying PathBuf.
    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }
}

impl AsRef<std::path::Path> for CanonicalPath {
    fn as_ref(&self) -> &std::path::Path {
        self.as_path()
    }
}

impl fmt::Display for CanonicalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

/// Normalizes `.` and `..` components without touching the filesystem.
fn normalize_lexically(path: &std::path::Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}

            std::path::Component::ParentDir => {
                if normalized.file_name().is_some() {
                    // Remove the preceding normal component.
                    normalized.pop();
                } else if !normalized.has_root() {
                    // Preserve leading `..` in relative paths.
                    normalized.push("..");
                }
                // For absolute paths, `..` cannot go above the root.
            }

            component => {
                normalized.push(component.as_os_str());
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

/// A dependency between units
pub struct UnitDep {
    /// The name of the unit that is depended on.
    pub unit: StrId,
    pub scope: DepScope,
}

pub enum DepScope {
    /// Modules must `import` the dependency's packages to use them.
    Explicit,
    /// Something that can be implicitly imported --- e.g. `zeta::result.Result`.
    Implicit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    Module(ModuleId),
    Item(ItemKey),
    Instance { item: ItemKey, args: Fingerprint },
}

#[derive(Clone, Debug)]
pub struct Edge<'a> {
    pub from: Owner,
    pub to: Target,
    pub edge_kind: EdgeKind,
    pub graph_ref_kind: GraphRefKind,
    pub span: SourceSpan<'a>,
    pub provenance: Provenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Provenance {
    Syntactic,
    Resolved,
}

pub struct ReverseEdge {
    pub owner: Owner,
    pub kind: EdgeKind,
}

pub struct ModuleDep {
    pub dependent: ModuleId,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Item(ItemKey),
    Module(ModuleId),
    Missing { scope: ModuleId, what: RefTarget },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MissingKey {
    pub scope: ModuleId,
    pub target: RefTarget,
}

pub struct Inner<'a, 'bump> {
    pub revision: Revision,

    pub modules: Vec<Option<Arc<ModuleEntry<'a, 'bump>>>>,
    pub units: Vec<UnitEntry>,

    pub indices: Indices,
    pub dependencies: DependencyGraph<'a>,
    pub sccs: SccGraph,

    pub diagnostics: Vec<Vec<GraphDiagnostic>>,
}

pub struct ModuleEntry<'a, 'bump> {
    pub id: ModuleId,
    pub unit: UnitId,
    /// Whether this file is re-exported by its owning package or not.
    pub import_mode: ModuleImportMode,
    pub decl: Arc<ModuleFacts<'a, 'bump>>,
    pub resolved: ResolvedModule,

    pub source_changed_at: Revision,
    pub iface_changed_at: Revision,
    pub body_changed_at: Revision,
}

pub struct DependencyGraph<'a> {
    pub owners: HashMap<Owner, OwnerEdges<'a>>,

    pub reverse: HashMap<Target, SmallVec<ReverseEdge, 2>>,

    pub module_reverse: Vec<SmallVec<ModuleDep, 4>>,

    pub missing: HashMap<MissingKey, SmallVec<Owner, 2>>,
}

pub struct OwnerEdges<'a> {
    pub resolved: Option<Vec<Edge<'a>>>,
    pub syntactic: Vec<Edge<'a>>,
}

pub struct ResolvedModule {
    pub symbols: ModuleSymbols,
    pub imports: Vec<ModuleId>,
}

pub struct ModuleSymbols {
    pub items: HashMap<SymbolName, Vec<ItemKey>>,
    pub methods: HashMap<MethodName, Vec<ItemKey>>,
}

pub struct Indices {
    pub paths: HashMap<CanonicalPath, ModuleId>,

    pub units: UnitIndices,

    pub item_modules: HashMap<ItemKey, ModuleId>,
}

pub struct UnitIndices {
    pub packages: HashMap<UnitId, PackageIndex>,
    pub methods: HashMap<UnitId, MethodIndex>,
}

pub type PackageIndex = HashMap<Vec<StrId>, Vec<ModuleId>>;

pub type MethodIndex = HashMap<(TypeIdentity, StrId), Vec<ItemKey>>;

/// Source-level namespace/category used during name lookup.
///
/// This is deliberately separate from internal compiler roles such as
/// FuncSig and FuncBody.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Module,
    Static,
    Const,
    Type,
    Trait,
    Function,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymbolName {
    pub name: StrId,
    pub kind: SymbolKind,
}

/// Identity of a nominal type that can own methods.
///
/// This can be expanded later if methods on other kinds of types
/// need to participate in lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeIdentity(pub ItemKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodName {
    pub owner: TypeIdentity,
    pub name: StrId,
}

/// A syntactic name/path being resolved.
///
/// This is also used to track failed lookups so that introducing
/// a previously missing declaration can invalidate its dependents.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RefTarget {
    Name { name: StrId, kind: SymbolKind },
    Path(Vec<StrId>),
    Method { owner: TypeIdentity, name: StrId },
}

pub struct UnitEntry {
    pub id: UnitId,
    pub decl: UnitDecl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SccId(pub u32);

pub struct SccGraph {
    pub components: Vec<Scc>,
    pub module_to_scc: Vec<Option<SccId>>,
}

pub struct Scc {
    pub id: SccId,
    pub modules: Vec<ModuleId>,
    pub dependencies: Vec<SccId>,
}

pub struct GraphDiagnostic {
    pub module: ModuleId,
    pub kind: GraphDiagnosticKind,
}

#[derive(Clone, Debug)]
pub enum GraphDiagnosticKind {
    DuplicateModule,
    DuplicatePackage,
    DuplicateUnit,
    MissingDependency,
    UnitCycle,
    ImportCycle,
    MissingReference,
    AmbiguousReference,
}

/// A problem found while scanning one file that doesn't need any other file
/// to detect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScanDiag<'a> {
    /// A statement that is only valid inside a function body (`let`, `if`,
    /// `while`, `return`, ...) appeared at module level.
    StrayTopLevel {
        /// Location of the offending statement.
        span: SourceSpan<'a>,
    },

    /// A construct the scanner doesn't handle yet.
    Unsupported {
        /// Location of the construct.
        span: SourceSpan<'a>,
        /// Short description, e.g. `"interface declaration"`.
        what: &'static str,
    },

    /// `dyn Trait` used by value.
    ByValueDyn {
        /// Location of the `dyn` type.
        span: SourceSpan<'a>,
    },

    /// More than one `package` declaration in a file.
    MultiplePackages {
        /// Location of the first declaration.
        first: SourceSpan<'a>,
        /// Location of the extra declarations.
        extra: SourceSpan<'a>,
    },
    RequiredBodyAbsent {
        span: SourceSpan<'a>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScanKey {
    pub name: StrId,
    pub item_key: ItemKey,
    pub ordinal: u64,
}

/// One step of a "this edge only exists if..." chain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ViaArg {
    /// The generic type the argument was passed to (unresolved name or path).
    pub generic: RefTarget,
    /// Which parameter of that generic the argument fills (0-based).
    pub index: u32,
}

/// How a generic item uses one of its own type parameters in its layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamUse {
    /// Index of the parameter in the item's own generic parameter list.
    pub param: u32,
    pub how: ParamHow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamHow {
    /// The parameter is embedded directly such as `struct Box2<T> { v: T }`.
    Direct,
    /// The parameter is embedded only if every step in the chain embeds its
    /// argument such as `struct W<T> { v: Vec<Option<T>> }`
    Through(Vec<ViaArg>),
}

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum BodyRule {
    Optional,
    Required,
}
