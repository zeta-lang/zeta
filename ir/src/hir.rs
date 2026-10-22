use crate::ast::LambdaModifier;
use crate::ast::MutabilityState;
pub use crate::ast::{FuncModifiers, Path, Visibility};
use crate::hir::DropKind::Undroppable;
use crate::span::SourceSpan;
use std::cell::UnsafeCell;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;
use zetaruntime::string_pool::VmString;

/// A reference to an interned string in the global string pool
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrId(pub VmString);

pub trait JoinStr {
    fn join(&self, sep: &str) -> String;
}

impl<T: AsRef<str>> JoinStr for &[T] {
    fn join(&self, sep: &str) -> String {
        let mut out = String::new();
        for (i, p) in self.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            out.push_str(p.as_ref());
        }
        out
    }
}

impl<T: AsRef<str>> JoinStr for Vec<T> {
    fn join(&self, sep: &str) -> String {
        let mut out = String::new();
        for (i, p) in self.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            out.push_str(p.as_ref());
        }
        out
    }
}

impl<'a, 'bump> JoinStr for Path<'a, 'bump> {
    fn join(&self, sep: &str) -> String {
        let mut out = String::new();
        for (i, p) in self.path.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            out.push_str(p.as_ref());
        }
        out
    }
}

impl Default for StrId {
    fn default() -> Self {
        StrId::from_static("DEFAULT_STR_ID")
    }
}

pub struct Split<'s> {
    inner: std::str::Split<'static, &'s str>,
}

impl<'s> Iterator for Split<'s> {
    type Item = StrId;

    #[inline]
    fn next(&mut self) -> Option<StrId> {
        self.inner.next().map(StrId::from_static)
    }
}

impl AsRef<str> for StrId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl StrId {
    pub fn split<'s>(&self, sep: &'s str) -> Split<'s> {
        Split {
            inner: self.as_str().split(sep),
        }
    }

    pub fn join<I>(parts: I, sep: &str, pool: &StringPool) -> StrId
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        StrId(pool.intern(&Self::join_string(parts, sep)))
    }

    pub fn join_string<I>(parts: I, sep: &str) -> String
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut out = String::new();
        for (i, p) in parts.into_iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            out.push_str(p.as_ref());
        }
        out
    }
}

impl Ord for StrId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for StrId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for StrId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Display for StrId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Deref for StrId {
    type Target = VmString;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StrId {
    pub fn new(vm_string: VmString) -> Self {
        StrId(vm_string)
    }

    pub const fn from_static(s: &'static str) -> Self {
        Self(VmString::from_static(s))
    }

    pub fn into_inner(self) -> VmString {
        self.0
    }

    pub fn as_str(&self) -> &'static str {
        self.0.as_str()
    }

    pub fn len(&self) -> usize {
        self.0.length
    }

    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PartialEq<&StrId> for StrId {
    fn eq(&self, other: &&StrId) -> bool {
        self == *other
    }

    fn ne(&self, other: &&StrId) -> bool {
        self != *other
    }
}

impl PartialEq<&str> for StrId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }

    fn ne(&self, other: &&str) -> bool {
        self.as_str() != *other
    }
}

impl PartialEq<str> for StrId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }

    fn ne(&self, other: &str) -> bool {
        self.as_str() != other
    }
}

impl From<VmString> for StrId {
    fn from(vm_string: VmString) -> Self {
        StrId(vm_string)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hir<'a, 'bump>
where
    'bump: 'a,
{
    Module(&'bump HirModule<'a, 'bump>),
    Func(&'bump HirFunc<'a, 'bump>),
    Struct(&'bump HirStruct<'a, 'bump>),
    Interface(&'bump HirInterface<'a, 'bump>),
    Impl(&'bump HirImpl<'a, 'bump>),
    Enum(&'bump HirEnum<'a, 'bump>),
    Const(&'bump ConstStmt<'a, 'bump>),
    Stmt(&'bump HirStmt<'a, 'bump>),
    Expr(&'bump HirExpr<'a, 'bump>),
}

impl<'a, 'bump> Hir<'a, 'bump> {
    pub fn is_func(&self) -> bool {
        if let Hir::Func(_) = self {
            return true;
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicKind {
    SizeOf,
    AlignOf,
    AssertAlign,
    TypeName,
    Own,
    CpuRelax,
    Unreachable,
    Reinterpret,
    Replace,
    Leak,
    FnPtr,
    DropInPlace,
    MemForget,
    AssumeInit,

    AtomicLoad,
    AtomicStore,
    AtomicSwap,
    AtomicCas,
    AtomicFetchAdd,
    AtomicFetchSub,
    AtomicFetchAnd,
    AtomicFetchOr,
    AtomicFetchXor,
    AtomicFence,

    PtrRead,
    PtrWrite,
    PtrCopy,
    PtrCopyNonoverlapping,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProvenanceRoot {
    Var(StrId),
    ThisRoot,
    Global { module_idx: usize, name: StrId },
    ImplicitParam(u32),
    Static,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProvenancePathSegment {
    Field(StrId),
    Deref,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub struct ProvenanceAnnotation<'bump> {
    pub root: ProvenanceRoot,
    pub path: &'bump [ProvenancePathSegment],
}

impl fmt::Display for ProvenanceRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceRoot::Var(name) => write!(f, "{name}"),
            ProvenanceRoot::ThisRoot => write!(f, "self"),
            ProvenanceRoot::Global { name, .. } => write!(f, "{name}"),
            ProvenanceRoot::ImplicitParam(i) => write!(f, "'{i}"),
            ProvenanceRoot::Static => write!(f, "static"),
        }
    }
}

impl fmt::Display for ProvenancePathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenancePathSegment::Field(field) => write!(f, ".{field}"),
            ProvenancePathSegment::Deref => write!(f, "*"),
        }
    }
}

impl<'bump> ProvenanceAnnotation<'bump> {
    #[inline]
    pub fn is_static(&self) -> bool {
        matches!(self.root, ProvenanceRoot::Static)
    }
}

impl<'bump> fmt::Display for ProvenanceAnnotation<'bump> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)?;

        for segment in self.path {
            match segment {
                ProvenancePathSegment::Field(field) => {
                    write!(f, ".{field}")?;
                }
                ProvenancePathSegment::Deref => {
                    write!(f, "*")?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HirModule<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub imports: &'bump [Path<'a, 'bump>],
    pub items: &'bump [Hir<'a, 'bump>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirReturnBinder<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub arg_path: &'bump [usize],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirEnsures<'a, 'bump>
where
    'bump: 'a,
{
    pub pattern: Option<HirPattern<'a, 'bump>>,
    pub condition: HirExpr<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirFunc<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub function_metadata: FuncModifiers,
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub params: Option<&'bump [HirParam<'a, 'bump>]>,
    pub return_type: Option<HirType<'a, 'bump>>,
    pub ensures: &'bump [HirEnsures<'a, 'bump>],
    pub return_binders: &'bump [HirReturnBinder<'a, 'bump>],
    pub body: Option<HirStmt<'a, 'bump>>,
    pub unmangled_name: StrId,
    pub declaring_module_idx: usize,
    pub impl_target: Option<StrId>,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirStruct<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub visibility: Visibility,
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub fields: &'bump [HirField<'a, 'bump>],
    pub unmangled_name: StrId,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirImpl<'a, 'bump>
where
    'bump: 'a,
{
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub interface: Option<StrId>,
    pub interface_generics: Option<&'bump [HirType<'a, 'bump>]>,
    pub target: StrId,
    pub target_generics: Option<&'bump [HirType<'a, 'bump>]>,
    pub methods: Option<&'bump [HirFunc<'a, 'bump>]>,
    pub is_unsafe: bool,

    pub target_span: SourceSpan<'a>,
    pub interface_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
    pub declaring_module_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirInterface<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub visibility: Visibility,
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub methods: Option<&'bump [HirFunc<'a, 'bump>]>,
    pub unmangled_name: StrId,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
    pub declaring_module_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirEnum<'a, 'bump> {
    pub name: StrId,
    pub visibility: Visibility,
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub variants: &'bump [HirEnumVariant<'a, 'bump>],
    pub unmangled_name: StrId,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct HirEnumVariant<'a, 'bump> {
    pub name: StrId,
    pub fields: &'bump [HirField<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct HirField<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub field_type: HirType<'a, 'bump>,
    pub visibility: Visibility,
    pub manual: bool,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThisPassingKind {
    /// `&this`
    RefConst,
    /// `&mut this`
    RefMut,
    /// `*mut this`
    MutSafePtr,
    /// `*const this`
    ConstSafePtr,
    /// `[*]mut this`
    MutUnsafePtr,
    /// `[*]const this`
    ConstUnsafePtr,
    /// `this`
    Move,
    /// `mut this`
    MoveMut,
    MultiPlace,
    RefAlias,
}

impl Display for ThisPassingKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ThisPassingKind::RefConst => write!(f, "&this"),
            ThisPassingKind::RefMut => write!(f, "&mut this"),
            ThisPassingKind::MutSafePtr => write!(f, "*mut this"),
            ThisPassingKind::ConstSafePtr => write!(f, "*const this"),
            ThisPassingKind::MutUnsafePtr => write!(f, "[*]mut this"),
            ThisPassingKind::ConstUnsafePtr => write!(f, "[*]const this"),
            ThisPassingKind::Move => write!(f, "this"),
            ThisPassingKind::MoveMut => write!(f, "mut this"),
            ThisPassingKind::MultiPlace => write!(f, "this.{{..}}"),
            ThisPassingKind::RefAlias => write!(f, "&alias this"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirParam<'a, 'bump>
where
    'bump: 'a,
{
    Normal {
        name: StrId,
        param_type: HirType<'a, 'bump>,
        multi_place: Option<&'bump [HirEffectAccess<'bump>]>,
        span: SourceSpan<'a>,
    },
    This {
        kind: ThisPassingKind,
        multi_place: Option<&'bump [HirEffectAccess<'bump>]>,
        span: SourceSpan<'a>,
    },
}

impl<'a, 'bump> HirParam<'a, 'bump> {
    pub fn get_type(&self) -> Option<HirType<'a, 'bump>> {
        match self {
            HirParam::Normal { param_type, .. } => Some(*param_type),
            HirParam::This { span, .. } => Some(HirType::this(*span)),
        }
    }

    /// Returns true when `this` is passed by pointer (all variants except Move/MoveMut).
    pub fn this_is_ptr(&self) -> bool {
        matches!(
            self,
            HirParam::This {
                kind: ThisPassingKind::RefConst
                    | ThisPassingKind::RefMut
                    | ThisPassingKind::MutSafePtr
                    | ThisPassingKind::ConstSafePtr
                    | ThisPassingKind::MutUnsafePtr
                    | ThisPassingKind::ConstUnsafePtr
                    | ThisPassingKind::MultiPlace,
                ..
            }
        )
    }

    pub fn get_span(&self) -> SourceSpan<'a> {
        match self {
            HirParam::Normal { span, .. } => *span,
            HirParam::This { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirGeneric<'a, 'bump> {
    pub name: StrId,
    pub constraints: &'bump [HirType<'a, 'bump>],
    pub default_type: Option<HirType<'a, 'bump>>,
    pub min_provenance: Option<ProvenanceAnnotation<'bump>>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HirTypeKind<'a, 'bump>
where
    'bump: 'a,
{
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    I128,
    U128,
    Boolean,
    String,
    Struct {
        name: StrId,
        unmangled_name: StrId,
        field_types: &'bump [HirType<'a, 'bump>],
        type_args: &'bump [HirType<'a, 'bump>],
    },
    DynInterface {
        name: StrId,
        unmangled_name: StrId,
        type_args: &'bump [HirType<'a, 'bump>],
    },
    Enum {
        name: StrId,
        unmangled_name: StrId,
        variants: &'bump [HirEnumVariant<'a, 'bump>],
        type_args: &'bump [HirType<'a, 'bump>],
    },
    SafePointer {
        inner: &'a HirType<'a, 'bump>,
        mutability_state: MutabilityState,
    },
    Ref {
        inner: &'a HirType<'a, 'bump>,
        ref_kind: RefKind,
        provenance: Option<ProvenanceAnnotation<'bump>>,
    },
    UnsafePointer {
        inner: &'a HirType<'a, 'bump>,
        mutability_state: MutabilityState,
    },
    OwnedPointer {
        inner: &'bump HirType<'a, 'bump>,
        allocator: Option<ProvenanceAnnotation<'bump>>,
    },
    Lambda {
        params: &'bump [HirType<'a, 'bump>],
        return_type: &'a HirType<'a, 'bump>,
    },
    Generic(StrId),
    Void,
    This,
    Null,
    Char,
    /// Placeholder used when a type couldn't be determined, e.g. after a
    /// type error has already been reported for the expression.
    Unknown,
    Nullable(&'a HirType<'a, 'bump>),
    Dyn {
        bounds: &'bump [HirType<'a, 'bump>],
    },
    Tuple(&'bump [HirType<'a, 'bump>]),
    Array(&'a HirType<'a, 'bump>, usize),
    Slice(&'a HirType<'a, 'bump>),
    Usize,
    Isize,
    Never,
    Range {
        elem: &'a HirType<'a, 'bump>,
        inclusive: bool,
    },
}

/// A type plus where it was written. `span` is `SourceSpan::default()` for
/// synthesized types. It is deliberately ignored by `==` and `Hash`.
#[derive(Debug, Clone, Copy)]
pub struct HirType<'a, 'bump>
where
    'bump: 'a,
{
    pub kind: HirTypeKind<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> PartialEq for HirType<'a, 'bump> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
impl<'a, 'bump> Eq for HirType<'a, 'bump> {}

impl<'a, 'bump> Hash for HirType<'a, 'bump> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl<'a, 'bump> From<HirTypeKind<'a, 'bump>> for HirType<'a, 'bump> {
    fn from(kind: HirTypeKind<'a, 'bump>) -> Self {
        HirType::synthetic(kind)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub ty: HirType<'a, 'bump>,
    pub value: HirExpr<'a, 'bump>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirStmt<'a, 'bump>
where
    'bump: 'a,
{
    Let {
        name: StrId,
        ty: HirType<'a, 'bump>,
        value: HirExpr<'a, 'bump>,
        mutable: bool,
        is_static: bool,
        catch_pattern: Option<HirErrorHandlerPattern<'a, 'bump>>,
        span: SourceSpan<'a>,
        manual: bool,
        visibility: Visibility,
        inferred_ty: bool,
    },
    Const(&'bump ConstStmt<'a, 'bump>),
    Return(Option<&'bump HirExpr<'a, 'bump>>, SourceSpan<'a>),
    Expr(&'bump HirExpr<'a, 'bump>),
    If {
        cond: HirExpr<'a, 'bump>,
        then_block: &'bump [HirStmt<'a, 'bump>],
        else_block: Option<&'bump HirStmt<'a, 'bump>>,
        span: SourceSpan<'a>,
    },
    While {
        cond: &'bump HirExpr<'a, 'bump>,
        body: &'bump HirStmt<'a, 'bump>,
    },
    For {
        init: Option<&'bump HirStmt<'a, 'bump>>,
        condition: Option<&'bump HirExpr<'a, 'bump>>,
        increment: Option<&'bump HirExpr<'a, 'bump>>,
        body: &'bump HirStmt<'a, 'bump>,
    },
    Match {
        expr: &'bump HirExpr<'a, 'bump>,
        arms: &'bump [HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    UnsafeBlock {
        body: &'bump HirStmt<'a, 'bump>,
    },
    Block {
        body: &'bump [HirStmt<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Break(Option<&'bump HirExpr<'a, 'bump>>, SourceSpan<'a>),
    Continue(SourceSpan<'a>),
    Defer(&'bump HirStmt<'a, 'bump>),
    Import(Path<'a, 'bump>, SourceSpan<'a>),
    Package(Path<'a, 'bump>, SourceSpan<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirErrorHandlerPattern<'a, 'bump>
where
    'bump: 'a,
{
    Single {
        error_type: HirType<'a, 'bump>,
        binding: Option<StrId>,
        body: &'bump [HirStmt<'a, 'bump>],
    },
    Multiple {
        branches: &'bump [HirErrorHandlerBranch<'a, 'bump>],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirErrorHandlerBranch<'a, 'bump>
where
    'bump: 'a,
{
    pub error_type: HirType<'a, 'bump>,
    pub binding: Option<StrId>,
    pub body: &'bump [HirStmt<'a, 'bump>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirMatchArm<'a, 'bump>
where
    'bump: 'a,
{
    pub pattern: HirPattern<'a, 'bump>,
    pub guard: Option<&'bump HirExpr<'a, 'bump>>,
    pub body: &'bump HirStmt<'a, 'bump>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirPatternKind<'a, 'bump> {
    Ident(StrId, bool),
    Number(i64),
    String(StrId),
    Boolean(bool),
    Tuple(&'bump [HirPattern<'a, 'bump>]),
    Array(&'bump [HirPattern<'a, 'bump>]),
    Struct {
        name: StrId,
        fields: &'bump [(StrId, HirPattern<'a, 'bump>)],
        variant_span: SourceSpan<'a>,
    },
    EnumVariant {
        variant: StrId,
        bindings: &'bump [StrId],
        variant_span: SourceSpan<'a>,
    },
    Or(&'bump [HirPattern<'a, 'bump>]),
    Wildcard,
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirPattern<'a, 'bump> {
    pub kind: HirPatternKind<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> HirPattern<'a, 'bump> {
    pub fn ident(span: SourceSpan<'a>, id: StrId, mutable: bool) -> Self {
        Self {
            kind: HirPatternKind::Ident(id, mutable),
            span,
        }
    }

    pub fn number(span: SourceSpan<'a>, value: i64) -> Self {
        Self {
            kind: HirPatternKind::Number(value),
            span,
        }
    }

    pub fn string(span: SourceSpan<'a>, value: StrId) -> Self {
        Self {
            kind: HirPatternKind::String(value),
            span,
        }
    }

    pub fn boolean(span: SourceSpan<'a>, value: bool) -> Self {
        Self {
            kind: HirPatternKind::Boolean(value),
            span,
        }
    }

    pub fn tuple(span: SourceSpan<'a>, patterns: &'bump [HirPattern<'a, 'bump>]) -> Self {
        Self {
            kind: HirPatternKind::Tuple(patterns),
            span,
        }
    }

    pub fn array(span: SourceSpan<'a>, patterns: &'bump [HirPattern<'a, 'bump>]) -> Self {
        Self {
            kind: HirPatternKind::Array(patterns),
            span,
        }
    }

    pub fn struct_pattern(
        span: SourceSpan<'a>,
        name: StrId,
        fields: &'bump [(StrId, HirPattern<'a, 'bump>)],
        variant_span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirPatternKind::Struct {
                name,
                fields,
                variant_span,
            },
            span,
        }
    }

    pub fn enum_variant(
        span: SourceSpan<'a>,
        variant: StrId,
        bindings: &'bump [StrId],
        variant_span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirPatternKind::EnumVariant {
                variant,
                bindings,
                variant_span,
            },
            span,
        }
    }

    pub fn or(span: SourceSpan<'a>, patterns: &'bump [HirPattern<'a, 'bump>]) -> Self {
        Self {
            kind: HirPatternKind::Or(patterns),
            span,
        }
    }

    pub fn wildcard(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirPatternKind::Wildcard,
            span,
        }
    }

    pub fn null(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirPatternKind::Null,
            span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirFuncProto<'a, 'bump> {
    pub name: StrId,
    pub params: Option<&'bump [HirParam<'a, 'bump>]>,
    pub return_type: HirType<'a, 'bump>,
    pub function_metadata: FuncModifiers,
    pub generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    pub unmangled_name: StrId,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirFieldInit<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub name_span: SourceSpan<'a>,
    pub value: HirExpr<'a, 'bump>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirExpr<'a, 'bump>
where
    'bump: 'a,
{
    OrElse {
        value: &'bump HirExpr<'a, 'bump>,
        else_body: &'bump [HirStmt<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Intrinsic {
        kind: IntrinsicKind,
        type_args: &'bump [HirType<'a, 'bump>], // the <T> part
        args: &'bump [HirExpr<'a, 'bump>],      // runtime args, e.g. $assert_align(ptr, 8)
        span: SourceSpan<'a>,
    },
    Cast {
        expr: &'bump HirExpr<'a, 'bump>,
        target_type: HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Uninit {
        span: SourceSpan<'a>,
        ty: HirType<'a, 'bump>,
    },
    Null(SourceSpan<'a>),
    Number(i64, SourceSpan<'a>),
    Char(char, SourceSpan<'a>),
    String(StrId, SourceSpan<'a>),
    Boolean(bool, SourceSpan<'a>),
    Ident(StrId, SourceSpan<'a>),
    Tuple(&'bump [HirExpr<'a, 'bump>], SourceSpan<'a>),
    Decimal(f64, SourceSpan<'a>),
    Binary {
        left: &'bump HirExpr<'a, 'bump>,
        op: Operator,
        right: &'bump HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Call {
        callee: &'bump HirExpr<'a, 'bump>,
        args: &'bump [HirExpr<'a, 'bump>],
        type_args: Option<&'bump [HirType<'a, 'bump>]>,
        span: SourceSpan<'a>,
    },
    InterfaceCall {
        callee: &'bump HirExpr<'a, 'bump>,
        args: &'bump [HirExpr<'a, 'bump>],
        interface: StrId,
        span: SourceSpan<'a>,
    },
    FieldAccess {
        object: &'bump HirExpr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    },
    Assignment {
        target: &'bump HirExpr<'a, 'bump>,
        op: AssignmentOperator,
        value: &'bump HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    InterpolatedString(&'bump [InterpolationPart<'a, 'bump>]),
    EnumInit {
        enum_name: StrId,
        variant: StrId,
        args: &'bump [HirExpr<'a, 'bump>],
        type_args: Option<&'bump [HirType<'a, 'bump>]>,
        variant_span: SourceSpan<'a>,
        span: SourceSpan<'a>,
    },
    ExprList {
        list: &'bump [HirExpr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Get {
        object: &'bump HirExpr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    },
    StructInit {
        name: &'bump HirExpr<'a, 'bump>,
        args: &'bump [HirFieldInit<'a, 'bump>],
        type_args: Option<&'bump [HirType<'a, 'bump>]>,
        span: SourceSpan<'a>,
    },
    Comparison {
        left: &'bump HirExpr<'a, 'bump>,
        op: Operator,
        right: &'bump HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Deref {
        expr: &'bump HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Ref {
        expr: &'bump HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
    },
    This {
        span: SourceSpan<'a>,
    },
    ModuleAccess(&'bump HirModuleAccess<'a, 'bump>),
    Lambda {
        is_move: bool,
        modifier: Option<LambdaModifier>,
        params: &'bump [HirLambdaParam<'a, 'bump>],
        return_type: &'a HirType<'a, 'bump>,
        /// Unconditional `ensures` clauses (`pattern` is always `None` for lambdas).
        ensures: &'bump [HirEnsures<'a, 'bump>],
        body: &'bump HirStmt<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Index {
        object: &'bump HirExpr<'a, 'bump>,
        index: &'bump HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    ArrayLiteral {
        elements: &'bump [HirExpr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Undefined {
        span: SourceSpan<'a>,
        ty: HirType<'a, 'bump>,
    },

    UnknownIntrinsic {
        span: SourceSpan<'a>,
        name: StrId,
    },
    GenericIdent(StrId, &'bump [HirType<'a, 'bump>], SourceSpan<'a>),
    If {
        if_stmt: &'bump HirStmt<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Match {
        expr: &'bump HirExpr<'a, 'bump>,
        arms: &'bump [HirMatchArm<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Block {
        body: &'bump [HirStmt<'a, 'bump>],
        is_unsafe: bool,
        span: SourceSpan<'a>,
    },
    Range {
        start: &'bump HirExpr<'a, 'bump>,
        end: &'bump HirExpr<'a, 'bump>,
        inclusive: bool,
        span: SourceSpan<'a>,
    },
    Slice {
        object: &'bump HirExpr<'a, 'bump>,
        start: &'bump HirExpr<'a, 'bump>,
        end: &'bump HirExpr<'a, 'bump>,
        inclusive: bool,
        span: SourceSpan<'a>,
    },
}

impl<'a, 'bump> HirExpr<'a, 'bump>
where
    'bump: 'a,
{
    pub fn get_span(&self) -> SourceSpan<'a> {
        match self {
            Self::OrElse { span, .. }
            | Self::Intrinsic { span, .. }
            | Self::Cast { span, .. }
            | Self::Uninit { span, .. }
            | Self::Binary { span, .. }
            | Self::Call { span, .. }
            | Self::InterfaceCall { span, .. }
            | Self::FieldAccess { span, .. }
            | Self::Assignment { span, .. }
            | Self::EnumInit { span, .. }
            | Self::ExprList { span, .. }
            | Self::Get { span, .. }
            | Self::StructInit { span, .. }
            | Self::Comparison { span, .. }
            | Self::Deref { span, .. }
            | Self::Ref { span, .. }
            | Self::This { span, .. }
            | Self::Lambda { span, .. }
            | Self::Index { span, .. }
            | Self::ArrayLiteral { span, .. }
            | Self::Undefined { span, .. }
            | Self::UnknownIntrinsic { span, .. }
            | Self::If { span, .. }
            | Self::Match { span, .. }
            | Self::Block { span, .. }
            | Self::Range { span, .. }
            | Self::Slice { span, .. } => *span,

            Self::Null(span)
            | Self::Number(_, span)
            | Self::Char(_, span)
            | Self::String(_, span)
            | Self::Boolean(_, span)
            | Self::Ident(_, span)
            | Self::Tuple(_, span)
            | Self::Decimal(_, span)
            | Self::GenericIdent(_, _, span) => *span,

            // Default which gives a bogus source span.. but interpolated strings aren't really even working anyway.
            // Please no one remind me that this exists until it gives me a bug 4 years down the line when major companies depend on Zeta /j
            // - Wildered Oct 12 2026
            Self::InterpolatedString(_) => SourceSpan::default(),

            Self::ModuleAccess(access) => access.span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModuloAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShiftLeftAssign,
    ShiftRightAssign,
    BitNot,
    LogicalNot,
    LogicalOr,
    LogicalAnd,
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    DerefUnsafe,
    Deref,
    Ref,
    RefMut,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignmentOperator {
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModuloAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShiftLeftAssign,
    ShiftRightAssign,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationPart<'a, 'bump>
where
    'bump: 'a,
{
    String(StrId),
    Expr(&'bump HirExpr<'a, 'bump>),
}

impl<'a, 'bump> Display for HirType<'a, 'bump> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}

impl<'a, 'bump> HirTypeKind<'a, 'bump> {
    fn write_args(f: &mut Formatter, name: &StrId, args: &[HirType<'a, 'bump>]) -> fmt::Result {
        if args.is_empty() {
            write!(f, "{}", name)
        } else {
            let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
            write!(f, "{}<{}>", name, args_str.join(", "))
        }
    }
}

impl<'a, 'bump> Display for HirTypeKind<'a, 'bump> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use HirTypeKind as K;
        match self {
            K::I8 => write!(f, "i8"),
            K::I16 => write!(f, "i16"),
            K::I32 => write!(f, "i32"),
            K::I64 => write!(f, "i64"),
            K::U8 => write!(f, "u8"),
            K::U16 => write!(f, "u16"),
            K::U32 => write!(f, "u32"),
            K::U64 => write!(f, "u64"),
            K::I128 => write!(f, "i128"),
            K::U128 => write!(f, "u128"),
            K::Usize => write!(f, "usize"),
            K::Isize => write!(f, "isize"),
            K::F32 => write!(f, "f32"),
            K::F64 => write!(f, "f64"),
            K::Boolean => write!(f, "bool"),
            K::String => write!(f, "str"),
            K::Struct {
                unmangled_name: name,
                type_args: args,
                ..
            } => Self::write_args(f, name, args),

            K::DynInterface {
                unmangled_name: name,
                type_args,
                ..
            } => Self::write_args(f, name, type_args),
            K::Enum {
                unmangled_name: name,
                type_args,
                ..
            } => Self::write_args(f, name, type_args),
            K::Lambda {
                params,
                return_type,
            } => {
                let params_str: Vec<_> = params.iter().map(|p| p.to_string()).collect();
                write!(f, "fn({})", params_str.join(", "))?;

                write!(f, " -> {}", return_type)
            }
            K::Generic(name) => write!(f, "{}", name),
            K::Void => write!(f, "void"),
            K::This => write!(f, "this"),
            K::Null => write!(f, "null"),
            K::SafePointer {
                inner,
                mutability_state,
            } => write!(
                f,
                "*{} {}",
                if *mutability_state == MutabilityState::Const {
                    "const"
                } else {
                    "mut"
                },
                inner
            ),
            K::UnsafePointer {
                inner,
                mutability_state,
            } => write!(
                f,
                "[*]{} {}",
                if *mutability_state == MutabilityState::Const {
                    "const"
                } else {
                    "mut"
                },
                inner
            ),
            K::Char => write!(f, "char"),
            K::Ref {
                inner,
                ref_kind,
                provenance,
            } => {
                write!(f, "&")?;
                if let Some(p) = provenance {
                    write!(f, "{} ", p)?;
                }
                match ref_kind {
                    RefKind::Unique => write!(f, "mut ")?,
                    RefKind::Alias => write!(f, "alias ")?,
                    RefKind::Shared => {}
                }
                write!(f, "{inner}")
            }
            K::Nullable(hir_type) => write!(f, "{}?", hir_type),
            K::Dyn { bounds } => {
                write!(f, "dyn ")?;
                let mut is_start = true;
                let mut iter = bounds.into_iter();
                while let Some(bound) = iter.next() {
                    write!(f, "{}", bound)?;
                    if !is_start {
                        write!(f, " + ")?;
                    }
                    is_start = false;
                }
                Ok(())
            }
            K::Unknown => write!(f, "<unknown>"),
            K::Tuple(args) => {
                let params_str: String = args
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");

                write!(f, "({})", params_str)
            }
            K::Array(hir_type, len) => write!(f, "&[{}]{}", len, hir_type),
            K::Slice(hir_type) => write!(f, "&[]{}", hir_type),
            K::OwnedPointer { inner, allocator } => {
                write!(
                    f,
                    "^{} {}",
                    allocator
                        .map(|all| all.to_string())
                        .unwrap_or(String::from("")),
                    inner
                )
            }
            K::Never => write!(f, "never"),
            K::Range { elem, inclusive } => {
                write!(
                    f,
                    "range<{}>{}",
                    elem,
                    if *inclusive { " (inclusive)" } else { "" }
                )
            }
        }
    }
}

impl<'a, 'bump> HirType<'a, 'bump>
where
    'bump: 'a,
{
    pub fn new(kind: HirTypeKind<'a, 'bump>, span: SourceSpan<'a>) -> Self {
        Self { kind, span }
    }

    /// For types that have no source location (inferred, built by the checker).
    pub fn synthetic(kind: HirTypeKind<'a, 'bump>) -> Self {
        Self {
            kind,
            span: SourceSpan::default(),
        }
    }

    pub fn with_span(mut self, span: SourceSpan<'a>) -> Self {
        self.span = span;
        self
    }

    pub fn has_span(&self) -> bool {
        self.span.line != 0
    }

    pub fn is_owned_slice(&self) -> bool {
        matches!(
            self.kind,
            HirTypeKind::OwnedPointer { inner, .. } if matches!(inner.kind, HirTypeKind::Slice(_))
        )
    }

    pub fn is_safe_pointer(&self) -> bool {
        matches!(self.kind, HirTypeKind::SafePointer { .. })
    }

    pub fn is_unsafe_pointer(&self) -> bool {
        matches!(self.kind, HirTypeKind::UnsafePointer { .. })
    }

    pub fn is_pointer_literal(&self) -> bool {
        matches!(
            self.kind,
            HirTypeKind::SafePointer { .. } | HirTypeKind::UnsafePointer { .. }
        )
    }

    pub fn is_pointer_semantics(&self) -> bool {
        matches!(
            self.kind,
            HirTypeKind::SafePointer { .. }
                | HirTypeKind::UnsafePointer { .. }
                | HirTypeKind::Ref { .. }
        )
    }

    pub fn as_safe_pointer(&self) -> Option<&'a HirType<'a, 'bump>> {
        match self.kind {
            HirTypeKind::SafePointer { inner, .. } => Some(inner),
            _ => None,
        }
    }

    pub fn as_unsafe_pointer(&self) -> Option<&'a HirType<'a, 'bump>> {
        match self.kind {
            HirTypeKind::UnsafePointer { inner, .. } => Some(inner),
            _ => None,
        }
    }

    pub fn as_pointer(&self) -> Option<&'a HirType<'a, 'bump>> {
        match self.kind {
            HirTypeKind::SafePointer { inner, .. } | HirTypeKind::UnsafePointer { inner, .. } => {
                Some(inner)
            }
            _ => None,
        }
    }

    pub fn drop_kind(&self) -> DropKind<'a, 'bump> {
        self.drop_kind_with(None)
    }

    pub fn drop_kind_with(
        &self,
        default_alloc: Option<ProvenanceAnnotation<'bump>>,
    ) -> DropKind<'a, 'bump> {
        self.try_drop_kind_with(default_alloc)
            .expect("owned pointer has no allocator and no default was supplied")
    }

    pub fn try_drop_kind_with(
        &self,
        default_alloc: Option<ProvenanceAnnotation<'bump>>,
    ) -> Option<DropKind<'a, 'bump>> {
        Some(match self.kind {
            HirTypeKind::OwnedPointer { inner, allocator } => DropKind::OwnedPointer {
                pointee: Box::new(inner.try_drop_kind_with(default_alloc)?),
                pointee_ty: *inner,
                allocator: allocator.or(default_alloc)?,
            },
            HirTypeKind::Nullable(inner) => match inner.try_drop_kind_with(default_alloc) {
                None | Some(DropKind::Undroppable) => DropKind::Undroppable,
                Some(k) => DropKind::Nullable {
                    inner: Box::new(k),
                    inner_ty: *inner,
                },
            },
            HirTypeKind::Struct { name, .. }
            | HirTypeKind::Enum { name, .. }
            | HirTypeKind::DynInterface { name, .. } => DropKind::Type(name),
            HirTypeKind::Slice(inner) => {
                let element = inner.try_drop_kind_with(default_alloc)?;
                if element.is_droppable() {
                    DropKind::Slice {
                        element: Box::new(element),
                        element_ty: *inner,
                    }
                } else {
                    DropKind::Undroppable
                }
            }
            _ => DropKind::Undroppable,
        })
    }

    pub fn nominal_type_name(&self) -> Option<StrId> {
        match self.kind {
            HirTypeKind::Struct { name, .. }
            | HirTypeKind::Enum { name, .. }
            | HirTypeKind::DynInterface { name, .. } => Some(name),

            HirTypeKind::Ref { inner, .. }
            | HirTypeKind::SafePointer { inner, .. }
            | HirTypeKind::UnsafePointer { inner, .. }
            | HirTypeKind::OwnedPointer { inner, .. }
            | HirTypeKind::Nullable(inner)
            | HirTypeKind::Array(inner, _)
            | HirTypeKind::Slice(inner) => inner.nominal_type_name(),

            _ => None,
        }
    }

    pub const fn i8(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::I8,
            span,
        }
    }

    pub const fn i16(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::I16,
            span,
        }
    }

    pub const fn i32(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::I32,
            span,
        }
    }

    pub const fn i64(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::I64,
            span,
        }
    }

    pub const fn i128(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::I128,
            span,
        }
    }

    pub const fn u8(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::U8,
            span,
        }
    }

    pub const fn u16(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::U16,
            span,
        }
    }

    pub const fn u32(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::U32,
            span,
        }
    }

    pub const fn u64(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::U64,
            span,
        }
    }

    pub const fn u128(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::U128,
            span,
        }
    }

    pub const fn usize(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Usize,
            span,
        }
    }

    pub const fn isize(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Isize,
            span,
        }
    }

    pub const fn f32(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::F32,
            span,
        }
    }

    pub const fn f64(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::F64,
            span,
        }
    }

    pub const fn boolean(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Boolean,
            span,
        }
    }

    pub const fn string(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::String,
            span,
        }
    }

    pub const fn char(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Char,
            span,
        }
    }

    pub const fn void(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Void,
            span,
        }
    }

    pub const fn this(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::This,
            span,
        }
    }

    pub const fn null(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Null,
            span,
        }
    }

    pub const fn unknown(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Unknown,
            span,
        }
    }

    pub const fn never(span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Never,
            span,
        }
    }

    pub const fn generic(name: StrId, span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Generic(name),
            span,
        }
    }

    pub const fn struct_type(
        name: StrId,
        unmangled_name: StrId,
        field_types: &'bump [HirType<'a, 'bump>],
        type_args: &'bump [HirType<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirTypeKind::Struct {
                name,
                unmangled_name,
                field_types,
                type_args,
            },
            span,
        }
    }

    pub const fn nullable(inner: &'a HirType<'a, 'bump>, span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Nullable(inner),
            span,
        }
    }

    pub const fn slice(elem: &'a HirType<'a, 'bump>, span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Slice(elem),
            span,
        }
    }

    pub const fn array(elem: &'a HirType<'a, 'bump>, len: usize, span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Array(elem, len),
            span,
        }
    }

    pub const fn tuple(elems: &'bump [HirType<'a, 'bump>], span: SourceSpan<'a>) -> Self {
        Self {
            kind: HirTypeKind::Tuple(elems),
            span,
        }
    }

    pub const fn lambda(
        params: &'bump [HirType<'a, 'bump>],
        return_type: &'a HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirTypeKind::Lambda {
                params,
                return_type,
            },
            span,
        }
    }

    pub const fn safe_ptr_to(
        inner: &'a HirType<'a, 'bump>,
        mutability_state: MutabilityState,
        span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirTypeKind::SafePointer {
                inner,
                mutability_state,
            },
            span,
        }
    }

    pub const fn unsafe_ptr_to(
        inner: &'a HirType<'a, 'bump>,
        mutability_state: MutabilityState,
        span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: HirTypeKind::UnsafePointer {
                inner,
                mutability_state,
            },
            span,
        }
    }

    pub const fn struct_name(&self) -> Option<StrId> {
        match self.kind {
            HirTypeKind::Struct { name, .. } => Some(name),
            _ => None,
        }
    }

    pub const fn enum_name(&self) -> Option<StrId> {
        match self.kind {
            HirTypeKind::Enum { name, .. } => Some(name),
            _ => None,
        }
    }

    pub const fn type_args(&self) -> &'bump [HirType<'a, 'bump>] {
        match self.kind {
            HirTypeKind::Struct { type_args, .. } | HirTypeKind::Enum { type_args, .. } => {
                type_args
            }
            _ => &[],
        }
    }

    pub const fn generic_name(&self) -> Option<StrId> {
        match self.kind {
            HirTypeKind::Generic(name) => Some(name),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DropKind<'a, 'bump> {
    Type(StrId),
    OwnedPointer {
        pointee: Box<DropKind<'a, 'bump>>,
        pointee_ty: HirType<'a, 'bump>,
        allocator: ProvenanceAnnotation<'bump>,
    },
    Slice {
        element: Box<DropKind<'a, 'bump>>,
        element_ty: HirType<'a, 'bump>,
    },
    Undroppable,
    Nullable {
        inner: Box<DropKind<'a, 'bump>>,
        inner_ty: HirType<'a, 'bump>,
    },
}

impl<'a, 'bump> DropKind<'a, 'bump> {
    pub fn is_droppable(&self) -> bool {
        match self {
            Undroppable => false,
            _ => true,
        }
    }
}

pub struct HirSlice<'bump, T> {
    inner: UnsafeCell<&'bump [T]>,
}

impl<'bump, T> HirSlice<'bump, T> {
    pub fn new(inner: &'bump [T]) -> Self {
        Self {
            inner: UnsafeCell::new(inner),
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // Safe because shared access doesn't mutate
        unsafe { *self.inner.get() }
    }

    #[inline(always)]
    #[allow(invalid_reference_casting)]
    pub fn as_mut_slice(&self) -> &mut [T] {
        // SAFETY: multiple mutable borrows would be UB,
        // caller must guarantee exclusive access
        unsafe { &mut *(*self.inner.get() as *const [T] as *mut [T]) }
    }
}

/// A module-qualified access: `std::io.println` or `std::io.File`.
/// `path` is the `::` separated module segments, `member` is the name
/// after the first `.`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirModuleAccess<'a, 'bump> {
    pub path: &'bump [StrId], // ["std", "io"]
    pub member: StrId,        // "println" / "File" / "out"
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirLambdaParam<'a, 'bump> {
    pub name: StrId,
    pub param_type: Option<HirType<'a, 'bump>>,
    pub multi_place: Option<&'bump [HirEffectAccess<'bump>]>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirEffectAccess<'bump> {
    pub ref_kind: RefKind,
    pub path: &'bump [HirEffectSegment<'bump>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HirEffectSegment<'bump> {
    Field(StrId),
    Index(EffectIndexKey<'bump>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectIndexKey<'bump> {
    Const(i64),
    Place { root: StrId, path: &'bump [StrId] },
    Dynamic,
}

impl<'bump> EffectIndexKey<'bump> {
    pub fn provably_same(&self, other: &Self) -> bool {
        match (self, other) {
            (EffectIndexKey::Const(a), EffectIndexKey::Const(b)) => a == b,
            (
                EffectIndexKey::Place { root: ra, path: pa },
                EffectIndexKey::Place { root: rb, path: pb },
            ) => ra == rb && pa == pb,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefKind {
    Shared, // &
    Alias,  // &alias
    Unique, // &mut
}

impl RefKind {
    /// Unique -> Alias -> Shared.
    pub fn coerces_to(self, target: RefKind) -> bool {
        use RefKind::*;
        matches!(
            (self, target),
            (Unique, Unique)
                | (Unique, Alias)
                | (Unique, Shared)
                | (Alias, Alias)
                | (Alias, Shared)
                | (Shared, Shared)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    ByValue,
    ByRef(RefKind),
}

#[derive(Debug, Clone, Copy)]
pub struct HirClosureCapture<'bump> {
    pub name: StrId,
    pub mode: CaptureMode,
    pub source: StrId,
    pub source_path: &'bump [HirEffectSegment<'bump>],
}

/// Derived from what the body does to its captures.
///   Fn     - only reads
///   FnMut  - mutates at least one capture
///   FnOnce - moves at least one capture out
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

/// Everything the hoister needs to closure-convert one lambda. Produced by the
/// type checker (it needs types to infer modes), keyed by the address of the
/// lambda's `body: &'bump HirStmt`, which is stable across the by-value copies
/// the hoister makes of HirExpr/HirStmt.
#[derive(Clone)]
pub struct ClosureLowering<'a, 'bump> {
    pub env_name: StrId, // `__closure_env_N`, the synthetic struct type
    pub fn_name: StrId,  // `__closure_fn_N`,  the hoisted function
    /// HirType::Struct { name: env_name, field_types: <one per capture> }
    pub env_ty: HirType<'a, 'bump>,
    pub captures: Vec<HirClosureCapture<'bump>>,
    /// Resolved parameter / return types (annotation, else from the generic
    /// constraint).
    pub param_tys: Vec<HirType<'a, 'bump>>,
    pub ret_ty: HirType<'a, 'bump>,
    pub kind: ClosureKind,
}

pub fn static_effect_path<'x, 'bump>(
    expr: &HirExpr<'x, 'bump>,
    this_id: StrId,
    bump: &GrowableBump<'bump>,
) -> Option<(StrId, Vec<HirEffectSegment<'bump>>)> {
    match expr {
        HirExpr::Ident(name, _) => Some((*name, Vec::new())),
        HirExpr::This { .. } => Some((this_id, Vec::new())),
        HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
            let (root, mut path) = static_effect_path(object, this_id, bump)?;
            path.push(HirEffectSegment::Field(*field));
            Some((root, path))
        }
        HirExpr::Index { object, index, .. } => {
            let (root, mut path) = static_effect_path(object, this_id, bump)?;
            let key = static_index_key(index, this_id, bump)?;
            path.push(HirEffectSegment::Index(key));
            Some((root, path))
        }
        _ => None,
    }
}

fn static_index_key<'x, 'bump>(
    index: &HirExpr<'x, 'bump>,
    this_id: StrId,
    bump: &GrowableBump<'bump>,
) -> Option<EffectIndexKey<'bump>> {
    match index {
        HirExpr::Number(n, _) => Some(EffectIndexKey::Const(*n)),
        other => {
            let (root, path) = static_effect_path(other, this_id, bump)?;
            let field_path: Vec<StrId> = path
                .into_iter()
                .map(|seg| match seg {
                    HirEffectSegment::Field(f) => Some(f),
                    HirEffectSegment::Index(_) => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(EffectIndexKey::Place {
                root,
                path: bump.alloc_slice(&field_path),
            })
        }
    }
}

pub fn effect_path_is_prefix(
    prefix: &[HirEffectSegment<'_>],
    full: &[HirEffectSegment<'_>],
) -> bool {
    if prefix.len() > full.len() {
        return false;
    }
    prefix
        .iter()
        .zip(full.iter())
        .all(|(a, b)| effect_segment_eq(a, b))
}

pub fn effect_segment_eq(a: &HirEffectSegment<'_>, b: &HirEffectSegment<'_>) -> bool {
    match (a, b) {
        (HirEffectSegment::Field(x), HirEffectSegment::Field(y)) => x == y,
        (HirEffectSegment::Index(x), HirEffectSegment::Index(y)) => effect_index_key_eq(x, y),
        _ => false,
    }
}

pub fn effect_index_key_eq(a: &EffectIndexKey<'_>, b: &EffectIndexKey<'_>) -> bool {
    match (a, b) {
        (EffectIndexKey::Const(x), EffectIndexKey::Const(y)) => x == y,
        (
            EffectIndexKey::Place { root: rx, path: px },
            EffectIndexKey::Place { root: ry, path: py },
        ) => rx == ry && px == py,
        (EffectIndexKey::Dynamic, EffectIndexKey::Dynamic) => true,
        _ => false,
    }
}

/// Names a pattern binds (syntactic; shared by the checker and the hoister).
pub fn pattern_bound_names(pat: &HirPattern<'_, '_>, out: &mut Vec<StrId>) {
    match pat.kind {
        HirPatternKind::Ident(n, _) => out.push(n),
        HirPatternKind::EnumVariant { bindings, .. } => out.extend(bindings.iter().copied()),
        HirPatternKind::Tuple(ps) | HirPatternKind::Array(ps) => {
            for p in ps.iter() {
                pattern_bound_names(p, out);
            }
        }
        HirPatternKind::Struct { fields, .. } => {
            for (_, p) in fields.iter() {
                pattern_bound_names(p, out);
            }
        }
        HirPatternKind::Or(ps) => {
            if let Some(first) = ps.first() {
                pattern_bound_names(first, out);
            }
        }
        HirPatternKind::Wildcard
        | HirPatternKind::Number(_)
        | HirPatternKind::String(_)
        | HirPatternKind::Boolean(_)
        | HirPatternKind::Null => {}
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HirAttrArg<'bump> {
    Ident(StrId),
    Str(StrId),
    Number(i64),
    Bool(bool),
    KeyValue {
        key: StrId,
        value: &'bump HirAttrArg<'bump>,
    },
    Call {
        name: StrId,
        args: &'bump [HirAttrArg<'bump>],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HirAttribute<'a, 'bump> {
    pub name: StrId,
    pub args: &'bump [HirAttrArg<'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Default)]
pub struct AttrTable<'a, 'bump> {
    pub map:
        crate::ir_hasher::FxHashMap<crate::attributes::AttrTarget, Vec<HirAttribute<'a, 'bump>>>,
}
impl<'a, 'bump> AttrTable<'a, 'bump> {
    pub fn get(&self, t: crate::attributes::AttrTarget) -> &[HirAttribute<'a, 'bump>] {
        self.map.get(&t).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn find(
        &self,
        t: crate::attributes::AttrTarget,
        name: &str,
    ) -> Option<&HirAttribute<'a, 'bump>> {
        self.get(t).iter().find(|a| a.name == name)
    }
}
