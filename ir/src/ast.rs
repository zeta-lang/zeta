use crate::hir::StrId;
use crate::span::SourceSpan;
use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stmt<'a, 'bump>
where
    'bump: 'a,
{
    Import(&'bump ImportStmt<'a, 'bump>),
    Package(&'bump PackageStmt<'a, 'bump>),
    Let(&'bump LetStmt<'a, 'bump>),
    Const(&'bump ConstStmt<'a, 'bump>),
    Return(&'bump ReturnStmt<'a, 'bump>),
    If(&'bump IfStmt<'a, 'bump>),
    While(&'bump WhileStmt<'a, 'bump>),
    For(&'bump ForStmt<'a, 'bump>),
    Match(&'bump MatchStmt<'a, 'bump>),
    UnsafeBlock(&'bump UnsafeBlock<'a, 'bump>),
    Block(&'bump Block<'a, 'bump>),
    FuncDecl(&'bump FuncDecl<'a, 'bump>),
    StructDecl(&'bump StructDecl<'a, 'bump>),
    InterfaceDecl(&'bump InterfaceDecl<'a, 'bump>),
    ImplDecl(&'bump ImplDecl<'a, 'bump>),
    EnumDecl(&'bump EnumDecl<'a, 'bump>),
    StateMachineDecl(&'bump StateMachineDecl<'a, 'bump>),
    Defer(&'bump DeferStmt<'a, 'bump>),
    Module(&'bump ModuleDecl<'a, 'bump>),
    ExprStmt(&'bump InternalExprStmt<'a, 'bump>),
    Break(Option<&'bump Expr<'a, 'bump>>, SourceSpan<'a>),
    TypeAliasDecl(&'bump TypeAliasDecl<'a, 'bump>),
    Continue(SourceSpan<'a>),
    Attributed(&'bump AttributedStmt<'a, 'bump>),
}

impl<'a, 'bump> Stmt<'a, 'bump> {
    /// Strips Attributed wrappers (any depth) -> (attrs, real stmt).
    pub fn peel(self) -> (Vec<Attribute<'a, 'bump>>, Stmt<'a, 'bump>) {
        let mut attrs = Vec::new();
        let mut cur = self;
        while let Stmt::Attributed(a) = cur {
            attrs.extend_from_slice(a.attrs);
            cur = a.inner;
        }
        (attrs, cur)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributedStmt<'a, 'bump> {
    pub attrs: &'bump [Attribute<'a, 'bump>],
    pub inner: Stmt<'a, 'bump>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeAliasDecl<'a, 'bump> {
    pub visibility: Visibility,
    pub name: StrId,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub ty: Type<'a, 'bump>,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportStmt<'a, 'bump> {
    pub path: &'bump Path<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackageStmt<'a, 'bump> {
    pub path: &'bump Path<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Path<'a, 'bump> {
    pub path: &'bump [StrId],
    pub member: Option<StrId>,
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> Display for Path<'a, 'bump> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for i in 0..self.path.len() {
            let element = self.path[i];
            write!(f, "{element}")?;
            if i + 1 < self.path.len() {
                write!(f, "::")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub ident: StrId,
    pub type_annotation: Type<'a, 'bump>,
    pub value: &'bump Expr<'a, 'bump>,
    pub mutable: bool,
    pub is_static: bool,
    pub catch_pattern: Option<ErrorHandlerPattern<'a, 'bump>>,
    pub span: SourceSpan<'a>,
    pub manual: bool,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub ident: StrId,
    pub type_annotation: Type<'a, 'bump>,
    pub value: &'bump Expr<'a, 'bump>,
    pub span: SourceSpan<'a>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImplDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub target: Type<'a, 'bump>,
    pub interface: Option<Type<'a, 'bump>>,
    pub methods: Option<&'bump [FuncDecl<'a, 'bump>]>,
    pub span: SourceSpan<'a>,
    pub constants: Option<&'bump [ConstStmt<'a, 'bump>]>,
    pub is_unsafe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub visibility: Visibility,
    pub sealed: bool,
    pub permits: Option<&'bump PermitsExpr<'a, 'bump>>,
    pub methods: Option<&'bump [FuncDecl<'a, 'bump>]>,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnumDecl<'a, 'bump> {
    pub name: StrId,
    pub visibility: Visibility,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub variants: &'bump [EnumVariant<'a, 'bump>],
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnumVariant<'a, 'bump> {
    pub name: StrId,
    pub fields: &'bump [Field<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Field<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub field_type: Type<'a, 'bump>,
    pub visibility: Visibility,
    pub span: SourceSpan<'a>,
    pub manual: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub value: Option<&'bump Expr<'a, 'bump>>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IfStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub condition: &'bump Expr<'a, 'bump>,
    pub then_branch: &'bump Block<'a, 'bump>,
    pub else_branch: Option<&'bump ElseBranch<'a, 'bump>>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElseBranch<'a, 'bump>
where
    'bump: 'a,
{
    If(&'bump IfStmt<'a, 'bump>),
    Else(&'bump Block<'a, 'bump>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub condition: &'bump Expr<'a, 'bump>,
    pub block: &'bump Block<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub kind: ForKind<'a, 'bump>,
    pub block: &'bump Block<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForKind<'a, 'bump>
where
    'bump: 'a,
{
    /// C-style: for (let i = 0; i < 10; i++)
    CStyle {
        let_stmt: Option<&'bump LetStmt<'a, 'bump>>,
        condition: Option<&'bump Expr<'a, 'bump>>,
        increment: Option<&'bump Expr<'a, 'bump>>,
    },
    /// Range-based: for i in 0..10 or for i in range
    RangeBased {
        variable: StrId,
        iterable: &'bump Expr<'a, 'bump>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub expr: &'bump Expr<'a, 'bump>,
    pub arms: &'bump [MatchArm<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchArm<'a, 'bump>
where
    'bump: 'a,
{
    pub pattern: Pattern<'a, 'bump>,
    pub guard: Option<&'bump Expr<'a, 'bump>>,
    pub block: &'bump Block<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternKind<'a, 'bump> {
    Ident(StrId, bool),
    Number(i64),
    String(StrId),
    Boolean(bool),
    Tuple(&'bump [Pattern<'a, 'bump>]),
    Array(&'bump [Pattern<'a, 'bump>]),
    Struct {
        name: StrId,
        fields: &'bump [(StrId, Pattern<'a, 'bump>)],
        variant_span: SourceSpan<'a>,
    },
    EnumVariant {
        name: StrId,
        bindings: &'bump [Pattern<'a, 'bump>],
        variant_span: SourceSpan<'a>,
    },
    Or(&'bump [Pattern<'a, 'bump>]),
    Wildcard,
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pattern<'a, 'bump> {
    pub kind: PatternKind<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> Pattern<'a, 'bump> {
    pub fn ident(span: SourceSpan<'a>, id: StrId, mutable: bool) -> Self {
        Self {
            kind: PatternKind::Ident(id, mutable),
            span,
        }
    }

    pub fn number(span: SourceSpan<'a>, value: i64) -> Self {
        Self {
            kind: PatternKind::Number(value),
            span,
        }
    }

    pub fn string(span: SourceSpan<'a>, value: StrId) -> Self {
        Self {
            kind: PatternKind::String(value),
            span,
        }
    }

    pub fn boolean(span: SourceSpan<'a>, value: bool) -> Self {
        Self {
            kind: PatternKind::Boolean(value),
            span,
        }
    }

    pub fn tuple(span: SourceSpan<'a>, patterns: &'bump [Pattern<'a, 'bump>]) -> Self {
        Self {
            kind: PatternKind::Tuple(patterns),
            span,
        }
    }

    pub fn array(span: SourceSpan<'a>, patterns: &'bump [Pattern<'a, 'bump>]) -> Self {
        Self {
            kind: PatternKind::Array(patterns),
            span,
        }
    }

    pub fn struct_pattern(
        span: SourceSpan<'a>,
        name: StrId,
        fields: &'bump [(StrId, Pattern<'a, 'bump>)],
        variant_span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: PatternKind::Struct {
                name,
                fields,
                variant_span,
            },
            span,
        }
    }

    pub fn enum_variant(
        span: SourceSpan<'a>,
        name: StrId,
        bindings: &'bump [Pattern<'a, 'bump>],
        variant_span: SourceSpan<'a>,
    ) -> Self {
        Self {
            kind: PatternKind::EnumVariant {
                name,
                bindings,
                variant_span,
            },
            span,
        }
    }

    pub fn or(span: SourceSpan<'a>, patterns: &'bump [Pattern<'a, 'bump>]) -> Self {
        Self {
            kind: PatternKind::Or(patterns),
            span,
        }
    }

    pub fn wildcard(span: SourceSpan<'a>) -> Self {
        Self {
            kind: PatternKind::Wildcard,
            span,
        }
    }

    pub fn null(span: SourceSpan<'a>) -> Self {
        Self {
            kind: PatternKind::Null,
            span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnsafeBlock<'a, 'bump>
where
    'bump: 'a,
{
    pub block: &'bump Block<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnBinder<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    /// Generic-argument indices from the root return type:
    /// `: n <- usize` -> [], `Result<n <- usize, E>` -> [0], `Option<Result<n <- usize, E>>` -> [0, 0].
    pub arg_path: &'bump [usize],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsuresClause<'a, 'bump>
where
    'bump: 'a,
{
    /// `ensures case <pattern> -> <cond>`; `None` means unconditional.
    pub pattern: Option<&'bump Pattern<'a, 'bump>>,
    pub condition: &'bump Expr<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuncDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub function_metadata: FuncModifiers,
    pub name: StrId,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub params: Option<&'bump [Param<'a, 'bump>]>,
    pub return_type: Option<Type<'a, 'bump>>,
    pub body: Option<&'bump Block<'a, 'bump>>,
    pub return_binders: &'bump [ReturnBinder<'a, 'bump>],
    pub ensures: &'bump [EnsuresClause<'a, 'bump>],
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
    pub attrs: &'bump [Attribute<'a, 'bump>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub visibility: Visibility,
    pub name: StrId,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub params: Option<&'bump [Param<'a, 'bump>]>,
    pub name_span: SourceSpan<'a>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param<'a, 'bump>
where
    'bump: 'a,
{
    Normal(&'bump NormalParam<'a, 'bump>),
    This(&'bump ThisParam<'a, 'bump>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThisParam<'a, 'bump> {
    pub passing_kind: ParamPassingKind,
    pub multi_place: Option<&'bump [EffectAccess<'a, 'bump>]>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalParam<'a, 'bump>
where
    'bump: 'a,
{
    pub is_mut: bool,
    pub name: StrId,
    pub type_annotation: Type<'a, 'bump>,
    pub visibility: Visibility,
    pub default_value: Option<Expr<'a, 'bump>>,
    pub multi_place: Option<&'bump [EffectAccess<'a, 'bump>]>,
    pub span: SourceSpan<'a>,
    pub manual: bool,
}

#[derive(Debug, Clone, PartialEq, Copy, Default)]
pub struct Block<'a, 'bump>
where
    'bump: 'a,
{
    pub block: &'bump [Stmt<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> IntoIterator for &'bump Block<'a, 'bump>
where
    'bump: 'a,
{
    type Item = &'bump Stmt<'a, 'bump>;
    type IntoIter = std::slice::Iter<'bump, Stmt<'a, 'bump>>;

    fn into_iter(self) -> Self::IntoIter {
        self.block.iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Generic<'a, 'bump> {
    pub type_name: StrId,
    pub span: SourceSpan<'a>,
    pub is_const: bool,
    pub is_static: bool,
    pub constraints: &'bump [Type<'a, 'bump>],
    pub default_type: Option<Type<'a, 'bump>>,
    pub min_provenance: Option<ProvenanceAnnotation<'bump>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InternalExprStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub expr: &'bump Expr<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldInit<'a, 'bump> {
    pub name: StrId,
    pub name_span: SourceSpan<'a>,
    pub value: Expr<'a, 'bump>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Expr<'a, 'bump>
where
    'bump: 'a,
{
    OrElse {
        value: &'bump Expr<'a, 'bump>,
        else_block: &'bump Block<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Attributed {
        attrs: &'bump [Attribute<'a, 'bump>],
        expr: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Intrinsic {
        name: StrId,
        generic_args: &'bump [Type<'a, 'bump>],
        arguments: &'bump [Expr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Null {
        span: SourceSpan<'a>,
    },
    Cast {
        expr: &'bump Expr<'a, 'bump>,
        target_type: Type<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Number {
        value: i64,
        span: SourceSpan<'a>,
    },
    Decimal {
        value: f64,
        span: SourceSpan<'a>,
    },
    String {
        value: StrId,
        span: SourceSpan<'a>,
    },
    Ident {
        name: StrId,
        span: SourceSpan<'a>,
    },
    /// An identifier with generic arguments, used for struct initialization with explicit generics:
    /// `Container<i32> { value: 42 }`
    GenericIdent {
        name: StrId,
        generic_args: &'bump [Type<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Boolean {
        value: bool,
        span: SourceSpan<'a>,
    },
    Comparison {
        lhs: &'bump Expr<'a, 'bump>,
        op: Op,
        rhs: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    StructInit {
        callee: &'bump Expr<'a, 'bump>,
        arguments: &'bump [FieldInit<'a, 'bump>],
        type_args: &'bump [Type<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    FieldAccess {
        object: &'bump Expr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    },
    Binary {
        left: &'bump Expr<'a, 'bump>,
        op: Op,
        right: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Call {
        callee: &'bump Expr<'a, 'bump>,
        generic_args: &'bump [Type<'a, 'bump>],
        arguments: &'bump [Expr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Get {
        object: &'bump Expr<'a, 'bump>,
        field: StrId,
        span: SourceSpan<'a>,
    },
    If {
        if_stmt: &'bump IfStmt<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Match {
        match_stmt: &'bump MatchStmt<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Assignment {
        lhs: &'bump Expr<'a, 'bump>,
        op: Op,
        rhs: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    ExprList {
        expressions: &'bump [Expr<'a, 'bump>],
        span: SourceSpan<'a>,
    },

    Char {
        value: char,
        span: SourceSpan<'a>,
    },
    Unary {
        op: Op,
        operand: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    ArrayIndex {
        expr: &'bump Expr<'a, 'bump>,
        index: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Deref {
        expr: &'bump Expr<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    This {
        span: SourceSpan<'a>,
    },
    Ref {
        expr: &'bump Expr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
    },
    Lambda {
        is_move: bool,
        modifiers: Option<LambdaModifier>,
        params: &'bump [LambdaParam<'a, 'bump>],
        return_type: Option<Type<'a, 'bump>>,
        /// `fn (a, b) ensures a != b { .. }`: unconditional clauses only (`pattern` is `None`).
        ensures: &'bump [EnsuresClause<'a, 'bump>],
        body: &'bump Block<'a, 'bump>,
        span: SourceSpan<'a>,
    },
    Tuple {
        values: &'bump [Expr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    ModulePath {
        segments: &'bump [StrId],
        span: SourceSpan<'a>,
    },
    ModuleAccess {
        segments: &'bump [StrId],
        member: StrId,
        span: SourceSpan<'a>,
    },
    ArrayLiteral {
        elements: &'bump [Expr<'a, 'bump>],
        span: SourceSpan<'a>,
    },
    Undefined {
        span: SourceSpan<'a>,
    },
    Uninit {
        span: SourceSpan<'a>,
    },
    Block(&'bump Block<'a, 'bump>),
    UnsafeBlock(&'bump UnsafeBlock<'a, 'bump>),
    ComptimeBlock(&'bump Block<'a, 'bump>),
}

/// A closure/anonymous-function parameter. Unlike `NormalParam`, the type
/// annotation is optional since closures can omit param types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaParam<'a, 'bump> {
    pub name: StrId,
    pub type_annotation: Option<Type<'a, 'bump>>,
    pub multi_place: Option<&'bump [EffectAccess<'a, 'bump>]>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaModifier {
    Suspend,
    Nosuspend,
    Blocking,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorHandlerPattern<'a, 'bump>
where
    'bump: 'a,
{
    /// `catch (MyError err) => { ... }`
    Single {
        error_type: Type<'a, 'bump>,
        binding: Option<StrId>,
        body: &'bump Block<'a, 'bump>,
    },
    /// `catch { MyError e => {}, MyOtherError e => {} }`
    Multiple {
        branches: &'bump [ErrorHandlerBranch<'a, 'bump>],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorHandlerBranch<'a, 'bump>
where
    'bump: 'a,
{
    pub error_type: Type<'a, 'bump>,
    pub binding: Option<StrId>,
    pub body: &'bump Block<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct Type<'a, 'bump>
where
    'bump: 'a,
{
    pub kind: TypeKind<'a, 'bump>,
    pub nullable: bool,
    pub span: SourceSpan<'a>,
}

impl<'a, 'bump> PartialEq for Type<'a, 'bump> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.nullable == other.nullable
    }
}

impl<'a, 'bump> Type<'a, 'bump> {
    pub fn struct_name(&self) -> Option<StrId> {
        match self.kind {
            TypeKind::Struct { name, .. } => Some(name),
            _ => None,
        }
    }

    pub fn struct_name_path(&self) -> Option<(StrId, &'bump [StrId])> {
        match self.kind {
            TypeKind::Struct { name, path, .. } => Some((name, path)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvenanceRoot {
    Var(StrId),
    ThisRoot,
    Global { module_idx: usize, name: StrId },
    ImplicitParam(u32),
    Static,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvenancePathSegment {
    Field(StrId),
    Deref,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
            ProvenanceRoot::ImplicitParam(i) => write!(f, "{i}"),
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypeKind<'a, 'bump>
where
    'bump: 'a,
{
    U8,
    I8,
    U16,
    I16,
    I32,
    F32,
    F64,
    I64,
    String,
    Boolean,
    UF32,
    U32,
    U64,
    I128,
    U128,
    UF64,
    Void,
    Infer,
    SafePointer {
        inner: &'a Type<'a, 'bump>,
        mutability_state: MutabilityState,
    },
    UnsafePointer {
        inner: &'a Type<'a, 'bump>,
        mutability_state: MutabilityState,
    },
    Array {
        inner: &'a Type<'a, 'bump>,
        length: usize,
    },
    Tuple {
        values: &'bump [Type<'a, 'bump>],
    },
    Slice {
        inner: &'a Type<'a, 'bump>,
    },
    Lambda {
        params: &'bump [Type<'a, 'bump>],
        return_type: &'a Type<'a, 'bump>,
    },
    Struct {
        name: StrId,
        path: &'bump [StrId],
        generics: &'bump [Type<'a, 'bump>],
    },
    This,
    Char,
    Ref {
        inner: &'a Type<'a, 'bump>,
        ref_kind: RefKind,
        provenance: Option<ProvenanceAnnotation<'bump>>,
    },
    Dyn {
        bounds: &'bump [Type<'a, 'bump>],
    },
    OwnedPointer {
        inner: &'a Type<'a, 'bump>,
        allocator: Option<ProvenanceAnnotation<'bump>>, // None = unbound, infer/monomorphize
    },
    Usize,
    Isize,
    Never,
    AnySlice,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateMachineDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub transitions: &'bump [Transition<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition<'a, 'bump>
where
    'bump: 'a,
{
    pub from_state: StateRef,
    pub to_state: StateRef,
    pub action: Option<&'bump Block<'a, 'bump>>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateRef {
    Named(StrId),
    Wildcard, // *
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeferStmt<'a, 'bump>
where
    'bump: 'a,
{
    pub action: DeferAction<'a, 'bump>,
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeferAction<'a, 'bump>
where
    'bump: 'a,
{
    Block(&'bump Block<'a, 'bump>),
    Stmt(&'bump Stmt<'a, 'bump>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModuleDecl<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub visibility: Visibility,
    pub body: &'bump [Stmt<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PermitsExpr<'a, 'bump>
where
    'bump: 'a,
{
    pub types: &'bump [Type<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamPassingKind {
    MoveMut,
    Move,
    RefConst,
    RefMut,
    RefAlias,
    ConstSafePtr,
    ConstUnsafePtr,
    MutSafePtr,
    MutUnsafePtr,
    MultiPlace,
}

impl fmt::Display for MutabilityState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MutabilityState::Mut => write!(f, "mut"),
            MutabilityState::Const => write!(f, "const"),
        }
    }
}

// Function Metadata
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FuncModifiers {
    pub visibility: Visibility,
    pub extern_modifier: ExternModifier,
    pub inline_modifier: InlineModifier,
    pub func_safety: FuncSafety,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExternModifier {
    #[default]
    None,
    Abi(StrId),
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InlineModifier {
    Inline,
    Noinline,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FuncSafety {
    #[default]
    Safe,
    Unsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutabilityState {
    Mut,
    Const,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Default, Eq)]
pub enum Visibility {
    Public,
    #[default]
    Private,
    Module,
    Internal,
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let displayed = match self {
            Visibility::Public => "public ",
            Visibility::Private => "",
            Visibility::Module => "module ",
            Visibility::Internal => "internal ",
        };
        write!(f, "{displayed}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    BitOr,
    BitXor,
    BitAnd,
    LogicalAnd,
    LogicalOr,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    ShlAssign,
    ShrAssign,
    BitOrAssign,
    BitXorAssign,
    BitAndAssign,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    BitNot,
    LogicalNot,
    Range,     // .. (inclusive range)
    RangeExcl, // ..< (exclusive range)
    Deref,
    Ref,
    RefMut,
}

impl Op {
    pub fn binding_power(&self) -> (u8, u8) {
        match self {
            Op::Range | Op::RangeExcl => (25, 26),
            Op::LogicalOr => (30, 31),
            Op::LogicalAnd => (35, 36),
            Op::Assign => (10, 11),
            Op::AddAssign
            | Op::SubAssign
            | Op::MulAssign
            | Op::DivAssign
            | Op::ModAssign
            | Op::ShlAssign
            | Op::ShrAssign
            | Op::BitOrAssign
            | Op::BitXorAssign
            | Op::BitAndAssign => (10, 11),
            Op::BitOr => (42, 43),
            Op::BitXor => (43, 44),
            Op::BitAnd => (44, 45),
            Op::Eq | Op::Neq => (45, 46),
            Op::Lt | Op::Lte | Op::Gt | Op::Gte => (50, 51),
            Op::Shl | Op::Shr => (55, 56),
            Op::Add | Op::Sub => (60, 61),
            Op::Mul | Op::Div | Op::Mod => (70, 71),
            Op::BitNot | Op::LogicalNot => (0, 80),
            _ => (0, 0),
        }
    }
}

impl TryFrom<&str> for Op {
    type Error = &'static str;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Ok(match s {
            "+" => Op::Add,
            "-" => Op::Sub,
            "*" => Op::Mul,
            "/" => Op::Div,
            "%" => Op::Mod,
            "<<" => Op::Shl,
            ">>" => Op::Shr,
            "|" => Op::BitOr,
            "^" => Op::BitXor,
            "&" => Op::BitAnd,

            "=" => Op::Assign,
            "+=" => Op::AddAssign,
            "-=" => Op::SubAssign,
            "*=" => Op::MulAssign,
            "/=" => Op::DivAssign,
            "%=" => Op::ModAssign,
            "<<=" => Op::ShlAssign,
            ">>=" => Op::ShrAssign,
            "|=" => Op::BitOrAssign,
            "^=" => Op::BitXorAssign,
            "&=" => Op::BitAndAssign,

            "==" => Op::Eq,
            "!=" => Op::Neq,
            "<" => Op::Lt,
            "<=" => Op::Lte,
            ">" => Op::Gt,
            ">=" => Op::Gte,

            _ => return Err("Unknown operator string"),
        })
    }
}

impl Op {
    pub fn is_math_op(&self) -> bool {
        match self {
            Op::Assign | Op::Eq | Op::Neq | Op::Lt | Op::Lte | Op::Gt | Op::Gte => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorHandlerKind {
    Error,
    Null,
}

impl<'a, 'bump> Type<'a, 'bump> {
    pub fn is_numeric(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::U8
                | TypeKind::I8
                | TypeKind::U16
                | TypeKind::I16
                | TypeKind::I32
                | TypeKind::I64
                | TypeKind::I128
                | TypeKind::U32
                | TypeKind::U64
                | TypeKind::U128
                | TypeKind::F32
                | TypeKind::F64
                | TypeKind::UF32
                | TypeKind::UF64
                | TypeKind::Usize
                | TypeKind::Isize
        )
    }

    pub fn i8(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::I8,
            nullable: false,
            span,
        }
    }

    pub fn i16(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::I16,
            nullable: false,
            span,
        }
    }

    pub fn i32(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::I32,
            nullable: false,
            span,
        }
    }

    pub fn i64(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::I64,
            nullable: false,
            span,
        }
    }

    pub fn i128(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::I128,
            nullable: false,
            span,
        }
    }

    pub fn u8(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::U8,
            nullable: false,
            span,
        }
    }

    pub fn u16(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::U16,
            nullable: false,
            span,
        }
    }

    pub fn u32(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::U32,
            nullable: false,
            span,
        }
    }

    pub fn u64(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::U64,
            nullable: false,
            span,
        }
    }

    pub fn u128(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::U128,
            nullable: false,
            span,
        }
    }

    pub fn usize(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Usize,
            nullable: false,
            span,
        }
    }

    pub fn isize(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Isize,
            nullable: false,
            span,
        }
    }

    pub fn f32(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::F32,
            nullable: false,
            span,
        }
    }

    pub fn f64(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::F64,
            nullable: false,
            span,
        }
    }

    pub fn boolean(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Boolean,
            nullable: false,
            span,
        }
    }

    pub fn string(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::String,
            nullable: false,
            span,
        }
    }

    pub fn void(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Void,
            nullable: false,
            span,
        }
    }

    pub fn this(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::This,
            nullable: false,
            span,
        }
    }

    pub fn char(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Char,
            nullable: false,
            span,
        }
    }

    pub fn infer(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Infer,
            nullable: false,
            span,
        }
    }

    pub fn never(span: SourceSpan<'a>) -> Self {
        Type {
            kind: TypeKind::Never,
            nullable: false,
            span,
        }
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

impl<'a, 'bump> Display for Type<'a, 'bump> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TypeKind::U8 => f.write_str("u8"),
            TypeKind::I8 => f.write_str("i8"),
            TypeKind::U16 => f.write_str("u16"),
            TypeKind::I16 => f.write_str("i16"),
            TypeKind::I32 => f.write_str("i32"),
            TypeKind::F32 => f.write_str("f32"),
            TypeKind::F64 => f.write_str("f64"),
            TypeKind::I64 => f.write_str("i64"),
            TypeKind::String => f.write_str("str"),
            TypeKind::Boolean => f.write_str("bool"),
            TypeKind::UF32 => f.write_str("uf32"),
            TypeKind::U32 => f.write_str("u32"),
            TypeKind::U64 => f.write_str("u64"),
            TypeKind::I128 => f.write_str("i128"),
            TypeKind::U128 => f.write_str("u128"),
            TypeKind::Usize => f.write_str("usize"),
            TypeKind::Isize => f.write_str("isize"),
            TypeKind::UF64 => f.write_str("uf64"),
            TypeKind::Void => f.write_str("void"),
            TypeKind::AnySlice => f.write_str("[]"),
            TypeKind::Lambda {
                params,
                return_type,
            } => {
                write!(f, "func(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "): {}", return_type)
            }
            TypeKind::Struct {
                name,
                path: _,
                generics,
            } => {
                write!(f, "{}", name)?;
                if !generics.is_empty() {
                    write!(f, "<")?;
                    for (i, generic) in generics.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", generic)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            TypeKind::Char => f.write_str("char"),
            TypeKind::This => f.write_str("this"),
            TypeKind::Infer => f.write_str("_"),
            TypeKind::Ref {
                inner,
                ref_kind,
                provenance,
            } => {
                write!(f, "&")?;

                if let Some(provenance) = provenance {
                    write!(f, "{} ", provenance)?;
                }

                if let RefKind::Unique = ref_kind {
                    write!(f, "mut ")?;
                } else if let RefKind::Alias = ref_kind {
                    write!(f, "alias ")?;
                }

                write!(f, "{inner}")
            }
            TypeKind::SafePointer {
                inner,
                mutability_state,
            } => write!(f, "*{} {}", mutability_state, inner),
            TypeKind::UnsafePointer {
                inner,
                mutability_state,
            } => write!(f, "[*]{} {}", mutability_state, inner),
            TypeKind::Array { inner, length } => write!(f, "[{}]{}", length, inner),
            TypeKind::Slice { inner } => write!(f, "[]{}", inner),
            TypeKind::Dyn { bounds } => {
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
            TypeKind::OwnedPointer { inner, allocator } => {
                write!(f, "^")?;
                if let Some(allocator) = allocator {
                    write!(f, "{} ", allocator)?;
                }
                write!(f, "{}", inner)
            }
            TypeKind::Never => f.write_str("never"),
            TypeKind::Tuple { values: _ } => todo!(),
        }?;

        if self.nullable {
            f.write_str("?")?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectAccess<'a, 'bump> {
    pub ref_kind: RefKind,
    pub path: &'bump [EffectSegment<'a, 'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectSegment<'a, 'bump> {
    Field(StrId),
    Index(&'bump Expr<'a, 'bump>),
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attribute<'a, 'bump> {
    pub name: StrId,
    pub args: &'bump [AttrArg<'bump>],
    pub span: SourceSpan<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttrArg<'bump> {
    Ident(StrId),
    Str(StrId),
    Number(i64),
    Bool(bool),
    KeyValue {
        key: StrId,
        value: &'bump AttrArg<'bump>,
    },
    Call {
        name: StrId,
        args: &'bump [AttrArg<'bump>],
    },
}
