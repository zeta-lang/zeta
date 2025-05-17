#[derive(Debug, Clone)]
pub enum Stmt {
    Import(ImportStmt),
    Let(LetStmt),
    Return(ReturnStmt),
    If(IfStmt),
    Else(ElseBranch),
    While(WhileStmt),
    For(ForStmt),
    Match(MatchStmt),
    UnsafeBlock(UnsafeBlock),
    FuncDecl(FuncDecl),
    ClassDecl(ClassDecl),
    ExprStmt(InternalExprStmt),
}

#[derive(Debug, Clone)]
pub struct ImportStmt {
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub mutability: Option<MutKeyword>,
    pub ident: String,
    pub type_annotation: Option<String>,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_branch: Block,
    pub else_branch: Option<Box<ElseBranch>>,
}

#[derive(Debug, Clone)]
pub enum ElseBranch {
    If(Box<IfStmt>),
    Else(Block),
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Box<Expr>,
    pub block: Block,
}

#[derive(Debug, Clone)]
pub struct ForStmt {
    pub let_stmt: LetStmt,
    pub condition: Box<Expr>,
    pub increment: Box<Expr>,
    pub block: Block,
}

#[derive(Debug, Clone)]
pub struct MatchStmt {
    pub expr: Expr,
    pub arms: Vec<MatchArm>,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub block: Block,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Ident(String),
    Number(i64),
    String(String),
    Tuple(Vec<Pattern>),
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct UnsafeBlock {
    pub block: Block,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub visibility: Option<String>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub body: Block,
}


#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub visibility: Option<String>,
    pub name: String,
    pub params: Option<Vec<Param>>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_annotation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub block: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct InternalExprStmt {
    pub expr: Box<Expr>,
}

#[derive(Debug, Clone)]
pub enum MutKeyword {
    Mut,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Number(i64),
    String(String),
    Ident(String),
    Binary {
        left: Box<Expr>,
        op: String,
        right: Box<Expr>,
    },
    Assignment(Box<Expr>, String, Box<Expr>),
}