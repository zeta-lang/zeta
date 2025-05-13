
#[derive(Debug)]
pub enum Stmt {
    Import(ImportStmt),
    Let(LetStmt),
    Return(ReturnStmt),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Match(MatchStmt),
    UnsafeBlock(UnsafeBlock),
    FuncDecl(FuncDecl),
    ClassDecl(ClassDecl),
    ExprStmt(ExprStmt),
}

#[derive(Debug)]
pub struct ImportStmt {
    pub path: String,  // For simplicity, storing the full path
}

#[derive(Debug)]
pub struct LetStmt {
    pub mutability: Option<MutKeyword>,
    pub ident: String,
    pub type_annotation: Option<String>,  // For simplicity
    pub value: Box<Expr>,
}

#[derive(Debug)]
pub struct ReturnStmt {
    pub value: Option<Box<Expr>>,
}

#[derive(Debug)]
pub struct IfStmt {
    pub condition: Box<Expr>,
    pub if_block: Block,
    pub else_if: Option<Box<IfStmt>>,
    pub else_block: Option<Block>,
}

#[derive(Debug)]
pub struct WhileStmt {
    pub condition: Box<Expr>,
    pub block: Block,
}

#[derive(Debug)]
pub struct ForStmt {
    pub let_stmt: LetStmt,
    pub condition: Box<Expr>,
    pub increment: Box<Expr>,
    pub block: Block,
}

#[derive(Debug)]
pub struct MatchStmt {
    pub expr: Expr,
    pub arms: Vec<MatchArm>,
}

#[derive(Debug)]
pub struct MatchArm {
    pub(crate) pattern: Pattern,
    pub(crate) block: Block,
}

#[derive(Debug)]
pub enum Pattern {
    Ident(String),
    Wildcard,
    Number(i32),
    String(String),
    Tuple(Vec<Pattern>),
}

#[derive(Debug)]
pub struct UnsafeBlock {
    block: Block,
}

#[derive(Debug)]
pub struct FuncDecl {
    name: String,
    params: Vec<Param>,
    return_type: Option<String>,
    body: Block,
}

#[derive(Debug)]
pub struct Param {
    name: String,
    type_annotation: Option<String>,
}

#[derive(Debug)]
pub struct ClassDecl {
    name: String,
    params: Option<Vec<Param>>,
    body: Block,
}

#[derive(Debug)]
pub struct ExprStmt {
    expr: Box<Expr>,
}

#[derive(Debug)]
pub struct Block {
    pub(crate) stmts: Vec<Stmt>,
}

#[derive(Debug)]
pub enum Expr {
    Number(i32),
    String(String),
    Ident(String),
    UnaryOp(UnaryOp, Box<Expr>),
    BinaryOp(BinaryOp, Box<Expr>, Box<Expr>),
    Assignment(Box<Expr>, AssignOp, Box<Expr>),
    FunctionCall(Box<Expr>, Vec<Expr>),
    Accessor(Box<Expr>, Vec<String>),  // Accessor path e.g., `a.b.c`
}

#[derive(Debug)]
pub enum UnaryOp {
    Not,
    Deref,
    AddressOf,
    Neg,
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
}

#[derive(Debug)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModuloAssign,
}

#[derive(Debug)]
pub enum MutKeyword {
    Mut,
}
