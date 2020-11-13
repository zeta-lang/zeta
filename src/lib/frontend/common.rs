//! Contains structures and methods common to multiple elements of the frontend

/// A location in a source string
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Loc {
  pub line: u32,
  pub column: u32,
  pub index: usize
}

/// All operators in a valid grammar
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operator {
  Add, Sub, Mul, Div, Rem, Pow,

  Concat,

  LShift, RShift,

  Eq, Ne, Lt, Gt, Le, Ge,

  LNot, BNot,
  
  LAnd, LOr,
  BAnd, BOr, BXOr,


  Assign,
  Comma, Dot, Semi, Colon,
  LParen, RParen,
  LBrace, RBrace,
  LBracket, RBracket,
}