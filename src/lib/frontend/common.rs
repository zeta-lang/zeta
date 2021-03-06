//! Contains structures and methods common to multiple elements of the frontend

use std::fmt;


/// A location in a source string
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Loc {
	pub line: u32,
	pub column: u32,
	pub index: usize
}

impl fmt::Debug for Loc {
	fn fmt (&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "@({}:{} / {})", self.line + 1, self.column + 1, self.index)
	}
}

impl fmt::Display for Loc {
	fn fmt (&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}:{}", self.line + 1, self.column + 1)
	}
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
	AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, PowAssign,
	Cast,
	Arrow,
	Path,
	Comma, Dot, Semi, Colon,
	LParen, RParen,
	LBrace, RBrace,
	LBracket, RBracket,
}



/// All keywords in a valid grammar
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Keyword {
	Global, Export, Import,
	Type, Local,
	Map, Record,
	Function, Fn,
	Loop, If, Else,
	Return, Break, Continue,
}


/// All constants in a valid grammar
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constant {
	Nil,
	Nan,
	Inf,
}