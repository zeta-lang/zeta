use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Integer,
    Float,
    Long,
    Double,
    String,
    Char,
    Boolean,
    Void,
    Byte,
    Short,

    UnsignedByte,
    UnsignedShort,

    Plus,
    Minus,
    Mul,
    Div,

    LParen,
    RParen,
    CurlyLParen,
    CurlyRParen,
    Identifier,
    Equal,
    Comma,
    Println,
    Colon,
    LAngleParen,
    RAngleParen,
    Exclamation,
    DoubleQuote,
    SingleQuote,
    Dot,
    End,
    Arrow,
    LSquareParen,
    RSquareParen,
    And,
    Ampersand,
    Or,

    Class,
    Interface,
    Enum,
    Data,
    Unsafe,
    Return,
    Let,
    Mut,
    If,
    Else,
    While,
    For,
    In,
    Break,
    Continue,
    True,
    False,
    Private,
    Public,
    Protected,
    UnsignedInteger,
    UnsignedDouble,
    UnsignedFloat,
    UnsignedLong,
    This,
    Super,
    Import
}

impl Display for TokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: Option<String>,
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}


impl Token {
    pub fn new(token_type: TokenType, value: Option<String>) -> Self {
        Token { token_type, value }
    }

    pub fn print_type(&self) {
        println!("{:?}", self.token_type);
    }
}