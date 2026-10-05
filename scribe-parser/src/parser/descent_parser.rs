use ir::ast::{AttrArg, Attribute, AttributedStmt, Stmt, Visibility};
use ir::diagnostics_context::{DiagnosticWarning, ParserDiagnosticsContext};
use ir::errors::error::{DiagnosticError, ParseErrorKind};
use ir::span::SourceSpan;
use std::sync::Arc;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

use crate::tokenizer::lexer::Lexer;
use ir::tokens::{Cursor, TokenKind, Tokens};

pub struct DescentParser<'a, 'bump>
where
    'bump: 'a,
{
    pub(crate) cursor: Cursor<'a, 'bump>,
    pub(crate) bump: &'bump GrowableBump<'bump>,
    pub(crate) string_pool: Arc<StringPool>,
    pub(crate) diag: ParserDiagnosticsContext<'a, 'bump>,
    pub(crate) pending_attrs: &'bump [Attribute<'a, 'bump>],
    pub(crate) pending_close_angle: u8, // To prevent mismatches when closing generics between `>`, `>>`, `>>>`, etc. So that they don't get confused with bitwise/comparison operators or fail entirely.
}

impl<'a, 'bump> DescentParser<'a, 'bump>
where
    'bump: 'a,
{
    pub fn parse(
        string_pool: Arc<StringPool>,
        bump: &'bump GrowableBump<'bump>,
        tokens: &'bump Tokens<'a, 'bump>,
    ) -> Result<Vec<Stmt<'a, 'bump>>, DiagnosticError<'a>> {
        let mut parser = DescentParser {
            cursor: Cursor::new(tokens, 0),
            string_pool,
            bump,
            diag: ParserDiagnosticsContext::new(true),
            pending_close_angle: 0,
            pending_attrs: &[],
        };

        parser.parse_toplevel()
    }

    fn parse_toplevel(&mut self) -> Result<Vec<Stmt<'a, 'bump>>, DiagnosticError<'a>> {
        let mut stmts = Vec::new();

        while self.cursor.peek() != TokenKind::EOF {
            match self.parse_stmt(Visibility::Public) {
                Ok(s) => stmts.push(s),
                Err(e) => {
                    self.diag.record(e);
                    let stop = self.diag.synchronize(&mut self.cursor);
                    if stop == TokenKind::EOF {
                        break;
                    }
                }
            }
        }

        Ok(stmts)
    }

    pub(crate) fn parse_stmt(
        &mut self,
        visibility: Visibility,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let kind = self.cursor.peek();
        match kind {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Import => self.parse_import(),
            TokenKind::Package => self.parse_package(),
            TokenKind::Const => self.parse_const_stmt(),
            TokenKind::Match => self.parse_match_stmt(),
            TokenKind::Defer => self.parse_defer_stmt(),
            TokenKind::Static => {
                self.cursor.advance(); // consume 'static'
                self.parse_static_let_stmt()
            }
            TokenKind::Func => self.parse_function_with_visibility(visibility),
            TokenKind::If => self.parse_if_stmt(),
            TokenKind::While => self.parse_while_stmt(),
            TokenKind::For => self.parse_for_stmt(),
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::Break => self.parse_break_stmt(),
            TokenKind::Continue => self.parse_continue_stmt(),

            TokenKind::Hashtag => {
                let attrs = self.parse_attributes()?;
                // FuncDecl takes pending_attrs itself; everything else gets wrapped.
                self.pending_attrs = attrs;
                let inner = self.parse_stmt(visibility)?;
                if self.pending_attrs.is_empty() {
                    Ok(inner) // consumed by a FuncDecl
                } else {
                    let attrs = std::mem::take(&mut self.pending_attrs);
                    Ok(Stmt::Attributed(
                        self.bump
                            .alloc_value_immutable(AttributedStmt { attrs, inner }),
                    ))
                }
            }

            TokenKind::Public => {
                self.cursor.advance();
                self.parse_stmt(Visibility::Public)
            }

            TokenKind::Private => {
                self.cursor.advance();
                self.parse_stmt(Visibility::Private)
            }

            TokenKind::Module => {
                self.cursor.advance();
                // `module Name { ... }` is a module *declaration*.
                // `module fn foo()` / `module struct Bar {}` is a visibility modifier.
                if self.cursor.peek() == TokenKind::Ident
                    && self.cursor.peek_n(1) == TokenKind::LBrace
                {
                    self.parse_module_decl(visibility)
                } else {
                    self.parse_stmt(Visibility::Module)
                }
            }

            TokenKind::Internal => {
                self.cursor.advance();
                self.parse_stmt(Visibility::Internal)
            }

            TokenKind::Unsafe => {
                // `unsafe { ... }` is a statement-level unsafe block.
                // `unsafe func` / `unsafe extern` are function modifiers.
                if self.cursor.peek_n(1) == TokenKind::LBrace {
                    self.parse_unsafe_block_stmt()
                } else if self.cursor.peek_n(1) == TokenKind::Impl {
                    self.parse_impl_decl(visibility)
                } else {
                    self.parse_function_with_visibility(visibility)
                }
            }

            TokenKind::LBrace => {
                let block = self.parse_block()?;
                Ok(Stmt::Block(self.bump.alloc_value_immutable(block)))
            }

            TokenKind::Inline | TokenKind::Noinline | TokenKind::Extern => {
                self.parse_function_with_visibility(visibility)
            }
            TokenKind::Struct => self.parse_struct_decl(visibility),
            TokenKind::Enum => self.parse_enum_decl(visibility),
            TokenKind::Impl => self.parse_impl_decl(visibility),
            TokenKind::Interface => self.parse_interface_decl(visibility, false),
            TokenKind::Sealed => {
                // sealed interface Name permits X, Y, Z { ... }
                self.cursor.advance();
                self.parse_interface_decl(visibility, true)
            }

            TokenKind::Ident => {
                // Check for shorthand let (ident := expr)
                if self.cursor.peek_n(1) == TokenKind::ColonAssign {
                    self.parse_shorthand_let_stmt()
                } else {
                    self.parse_expr_stmt()
                }
            }

            TokenKind::Type => self.parse_type_alias_decl(visibility),

            TokenKind::Mut => {
                // Check for shorthand let (ident := expr)
                if self.cursor.peek_n(1) == TokenKind::Ident
                    && self.cursor.peek_n(2) == TokenKind::ColonAssign
                {
                    self.parse_shorthand_let_stmt()
                } else {
                    self.parse_expr_stmt()
                }
            }

            _ => self.parse_expr_stmt(),
        }
    }

    fn at(&self, k: TokenKind) -> bool {
        self.cursor.peek() == k
    }

    fn eat(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.cursor.advance();
            true
        } else {
            false
        }
    }

    fn expect_tok(&mut self, k: TokenKind, what: &str) -> Result<(), DiagnosticError<'a>> {
        if self.eat(k) {
            Ok(())
        } else {
            let err = DiagnosticError {
                kind: ParseErrorKind::UnexpectedToken {
                    expected: k,
                    found: self.cursor.peek(),
                },
                span: self.cur_span(),
                context: Vec::new(),
                notes: vec![what.to_string()],
            };
            self.diag.record(err.clone());
            Err(err)
        }
    }

    fn cur_span(&self) -> SourceSpan<'a> {
        self.cursor.peek_token().span
    }

    /// Zero or more `#[a, b(1, "x"), c = ident]` groups. Returns an empty
    /// slice if there is no `#`, so it is safe to call before any item.
    pub(crate) fn parse_attributes(
        &mut self,
    ) -> Result<&'bump [Attribute<'a, 'bump>], DiagnosticError<'a>> {
        let mut out = Vec::new();
        while self.at(TokenKind::Hashtag) {
            self.cursor.advance();
            self.expect_tok(TokenKind::LBracket, "`[` after `#`")?;
            loop {
                let span = self.cur_span();
                let name = self.cursor.expect_ident()?.0;
                let args: &'bump [AttrArg<'bump>] = if self.eat(TokenKind::LParen) {
                    let a = self.parse_attr_args()?;
                    self.expect_tok(TokenKind::RParen, "`)` to close attribute arguments")?;
                    a
                } else {
                    &[]
                };
                out.push(Attribute { name, args, span });
                if !self.eat(TokenKind::Comma) {
                    break;
                }
                if self.at(TokenKind::RBracket) {
                    break;
                } // trailing comma
            }
            self.expect_tok(TokenKind::RBracket, "`]` to close attribute")?;
        }
        Ok(self.bump.alloc_slice(&out))
    }

    /// `arg (',' arg)* [',']`, stops before `)`.
    pub(crate) fn parse_attr_args(
        &mut self,
    ) -> Result<&'bump [AttrArg<'bump>], DiagnosticError<'a>> {
        let mut out = Vec::new();
        while !self.at(TokenKind::RParen) {
            out.push(self.parse_attr_arg()?);
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(self.bump.alloc_slice(&out))
    }

    /// ident | "str" | number | true | false | key = arg | name(args)
    fn parse_attr_arg(&mut self) -> Result<AttrArg<'bump>, DiagnosticError<'a>> {
        match self.cursor.peek() {
            TokenKind::String => Ok(AttrArg::Str(self.cursor.expect_string()?.0)),
            TokenKind::Number => {
                let tok = self.cursor.peek_token();
                let text = tok.text.unwrap_or_default();
                let value = text.as_str().parse::<i64>().unwrap_or(0);
                Ok(AttrArg::Number(value))
            }
            TokenKind::Ident => {
                let name = self.cursor.expect_ident()?.0;
                if self.eat(TokenKind::Assign) {
                    let value = self.parse_attr_arg()?;
                    Ok(AttrArg::KeyValue {
                        key: name,
                        value: self.bump.alloc_value_immutable(value),
                    })
                } else if self.eat(TokenKind::LParen) {
                    let args = self.parse_attr_args()?;
                    self.expect_tok(TokenKind::RParen, "`)`")?;
                    Ok(AttrArg::Call { name, args })
                } else if name == "true" {
                    Ok(AttrArg::Bool(true))
                } else if name == "false" {
                    Ok(AttrArg::Bool(false))
                } else {
                    Ok(AttrArg::Ident(name))
                }
            }

            _ => {
                let tok = self.cursor.peek_token();
                let text = tok.text.unwrap_or_default();
                let err = DiagnosticError {
                    kind: ParseErrorKind::UnexpectedTokenOneOf {
                        expected: vec![TokenKind::Ident, TokenKind::Number, TokenKind::String],
                        found: self.cursor.peek(),
                    },
                    span: self.cur_span(),
                    context: Vec::new(),
                    notes: vec![format!("expected an attribute argument, found `{text}`")],
                };
                self.diag.record(err.clone());
                Err(err)
            }
        }
    }
}

pub fn token_to_visibility(token_kind: TokenKind) -> Visibility {
    match token_kind {
        TokenKind::Private => Visibility::Private,
        TokenKind::Module => Visibility::Module,
        TokenKind::Package => Visibility::Internal,
        _ => Visibility::Public,
    }
}

#[derive(Debug, Clone)]
pub struct ParseResult<'a, 'bump> {
    pub statements: Vec<Stmt<'a, 'bump>>,
    pub diagnostics: ParserDiagnostics<'a>,
}

pub fn parse_program<'a, 'bump>(
    src: &str,
    file_name: &'bump str,
    context: Arc<StringPool>,
    bump: &'bump GrowableBump<'bump>,
) -> ParseResult<'a, 'bump> {
    let lexer: Lexer = Lexer::new(context.clone());
    let tokens: Tokens<'a, 'bump> = lexer.tokenize(src, file_name, bump);

    let tokenized_source = bump.alloc_value(tokens);

    let mut parser = DescentParser {
        cursor: Cursor::new(tokenized_source, 0),
        string_pool: context,
        bump: bump,
        diag: ParserDiagnosticsContext::new(false),
        pending_close_angle: 0,
        pending_attrs: &[],
    };

    let stmts: Vec<Stmt<'_, '_>> = match parser.parse_toplevel() {
        Ok(s) => s,
        Err(e) => {
            parser.diag.record(e);
            Vec::new()
        }
    };

    let (errors, warnings) = parser.diag.into_diagnostics();

    ParseResult {
        statements: stmts,
        diagnostics: ParserDiagnostics { errors, warnings },
    }
}

/// Collects parser diagnostics (errors, warnings, notes) instead of panicking
#[derive(Debug, Clone)]
pub struct ParserDiagnostics<'a> {
    pub errors: Vec<DiagnosticError<'a>>,
    pub warnings: Vec<DiagnosticWarning<'a>>,
}

impl<'a> ParserDiagnostics<'a> {
    pub fn new() -> Self {
        ParserDiagnostics {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: DiagnosticError<'a>) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: DiagnosticWarning<'a>) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}
