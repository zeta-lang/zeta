use crate::parser::descent_parser::{BinderCtx, DescentParser};
use ir::ast::{
    EnsuresClause, ExternModifier, FuncDecl, FuncModifiers, FuncSafety, InlineModifier,
    ReturnBinder, Stmt, Type, Visibility,
};
use ir::errors::error::{DiagnosticError, ParseErrorKind};
use ir::tokens::{Cursor, TokenKind};

impl<'a, 'bump> DescentParser<'a, 'bump>
where
    'bump: 'a,
{
    pub fn parse_function_signature(
        &mut self,
        visibility: Visibility,
    ) -> Result<FuncDecl<'a, 'bump>, DiagnosticError<'a>> {
        let attrs = std::mem::take(&mut self.pending_attrs);
        let function_metadata = Self::get_func_metadata(&mut self.cursor, visibility)?;
        let fn_token = self.cursor.expect(TokenKind::Func)?;

        let name_token = self.cursor.bump();
        let name_span = name_token.span;
        let name = match name_token.kind {
            TokenKind::Ident => match name_token.text {
                Some(text) if text.is_empty() => {
                    return Err(DiagnosticError::new(
                        ParseErrorKind::EmptyIdent,
                        name_token.span,
                    ));
                }
                Some(text) => text,
                None => {
                    return Err(DiagnosticError::new(
                        ParseErrorKind::EmptyIdent,
                        name_token.span,
                    ));
                }
            },
            _ => {
                return Err(DiagnosticError::new(
                    ParseErrorKind::InvalidFunctionName {
                        found: name_token.kind,
                    },
                    name_token.span,
                ));
            }
        };

        let generics = self.parse_generics()?;

        let params = self.parse_params()?;

        let (return_type, return_binders) = if self.cursor.consume(TokenKind::Colon) {
            self.parse_return_type()?
        } else {
            (None, &[][..])
        };

        let ensures = self.parse_ensures_clauses()?;

        let body = if self.cursor.consume(TokenKind::Semicolon) {
            None
        } else {
            let block = self.parse_block()?;
            Some(self.bump.alloc_value_immutable(block))
        };

        let func_decl = FuncDecl {
            function_metadata,
            name,
            generics,
            params,
            return_type,
            return_binders,
            ensures,
            body,
            name_span,
            span: fn_token.span,
            attrs,
        };

        Ok(func_decl)
    }

    pub fn parse_ensures_clauses(
        &mut self,
    ) -> Result<&'bump [EnsuresClause<'a, 'bump>], DiagnosticError<'a>> {
        let mut clauses = Vec::new();
        while self.cursor.peek() == TokenKind::Ensures {
            let kw = self.cursor.bump();
            let pattern = if self.cursor.consume(TokenKind::Case) {
                let p = self.parse_pattern()?;
                self.cursor.expect(TokenKind::Arrow)?;
                Some(self.bump.alloc_value_immutable(p))
            } else {
                None
            };
            let cond = self.parse_expr(0)?;
            clauses.push(EnsuresClause {
                pattern,
                condition: self.bump.alloc_value_immutable(cond),
                span: kw.span,
            });
        }
        Ok(self.bump.alloc_slice(&clauses))
    }

    /// Consumes `name <-` if present and records it. Does nothing outside a return type,
    /// so speculative generic parsing in expressions (`a < b<-1`) is unaffected.
    fn try_parse_binder(&mut self) -> Result<(), DiagnosticError<'a>> {
        Self::try_binder_in(self.bump, &mut self.cursor, &mut self.binder_ctx)
    }

    fn parse_return_type(
        &mut self,
    ) -> Result<(Option<Type<'a, 'bump>>, &'bump [ReturnBinder<'a, 'bump>]), DiagnosticError<'a>>
    {
        match self.cursor.peek() {
            TokenKind::Semicolon | TokenKind::LBrace | TokenKind::Ensures | TokenKind::EOF => {
                Ok((None, &[]))
            }
            _ => {
                let prev = self.binder_ctx.replace(BinderCtx {
                    path: Vec::new(),
                    found: Vec::new(),
                });
                let parsed = self.try_parse_binder().and_then(|_| self.parse_type()); // root binder
                let ctx = std::mem::replace(&mut self.binder_ctx, prev);
                let ty = parsed?;
                let found = ctx.map(|c| c.found).unwrap_or_default();
                Ok((Some(ty), self.bump.alloc_slice(&found)))
            }
        }
    }

    pub fn parse_function_with_visibility(
        &mut self,
        visibility: Visibility,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let func = self.parse_function_signature(visibility)?;

        let is_extern = !matches!(func.function_metadata.extern_modifier, ExternModifier::None);

        if func.body.is_none() && !is_extern {
            return Err(DiagnosticError::new(
                ParseErrorKind::ExpectedBlock,
                self.cursor.peek_token().span,
            ));
        }

        Ok(Stmt::FuncDecl(self.bump.alloc_value_immutable(func)))
    }

    pub fn get_func_metadata(
        cursor: &mut Cursor<'a, 'bump>,
        visibility: Visibility,
    ) -> Result<FuncModifiers, DiagnosticError<'a>> {
        let mut inline_modifier = InlineModifier::None;
        let mut extern_modifier = ExternModifier::None;
        let mut func_safety = FuncSafety::Safe;

        loop {
            match cursor.peek() {
                TokenKind::Inline => {
                    if let InlineModifier::Noinline = inline_modifier {
                        return Err(DiagnosticError::new(
                            ParseErrorKind::UnexpectedToken {
                                expected: TokenKind::Func,
                                found: cursor.peek(),
                            },
                            cursor.peek_token().span,
                        ));
                    }
                    if let InlineModifier::Inline = inline_modifier {
                        return Err(DiagnosticError::new(
                            ParseErrorKind::UnexpectedToken {
                                expected: TokenKind::Func,
                                found: cursor.peek(),
                            },
                            cursor.peek_token().span,
                        ));
                    }
                    inline_modifier = InlineModifier::Inline;
                    cursor.advance();
                }
                TokenKind::Noinline => {
                    if let InlineModifier::Inline = inline_modifier {
                        return Err(DiagnosticError::new(
                            ParseErrorKind::UnexpectedToken {
                                expected: TokenKind::Func,
                                found: cursor.peek(),
                            },
                            cursor.peek_token().span,
                        ));
                    }
                    if let InlineModifier::Noinline = inline_modifier {
                        return Err(DiagnosticError::new(
                            ParseErrorKind::UnexpectedToken {
                                expected: TokenKind::Func,
                                found: cursor.peek(),
                            },
                            cursor.peek_token().span,
                        ));
                    }
                    inline_modifier = InlineModifier::Noinline;
                    cursor.advance();
                }
                TokenKind::Extern => {
                    cursor.advance();

                    let (abi, _) = cursor.expect_string()?;
                    extern_modifier = ExternModifier::Abi(abi);
                }
                TokenKind::Unsafe => {
                    if func_safety == FuncSafety::Unsafe {
                        return Err(DiagnosticError::new(
                            ParseErrorKind::UnexpectedToken {
                                expected: TokenKind::Func,
                                found: cursor.peek(),
                            },
                            cursor.peek_token().span,
                        ));
                    }
                    func_safety = FuncSafety::Unsafe;
                    cursor.advance();
                }
                _ => break,
            }
        }

        Ok(FuncModifiers {
            visibility,
            extern_modifier,
            inline_modifier,
            func_safety,
        })
    }
}
