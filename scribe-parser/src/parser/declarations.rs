use crate::parser::descent_parser::{DescentParser, set_decl_sep, token_to_visibility};
use ir::ast::{
    ConstStmt, EnumDecl, EnumVariant, Expr, Field, Generic, ImplDecl, InterfaceDecl, Param,
    PermitsExpr, Stmt, StructDecl, Type, TypeAliasDecl, TypeKind, Visibility,
};
use ir::errors::error::{DiagnosticError, ParseErrorKind};
use ir::hir::StrId;
use ir::span::SourceSpan;
use ir::tokens::{Cursor, TokenKind};

#[derive(Clone, Copy, PartialEq, Eq)]
enum FuncScan {
    /// `func(..) { ... }`
    Body,
    /// `func(..): T;`
    Semicolon,
    /// ran into `}` / EOF first
    Unterminated,
}

/// Everything to the left of the declaration `::` in
/// `NAME<T>: Annotation :: rhs`.
pub(crate) struct DeclHeader<'a, 'bump>
where
    'bump: 'a,
{
    pub name: StrId,
    pub name_span: SourceSpan<'a>,
    pub generics: Option<&'bump [Generic<'a, 'bump>]>,
    pub annotation: Option<Type<'a, 'bump>>,
}

impl<'a, 'bump> DescentParser<'a, 'bump>
where
    'bump: 'a,
{
    /// Does the statement starting at the current `Ident` begin a `::` declaration?
    ///
    /// ```text
    /// NAME :: rhs;
    /// NAME: Type :: rhs;
    /// NAME<T> :: rhs;
    /// ```
    ///
    /// The one exception is a path call, `NAME :: x()` / `NAME::x<T>()`: calls are not allowed
    /// as the value of a `::` declaration (that's what `NAME :: comptime { ... }` is for), so
    /// `Ident :: Ident ... (` is always the expression statement.
    pub(crate) fn at_decl(&mut self) -> bool {
        match self.cursor.peek_n_kind(1) {
            TokenKind::Colon => true,
            TokenKind::ColonColon => {
                let mut c = self.cursor.clone();
                c.advance(); // NAME
                c.advance(); // ::
                !Self::path_call_follows(self.bump, &mut c)
            }
            TokenKind::Lt => {
                // `NAME<T, U: Display> ::`, parsed speculatively so `a < b;` is untouched.
                let checkpoint = self.cursor.pos();
                let saved_pending = self.pending_close_angle;
                self.cursor.advance(); // ident
                let is_decl = matches!(self.parse_generics(), Ok(Some(_)))
                    && self.cursor.peek() == TokenKind::ColonColon
                    && {
                        let mut c = self.cursor.clone();
                        c.advance(); // ::
                        !Self::path_call_follows(self.bump, &mut c)
                    };
                self.cursor.reset(checkpoint);
                self.pending_close_angle = saved_pending;
                is_decl
            }
            _ => false,
        }
    }

    /// With `c` just after a `::`: is this `a::b.c<T>::d(`, i.e. a path followed by a call?
    fn path_call_follows(bump: &'bump zetaruntime::bump::GrowableBump, c: &mut Cursor<'a>) -> bool {
        if c.peek() != TokenKind::Ident {
            return false;
        }
        c.advance();
        loop {
            match c.peek() {
                TokenKind::ColonColon | TokenKind::Dot => {
                    c.advance();
                    if c.peek() != TokenKind::Ident {
                        return false;
                    }
                    c.advance();
                }
                TokenKind::Lt => {
                    c.advance();
                    let mut pending = 0u8;
                    loop {
                        if Self::parse_type_impl(bump, c, &mut pending).is_err() {
                            return false;
                        }
                        if !c.consume(TokenKind::Comma) {
                            break;
                        }
                    }
                    if !Self::try_consume_close_angle(c, &mut pending) {
                        return false;
                    }
                }
                TokenKind::LParen => return true,
                _ => return false,
            }
        }
    }

    /// With the cursor at the start of an annotation (just after `NAME:`): the token index of
    /// the declaration `::`, which is the first `::` outside any `()`, `[]`, `{}` or `<>`.
    /// Qualified annotation types therefore need parentheses: `NAME: (a::B) :: x`.
    fn find_decl_sep(&self) -> Option<usize> {
        let mut c = self.cursor.clone();
        let mut depth = 0i32;
        loop {
            match c.peek() {
                TokenKind::EOF | TokenKind::Semicolon => return None,
                TokenKind::ColonColon if depth <= 0 => return Some(c.pos() as usize),
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace | TokenKind::Lt => {
                    depth += 1
                }
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace | TokenKind::Gt => {
                    depth -= 1
                }
                TokenKind::Shr => depth -= 2,
                TokenKind::UnsignedShr => depth -= 3,
                _ => {}
            }
            c.advance();
        }
    }

    /// `NAME<T>: Annotation ::`, with generics and the annotation both optional.
    pub(crate) fn parse_decl_header(
        &mut self,
    ) -> Result<DeclHeader<'a, 'bump>, DiagnosticError<'a>> {
        let (name, name_span) = self.cursor.expect_ident()?;
        let generics = self.parse_generics()?; // None when there is no `<`
        let annotation = if self.cursor.consume(TokenKind::Colon) {
            // Stop the type parser's `a::B` path loop at the declaration `::`.
            set_decl_sep(self.find_decl_sep());
            let ty = self.parse_type();
            set_decl_sep(None);
            Some(ty?)
        } else {
            None
        };
        self.cursor.expect(TokenKind::ColonColon)?;
        Ok(DeclHeader {
            name,
            name_span,
            generics,
            annotation,
        })
    }

    /// Parses any `[visibility] NAME ... :: rhs` statement. Visibility has already been
    /// consumed by `parse_stmt`, it belongs to the declaration itself.
    ///
    /// ```text
    /// NAME :: 5;                              // inferred const
    /// NAME: i64 :: 5;                         // annotated const
    /// Pair<T> :: Tuple<T, T>;                 // generic const (usually a type alias)
    /// Point :: struct { x: i32, y: i32 };
    /// Shape :: sealed interface permits A, B { area :: func(&this): f64; };
    /// Color :: enum { Red, Green };
    /// Bytes :: []u8;                          // type alias
    /// add :: func(a: i64, b: i64): i64 { return a + b; }
    /// ```
    pub(crate) fn parse_decl_stmt(
        &mut self,
        visibility: Visibility,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let header = self.parse_decl_header()?;
        self.parse_decl_rhs(visibility, header)
    }

    fn parse_decl_rhs(
        &mut self,
        visibility: Visibility,
        h: DeclHeader<'a, 'bump>,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let stmt = match self.cursor.peek() {
            TokenKind::Struct => {
                self.reject_annotation(&h, true);
                self.parse_struct_decl(visibility, h.name, h.name_span, h.generics)?
            }
            TokenKind::Enum => {
                self.reject_annotation(&h, true);
                self.parse_enum_decl(visibility, h.name, h.name_span, h.generics)?
            }
            TokenKind::Interface | TokenKind::Sealed => {
                self.reject_annotation(&h, true);
                let is_sealed = self.cursor.consume(TokenKind::Sealed);
                self.parse_interface_decl(visibility, is_sealed, h.name, h.name_span, h.generics)?
            }
            _ if self.at_func_decl() => {
                self.reject_annotation(&h, false);
                self.parse_function_decl(visibility, h.name, h.name_span, h.generics)?
            }
            _ if self.rhs_is_type_alias() => {
                self.reject_annotation(&h, true);
                let ty = self.parse_type()?;
                Stmt::TypeAliasDecl(self.bump.alloc_value_immutable(TypeAliasDecl {
                    visibility,
                    name: h.name,
                    generics: h.generics,
                    ty,
                    name_span: h.name_span,
                    span: h.name_span,
                }))
            }
            _ => Stmt::Const(self.parse_const_value(visibility, h)?),
        };

        self.cursor.consume(TokenKind::Semicolon);
        Ok(stmt)
    }

    /// `NAME[: T] :: value`, everything up to (not including) the trailing `;`.
    /// A missing annotation means the type is inferred (`type_annotation: None`).
    ///
    /// `value` is an expression, a `comptime { ... }` block, never a call and never a bare
    /// `{ ... }` block (`NAME :: x()` is reserved for the path call expression statement).
    fn parse_const_value(
        &mut self,
        visibility: Visibility,
        h: DeclHeader<'a, 'bump>,
    ) -> Result<&'bump ConstStmt<'a, 'bump>, DiagnosticError<'a>> {
        let value = if self.at_comptime_block() {
            self.cursor.advance(); // `comptime`
            let block = self.parse_block()?;
            Expr::ComptimeBlock(self.bump.alloc_value_immutable(block))
        } else if self.cursor.peek() == TokenKind::LBrace {
            return Err(DiagnosticError::new(
                ParseErrorKind::InvalidExpression {
                    found: TokenKind::LBrace,
                },
                self.cursor.peek_token().span,
            )
            .with_note(
                "a bare `{ ... }` isn't a valid const value; use `NAME :: comptime { ... };` to \
                 run code at compile time.",
            ));
        } else {
            let value = self.parse_expr(0)?;
            if let Some(span) = Self::first_call(&value) {
                self.diag.record(
                    DiagnosticError::new(
                        ParseErrorKind::InvalidExpression {
                            found: TokenKind::LParen,
                        },
                        span,
                    )
                    .with_note(
                        "functions can't be called directly in a `::` declaration; wrap it in \
                         `NAME :: comptime { ... };`.",
                    ),
                );
            }
            value
        };

        Ok(self.bump.alloc_value_immutable(ConstStmt {
            ident: h.name,
            generics: h.generics,
            type_annotation: h.annotation.unwrap_or(Type::infer(h.name_span)),
            value: self.bump.alloc_value_immutable(value),
            span: h.name_span,
            visibility,
        }))
    }

    /// `comptime { ... }`. `comptime` is matched as an identifier; if the lexer has a real
    /// keyword token for it, compare against that here instead.
    fn at_comptime_block(&self) -> bool {
        self.cursor.peek() == TokenKind::Ident
            && self
                .cursor
                .peek_token()
                .text
                .is_some_and(|t| t.as_str() == "comptime")
            && self.cursor.peek_n_kind(1) == TokenKind::LBrace
    }

    /// The span of the first call that would be evaluated by the value expression itself
    /// (calls inside lambda bodies run later, so they don't count).
    fn first_call(e: &Expr<'a, 'bump>) -> Option<SourceSpan<'a>> {
        match e {
            Expr::Call { span, .. } => Some(*span),
            Expr::Binary { left, right, .. } => {
                Self::first_call(left).or_else(|| Self::first_call(right))
            }
            Expr::Comparison { lhs, rhs, .. } => {
                Self::first_call(lhs).or_else(|| Self::first_call(rhs))
            }
            Expr::Unary { operand, .. } => Self::first_call(operand),
            Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::Deref { expr, .. }
            | Expr::ArrayIndex { expr, .. } => Self::first_call(expr),
            Expr::FieldAccess { object, .. } | Expr::Get { object, .. } => Self::first_call(object),
            Expr::Tuple { values, .. } => values.iter().find_map(Self::first_call),
            Expr::ArrayLiteral { elements, .. } => elements.iter().find_map(Self::first_call),
            _ => None,
        }
    }

    /// Struct/enum/interface/alias declarations are always of type `type`, functions have
    /// a type of their own, so `NAME: T :: struct {}` only makes sense with `T = type`.
    fn reject_annotation(&mut self, h: &DeclHeader<'a, 'bump>, allow_type: bool) {
        let Some(annotation) = h.annotation else {
            return;
        };
        if allow_type && matches!(annotation.kind, TypeKind::Type) {
            return;
        }
        let note = if allow_type {
            "this declaration is always of type `type`: write `Name :: ...` or `Name: type :: ...`."
        } else {
            "functions don't take a type annotation: write `name :: func(...) { ... }`."
        };
        self.diag.record(
            DiagnosticError::new(
                ParseErrorKind::UnexpectedToken {
                    expected: TokenKind::ColonColon,
                    found: TokenKind::Colon,
                },
                annotation.span,
            )
            .with_note(note),
        );
    }

    /// `func ... { body }`, or any of the function modifiers (`inline`, `noinline`,
    /// `extern "C"`, `unsafe func`). A plain `func(...)` that ends in `;` is a function *type*
    /// (`Callback :: func(i64): i64;`) and goes to the alias path. A `func(...)` that ends in
    /// neither is treated as a function so the missing body is reported.
    fn at_func_decl(&mut self) -> bool {
        match self.cursor.peek() {
            TokenKind::Func => self.scan_func() != FuncScan::Semicolon,
            _ => self.at_member_fn(),
        }
    }

    /// Methods (impl/interface members) may omit the body, so `func` always counts there.
    fn at_member_fn(&mut self) -> bool {
        match self.cursor.peek() {
            TokenKind::Func | TokenKind::Inline | TokenKind::Noinline | TokenKind::Extern => true,
            // `unsafe { ... }` is an expression, `unsafe func` is a function.
            TokenKind::Unsafe => matches!(
                self.cursor.peek_n_kind(1),
                TokenKind::Func | TokenKind::Extern | TokenKind::Inline | TokenKind::Noinline
            ),
            _ => false,
        }
    }

    /// Scans `func (...) [: ret] [ensures ...]` and reports what ends it.
    /// Anonymous `struct { ... }` / `enum { ... }` return types are skipped over.
    fn scan_func(&mut self) -> FuncScan {
        let mut c = self.cursor.clone();
        c.advance(); // `func`
        if c.peek() != TokenKind::LParen {
            return FuncScan::Unterminated;
        }
        let mut depth = 0i32;
        loop {
            match c.peek() {
                TokenKind::LParen | TokenKind::LBracket => {
                    depth += 1;
                    c.advance();
                }
                TokenKind::RParen | TokenKind::RBracket => {
                    depth -= 1;
                    c.advance();
                }
                TokenKind::LBrace if depth == 0 => return FuncScan::Body,
                TokenKind::Struct | TokenKind::Enum => {
                    c.advance();
                    Self::skip_braces(&mut c);
                }
                TokenKind::Semicolon if depth == 0 => return FuncScan::Semicolon,
                TokenKind::RBrace if depth == 0 => return FuncScan::Unterminated,
                TokenKind::EOF => return FuncScan::Unterminated,
                _ => {
                    c.advance();
                }
            }
        }
    }

    fn skip_braces(c: &mut Cursor<'a>) {
        if c.peek() != TokenKind::LBrace {
            return;
        }
        let mut depth = 0i32;
        loop {
            match c.peek() {
                TokenKind::EOF => return,
                TokenKind::LBrace => {
                    depth += 1;
                    c.advance();
                }
                TokenKind::RBrace => {
                    depth -= 1;
                    c.advance();
                    if depth == 0 {
                        return;
                    }
                }
                _ => {
                    c.advance();
                }
            }
        }
    }

    /// `X :: Y;` can't be told apart from a const holding the expression `Y` (the same goes
    /// for `X<Z> :: Y<Z>;`), so those stay `Const` and name resolution decides whether the
    /// value is a type. Right-hand sides that can *only* be types become a `TypeAliasDecl`.
    fn rhs_is_type_alias(&mut self) -> bool {
        let candidate = match self.cursor.peek() {
            TokenKind::Question | TokenKind::Dyn | TokenKind::BitXor | TokenKind::Func => true,
            TokenKind::Mul => matches!(
                self.cursor.peek_n_kind(1),
                TokenKind::Mut | TokenKind::Const
            ),
            TokenKind::LBracket => true,
            TokenKind::BitAnd => matches!(
                self.cursor.peek_n_kind(1),
                TokenKind::Mut
                    | TokenKind::Alias
                    | TokenKind::Static
                    | TokenKind::Dyn
                    | TokenKind::LBracket
            ),
            _ => false,
        };
        if !candidate {
            return false;
        }

        // `[1, 2]` and `&[1, 2]` are expressions, `[4]i32` and `&[]T` are types.
        let mut c = self.cursor.clone();
        let mut pending = 0u8;
        let is_func_type = self.cursor.peek() == TokenKind::Func;
        Self::parse_type_impl(self.bump, &mut c, &mut pending).is_ok()
            && match c.peek() {
                // A function type must be terminated by `;` (`at_func_decl` already routed
                // everything else to the function path).
                TokenKind::Semicolon => true,
                TokenKind::RBrace | TokenKind::EOF => !is_func_type,
                _ => false,
            }
    }

    /// Parse the right-hand side of `NAME :: struct ...`
    /// ```rs
    /// Point :: struct {
    ///     x: i32,
    ///     y: i32,
    /// };
    /// Pair :: struct(i32, i32);
    /// Marker :: struct;
    /// ```
    pub fn parse_struct_decl(
        &mut self,
        visibility: Visibility,
        name: StrId,
        name_span: SourceSpan<'a>,
        generics: Option<&'bump [Generic<'a, 'bump>]>,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let token = self.cursor.expect(TokenKind::Struct)?;

        let params = if self.cursor.peek() == TokenKind::Semicolon {
            // Unit struct: `Marker :: struct;`
            None
        } else if self.cursor.peek() == TokenKind::LParen {
            // Tuple struct: `Pair :: struct(i32, i32);`
            Some(self.parse_tuple_struct_fields()?)
        } else {
            self.cursor.expect(TokenKind::LBrace)?;

            let mut fields = Vec::new();

            while self.cursor.peek() != TokenKind::RBrace && self.cursor.peek() != TokenKind::EOF {
                let field_vis = self.eat_visibility(Visibility::Private);

                let manual = self.eat_manual_modifier();
                let (field_name, field_span) = self.cursor.expect_ident()?;
                self.cursor.expect(TokenKind::Colon)?;
                let field_type = self.parse_type()?;

                let unmerged_mp_span = self.cursor.peek_token().span;
                let multi_place = self.parse_multi_place();
                let multi_place_span =
                    SourceSpan::merge(self.cursor.peek_token().span, unmerged_mp_span);
                match multi_place {
                    Ok(None) => {}

                    Ok(Some(_)) => {
                        self.diag
                            .record(DiagnosticError::unsupported_disjoint_borrow(
                                multi_place_span,
                            ));
                        let stop = self.diag.synchronize(&mut self.cursor);
                        if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                            break;
                        }
                    }
                    Err(e) => {
                        self.diag
                            .record(DiagnosticError::unsupported_disjoint_borrow(
                                multi_place_span,
                            ));
                        self.diag.record(e);
                        let stop = self.diag.synchronize(&mut self.cursor);
                        if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                            break;
                        }
                    }
                }

                fields.push(Field {
                    name: field_name,
                    field_type,
                    visibility: field_vis,
                    span: field_span,
                    manual,
                });

                self.cursor.consume(TokenKind::Comma);

                if self.cursor.peek() == TokenKind::RBrace {
                    break;
                }
            }

            self.cursor.expect(TokenKind::RBrace)?;

            let fields_as_params: Vec<Param<'a, 'bump>> = fields
                .into_iter()
                .map(|f| {
                    Param::Normal(self.bump.alloc_value_immutable(ir::ast::NormalParam {
                        is_mut: false,
                        name: f.name,
                        type_annotation: f.field_type,
                        visibility: f.visibility,
                        default_value: None,
                        multi_place: None,
                        span: f.span,
                        manual: f.manual,
                    }))
                })
                .collect();

            if fields_as_params.is_empty() {
                None
            } else {
                Some(self.bump.alloc_slice(&fields_as_params))
            }
        };

        let struct_decl = StructDecl {
            visibility,
            name,
            generics,
            params,
            name_span,
            span: token.span,
        };

        Ok(Stmt::StructDecl(
            self.bump.alloc_value_immutable(struct_decl),
        ))
    }

    /// Parse an impl block
    /// ```rs
    /// impl Point {
    ///     new :: func(x: i32, y: i32): Point {
    ///         return Point { x, y };
    ///     }
    ///     ORIGIN :: Point { x: 0, y: 0 };
    /// }
    ///
    /// impl Point by Drawable {
    ///     draw :: func(&this) { ... }
    /// }
    /// ```
    pub fn parse_impl_decl(
        &mut self,
        _visibility: Visibility,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let is_unsafe = self.cursor.consume(TokenKind::Unsafe);
        let token = self.cursor.expect(TokenKind::Impl)?;

        // Check for generic impl: impl<T> ...
        let generics = if self.cursor.peek() == TokenKind::Lt {
            self.parse_generics()?
        } else {
            None
        };

        let target = self.parse_impl_target()?;

        let interface = if self.cursor.peek() == TokenKind::By {
            self.cursor.advance();
            let trait_name = self.parse_type()?;

            Some(trait_name)
        } else {
            None
        };

        if is_unsafe && interface.is_none() {
            self.diag.record(
                DiagnosticError::new(
                    ParseErrorKind::UnexpectedToken {
                        expected: TokenKind::By,
                        found: self.cursor.peek(),
                    },
                    token.span,
                )
                .with_note(
                    "`unsafe impl` must name an unsafe interface: `unsafe impl Type by Send {}`.",
                ),
            );
        }

        self.cursor.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        let mut constants = Vec::new();

        while self.cursor.peek() != TokenKind::RBrace && self.cursor.peek() != TokenKind::EOF {
            let attrs = self.parse_attributes()?; // empty slice if there's no `#`

            let inner_visibility = self.eat_visibility(Visibility::Public);

            let header = match self.parse_decl_header() {
                Ok(header) => header,
                Err(e) => {
                    self.diag.record(e);
                    let stop = self.diag.synchronize(&mut self.cursor);
                    if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                        break;
                    }
                    continue;
                }
            };

            if self.at_member_fn() {
                self.pending_attrs = attrs;
                match self.parse_function_decl(
                    inner_visibility,
                    header.name,
                    header.name_span,
                    header.generics,
                ) {
                    Ok(Stmt::FuncDecl(func)) => methods.push(*func),
                    Ok(_) => {}
                    Err(e) => {
                        self.diag.record(e);
                        let stop = self.diag.synchronize(&mut self.cursor);
                        if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                            break;
                        }
                    }
                }
            } else {
                match self.parse_const_value(inner_visibility, header) {
                    Ok(const_stmt) => {
                        constants.push(*const_stmt);
                        self.cursor.consume(TokenKind::Semicolon);
                    }
                    Err(e) => {
                        self.diag.record(e);
                        let stop = self.diag.synchronize(&mut self.cursor);
                        if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                            break;
                        }
                    }
                }
            }
        }

        self.cursor.expect(TokenKind::RBrace)?;

        let impl_decl = ImplDecl {
            generics,
            interface,
            target,
            methods: if methods.is_empty() {
                None
            } else {
                Some(self.bump.alloc_slice(&methods))
            },
            constants: if constants.is_empty() {
                None
            } else {
                Some(self.bump.alloc_slice(&constants))
            },
            span: token.span,
            is_unsafe,
        };

        Ok(Stmt::ImplDecl(self.bump.alloc_value_immutable(impl_decl)))
    }

    pub(crate) fn eat_visibility(&mut self, default: Visibility) -> Visibility {
        match token_to_visibility(self.cursor.peek()) {
            Some(vis) => {
                self.cursor.advance();
                vis
            }
            None => default,
        }
    }

    fn parse_tuple_struct_fields(
        &mut self,
    ) -> Result<&'bump [Param<'a, 'bump>], DiagnosticError<'a>> {
        self.cursor.expect(TokenKind::LParen)?;

        let mut fields = Vec::new();

        while self.cursor.peek() != TokenKind::RParen && self.cursor.peek() != TokenKind::EOF {
            let field_vis = self.eat_visibility(Visibility::Private);

            let peeked_token = self.cursor.peek_token();
            let manual = self.eat_manual_modifier();

            let field_type = self.parse_type()?;

            let unmerged_mp_span = self.cursor.peek_token().span;
            let multi_place = self.parse_multi_place();
            let multi_place_span =
                SourceSpan::merge(self.cursor.peek_token().span, unmerged_mp_span);
            match multi_place {
                Ok(None) => {}

                Ok(Some(_)) => {
                    self.diag
                        .record(DiagnosticError::unsupported_disjoint_borrow(
                            multi_place_span,
                        ));
                    let stop = self.diag.synchronize(&mut self.cursor);
                    if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                        break;
                    }
                }
                Err(e) => {
                    self.diag
                        .record(DiagnosticError::unsupported_disjoint_borrow(
                            multi_place_span,
                        ));
                    self.diag.record(e);
                    let stop = self.diag.synchronize(&mut self.cursor);
                    if stop == TokenKind::RBrace || stop == TokenKind::EOF {
                        break;
                    }
                }
            }

            fields.push(Param::Normal(
                self.bump.alloc_value_immutable(ir::ast::NormalParam {
                    is_mut: false,
                    name: StrId(
                        self.string_pool
                            .intern(format!("anonymous_tuple_field_{}", field_type).as_str()),
                    ),
                    type_annotation: field_type,
                    visibility: field_vis,
                    default_value: None,
                    multi_place: None,
                    span: peeked_token.span,
                    manual,
                }),
            ));

            if !self.cursor.consume(TokenKind::Comma) {
                break;
            }
        }

        self.cursor.expect(TokenKind::RParen)?;

        Ok(self.bump.alloc_slice(&fields))
    }

    /// Parse the right-hand side of `NAME :: [sealed] interface ...` (like a Rust trait).
    /// `sealed` has already been consumed when `is_sealed` is set.
    /// ```ignore
    /// Drawable :: interface {
    ///     draw :: func(&this);
    /// };
    ///
    /// SealedTrait :: sealed interface permits A, B, C {
    ///     method :: func(&this);
    /// };
    /// ```
    pub fn parse_interface_decl(
        &mut self,
        visibility: Visibility,
        is_sealed: bool,
        name: StrId,
        name_span: SourceSpan<'a>,
        generics: Option<&'bump [Generic<'a, 'bump>]>,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let token = self.cursor.expect(TokenKind::Interface)?;

        // Parse optional permits clause for sealed interfaces
        let permits = if self.cursor.peek() == TokenKind::Permits {
            Some(self.parse_permits_expr()?)
        } else {
            None
        };

        self.cursor.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();

        while self.cursor.peek() != TokenKind::RBrace && self.cursor.peek() != TokenKind::EOF {
            let method_vis = self.eat_visibility(Visibility::Private);

            if self.cursor.peek() == TokenKind::Ident
                && matches!(
                    self.cursor.peek_n_kind(1),
                    TokenKind::ColonColon | TokenKind::Lt | TokenKind::Colon
                )
            {
                let header = self.parse_decl_header()?;
                if self.at_member_fn() {
                    let func = self.parse_function_signature(
                        method_vis,
                        header.name,
                        header.name_span,
                        header.generics,
                    )?;
                    methods.push(func);
                } else {
                    // Associated consts, types, etc. aren't supported in interfaces yet:
                    // skip to the end of the member.
                    while !matches!(
                        self.cursor.peek(),
                        TokenKind::Semicolon | TokenKind::RBrace | TokenKind::EOF
                    ) {
                        self.cursor.advance();
                    }
                    self.cursor.consume(TokenKind::Semicolon);
                }
            } else {
                self.cursor.advance();
            }
        }

        self.cursor.expect(TokenKind::RBrace)?;

        let interface_decl = InterfaceDecl {
            name,
            visibility,
            sealed: is_sealed,
            permits,
            methods: if methods.is_empty() {
                None
            } else {
                Some(self.bump.alloc_slice(&methods))
            },
            generics,
            name_span,
            span: token.span,
        };

        Ok(Stmt::InterfaceDecl(
            self.bump.alloc_value_immutable(interface_decl),
        ))
    }

    fn parse_permits_expr(&mut self) -> Result<&'bump PermitsExpr<'a, 'bump>, DiagnosticError<'a>> {
        let token = self.cursor.expect(TokenKind::Permits)?;

        let mut types = Vec::new();

        loop {
            let ty = self.parse_type()?;
            types.push(ty);

            if !self.cursor.consume(TokenKind::Comma) {
                break;
            }
        }

        Ok(self.bump.alloc_value_immutable(ir::ast::PermitsExpr {
            types: self.bump.alloc_slice(&types),
            span: token.span,
        }))
    }

    /// Parse the right-hand side of `NAME :: enum { ... }`
    pub fn parse_enum_decl(
        &mut self,
        visibility: Visibility,
        name: StrId,
        name_span: SourceSpan<'a>,
        generics: Option<&'bump [Generic<'a, 'bump>]>,
    ) -> Result<Stmt<'a, 'bump>, DiagnosticError<'a>> {
        let token = self.cursor.expect(TokenKind::Enum)?;

        let mut variants = Vec::new();

        self.cursor.expect(TokenKind::LBrace)?;
        while self.cursor.peek() != TokenKind::RBrace && self.cursor.peek() != TokenKind::EOF {
            variants.push(self.parse_variant()?);
            if !self.cursor.consume(TokenKind::Comma) {
                break;
            }
        }
        self.cursor.expect(TokenKind::RBrace)?;
        Ok(Stmt::EnumDecl(self.bump.alloc_value_immutable(EnumDecl {
            name,
            visibility,
            generics,
            variants: self.bump.alloc_slice(&variants),
            name_span,
            span: token.span,
        })))
    }

    /// Allows `VariantName` or `VariantName { first_field: type, second_field: type }`
    fn parse_variant(&mut self) -> Result<EnumVariant<'a, 'bump>, DiagnosticError<'a>> {
        let (name, span) = self.cursor.expect_ident()?;

        if self.cursor.peek() == TokenKind::LParen {
            let paren_span = self.cursor.peek_token().span;
            self.diag.record(
                DiagnosticError::new(
                    ParseErrorKind::UnexpectedToken {
                        expected: TokenKind::LBrace,
                        found: TokenKind::LParen,
                    },
                    paren_span,
                )
                .with_note(
                    "Fix: Zeta enum variants don't support tuple syntax `Name(Type)`. \
                     Use named fields instead: `Name { field: Type }`.",
                ),
            );

            self.cursor.advance(); // consume '('
            let mut depth = 1i32;
            while depth > 0 && self.cursor.peek() != TokenKind::EOF {
                match self.cursor.peek() {
                    TokenKind::LParen => depth += 1,
                    TokenKind::RParen => depth -= 1,
                    _ => {}
                }
                self.cursor.advance();
            }

            return Ok(EnumVariant {
                name,
                fields: &[],
                span,
            });
        }

        if self.cursor.peek() != TokenKind::LBrace {
            Ok(EnumVariant {
                name,
                fields: &[],
                span,
            })
        } else {
            self.cursor.advance();
            let mut fields = Vec::new();
            while self.cursor.peek() != TokenKind::RBrace && self.cursor.peek() != TokenKind::EOF {
                let visibility =
                    token_to_visibility(self.cursor.peek()).unwrap_or(Visibility::Public);
                if token_to_visibility(self.cursor.peek()).is_some() {
                    self.cursor.advance();
                }

                let manual = self.eat_manual_modifier();
                let (field_name, field_span) = self.cursor.expect_ident()?;
                self.cursor.expect(TokenKind::Colon)?;
                let field_type = self.parse_type()?;

                fields.push(Field {
                    name: field_name,
                    field_type,
                    visibility,
                    span: field_span,
                    manual,
                });
                if !self.cursor.consume(TokenKind::Comma) {
                    break;
                }
            }

            self.cursor.expect(TokenKind::RBrace)?;
            Ok(EnumVariant {
                name,
                fields: self.bump.alloc_slice(&fields),
                span,
            })
        }
    }
}
