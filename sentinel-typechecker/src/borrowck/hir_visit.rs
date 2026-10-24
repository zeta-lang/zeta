use ir::hir::{HirErrorHandlerPattern, HirExpr, HirStmt, InterpolationPart, StrId};

pub trait HirVisitor<'a, 'bump> {
    fn should_continue(&self) -> bool {
        true
    }

    fn visit_expr(&mut self, expr: &HirExpr<'a, 'bump>) {
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        walk_stmt(self, stmt);
    }

    fn visit_ident(&mut self, _name: StrId) {}
}

pub fn walk_expr<'a, 'bump, V>(v: &mut V, expr: &HirExpr<'a, 'bump>)
where
    V: HirVisitor<'a, 'bump> + ?Sized,
{
    if !v.should_continue() {
        return;
    }
    match expr {
        HirExpr::Ident(name, _) => v.visit_ident(*name),
        HirExpr::Match { expr, arms, .. } => {
            v.visit_expr(expr);
            for arm in arms.iter() {
                if let Some(g) = arm.guard {
                    v.visit_expr(g);
                }
                v.visit_stmt(arm.body);
            }
        }
        HirExpr::Block { body, .. } => {
            for s in body.iter() {
                v.visit_stmt(s);
            }
        }
        HirExpr::ComptimeBlock { body, .. } => {
            for s in body.iter() {
                v.visit_stmt(s);
            }
        }
        HirExpr::Range { start, end, .. } => {
            v.visit_expr(start);
            v.visit_expr(end);
        }
        HirExpr::Slice {
            object, start, end, ..
        } => {
            v.visit_expr(object);
            v.visit_expr(start);
            v.visit_expr(end);
        }
        HirExpr::Tuple(exprs, _)
        | HirExpr::ArrayLiteral {
            elements: exprs, ..
        } => {
            for e in exprs.iter() {
                v.visit_expr(e);
            }
        }
        HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
            v.visit_expr(callee);
            for a in args.iter() {
                v.visit_expr(a);
            }
        }
        HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
            v.visit_expr(object);
        }
        HirExpr::Assignment { target, value, .. } => {
            v.visit_expr(target);
            v.visit_expr(value);
        }
        HirExpr::StructInit { args, .. } => {
            for f in args.iter() {
                v.visit_expr(&f.value);
            }
        }
        HirExpr::EnumInit { args, .. } => {
            for a in args.iter() {
                v.visit_expr(a);
            }
        }
        HirExpr::ExprList { list, .. } => {
            for e in list.iter() {
                v.visit_expr(e);
            }
        }
        HirExpr::Deref { expr, .. } | HirExpr::Cast { expr, .. } | HirExpr::Ref { expr, .. } => {
            v.visit_expr(expr);
        }
        HirExpr::Index { object, index, .. } => {
            v.visit_expr(object);
            v.visit_expr(index);
        }
        HirExpr::InterpolatedString(parts) => {
            for p in parts.iter() {
                if let InterpolationPart::Expr(e) = p {
                    v.visit_expr(e);
                }
            }
        }
        HirExpr::If { if_stmt, .. } => v.visit_stmt(if_stmt),
        HirExpr::Intrinsic { args, .. } => {
            for a in args.iter() {
                v.visit_expr(a);
            }
        }
        HirExpr::Lambda { body, .. } => v.visit_stmt(body),
        HirExpr::OrElse {
            value, else_body, ..
        } => {
            v.visit_expr(value);
            for s in else_body.iter() {
                v.visit_stmt(s);
            }
        }
        HirExpr::This { .. }
        | HirExpr::ModuleAccess(_)
        | HirExpr::GenericIdent(..)
        | HirExpr::Number(..)
        | HirExpr::Decimal(..)
        | HirExpr::String(..)
        | HirExpr::Boolean(..)
        | HirExpr::Null(_)
        | HirExpr::Undefined { .. }
        | HirExpr::Uninit { .. }
        | HirExpr::Char(_, _)
        | HirExpr::UnknownIntrinsic { .. } => {}
    }
}

pub fn walk_stmt<'a, 'bump, V>(v: &mut V, stmt: &HirStmt<'a, 'bump>)
where
    V: HirVisitor<'a, 'bump> + ?Sized,
{
    if !v.should_continue() {
        return;
    }
    match stmt {
        HirStmt::Let {
            value,
            catch_pattern,
            ..
        } => {
            v.visit_expr(value);
            if let Some(pattern) = catch_pattern {
                match pattern {
                    HirErrorHandlerPattern::Single { body, .. } => {
                        for s in body.iter() {
                            v.visit_stmt(s);
                        }
                    }
                    HirErrorHandlerPattern::Multiple { branches } => {
                        for branch in branches.iter() {
                            for s in branch.body.iter() {
                                v.visit_stmt(s);
                            }
                        }
                    }
                }
            }
        }
        HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => v.visit_expr(e),
        HirStmt::Return(None, _)
        | HirStmt::Break(None, _)
        | HirStmt::Continue(_)
        | HirStmt::Import(..)
        | HirStmt::Package(..) => {}
        HirStmt::Expr(e) => v.visit_expr(e),
        HirStmt::If {
            cond,
            then_block,
            else_block,
            span: _,
        } => {
            v.visit_expr(cond);
            for s in then_block.iter() {
                v.visit_stmt(s);
            }
            if let Some(e) = else_block {
                v.visit_stmt(e);
            }
        }
        HirStmt::While { cond, body } => {
            v.visit_expr(cond);
            v.visit_stmt(body);
        }
        HirStmt::For {
            init,
            condition,
            increment,
            body,
        } => {
            if let Some(i) = init {
                v.visit_stmt(i);
            }
            if let Some(c) = condition {
                v.visit_expr(c);
            }
            if let Some(inc) = increment {
                v.visit_expr(inc);
            }
            v.visit_stmt(body);
        }
        HirStmt::Block { body, span: _ } => {
            for s in body.iter() {
                v.visit_stmt(s);
            }
        }
        HirStmt::Const(c) => v.visit_expr(&c.value),
        HirStmt::Match {
            expr,
            arms,
            span: _,
        } => {
            v.visit_expr(expr);
            for arm in arms.iter() {
                if let Some(g) = arm.guard {
                    v.visit_expr(g);
                }
                v.visit_stmt(arm.body);
            }
        }
        HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => v.visit_stmt(body),
    }
}
