use ir::{
    ast::MutabilityState,
    borrow_checker::Bound,
    errors::type_error::TypeErrorKind,
    hir::{
        AssignmentOperator, EffectIndexKey, HirEffectAccess, HirEffectSegment,
        HirErrorHandlerPattern, HirExpr, HirStmt, HirType, InterpolationPart, RefKind, StrId,
    },
};

use crate::{str_id_to_string, TypeChecker};

#[derive(Clone)]
pub enum UsedIndex {
    Range(i64, i64),          // from a literal index or a slice
    Place(StrId, Vec<StrId>), // from a stable field-path index
}

pub type EffectUsage = (Vec<StrId>, Option<UsedIndex>);

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn static_field_path(&self, expr: &HirExpr<'a, 'bump>) -> Option<(StrId, Vec<StrId>)> {
        match expr {
            HirExpr::Ident(name, _) => Some((*name, Vec::new())),
            HirExpr::This { .. } => Some((self.this_id, Vec::new())),
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let (root, mut path) = self.static_field_path(object)?;
                path.push(*field);
                Some((root, path))
            }
            _ => None,
        }
    }

    pub fn effect_place_of(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<StrId>, Option<UsedIndex>)> {
        match expr {
            HirExpr::Index { object, index, .. } => {
                let (root, path) = self.static_field_path(object)?;
                let used = match index {
                    HirExpr::Number(i, _) => Some(UsedIndex::Range(*i, *i + 1)),
                    other => self
                        .static_field_path(other)
                        .map(|(r, p)| UsedIndex::Place(r, p)),
                };
                Some((root, path, used))
            }
            _ => self.static_field_path(expr).map(|(r, p)| (r, p, None)),
        }
    }

    pub fn collect_root_accesses_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
        root: StrId,
        writes: &mut Vec<EffectUsage>,
        reads: &mut Vec<EffectUsage>,
    ) {
        match expr {
            HirExpr::Assignment {
                target, op, value, ..
            } => {
                self.collect_root_accesses_expr(value, root, writes, reads);
                if let Some((r, path, range)) = self.effect_place_of(target) {
                    if r == root {
                        if !matches!(op, AssignmentOperator::Assign) {
                            writes.push((path.clone(), range.clone()));
                            reads.push((path, range)); // compound assign reads-then-writes
                        } else {
                            writes.push((path.clone(), range));
                        }
                        return;
                    }
                }
                self.collect_root_accesses_expr(target, root, writes, reads);
            }
            HirExpr::Ref {
                expr: inner,
                ref_kind,
                ..
            } => {
                if let Some((r, path, range)) = self.effect_place_of(inner) {
                    if r == root {
                        if *ref_kind != RefKind::Shared {
                            writes.push((path.clone(), range.clone()));
                            reads.push((path, range));
                        } else {
                            reads.push((path, range));
                        }
                        return;
                    }
                }
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Ident(name, _) if *name == root => reads.push((Vec::new(), None)),
            HirExpr::This { .. } if root == self.this_id => reads.push((Vec::new(), None)),
            HirExpr::Ident(_, _) | HirExpr::This { .. } => {}
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let Some((r, mut path)) = self.static_field_path(object) {
                    if r == root {
                        path.push(*field);
                        reads.push((path, None));
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
            }
            HirExpr::Index { object, index, .. } => {
                if let Some((r, path, range)) = self.effect_place_of(expr) {
                    if r == root {
                        reads.push((path, range));
                        self.collect_root_accesses_expr(index, root, writes, reads);
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
                self.collect_root_accesses_expr(index, root, writes, reads);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                if let Some((r, path)) = self.static_field_path(object) {
                    if r == root {
                        reads.push((path, None));
                        self.collect_root_accesses_expr(start, root, writes, reads);
                        self.collect_root_accesses_expr(end, root, writes, reads);
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
                self.collect_root_accesses_expr(start, root, writes, reads);
                self.collect_root_accesses_expr(end, root, writes, reads);
            }
            HirExpr::Deref { expr: inner, .. } => {
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Range { start, end, .. } => {
                self.collect_root_accesses_expr(start, root, writes, reads);
                self.collect_root_accesses_expr(end, root, writes, reads);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            }
            | HirExpr::ExprList { list: exprs, .. } => {
                for e in exprs.iter() {
                    self.collect_root_accesses_expr(e, root, writes, reads);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.collect_root_accesses_expr(left, root, writes, reads);
                self.collect_root_accesses_expr(right, root, writes, reads);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.collect_root_accesses_expr(callee, root, writes, reads);
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.collect_root_accesses_expr(&f.value, root, writes, reads);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.collect_root_accesses_expr(e, root, writes, reads);
                    }
                }
            }
            HirExpr::Cast { expr: inner, .. } => {
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::If { if_stmt, .. } => {
                self.collect_root_accesses_stmt(if_stmt, root, writes, reads);
            }
            HirExpr::Match {
                expr: scrutinee,
                arms,
                ..
            } => {
                self.collect_root_accesses_expr(scrutinee, root, writes, reads);
                for arm in arms.iter() {
                    if let Some(guard) = arm.guard {
                        self.collect_root_accesses_expr(guard, root, writes, reads);
                    }
                    self.collect_root_accesses_stmt(arm.body, root, writes, reads);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
            }
            HirExpr::Lambda { params, body, .. } => {
                let shadowed = params.iter().any(|p| p.name == root);
                if !shadowed {
                    self.collect_root_accesses_stmt(body, root, writes, reads);
                }
            }
            HirExpr::Null(_)
            | HirExpr::Number(_, _)
            | HirExpr::Char(_, _)
            | HirExpr::String(_, _)
            | HirExpr::Boolean(_, _)
            | HirExpr::Decimal(_, _)
            | HirExpr::Undefined { .. }
            | HirExpr::Uninit { .. }
            | HirExpr::GenericIdent(..)
            | HirExpr::ModuleAccess(_)
            | HirExpr::UnknownIntrinsic { .. } => {}
        }
    }

    pub fn collect_root_accesses_stmt(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        root: StrId,
        writes: &mut Vec<EffectUsage>,
        reads: &mut Vec<EffectUsage>,
    ) {
        match stmt {
            HirStmt::Let {
                value,
                else_block,
                catch_pattern,
                ..
            } => {
                self.collect_root_accesses_expr(value, root, writes, reads);
                if let Some(b) = else_block {
                    self.collect_root_accesses_stmt(b, root, writes, reads);
                }
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                self.collect_root_accesses_stmt(s, root, writes, reads);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    self.collect_root_accesses_stmt(s, root, writes, reads);
                                }
                            }
                        }
                    }
                }
            }
            HirStmt::Const(c) => self.collect_root_accesses_expr(&c.value, root, writes, reads),
            HirStmt::Return(Some(e), _span) | HirStmt::Break(Some(e), _span) => {
                self.collect_root_accesses_expr(e, root, writes, reads)
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => self.collect_root_accesses_expr(e, root, writes, reads),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.collect_root_accesses_expr(cond, root, writes, reads);
                for s in then_block.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
                if let Some(e) = else_block {
                    self.collect_root_accesses_stmt(e, root, writes, reads);
                }
            }
            HirStmt::While { cond, body } => {
                self.collect_root_accesses_expr(cond, root, writes, reads);
                self.collect_root_accesses_stmt(body, root, writes, reads);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.collect_root_accesses_stmt(i, root, writes, reads);
                }
                if let Some(c) = condition {
                    self.collect_root_accesses_expr(c, root, writes, reads);
                }
                if let Some(inc) = increment {
                    self.collect_root_accesses_expr(inc, root, writes, reads);
                }
                self.collect_root_accesses_stmt(body, root, writes, reads);
            }
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
            }
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.collect_root_accesses_expr(expr, root, writes, reads);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_root_accesses_expr(g, root, writes, reads);
                    }
                    self.collect_root_accesses_stmt(arm.body, root, writes, reads);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                self.collect_root_accesses_stmt(body, root, writes, reads)
            }
        }
    }

    pub fn declared_access_covers(
        declared_path: &[HirEffectSegment],
        used_fields: &[StrId],
        used_index: Option<&UsedIndex>,
    ) -> bool {
        let mut d_fields = Vec::new();
        let mut d_index: Option<&EffectIndexKey> = None;
        for seg in declared_path {
            match seg {
                HirEffectSegment::Field(f) => d_fields.push(*f),
                HirEffectSegment::Index(key) => d_index = Some(key),
            }
        }
        if d_fields != used_fields {
            return false;
        }
        match (d_index, used_index) {
            (None, _) => true,
            (Some(EffectIndexKey::Dynamic), _) => true, // can't correlate, so grant the whole slot space
            (Some(EffectIndexKey::Const(d_i)), Some(UsedIndex::Range(u_s, u_e))) => {
                *d_i <= *u_s && *u_e <= *d_i + 1
            }
            (
                Some(EffectIndexKey::Place { root, path }),
                Some(UsedIndex::Place(u_root, u_path)),
            ) => *root == *u_root && *path == u_path.as_slice(),
            _ => false,
        }
    }

    pub fn validate_multi_place_declaration(
        &mut self,
        root: StrId,
        declared: &[HirEffectAccess<'bump>],
        body: &HirStmt<'a, 'bump>,
    ) {
        let mut writes = Vec::new();
        let mut reads = Vec::new();
        self.collect_root_accesses_stmt(body, root, &mut writes, &mut reads);

        for (path, range) in &writes {
            let covered = declared.iter().any(|d| {
                d.ref_kind != RefKind::Shared
                    && Self::declared_access_covers(d.path, path, range.as_ref())
            });
            if !covered {
                self.record(TypeErrorKind::Generic(format!(
                    "writes to `{}{}` but the declared effects don't grant `&mut` access there",
                    str_id_to_string(root),
                    Self::path_display(path, range.as_ref()),
                )));
            }
        }
        for (path, range) in &reads {
            let covered = declared
                .iter()
                .any(|d| Self::declared_access_covers(d.path, path, range.as_ref()));
            if !covered {
                self.record(TypeErrorKind::Generic(format!(
                    "reads `{}{}` but it isn't listed in the declared effects",
                    str_id_to_string(root),
                    Self::path_display(path, range.as_ref()),
                )));
            }
        }
    }

    pub fn validate_multi_place_signature(
        &mut self,
        param_type: Option<&HirType<'a, 'bump>>,
        accesses: &[HirEffectAccess<'bump>],
    ) {
        let outer_mut = match param_type {
            Some(HirType::Ref { ref_kind, .. }) => *ref_kind != RefKind::Shared,
            Some(HirType::SafePointer {
                mutability_state, ..
            })
            | Some(HirType::UnsafePointer {
                mutability_state, ..
            }) => *mutability_state == MutabilityState::Mut,
            Some(HirType::This) => true,
            None => true,
            _ => {
                self.record(TypeErrorKind::Generic(
                    "`.{...}` access lists are only valid on reference or pointer parameters"
                        .to_string(),
                ));
                return;
            }
        };
        if !outer_mut && accesses.iter().any(|a| a.ref_kind == RefKind::Unique) {
            self.record(TypeErrorKind::Generic(
                "declares `&mut` access through a parameter that isn't itself `&mut`".to_string(),
            ));
        }
    }

    pub fn effect_index_key_to_bound(&mut self, key: &EffectIndexKey) -> Bound {
        match key {
            EffectIndexKey::Const(i) => Bound::Const(*i),
            EffectIndexKey::Place { root, path } => {
                let mut name = str_id_to_string(*root);
                for seg in path.iter() {
                    name.push('.');
                    name.push_str(&str_id_to_string(*seg));
                }
                Bound::Symbol(StrId(self.context.string_pool.intern(&name)))
            }
            EffectIndexKey::Dynamic => self.fresh_opaque(),
        }
    }
}
