use ir::{
    borrow_checker::{
        Bound, IndexContainer, IndexTemplate, Interval, PlaceId, RefTemplate, TemplateBase,
        TemplateProjection,
    },
    errors::type_error::TypeErrorKind,
    hir::{
        HirExpr, HirFunc, HirParam, HirStmt, HirType, IntrinsicKind, ProvenanceAnnotation,
        ProvenancePathSegment, ProvenanceRoot, RefKind, StrId,
    },
    ir_hasher::FxHashMap,
};

use crate::{
    naming::provenance_to_string, ref_effects::UsedIndex, str_id_to_string, TypeChecker,
    TypeContext,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn infer_provenance(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let mut segments = Vec::new();
        let root = self.infer_provenance_root(expr, &mut segments)?;
        segments.reverse();
        Some(ProvenanceAnnotation {
            root,
            path: self.context.bump.alloc_slice(&segments),
        })
    }

    pub fn infer_provenance_root(
        &self,
        expr: &HirExpr<'a, 'bump>,
        segments: &mut Vec<ProvenancePathSegment>,
    ) -> Option<ProvenanceRoot> {
        match expr {
            HirExpr::Ident(name, _) => Some(ProvenanceRoot::Var(*name)),
            HirExpr::This { .. } => Some(ProvenanceRoot::ThisRoot),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                segments.push(ProvenancePathSegment::Field(*field));
                self.infer_provenance_root(object, segments)
            }

            HirExpr::Deref { expr: inner, .. } => {
                segments.push(ProvenancePathSegment::Deref);
                self.infer_provenance_root(inner, segments)
            }

            HirExpr::Index { object, .. } => self.infer_provenance_root(object, segments),

            HirExpr::ModuleAccess(access) => {
                let module_idx = self
                    .context
                    .dep_graph
                    .borrow()
                    .resolve_module_path(access.path)?;
                self.context
                    .dep_graph
                    .borrow()
                    .resolve_global_const(module_idx, access.member)?;
                Some(ProvenanceRoot::Global {
                    module_idx,
                    name: access.member,
                })
            }

            _ => None,
        }
    }

    pub fn extend_provenance(
        base: Option<ProvenanceAnnotation<'bump>>,
        field: StrId,
        ctx: &TypeContext<'a, 'bump>,
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let base = base?;
        let mut path: Vec<ProvenancePathSegment> = base.path.to_vec();
        path.push(ProvenancePathSegment::Field(field));
        Some(ProvenanceAnnotation {
            root: base.root,
            path: ctx.bump.alloc_slice(&path),
        })
    }

    pub fn analyze_ref_template(&mut self, func: &HirFunc<'a, 'bump>) -> RefTemplate {
        if let Some(t) = self.ref_templates.get(&func.name) {
            return t.clone();
        }

        self.ref_templates.insert(func.name, RefTemplate::Opaque);

        let template = Self::build_ref_template(func);
        self.ref_templates.insert(func.name, template.clone());
        template
    }

    pub fn check_return_provenance(&mut self, func: &HirFunc<'a, 'bump>) {
        let Some(HirType::Ref {
            provenance: Some(ann),
            ..
        }) = func.return_type
        else {
            return;
        };

        let template = Self::build_ref_template(func);
        let RefTemplate::Path { base, .. } = template else {
            self.record(TypeErrorKind::Generic(format!(
                "return type declares provenance `{}` but the body's returned reference isn't a simple projection",
                provenance_to_string(&ann)
            )));
            return;
        };

        let root_matches = match (ann.root, base) {
            (ProvenanceRoot::Var(name), TemplateBase::Param(idx)) => func
                .params
                .map(|p| {
                    p.iter()
                        .filter(|pp| matches!(pp, HirParam::Normal { .. }))
                        .collect::<Vec<_>>()
                })
                .and_then(|normals| normals.get(idx).copied())
                .is_some_and(
                    |p| matches!(p, HirParam::Normal { name: pname, .. } if name == *pname),
                ),
            (ProvenanceRoot::ThisRoot, TemplateBase::This) => true,
            _ => false,
        };

        if !root_matches {
            self.record(TypeErrorKind::Generic(format!(
                "declared provenance `{}` doesn't match the parameter the returned reference is actually rooted in",
                provenance_to_string(&ann)
            )));
        }
    }

    pub fn build_ref_template(func: &HirFunc<'a, 'bump>) -> RefTemplate {
        let Some(params) = func.params else {
            return RefTemplate::Opaque;
        };

        let mut param_index: FxHashMap<StrId, usize> = FxHashMap::default();
        let mut has_this = false;
        let mut normal_idx = 0usize;
        for p in params.iter() {
            match p {
                HirParam::Normal { name, .. } => {
                    param_index.insert(*name, normal_idx);
                    normal_idx += 1;
                }
                HirParam::This { .. } => has_this = true,
            }
        }

        let Some(HirStmt::Block { body, span: _ }) = func.body else {
            return RefTemplate::Opaque;
        };

        let [HirStmt::Return(Some(ret_expr), _span)] = body else {
            return RefTemplate::Opaque;
        };
        let (expr, ref_kind) = match ret_expr {
            HirExpr::Ref {
                expr,
                ref_kind: mutable,
                ..
            } => (*expr, *mutable),
            HirExpr::Intrinsic {
                kind: IntrinsicKind::Own,
                args,
                ..
            } if args.len() == 2 => (&args[0], RefKind::Unique),
            _ => return RefTemplate::Opaque,
        };

        let Some((base, projections)) = Self::expr_to_template(expr, &param_index, has_this) else {
            return RefTemplate::Opaque;
        };

        RefTemplate::Path {
            base,
            ref_kind,
            projections,
        }
    }

    pub fn expr_to_template(
        expr: &HirExpr<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
    ) -> Option<(TemplateBase, Vec<TemplateProjection>)> {
        match expr {
            HirExpr::Ident(name, _) => {
                Some((TemplateBase::Param(*param_index.get(name)?), Vec::new()))
            }

            HirExpr::This { .. } if has_this => Some((TemplateBase::This, Vec::new())),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let (base, mut proj) = Self::expr_to_template(object, param_index, has_this)?;
                proj.push(TemplateProjection::Field(*field));
                Some((base, proj))
            }

            HirExpr::Deref { expr, .. } => {
                let (base, mut proj) = Self::expr_to_template(expr, param_index, has_this)?;
                proj.push(TemplateProjection::Deref);
                Some((base, proj))
            }

            HirExpr::Index { object, index, .. } => {
                let (base, mut proj) = Self::expr_to_template(object, param_index, has_this)?;
                let idx = match &**index {
                    HirExpr::Number(n, _) => IndexTemplate::Const(*n),
                    HirExpr::Ident(name, _) => param_index
                        .get(name)
                        .map(|&i| IndexTemplate::Param(i))
                        .unwrap_or(IndexTemplate::Opaque),
                    _ => IndexTemplate::Opaque,
                };
                proj.push(TemplateProjection::Index(idx));
                Some((base, proj))
            }

            _ => None,
        }
    }

    pub fn provenance_from_template(
        &self,
        template: &RefTemplate,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let RefTemplate::Path {
            base, projections, ..
        } = template
        else {
            return None;
        };

        let base_expr = match base {
            TemplateBase::This => receiver?,
            TemplateBase::Param(i) => {
                let arg = args.get(*i)?;
                match arg {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                }
            }
        };

        let mut base_provenance = self.infer_provenance(base_expr)?;

        let mut path: Vec<ProvenancePathSegment> = base_provenance.path.to_vec();
        for proj in projections {
            match proj {
                TemplateProjection::Field(f) => path.push(ProvenancePathSegment::Field(*f)),
                TemplateProjection::Deref => path.push(ProvenancePathSegment::Deref),
                TemplateProjection::Index(_) => {}
            }
        }
        base_provenance.path = self.context.bump.alloc_slice(&path);
        Some(base_provenance)
    }

    pub fn resolve_template_place(
        &mut self,
        template: &RefTemplate,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Option<PlaceId> {
        let RefTemplate::Path {
            base, projections, ..
        } = template
        else {
            return None;
        };

        let base_expr = match base {
            TemplateBase::This => receiver?,
            TemplateBase::Param(i) => {
                let arg = args.get(*i)?;
                match arg {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                }
            }
        };

        let mut place = self.resolve_place(base_expr)?;

        for proj in projections {
            place = match proj {
                TemplateProjection::Field(f) => self.borrow_checker.project_field(place, *f),
                TemplateProjection::Deref => self.borrow_checker.project_deref(place),
                TemplateProjection::Index(idx_template) => {
                    let bound = match idx_template {
                        IndexTemplate::Const(c) => Bound::Const(*c),
                        IndexTemplate::Param(i) => self.expr_to_bound(args.get(*i)?),
                        IndexTemplate::Opaque => Bound::Opaque(0),
                    };
                    let interval = Interval {
                        lower: bound.clone(),
                        upper: bound,
                    };
                    self.borrow_checker
                        .project_index(place, interval, IndexContainer::Primitive)
                }
            };
        }

        Some(place)
    }

    pub fn return_type_may_alias(&self, ty: &HirType<'a, 'bump>) -> bool {
        match ty {
            HirType::Ref { .. }
            | HirType::SafePointer { .. }
            | HirType::UnsafePointer { .. }
            | HirType::OwnedPointer { .. } => true,

            HirType::Nullable(inner) => self.return_type_may_alias(inner),

            HirType::Array(inner, _) => self.return_type_may_alias(inner),

            HirType::Tuple(elems) => elems.iter().any(|e| self.return_type_may_alias(e)),

            HirType::Struct { name, .. } => {
                let name_str = str_id_to_string(*name);
                match self.context.get_struct(&name_str) {
                    Some(def) => def
                        .fields
                        .iter()
                        .any(|f| self.return_type_may_alias(&f.field_type)),
                    None => true,
                }
            }

            HirType::Enum { name, .. } => {
                let name_str = str_id_to_string(*name);
                match self.context.get_enum(&name_str) {
                    Some(def) => def
                        .variants
                        .iter()
                        .flat_map(|v| v.fields.iter())
                        .any(|f| self.return_type_may_alias(&f.field_type)),
                    None => true,
                }
            }

            HirType::Dyn { .. } | HirType::DynInterface(..) => true,

            _ => false,
        }
    }

    pub fn path_display(path: &[StrId], range: Option<&UsedIndex>) -> String {
        let mut s = String::new();
        for seg in path {
            s.push('.');
            s.push_str(&seg.to_string());
        }
        if let Some(idx) = range {
            match idx {
                UsedIndex::Range(start, end) if *end == *start + 1 => {
                    s.push_str(&format!("[{}]", start));
                }
                UsedIndex::Range(start, end) => {
                    s.push_str(&format!("[{}..{}]", start, end));
                }
                UsedIndex::Place(root, p) => {
                    s.push('[');
                    s.push_str(&root.to_string());
                    for seg in p {
                        s.push('.');
                        s.push_str(&seg.to_string());
                    }
                    s.push(']');
                }
            }
        }
        s
    }
}
