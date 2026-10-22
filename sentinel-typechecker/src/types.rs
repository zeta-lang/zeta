use ir::{
    ast::MutabilityState,
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{HirExpr, HirType, HirTypeKind, RefKind},
};

use crate::{
    TypeChecker,
    naming::type_to_string,
    type_checker::{LocalSymbolId, SymbolId},
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn nullable_equality_compatible(
        &self,
        left: &HirType<'a, 'bump>,
        right: &HirType<'a, 'bump>,
    ) -> bool {
        let unwrap_nullable = |t: &HirType<'a, 'bump>| match &t.kind {
            HirTypeKind::Nullable(inner) => Some(**inner),
            _ => None,
        };

        let payload_compatible = |a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>| {
            (self.is_comparable(a) && self.is_comparable(b) && self.types_structurally_equal(a, b))
                || (self.is_reference_like(a)
                    && self.is_reference_like(b)
                    && self.types_structurally_equal(a, b))
        };

        match (unwrap_nullable(left), unwrap_nullable(right)) {
            (Some(li), Some(ri)) => payload_compatible(&li, &ri),
            (Some(li), None) => {
                payload_compatible(&li, right) || matches!(right.kind, HirTypeKind::Null)
            }
            (None, Some(ri)) => {
                payload_compatible(left, &ri) || matches!(left.kind, HirTypeKind::Null)
            }
            (None, None) => false,
        }
    }

    pub fn types_structurally_equal(&self, a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>) -> bool {
        use HirTypeKind::*;
        if matches!(a.kind, Generic(_)) || matches!(b.kind, Generic(_)) {
            return true;
        }

        match (&a.kind, &b.kind) {
            (Nullable(inner), _) => self.types_structurally_equal(inner, b),
            (_, Nullable(inner)) => self.types_structurally_equal(a, inner),
            (
                Struct {
                    name: na,
                    type_args: ta,
                    ..
                },
                Struct {
                    name: nb,
                    type_args: tb,
                    ..
                },
            ) => {
                na == nb
                    && ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (
                Ref {
                    inner: ia,
                    ref_kind: ma,
                    ..
                },
                Ref {
                    inner: ib,
                    ref_kind: mb,
                    ..
                },
            ) => {
                // Deliberately ignore provenance here, it's borrow-checker
                // bookkeeping about where a reference came from, not part of
                // the reference's type identity. Two `&mut i64` are the same
                // type regardless of which place each one happens to alias.
                ma == mb && self.types_structurally_equal(ia, ib)
            }
            #[allow(unreachable_patterns)]
            (Nullable(ia), Nullable(ib)) => self.types_structurally_equal(ia, ib),
            (
                SafePointer {
                    inner: ia,
                    mutability_state: ma,
                },
                SafePointer {
                    inner: ib,
                    mutability_state: mb,
                },
            ) => ma == mb && self.types_structurally_equal(ia, ib),
            (
                UnsafePointer {
                    inner: ia,
                    mutability_state: ma,
                },
                UnsafePointer {
                    inner: ib,
                    mutability_state: mb,
                },
            ) => ma == mb && self.types_structurally_equal(ia, ib),

            (
                OwnedPointer {
                    inner: ia,
                    allocator: _,
                },
                OwnedPointer {
                    inner: ib,
                    allocator: _,
                },
            ) => self.types_structurally_equal(ia, ib),
            (Array(ia, la), Array(ib, lb)) => la == lb && self.types_structurally_equal(ia, ib),
            (Slice(ia), Slice(ib)) => self.types_structurally_equal(ia, ib),
            (Tuple(ta), Tuple(tb)) => {
                ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (Dyn { bounds: ba }, Dyn { bounds: bb }) => {
                ba.len() == bb.len()
                    && ba
                        .iter()
                        .zip(bb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (
                Lambda {
                    params: pa,
                    return_type: ra,
                },
                Lambda {
                    params: pb,
                    return_type: rb,
                },
            ) => {
                pa.len() == pb.len()
                    && pa
                        .iter()
                        .zip(pb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
                    && self.types_structurally_equal(ra, rb)
            }
            (
                Enum {
                    name: na,
                    type_args: ta,
                    ..
                },
                Enum {
                    name: nb,
                    type_args: tb,
                    ..
                },
            ) => {
                na == nb
                    && ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            _ => a == b,
        }
    }

    pub(crate) fn owned_ref_coerces(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> bool {
        let HirTypeKind::Ref {
            inner: f_inner,
            ref_kind: fk,
            ..
        } = &found.kind
        else {
            return false;
        };

        match (&expected.kind, &f_inner.kind) {
            // &^T  ->  &T
            (
                HirTypeKind::Ref {
                    inner: e_inner,
                    ref_kind: ek,
                    ..
                },
                HirTypeKind::OwnedPointer { inner: pointee, .. },
            ) if !matches!(pointee.kind, HirTypeKind::Slice(_)) => {
                fk.coerces_to(*ek) && self.types_compatible(e_inner, pointee).is_ok()
            }
            // &?^T  ->  ?&T
            (HirTypeKind::Nullable(e), HirTypeKind::Nullable(f_nullable_inner)) => {
                match (&e.kind, &f_nullable_inner.kind) {
                    (
                        HirTypeKind::Ref {
                            inner: e_inner,
                            ref_kind: ek,
                            ..
                        },
                        HirTypeKind::OwnedPointer { inner: pointee, .. },
                    ) if !matches!(pointee.kind, HirTypeKind::Slice(_)) => {
                        fk.coerces_to(*ek) && self.types_compatible(e_inner, pointee).is_ok()
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn types_compatible(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> TypeCheckResult<'a, ()> {
        if matches!(found.kind, HirTypeKind::Never) {
            return Ok(());
        }

        if let HirTypeKind::Ref { inner, .. } = &expected.kind {
            if matches!(inner.kind, HirTypeKind::String)
                && matches!(found.kind, HirTypeKind::String)
            {
                return Ok(());
            }
        }

        if matches!(expected.kind, HirTypeKind::String) {
            if let HirTypeKind::Ref { inner, .. } = &found.kind {
                if matches!(inner.kind, HirTypeKind::String) {
                    return Ok(());
                }
            }
        }

        if self.owned_ref_coerces(expected, found) {
            return Ok(());
        }

        if let HirTypeKind::Ref {
            inner: ei,
            ref_kind: erk,
            ..
        } = &expected.kind
        {
            if let HirTypeKind::Slice(es) = &ei.kind {
                match &found.kind {
                    HirTypeKind::Slice(fs) if self.types_structurally_equal(es, fs) => {
                        return Ok(());
                    }
                    HirTypeKind::Ref {
                        inner: fi,
                        ref_kind: frk,
                        ..
                    } => {
                        if let HirTypeKind::Array(fe, _) | HirTypeKind::Slice(fe) = &fi.kind {
                            if self.types_structurally_equal(es, fe) && frk.coerces_to(*erk) {
                                return Ok(());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if let HirTypeKind::Slice(es) = &expected.kind {
            if let HirTypeKind::Ref { inner: fi, .. } = &found.kind {
                if let HirTypeKind::Array(fe, _) | HirTypeKind::Slice(fe) = &fi.kind {
                    if self.types_structurally_equal(es, fe) {
                        return Ok(());
                    }
                }
            }
        }

        if let (
            HirTypeKind::OwnedPointer { inner: ei, .. },
            HirTypeKind::OwnedPointer { inner: fi, .. },
        ) = (&expected.kind, &found.kind)
        {
            if self.types_structurally_equal(ei, fi) {
                return Ok(());
            }
        }

        if let (
            HirTypeKind::Ref {
                inner: ri,
                ref_kind,
                ..
            },
            HirTypeKind::SafePointer {
                inner: pi,
                mutability_state,
            }
            | HirTypeKind::UnsafePointer {
                inner: pi,
                mutability_state,
            },
        ) = (&found.kind, &expected.kind)
        {
            let elem: &HirType<'a, 'bump> = match &ri.kind {
                HirTypeKind::Array(e, _) | HirTypeKind::Slice(e) => e,
                _ => ri,
            };
            // `&` may only become a const pointer; otherwise a shared borrow could be written through.
            let mut_ok =
                *ref_kind != RefKind::Shared || *mutability_state == MutabilityState::Const;
            if mut_ok
                && (self.types_structurally_equal(ri, pi)
                    || self.types_structurally_equal(elem, pi))
            {
                return Ok(());
            }
        }

        if let (
            HirTypeKind::Ref {
                inner: ei,
                ref_kind: erk,
                ..
            },
            HirTypeKind::Ref {
                inner: fi,
                ref_kind: frk,
                ..
            },
        ) = (&expected.kind, &found.kind)
        {
            if self.types_structurally_equal(ei, fi) && frk.coerces_to(*erk) {
                return Ok(());
            }
        }

        if self.types_structurally_equal(expected, found) {
            return Ok(());
        }

        if expected.kind == HirTypeKind::Unknown || found.kind == HirTypeKind::Unknown {
            return Ok(());
        }

        if self.struct_satisfies_interface_type(expected, found)
            || self.struct_satisfies_interface_type(found, expected)
        {
            return Ok(());
        }

        match expected.kind {
            HirTypeKind::UnsafePointer { .. } | HirTypeKind::Nullable(_) => {
                if found.kind == HirTypeKind::Null {
                    return Ok(());
                }
            }
            _ => {}
        }

        Err(TypeErrorKind::TypeMismatch {
            expected: type_to_string(expected, &self.context.string_pool).to_string(),
            found: type_to_string(found, &self.context.string_pool).to_string(),
        }
        .at(self.current_span))
    }

    pub fn struct_satisfies_interface_type(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> bool {
        let expected_inner = Self::strip_ref(expected);
        let found_inner = Self::strip_ref(found);

        let interface_name = match &expected_inner.kind {
            HirTypeKind::DynInterface { name, .. } => Some(name),
            HirTypeKind::Dyn { bounds } => bounds.iter().find_map(|b| match &b.kind {
                HirTypeKind::DynInterface { name, .. } => Some(name),
                HirTypeKind::Struct { name, .. } => {
                    if self.context.get_interface(name).is_some() {
                        Some(name)
                    } else {
                        None
                    }
                }
                _ => None,
            }),
            _ => None,
        };

        let Some(interface_name) = interface_name else {
            return false;
        };

        let struct_name = match &found_inner.kind {
            HirTypeKind::Struct { name, .. } => name,
            _ => return false,
        };

        self.context.struct_implements(struct_name, interface_name)
    }

    pub fn strip_ref<'x>(ty: &'x HirType<'a, 'bump>) -> &'x HirType<'a, 'bump> {
        match &ty.kind {
            HirTypeKind::Ref { inner, .. } => inner,
            HirTypeKind::OwnedPointer { inner, .. } => inner,
            _ => ty,
        }
    }

    pub fn is_numeric(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty.kind,
            HirTypeKind::I8
                | HirTypeKind::I16
                | HirTypeKind::I32
                | HirTypeKind::I64
                | HirTypeKind::U8
                | HirTypeKind::U16
                | HirTypeKind::U32
                | HirTypeKind::U64
                | HirTypeKind::F32
                | HirTypeKind::F64
                | HirTypeKind::I128
                | HirTypeKind::U128
                | HirTypeKind::Usize
                | HirTypeKind::Isize
        )
    }

    pub fn is_integer(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty.kind,
            HirTypeKind::I8
                | HirTypeKind::I16
                | HirTypeKind::I32
                | HirTypeKind::I64
                | HirTypeKind::U8
                | HirTypeKind::U16
                | HirTypeKind::U32
                | HirTypeKind::U64
                | HirTypeKind::I128
                | HirTypeKind::U128
                | HirTypeKind::Usize
                | HirTypeKind::Isize
        )
    }

    pub fn is_comparable(&self, ty: &HirType<'a, 'bump>) -> bool {
        self.is_numeric(ty) || matches!(ty.kind, HirTypeKind::Boolean | HirTypeKind::String)
    }

    pub fn peek_type(&self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, _) => {
                self.context
                    .get_variable(name)
                    .unwrap_or((
                        SymbolId::Local(LocalSymbolId(u32::MAX)),
                        HirType::synthetic(HirTypeKind::Unknown),
                    ))
                    .1
            }
            HirExpr::This { .. } => {
                self.context
                    .get_variable("this")
                    .unwrap_or((
                        SymbolId::Local(LocalSymbolId(u32::MAX)),
                        HirType::synthetic(HirTypeKind::This),
                    ))
                    .1
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                match Self::peel_indirections(self.peek_type(object)).kind {
                    HirTypeKind::Struct {
                        name: struct_name,
                        type_args,
                        ..
                    } => {
                        let Some(struct_def) = self.context.get_struct(struct_name) else {
                            return HirType::synthetic(HirTypeKind::Unknown);
                        };
                        let Some(field_idx) =
                            struct_def.fields.iter().position(|f| f.name == field)
                        else {
                            return HirType::synthetic(HirTypeKind::Unknown);
                        };
                        if type_args.is_empty() {
                            struct_def.fields[field_idx].field_type
                        } else {
                            self.instantiate_struct(struct_name, type_args)
                                .map(|fields| fields[field_idx])
                                .unwrap_or(struct_def.fields[field_idx].field_type)
                        }
                    }
                    _ => HirType::synthetic(HirTypeKind::Unknown),
                }
            }
            HirExpr::Deref { expr, .. } => match self.peek_type(expr).kind {
                HirTypeKind::Ref { inner, .. } => *inner,
                HirTypeKind::SafePointer { inner, .. } => *inner,
                HirTypeKind::UnsafePointer { inner, .. } => *inner,
                HirTypeKind::OwnedPointer { inner, .. } => *inner,
                _ => HirType::synthetic(HirTypeKind::Unknown),
            },
            HirExpr::Index { object, .. } => match self.peek_type(object).kind {
                HirTypeKind::Array(inner, _) => *inner,
                HirTypeKind::Slice(inner) => *inner,
                _ => HirType::synthetic(HirTypeKind::Unknown),
            },
            HirExpr::Cast { target_type, .. } => *target_type,
            _ => HirType::synthetic(HirTypeKind::Unknown),
        }
    }

    pub fn is_zeroable(&self, ty: &HirType<'a, 'bump>) -> bool {
        match &ty.kind {
            HirTypeKind::I8
            | HirTypeKind::I16
            | HirTypeKind::I32
            | HirTypeKind::I64
            | HirTypeKind::I128
            | HirTypeKind::U8
            | HirTypeKind::U16
            | HirTypeKind::U32
            | HirTypeKind::U64
            | HirTypeKind::U128
            | HirTypeKind::F32
            | HirTypeKind::F64 => true,

            HirTypeKind::Array(inner, _) => self.is_zeroable(inner),

            HirTypeKind::Tuple(elems) => elems.iter().all(|e| self.is_zeroable(e)),

            HirTypeKind::Struct { name, .. } => match self.context.get_struct(name) {
                Some(def) => def.fields.iter().all(|f| self.is_zeroable(&f.field_type)),
                None => false,
            },

            HirTypeKind::Nullable(inner) => match inner.kind {
                // Pointer-shaped: all-zero bits legitimately means "null". Safe to zero-init.
                HirTypeKind::SafePointer { .. }
                | HirTypeKind::UnsafePointer { .. }
                | HirTypeKind::OwnedPointer { .. }
                | HirTypeKind::Ref { .. } => true,
                // Non-pointer nullable (e.g. i32?) needs a discriminant/tag, not just zero
                // bits
                // We could probably add some optimizations like `NonZero<u32>` like in Rust, but for now, this is good enough.
                _ => false,
            },

            // Impermissible: bool, char, string, enums, interfaces/dyn, lambdas,
            // pointers/refs, nullable, slices, generics, void/null/this/unknown.
            // Zeroing these either produces an invalid bit pattern (bool/char/enum
            // discriminants), a dangling/null reference where one shouldn't
            // silently appear (pointers), or is simply meaningless (lambda, dyn).
            _ => false,
        }
    }
}
