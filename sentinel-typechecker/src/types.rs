use ir::{
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{HirExpr, HirType},
};

use crate::{
    naming::type_to_string,
    str_id_to_string,
    type_checker::{LocalSymbolId, SymbolId},
    TypeChecker,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn nullable_equality_compatible(
        &self,
        left: &HirType<'a, 'bump>,
        right: &HirType<'a, 'bump>,
    ) -> bool {
        let unwrap_nullable = |t: &HirType<'a, 'bump>| match t {
            HirType::Nullable(inner) => Some(**inner),
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
            (Some(li), None) => payload_compatible(&li, right) || matches!(right, HirType::Null),
            (None, Some(ri)) => payload_compatible(left, &ri) || matches!(left, HirType::Null),
            (None, None) => false,
        }
    }

    pub fn types_structurally_equal(&self, a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>) -> bool {
        use HirType::*;
        if matches!(a, Generic(_)) || matches!(b, Generic(_)) {
            return true;
        }
        match (a, b) {
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
                    allocator: aa,
                },
                OwnedPointer {
                    inner: ib,
                    allocator: ab,
                },
            ) => aa == ab && self.types_structurally_equal(ia, ib),
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

    pub fn types_compatible(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> TypeCheckResult<'a, ()> {
        if matches!(found, HirType::Never) {
            return Ok(());
        }

        if let (
            HirType::Ref {
                inner: ei,
                ref_kind: erk,
                ..
            },
            HirType::Ref {
                inner: fi,
                ref_kind: frk,
                ..
            },
        ) = (expected, found)
        {
            if self.types_structurally_equal(ei, fi) && frk.coerces_to(*erk) {
                return Ok(());
            }
        }

        if self.types_structurally_equal(expected, found) {
            return Ok(());
        }

        if *expected == HirType::Unknown || *found == HirType::Unknown {
            return Ok(());
        }

        if self.struct_satisfies_interface_type(expected, found)
            || self.struct_satisfies_interface_type(found, expected)
        {
            return Ok(());
        }

        if let HirType::Nullable(_) = expected {
            if found == &HirType::Null {
                return Ok(());
            }
        }

        Err(TypeErrorKind::TypeMismatch {
            expected: type_to_string(expected),
            found: type_to_string(found),
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

        let interface_name = match expected_inner {
            HirType::DynInterface(name, _) => Some(name.to_string()),
            HirType::Dyn { bounds } => bounds.iter().find_map(|b| match b {
                HirType::DynInterface(name, _) => Some(name.to_string()),
                HirType::Struct { name, .. } => {
                    let name_str = name.to_string();
                    if self.context.get_interface(&name_str).is_some() {
                        Some(name_str)
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

        let struct_name = match found_inner {
            HirType::Struct { name, .. } => name.to_string(),
            _ => return false,
        };

        self.context
            .struct_implements(&struct_name, &interface_name)
    }

    pub fn strip_ref<'x>(ty: &'x HirType<'a, 'bump>) -> &'x HirType<'a, 'bump> {
        match ty {
            HirType::Ref { inner, .. } => inner,
            HirType::SafePointer { inner, .. } => inner,
            HirType::OwnedPointer { inner, .. } => inner,
            _ => ty,
        }
    }

    pub fn is_numeric(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::F32
                | HirType::F64
                | HirType::I128
                | HirType::U128
                | HirType::Usize
                | HirType::Isize
        )
    }

    pub fn is_integer(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::I128
                | HirType::U128
                | HirType::Usize
                | HirType::Isize
        )
    }

    pub fn is_comparable(&self, ty: &HirType<'a, 'bump>) -> bool {
        self.is_numeric(ty) || matches!(ty, HirType::Boolean | HirType::String)
    }

    pub fn peek_type(&self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, _) => {
                let var_name = str_id_to_string(*name);
                self.context
                    .get_variable(&var_name)
                    .unwrap_or((SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::Unknown))
                    .1
            }
            HirExpr::This { .. } => {
                self.context
                    .get_variable("this")
                    .unwrap_or((SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::This))
                    .1
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                match self.peek_type(object) {
                    HirType::Struct {
                        name: struct_name,
                        type_args,
                        ..
                    } => {
                        let struct_name_str = str_id_to_string(struct_name);
                        let field_name = str_id_to_string(*field);
                        let Some(struct_def) = self.context.get_struct(&struct_name_str) else {
                            return HirType::Unknown;
                        };
                        let Some(field_idx) = struct_def
                            .fields
                            .iter()
                            .position(|f| str_id_to_string(f.name) == field_name)
                        else {
                            return HirType::Unknown;
                        };
                        if type_args.is_empty() {
                            struct_def.fields[field_idx].field_type
                        } else {
                            self.instantiate_struct(struct_name, type_args)
                                .map(|fields| fields[field_idx])
                                .unwrap_or(struct_def.fields[field_idx].field_type)
                        }
                    }
                    _ => HirType::Unknown,
                }
            }
            HirExpr::Deref { expr, .. } => match self.peek_type(expr) {
                HirType::Ref { inner, .. } => *inner,
                HirType::SafePointer { inner, .. } => *inner,
                HirType::UnsafePointer { inner, .. } => *inner,
                HirType::OwnedPointer { inner, .. } => *inner,
                _ => HirType::Unknown,
            },
            HirExpr::Index { object, .. } => match self.peek_type(object) {
                HirType::Array(inner, _) => *inner,
                HirType::Slice(inner) => *inner,
                _ => HirType::Unknown,
            },
            HirExpr::Cast { target_type, .. } => *target_type,
            _ => HirType::Unknown,
        }
    }

    pub fn is_zeroable(&self, ty: &HirType<'a, 'bump>) -> bool {
        match ty {
            HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::F32
            | HirType::F64 => true,

            HirType::Array(inner, _) => self.is_zeroable(inner),

            HirType::Tuple(elems) => elems.iter().all(|e| self.is_zeroable(e)),

            HirType::Struct { name, .. } => {
                let name_str = str_id_to_string(*name);
                match self.context.get_struct(&name_str) {
                    Some(def) => def.fields.iter().all(|f| self.is_zeroable(&f.field_type)),
                    None => false,
                }
            }

            HirType::Nullable(inner) => match **inner {
                // Pointer-shaped: all-zero bits legitimately means "null". Safe to zero-init.
                HirType::SafePointer { .. }
                | HirType::UnsafePointer { .. }
                | HirType::OwnedPointer { .. }
                | HirType::Ref { .. } => true,
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
