use ir::{
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirType, RefKind, StrId},
    ir_hasher::{FxHashMap, HashSet},
};

use crate::{
    TypeChecker,
    closures::{capture_mode_text, required_for_capture},
    naming::type_to_string,
    str_id_to_string,
    type_checker::ImplCondition,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutoTrait {
    Send,
    Sync,
}

impl AutoTrait {
    pub fn name(self) -> &'static str {
        match self {
            AutoTrait::Send => "Send",
            AutoTrait::Sync => "Sync",
        }
    }

    /// Accepts bare or path-qualified names (`Send`, `traits::Send`).
    /// TODO: replace this with proper interface lookups. this is a hack.
    pub fn from_interface_name(name: &str) -> Option<Self> {
        let name = name.trim();
        let name = name.strip_prefix("interface ").unwrap_or(name);
        match name.rsplit("::").next()? {
            "Send" => Some(AutoTrait::Send),
            "Sync" => Some(AutoTrait::Sync),
            _ => None,
        }
    }
}

/// Interfaces that may (and must) be implemented with `unsafe impl`.
pub fn is_unsafe_marker_interface(name: &str) -> bool {
    AutoTrait::from_interface_name(name).is_some()
}

fn last_segment(name: &str) -> &str {
    let name = name
        .trim()
        .strip_prefix("interface ")
        .unwrap_or(name.trim());
    name.rsplit("::").next().unwrap_or(name)
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn implements_auto(&mut self, ty: &HirType<'a, 'bump>, tr: AutoTrait) -> bool {
        let mut visiting: HashSet<(StrId, AutoTrait)> = HashSet::default();
        self.auto_in(ty, tr, &FxHashMap::default(), &mut visiting)
    }

    fn auto_in(
        &mut self,
        ty: &HirType<'a, 'bump>,
        tr: AutoTrait,
        env: &FxHashMap<StrId, HirType<'a, 'bump>>,
        visiting: &mut HashSet<(StrId, AutoTrait)>,
    ) -> bool {
        match ty {
            HirType::Ref {
                inner, ref_kind, ..
            } => match ref_kind {
                // &T: Send <-> T: Sync, and &T: Sync <-> T: Sync
                RefKind::Shared => self.auto_in(inner, AutoTrait::Sync, env, visiting),
                // &mut T: Send <-> T: Send, &mut T: Sync <-> T: Sync
                RefKind::Unique => self.auto_in(inner, tr, env, visiting),
                // &alias T needs Send + Sync for either trait: invariant of `&alias`.
                RefKind::Alias => {
                    self.auto_in(inner, AutoTrait::Send, env, visiting)
                        && self.auto_in(inner, AutoTrait::Sync, env, visiting)
                }
            },

            // Raw pointers opt in through `unsafe impl` on the wrapping type.
            HirType::SafePointer { .. } | HirType::UnsafePointer { .. } => false,

            // ^T: Send <-> T: Send, ^T: Sync <-> T: Sync.
            // TODO: allocator rules (the allocator provenance must also be `tr`).
            HirType::OwnedPointer { inner, .. } => self.auto_in(inner, tr, env, visiting),

            HirType::Nullable(inner) | HirType::Array(inner, _) | HirType::Slice(inner) => {
                self.auto_in(inner, tr, env, visiting)
            }

            HirType::Tuple(elems) => {
                for e in elems.iter() {
                    if !self.auto_in(e, tr, env, visiting) {
                        return false;
                    }
                }
                true
            }

            HirType::Struct {
                name, type_args, ..
            } => self.auto_adt(*name, type_args, tr, env, visiting),

            HirType::Enum {
                name, type_args, ..
            } => self.auto_adt(*name, type_args, tr, env, visiting),

            HirType::Generic(name) => match env.get(name).copied() {
                Some(t) => self.auto_in(&t, tr, &FxHashMap::default(), visiting),
                None => self.generic_has_bound(*name, tr),
            },

            HirType::Dyn { .. } => {
                let s = type_to_string(ty);
                s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .any(|w| w == tr.name())
            }
            HirType::DynInterface(name, _) => str_id_to_string(*name) == tr.name(),

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
            | HirType::Usize
            | HirType::Isize
            | HirType::F32
            | HirType::F64
            | HirType::Boolean
            | HirType::Char => tr == AutoTrait::Send,

            _ => true,
        }
    }

    fn auto_adt(
        &mut self,
        name: StrId,
        type_args: &[HirType<'a, 'bump>],
        tr: AutoTrait,
        env: &FxHashMap<StrId, HirType<'a, 'bump>>,
        visiting: &mut HashSet<(StrId, AutoTrait)>,
    ) -> bool {
        let name_str = str_id_to_string(name);

        // Explicit `unsafe impl X by Send/Sync` replaces structural derivation (as in Rust):
        // if its bounds don't hold the type is simply not Send/Sync.
        if self
            .context
            .struct_interfaces
            .get(&name_str)
            .is_some_and(|s| s.contains(tr.name()))
        {
            return self.explicit_impl_holds(name, &name_str, type_args, tr, env, visiting);
        }

        // Coinductive: a recursive type (`struct Node { next: ^Node }`) is Send if its
        // other fields are.
        if !visiting.insert((name, tr)) {
            return true;
        }

        let (generic_names, fields): (Vec<StrId>, Vec<HirType<'a, 'bump>>) =
            if let Some(def) = self.context.get_struct(&name_str) {
                (
                    def.generics.unwrap_or(&[]).iter().map(|g| g.name).collect(),
                    def.fields.iter().map(|f| f.field_type).collect(),
                )
            } else if let Some(def) = self.context.get_enum(&name_str) {
                (
                    def.generics.unwrap_or(&[]).iter().map(|g| g.name).collect(),
                    def.variants
                        .iter()
                        .flat_map(|v| v.fields.iter())
                        .map(|f| f.field_type)
                        .collect(),
                )
            } else {
                visiting.remove(&(name, tr));
                return true;
            };

        let mut new_env: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
        for (g, a) in generic_names.iter().zip(type_args.iter()) {
            let resolved = match a {
                HirType::Generic(n) => env.get(n).copied().unwrap_or(*a),
                other => *other,
            };
            new_env.insert(*g, resolved);
        }

        let mut ok = true;
        for f in fields.iter() {
            if !self.auto_in(f, tr, &new_env, visiting) {
                ok = false;
                break;
            }
        }
        visiting.remove(&(name, tr));
        ok
    }

    /// Does any registered `unsafe impl ... by tr` for `name` apply to these type arguments?
    fn explicit_impl_holds(
        &mut self,
        name: StrId,
        name_str: &str,
        type_args: &[HirType<'a, 'bump>],
        tr: AutoTrait,
        env: &FxHashMap<StrId, HirType<'a, 'bump>>,
        visiting: &mut HashSet<(StrId, AutoTrait)>,
    ) -> bool {
        let conds = self.context.conditional_impls_for(name_str, tr.name());
        if conds.is_empty() {
            return true; // registered without conditions
        }
        if !visiting.insert((name, tr)) {
            return true;
        }

        let actual: Vec<HirType<'a, 'bump>> = type_args
            .iter()
            .map(|a| match a {
                HirType::Generic(n) => env.get(n).copied().unwrap_or(*a),
                other => *other,
            })
            .collect();

        let mut holds = false;
        for c in conds.iter() {
            if self.impl_condition_holds(c, &actual, visiting) {
                holds = true;
                break;
            }
        }
        visiting.remove(&(name, tr));
        holds
    }

    fn impl_condition_holds(
        &mut self,
        c: &ImplCondition<'a, 'bump>,
        actual: &[HirType<'a, 'bump>],
        visiting: &mut HashSet<(StrId, AutoTrait)>,
    ) -> bool {
        // Bind the impl's generics to the concrete type arguments.
        let mut binding: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();

        if c.target_args.is_empty() {
            // Target args weren't lowered: assume impl generics line up with the struct's.
            for (i, (gname, _)) in c.generics.iter().enumerate() {
                if let Some(act) = actual.get(i) {
                    binding.insert(*gname, *act);
                }
            }
        } else {
            for (i, formal) in c.target_args.iter().enumerate() {
                let Some(act) = actual.get(i) else { continue };
                match formal {
                    HirType::Generic(g) if c.generics.iter().any(|(n, _)| n == g) => {
                        binding.entry(*g).or_insert(*act);
                    }
                    // Specialised impl (`unsafe impl Wrapper<i32> by Send`): must match exactly.
                    other => {
                        if !matches!(act, HirType::Unknown)
                            && !self.types_structurally_equal(other, act)
                        {
                            return false;
                        }
                    }
                }
            }
        }

        // Evaluate the bounds on each impl generic against what it was bound to.
        for (gname, bounds) in c.generics.iter() {
            let Some(act) = binding.get(gname).copied() else {
                continue;
            };
            for b in bounds {
                let ok = match AutoTrait::from_interface_name(b) {
                    Some(t) => self.auto_in(&act, t, &FxHashMap::default(), visiting),
                    None => self.satisfies_interface_bound(&act, b),
                };
                if !ok {
                    return false;
                }
            }
        }
        true
    }

    /// Non-auto bounds on an impl generic (`unsafe impl<T: Clone + Send> ...`). Lenient:
    /// only nominal types and bounded generics are checked.
    fn satisfies_interface_bound(&self, ty: &HirType<'a, 'bump>, bound: &str) -> bool {
        let short = last_segment(bound);
        match ty {
            HirType::Struct { name, .. } | HirType::Enum { name, .. } => {
                let n = str_id_to_string(*name);
                self.context.struct_implements(&n, bound)
                    || self.context.struct_implements(&n, short)
            }
            HirType::Generic(n) => self
                .current_fn_bounds
                .get(n)
                .is_some_and(|bs| bs.iter().any(|b| last_segment(b) == short)),
            _ => true,
        }
    }

    fn generic_has_bound(&self, name: StrId, tr: AutoTrait) -> bool {
        self.current_fn_bounds.get(&name).is_some_and(|bounds| {
            bounds
                .iter()
                .any(|b| AutoTrait::from_interface_name(b) == Some(tr))
        })
    }

    /// `impl X by Send` must be `unsafe impl`, and `unsafe impl` is only for unsafe interfaces.
    pub fn check_unsafe_marker_impl(&mut self, target: &str, interface: &str, is_unsafe: bool) {
        let marker = is_unsafe_marker_interface(interface);
        let iface = last_segment(interface);
        match (marker, is_unsafe) {
            (true, false) => self.record(TypeErrorKind::Generic(format!(
                "implementing `{iface}` for `{target}` is unsafe; write `unsafe impl {target} by {iface}`"
            ))),
            (false, true) => self.record(TypeErrorKind::Generic(format!(
                "`unsafe impl` is only allowed for unsafe interfaces (`Send`, `Sync`); \
                 `{iface}` is safe to implement"
            ))),
            _ => {}
        }
    }

    /// Registers an `impl ... by Interface` (replaces the bare `add_struct_interface` call in
    /// `register_module`). `generics` are the impl's own generic params with their bound
    /// names; `target_args` are the impl target's type arguments.
    pub fn register_interface_impl(
        &mut self,
        target: &str,
        interface: &str,
        is_unsafe: bool,
        generics: impl Iterator<Item = (StrId, Vec<String>)>,
        target_args: &[HirType<'a, 'bump>],
    ) {
        self.check_unsafe_marker_impl(target, interface, is_unsafe);
        self.context
            .add_struct_interface(target, interface.to_string());
        if is_unsafe_marker_interface(interface) {
            self.context.add_conditional_impl(
                target,
                interface,
                ImplCondition {
                    generics: generics.collect(),
                    target_args: target_args.to_vec(),
                },
            );
        }
    }

    pub fn check_closure_auto_bounds(&mut self, lambda: &HirExpr<'a, 'bump>, traits: &[AutoTrait]) {
        let captures = self.infer_closure_captures(lambda);
        for cap in captures {
            for &tr in traits {
                for &need in required_for_capture(cap.mode, tr) {
                    if !self.implements_auto(&cap.ty, need) {
                        self.record(TypeErrorKind::Generic(format!(
                            "closure must be `{}`, but it captures `{}` {} and `{}` is not `{}`",
                            tr.name(),
                            str_id_to_string(cap.name),
                            capture_mode_text(cap.mode),
                            type_to_string(&cap.ty),
                            need.name(),
                        )));
                    }
                }
            }
        }

        if let HirExpr::Lambda { body, span, .. } = lambda {
            if !traits.is_empty() {
                if let Some(uses) = self.closure_static_uses.get(&Self::stmt_key(body)).cloned() {
                    self.check_static_thread_bounds(&uses, *span);
                }
            }
        }
    }

    pub fn check_arg_auto_bounds(&mut self, arg: &HirExpr<'a, 'bump>, traits: &[AutoTrait]) {
        if matches!(arg, HirExpr::Lambda { .. }) {
            self.check_closure_auto_bounds(arg, traits);
            return;
        }
        let ty = self.peek_type(arg);
        // A closure stored in a variable can't be re-analysed here.
        if matches!(ty, HirType::Lambda { .. }) {
            return;
        }
        for &tr in traits {
            if !self.implements_auto(&ty, tr) {
                self.record(TypeErrorKind::Generic(format!(
                    "`{}` cannot be used here: it is not `{}`",
                    type_to_string(&ty),
                    tr.name()
                )));
            }
        }
    }
}
