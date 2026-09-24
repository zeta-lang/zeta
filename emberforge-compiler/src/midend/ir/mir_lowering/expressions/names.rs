use ir::{
    hir::{self, HirExpr, StrId},
    ssa_ir::{Instruction, Operand, SsaType, Value},
};
use zetaruntime::intern_fmt;

use crate::{midend::ir::mir_lowering::FunctionLowerer, optimized_string_buffering};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(super) fn resolve_receiver_target_key(&self, ty: &SsaType) -> Option<StrId> {
        match ty {
            SsaType::User(name, _) => Some(*name),
            SsaType::Enum { name, .. } => Some(*name),
            SsaType::Interface(name) => Some(*name),
            SsaType::Pointer(inner) | SsaType::Owned(inner) => {
                self.resolve_receiver_target_key(inner)
            }
            other => self.builtin_target_key(other),
        }
    }

    pub(super) fn builtin_target_key(&self, ty: &SsaType) -> Option<StrId> {
        let prim = |s: &str| Some(StrId(self.context.thread_local().intern(s)));

        match ty {
            SsaType::I8 => prim("i8"),
            SsaType::I16 => prim("i16"),
            SsaType::I32 => prim("i32"),
            SsaType::I64 => prim("i64"),
            SsaType::I128 => prim("i128"),
            SsaType::U8 => prim("u8"),
            SsaType::U16 => prim("u16"),
            SsaType::U32 => prim("u32"),
            SsaType::U64 => prim("u64"),
            SsaType::U128 => prim("u128"),
            SsaType::Isize => prim("isize"),
            SsaType::Usize => prim("usize"),
            SsaType::F32 => prim("f32"),
            SsaType::F64 => prim("f64"),
            SsaType::Bool => prim("bool"),
            SsaType::String => prim("str"),
            SsaType::Char => prim("char"),
            SsaType::Slice(elem) | SsaType::Array(elem, _) => {
                let elem_key = match elem.as_ref() {
                    SsaType::User(n, _) => Some(*n),
                    other => self.builtin_target_key(other),
                };
                if let Some(ek) = elem_key {
                    let specialized = StrId(intern_fmt!(self.context, "slice_{}", ek));
                    if self.struct_mangled_map.contains_key(&specialized) {
                        return Some(specialized);
                    }
                }
                prim("slice")
            }
            SsaType::Owned(inner) => self.resolve_receiver_target_key(inner),
            _ => None,
        }
    }

    pub(crate) fn try_flatten_module_path(&self, expr: &HirExpr<'a, 'bump>) -> Option<StrId> {
        match expr {
            HirExpr::ModuleAccess(acc) => self.resolve_module_access_callee(acc.path, acc.member),
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::ModuleAccess(acc) = object {
                    Some(self.resolve_module_qualified_name(acc.path, acc.member, Some(*field)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn resolve_module_qualified_name(
        &self,
        path: &[StrId],
        member: StrId,
        extra: Option<StrId>,
    ) -> StrId {
        let bare_name = extra.unwrap_or(member);

        if self.extern_c_names.contains(&bare_name) {
            return bare_name;
        }

        let target_module_idx = self
            .dep_graph
            .borrow()
            .resolve_module_path(path)
            .or_else(|| {
                if path.len() == 1 {
                    self.module_import_aliases
                        .get(&self.module_idx)
                        .and_then(|aliases| aliases.get(&path[0]))
                        .copied()
                } else {
                    None
                }
            });

        match extra {
            Some(method_name) => {
                let mut segments: Vec<StrId> = Vec::with_capacity(path.len() + 1);
                segments.push(member);
                segments.extend_from_slice(path);
                optimized_string_buffering::build_module_scoped_name(
                    &segments,
                    method_name,
                    None,
                    self.context.clone(),
                )
            }
            None => {
                let Some(target_idx) = target_module_idx else {
                    return optimized_string_buffering::build_module_scoped_name(
                        path,
                        member,
                        None,
                        self.context.clone(),
                    );
                };
                let real_idx = self
                    .dep_graph
                    .borrow()
                    .canonical_member_module(target_idx, member);
                let Some(pkg) = self.dep_graph.borrow().get_module_package(real_idx) else {
                    return member;
                };
                let pkg_str = pkg.to_string();
                let segments: Vec<StrId> = pkg_str
                    .split("::")
                    .map(|s| StrId(self.context.thread_local().intern(s)))
                    .collect();
                optimized_string_buffering::build_module_scoped_name(
                    &segments,
                    member,
                    None,
                    self.context.clone(),
                )
            }
        }
    }

    pub(crate) fn resolve_static_receiver_struct_name(
        &self,
        bare_name: StrId,
        field: StrId,
    ) -> Option<StrId> {
        if let Some(type_module_idx) = self
            .module_named_imports
            .get(&self.module_idx)
            .and_then(|named| named.get(&bare_name))
            .copied()
        {
            let mangled =
                self.dep_graph
                    .borrow()
                    .mangle_type_name(type_module_idx, bare_name, &self.context);
            if let Some(mmap) = self.struct_mangled_map.get(&mangled) {
                if mmap.contains_key(&field) {
                    return Some(mangled);
                }
            }
        }

        let pkg = self.dep_graph.borrow().get_module_package(self.module_idx);
        let bare_str = self.context.resolve_string(&bare_name);
        let candidate = pkg.map(|p| {
            let pkg_str = self.context.resolve_string(&p).replace("::", "_");
            StrId(intern_fmt!(self.context, "{}_{}", pkg_str, bare_str))
        });

        if let Some(cand) = candidate {
            if let Some(mmap) = self.struct_mangled_map.get(&cand) {
                if mmap.contains_key(&field) {
                    return Some(cand);
                }
            }
        }

        None
    }

    pub(crate) fn lower_module_access_expr(
        &mut self,
        hir_module_access: &hir::HirModuleAccess<'a, 'bump>,
    ) -> Value {
        if let Some(v) =
            self.try_lower_bare_enum_variant(&hir_module_access.member, &hir_module_access.member)
        {
            return v;
        }
        for path_seg in hir_module_access.path.iter().rev() {
            if let Some(v) = self.try_lower_bare_enum_variant(path_seg, &hir_module_access.member) {
                return v;
            }
        }
        let mangled = {
            let dg = self.dep_graph.borrow();
            match dg.resolve_module_path(hir_module_access.path) {
                Some(module_idx) => {
                    let real_idx = dg.canonical_member_module(module_idx, hir_module_access.member);
                    match dg.get_module_package(real_idx) {
                        Some(pkg) => {
                            let pkg_str = pkg.to_string();
                            let segments: Vec<StrId> = pkg_str
                                .split("::")
                                .map(|s| StrId(self.context.thread_local().intern(s)))
                                .collect();
                            optimized_string_buffering::build_module_scoped_name(
                                &segments,
                                hir_module_access.member,
                                None,
                                self.context.clone(),
                            )
                        }
                        None => optimized_string_buffering::build_module_scoped_name(
                            hir_module_access.path,
                            hir_module_access.member,
                            None,
                            self.context.clone(),
                        ),
                    }
                }
                None => optimized_string_buffering::build_module_scoped_name(
                    hir_module_access.path,
                    hir_module_access.member,
                    None,
                    self.context.clone(),
                ),
            }
        };

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest,
            ty: SsaType::I64, // TODO: this is a placeholder; refined once type info flows through
            value: Operand::GlobalRef(mangled),
        });
        self.current_block_data
            .value_types
            .insert(dest, SsaType::I64);
        dest
    }

    pub(crate) fn resolve_module_access_callee(
        &self,
        path: &[StrId],
        member: StrId,
    ) -> Option<StrId> {
        let module_target = self
            .dep_graph
            .borrow()
            .resolve_module_path(path)
            .or_else(|| {
                if path.len() == 1 {
                    self.module_import_aliases
                        .get(&self.module_idx)
                        .and_then(|aliases| aliases.get(&path[0]))
                        .copied()
                } else {
                    None
                }
            });
        if let Some(target_idx) = module_target {
            if self.extern_c_names.contains(&member) {
                return Some(member);
            }
            let real_idx = self
                .dep_graph
                .borrow()
                .canonical_member_module(target_idx, member);
            return Some(self.dep_graph.borrow().mangle_free_function(
                real_idx,
                member,
                false,
                &self.context,
            ));
        }

        if path.len() == 1 {
            let type_module_idx = self
                .module_named_imports
                .get(&self.module_idx)
                .and_then(|named| named.get(&path[0]))
                .copied()
                .unwrap_or(self.module_idx);

            let mangled_type =
                self.dep_graph
                    .borrow()
                    .mangle_type_name(type_module_idx, path[0], &self.context);

            if let Some(mangled_method) = self
                .struct_mangled_map
                .get(&mangled_type)
                .and_then(|methods| methods.get(&member))
            {
                return Some(*mangled_method);
            }
        }

        if self.extern_c_names.contains(&member) {
            return Some(member);
        }

        None
    }
}
