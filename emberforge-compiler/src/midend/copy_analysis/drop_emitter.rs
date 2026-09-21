use crate::midend::copy_analysis::drop_glue::DropGlueRegistry;
use crate::midend::copy_analysis::drop_tracking::DropMoveState;
use crate::midend::ir::block_data::CurrentBlockData;
use codex_dependency_graph::DepGraph;
use ir::hir::{self, DropKind, HirEnum, HirStruct, HirType, ProvenanceAnnotation, StrId};
use ir::hir_utils::type_suffix_with_pool;
use ir::ir_conversion::lower_type_hir;
use ir::ir_hasher::HashMap;
use ir::layout::{Layout, TargetInfo, layout_of_ssa, sizeof_ssa};
use ir::span::SourceSpan;
use ir::ssa_ir::{
    AllocatorKind, BinOp, BlockId, Instruction, IntrinsicOp, Operand, SsaType, Value, cast_kind,
};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::Arc;
use zetaruntime::intern_fmt;
use zetaruntime::string_pool::StringPool;

pub trait AllocatorResolver {
    fn resolve_root(&mut self, root: &hir::ProvenanceRoot) -> Value;
    fn lower_global_ref(&mut self, _module_idx: usize, _name: StrId) -> StrId {
        panic!("lower_global_ref not supported by this allocator resolver");
    }
}

pub struct FnAllocatorResolver<'r> {
    pub var_map: &'r HashMap<StrId, Value>,
    pub context: Arc<StringPool>,
    pub dep_graph: &'r RefCell<DepGraph>,
}

impl<'r> AllocatorResolver for FnAllocatorResolver<'r> {
    fn resolve_root(&mut self, root: &hir::ProvenanceRoot) -> Value {
        match root {
            hir::ProvenanceRoot::Var(name) => *self.var_map.get(name).unwrap_or_else(|| {
                panic!(
                    "resolve_allocator_value: allocator root `{:?}` not bound",
                    name
                )
            }),
            hir::ProvenanceRoot::ThisRoot => {
                let this_name = StrId::from_static("this");
                *self.var_map.get(&this_name).unwrap_or_else(|| {
                    panic!(
                        "resolve_allocator_value: `this` not bound but allocator root is ThisRoot"
                    )
                })
            }
            hir::ProvenanceRoot::Global { .. } => {
                unreachable!(
                    "Global root is handled directly by DropEmitter::resolve_allocator_value"
                )
            }
            hir::ProvenanceRoot::ImplicitParam(_) => {
                unreachable!("resolved by monomorphization before MIR")
            }
        }
    }

    fn lower_global_ref(&mut self, module_idx: usize, name: StrId) -> StrId {
        let pkg = self.dep_graph.borrow().get_module_package(module_idx);
        let segments: Vec<StrId> = match pkg {
            Some(pkg) => {
                let pkg_str = self.context.resolve_string(&pkg);
                pkg_str
                    .split("::")
                    .map(|s| StrId(self.context.intern(s)))
                    .collect()
            }
            None => Vec::new(),
        };
        crate::optimized_string_buffering::build_module_scoped_name(
            &segments,
            name,
            None,
            self.context.clone(),
        )
    }
}

struct ThisRebased<'r> {
    this_val: Value,
    inner: &'r mut dyn AllocatorResolver,
}

impl<'r> AllocatorResolver for ThisRebased<'r> {
    fn resolve_root(&mut self, root: &hir::ProvenanceRoot) -> Value {
        match root {
            hir::ProvenanceRoot::ThisRoot => self.this_val,
            other => self.inner.resolve_root(other),
        }
    }

    fn lower_global_ref(&mut self, module_idx: usize, name: StrId) -> StrId {
        self.inner.lower_global_ref(module_idx, name)
    }
}

/// A field that owns the head of a singly-linked chain, e.g.
/// `LinkedList.head: ?^Node<T>` where `Node<T>.next: ?^Node<T>`.
#[derive(Clone, Debug)] // Wouldn't want this to be Copy, since it's quite chonky
pub struct OwnedChain<'a, 'bump> {
    pub head_field: StrId,
    pub head_offset: usize,
    pub head_field_ty: HirType<'a, 'bump>,
    pub node_struct: StrId,
    pub node_ty: HirType<'a, 'bump>,
    pub link_field: StrId,
    pub link_offset: usize,
    pub link_field_ty: HirType<'a, 'bump>,
    pub allocator: ChainAlloc<'bump>,
}

#[derive(Clone, Copy, Debug)]
pub enum ChainAlloc<'bump> {
    /// Type carried `?^Node` with a provenance annotation.
    Annotated(ProvenanceAnnotation<'bump>),
    /// No annotation: use the containing struct's allocator field (`this.<field>`).
    Field(StrId),
}

/// `?^S` / `^S` -> (S as a HirType, S's name, allocator annotation)
pub fn owned_link_parts<'a, 'bump>(
    ty: &HirType<'a, 'bump>,
) -> Option<(
    HirType<'a, 'bump>,
    StrId,
    Option<ProvenanceAnnotation<'bump>>,
)> {
    let owned: HirType<'a, 'bump> = match ty {
        HirType::Nullable(inner) => **inner,
        other => *other,
    };
    let HirType::OwnedPointer { inner, allocator } = owned else {
        return None;
    };
    let pointee: HirType<'a, 'bump> = *inner;
    let HirType::Struct { name, .. } = pointee else {
        return None;
    };
    Some((pointee, name, allocator))
}

/// The single field of `node` that owns another `node` (`next: ?^Node<T>`).
/// `None` if there isn't exactly one. Two self-links means a tree, which
/// needs a worklist rather than a simple loop, so it is not handled here.
pub fn self_link_of<'a, 'bump>(
    structs: &HashMap<StrId, HirStruct<'a, 'bump>>,
    node: StrId,
) -> Option<(StrId, HirType<'a, 'bump>)> {
    let hir_struct = structs.get(&node)?;
    let mut found: Option<(StrId, HirType<'a, 'bump>)> = None;
    for f in hir_struct.fields.iter() {
        if let Some((_, pointee, _)) = owned_link_parts(&f.field_type) {
            if pointee == node {
                if found.is_some() {
                    return None;
                }
                found = Some((f.name, f.field_type));
            }
        }
    }
    found
}

pub fn find_allocator_field<'a, 'bump>(
    structs: &HashMap<StrId, HirStruct<'a, 'bump>>,
    allocator_kind: &HashMap<StrId, AllocatorKind>,
    struct_name: StrId,
) -> Option<StrId> {
    structs
        .get(&struct_name)?
        .fields
        .iter()
        .find_map(|f| match &f.field_type {
            HirType::Struct { name, .. } if allocator_kind.contains_key(name) => Some(f.name),
            _ => None,
        })
}

pub fn owned_chains_of<'a, 'bump>(
    structs: &HashMap<StrId, HirStruct<'a, 'bump>>,
    field_offsets: &HashMap<StrId, HashMap<StrId, usize>>,
    allocator_kind: &HashMap<StrId, AllocatorKind>,
    struct_name: StrId,
) -> Vec<OwnedChain<'a, 'bump>> {
    let mut out = Vec::new();
    let Some(hir_struct) = structs.get(&struct_name) else {
        return out;
    };
    let Some(offsets) = field_offsets.get(&struct_name) else {
        return out;
    };
    for f in hir_struct.fields.iter() {
        let Some((node_ty, node_struct, ann)) = owned_link_parts(&f.field_type) else {
            continue;
        };
        let allocator = match ann {
            Some(a) => ChainAlloc::Annotated(a),
            None => match find_allocator_field(structs, allocator_kind, struct_name) {
                Some(field) => ChainAlloc::Field(field),
                None => continue,
            },
        };
        let Some((link_field, link_field_ty)) = self_link_of(structs, node_struct) else {
            continue;
        };
        let Some(&head_offset) = offsets.get(&f.name) else {
            continue;
        };
        let Some(&link_offset) = field_offsets
            .get(&node_struct)
            .and_then(|o| o.get(&link_field))
        else {
            continue;
        };
        out.push(OwnedChain {
            head_field: f.name,
            head_offset,
            head_field_ty: f.field_type,
            node_struct,
            node_ty,
            link_field,
            link_offset,
            link_field_ty,
            allocator,
        });
    }
    out
}

// Performant reimplementation of owned_chains_of that is specialized
pub(crate) fn is_struct_owns_chain<'a, 'bump>(
    structs: &HashMap<StrId, HirStruct<'a, 'bump>>,
    field_offsets: &HashMap<StrId, HashMap<StrId, usize>>,
    struct_name: StrId,
) -> bool {
    let Some(hir_struct) = structs.get(&struct_name) else {
        return false;
    };
    let Some(offsets) = field_offsets.get(&struct_name) else {
        return false;
    };
    for f in hir_struct.fields.iter() {
        let Some((_, node_struct, _)) = owned_link_parts(&f.field_type) else {
            continue;
        };
        let Some((link_field, _)) = self_link_of(structs, node_struct) else {
            continue;
        };
        let Some(_) = offsets.get(&f.name) else {
            continue;
        };
        let Some(_) = field_offsets
            .get(&node_struct)
            .and_then(|o| o.get(&link_field))
        else {
            continue;
        };
        return true;
    }
    false
}

pub struct DropEmitter<'x, 'a, 'bump, 'f> {
    pub current_block_data: &'x mut CurrentBlockData<'f>,
    pub context: Arc<StringPool>,
    pub struct_mangled_map: &'x HashMap<StrId, HashMap<StrId, StrId>>,
    pub struct_field_offsets: &'x HashMap<StrId, HashMap<StrId, usize>>,
    pub structs: &'x HashMap<StrId, HirStruct<'a, 'bump>>,
    pub enums: &'x HashMap<StrId, HirEnum<'a, 'bump>>,
    pub allocator_kind: &'x HashMap<StrId, AllocatorKind>,
    pub glue_registry: &'x DropGlueRegistry,
}

impl<'x, 'a, 'bump, 'f> DropEmitter<'x, 'a, 'bump, 'f> {
    pub fn new(
        current_block_data: &'x mut CurrentBlockData<'f>,
        context: Arc<StringPool>,
        struct_mangled_map: &'x HashMap<StrId, HashMap<StrId, StrId>>,
        struct_field_offsets: &'x HashMap<StrId, HashMap<StrId, usize>>,
        structs: &'x HashMap<StrId, HirStruct<'a, 'bump>>,
        enums: &'x HashMap<StrId, HirEnum<'a, 'bump>>,
        allocator_kind: &'x HashMap<StrId, AllocatorKind>,
        glue_registry: &'x DropGlueRegistry,
    ) -> Self {
        Self {
            current_block_data,
            context,
            struct_mangled_map,
            struct_field_offsets,
            structs,
            enums,
            allocator_kind,
            glue_registry,
        }
    }

    pub fn emit(&mut self, instruction: Instruction) {
        self.current_block_data.bb().instructions.push(instruction);
    }

    pub fn resolve_allocator_value<R: AllocatorResolver>(
        &mut self,
        allocator: &ProvenanceAnnotation<'bump>,
        resolver: &mut R,
    ) -> Value {
        self.resolve_allocator_value_named(allocator, resolver).0
    }

    fn resolve_allocator_value_named<R: AllocatorResolver>(
        &mut self,
        allocator: &ProvenanceAnnotation<'bump>,
        resolver: &mut R,
    ) -> (Value, Option<StrId>) {
        let mut base = match &allocator.root {
            hir::ProvenanceRoot::Global { module_idx, name } => {
                let mangled = resolver.lower_global_ref(*module_idx, *name);
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Const {
                    dest,
                    ty: SsaType::I64,
                    value: Operand::GlobalRef(mangled),
                });
                self.current_block_data
                    .value_types
                    .insert(dest, SsaType::I64);
                dest
            }
            root => resolver.resolve_root(root),
        };

        let mut known_name: Option<StrId> = None;
        for seg in allocator.path {
            match seg {
                hir::ProvenancePathSegment::Field(field) => {
                    let (addr, _field_ty, name) = self.field_addr_on_value(base, *field);
                    base = addr;
                    known_name = name;
                }
                hir::ProvenancePathSegment::Deref => {
                    let dest = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest,
                        ptr: Operand::Value(base),
                    });
                    let pointee_ty = match self.current_block_data.value_types.get(&base) {
                        Some(SsaType::Pointer(inner)) => (**inner).clone(),
                        other => {
                            panic!("[resolve_allocator_value] Deref of non-pointer {:?}", other)
                        }
                    };
                    self.current_block_data.value_types.insert(dest, pointee_ty);
                    base = dest;
                    known_name = None;
                }
            }
        }

        (base, known_name)
    }

    pub fn apply_provenance_path(
        &mut self,
        mut base: Value,
        path: &[hir::ProvenancePathSegment],
    ) -> Value {
        for seg in path {
            match seg {
                hir::ProvenancePathSegment::Field(field) => {
                    let (addr, _field_ty, _known_name) = self.field_addr_on_value(base, *field);
                    base = addr;
                }
                hir::ProvenancePathSegment::Deref => {
                    let dest = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest,
                        ptr: Operand::Value(base),
                    });
                    let pointee_ty = match self.current_block_data.value_types.get(&base) {
                        Some(SsaType::Pointer(inner)) => (**inner).clone(),
                        other => panic!("[apply_provenance_path] Deref of non-pointer {:?}", other),
                    };
                    self.current_block_data.value_types.insert(dest, pointee_ty);
                    base = dest;
                }
            }
        }
        base
    }

    pub fn field_addr_on_value(
        &mut self,
        obj_val: Value,
        field: StrId,
    ) -> (Value, SsaType, Option<StrId>) {
        let cls_name = match self.current_block_data.value_types.get(&obj_val) {
            Some(SsaType::User(name, _)) => *name,
            Some(SsaType::Pointer(inner)) => match inner.as_ref() {
                SsaType::User(name, _) => *name,
                other => panic!(
                    "[field_addr_on_value] pointer to non-User type: {:?}",
                    other
                ),
            },
            other => panic!(
                "[field_addr_on_value] could not determine struct type: {:?}",
                other
            ),
        };

        let offsets = self
            .struct_field_offsets
            .get(&cls_name)
            .unwrap_or_else(|| panic!("Unknown struct {} in provenance path", cls_name));
        let offset = *offsets
            .get(&field)
            .unwrap_or_else(|| panic!("Unknown field {} on struct {}", field, cls_name));

        let hir_field = self
            .structs
            .get(&cls_name)
            .unwrap_or_else(|| {
                panic!(
                    "[field_addr_on_value] struct `{}` not found in DropEmitter's struct map \
                     (looking up field `{}`), even though struct_field_offsets has an entry for \
                     it.",
                    cls_name, field
                )
            })
            .fields
            .iter()
            .find(|f| f.name == field)
            .unwrap_or_else(|| {
                panic!(
                    "[field_addr_on_value] struct `{}` has no field `{}`",
                    cls_name, field
                )
            });

        if let HirType::Generic(param) = hir_field.field_type {
            panic!(
                "[field_addr_on_value] field `{}` on struct `{}` has unresolved generic type \
                 parameter `{}`.",
                field, cls_name, param
            );
        }

        let known_name = match &hir_field.field_type {
            HirType::Struct { name, .. }
            | HirType::Enum { name, .. }
            | HirType::DynInterface(name, _) => Some(*name),
            _ => None,
        };

        let field_ty = lower_type_hir(&hir_field.field_type, self.enums);

        let addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: addr,
            base: Operand::Value(obj_val),
            offset,
        });
        self.current_block_data
            .value_types
            .insert(addr, SsaType::Pointer(Box::new(field_ty.clone())));

        (addr, field_ty, known_name)
    }

    pub fn mangled_method_name(&self, struct_name: StrId, method_name: &str) -> StrId {
        let method_id = StrId(self.context.intern(method_name));
        self.struct_mangled_map
            .get(&struct_name)
            .and_then(|m| m.get(&method_id))
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "mangled_method_name: no mangled entry for `{}::{}`",
                    struct_name, method_name
                )
            })
    }

    pub fn monomorphized_method_name(
        &self,
        struct_name: StrId,
        method_name: &str,
        ty: &HirType,
    ) -> StrId {
        let base = self.mangled_method_name(struct_name, method_name);
        let suffix = type_suffix_with_pool(self.context.clone(), ty);
        let (base_s, suffix_s) = (base.as_str(), suffix.as_str());

        // The map may already hold the instantiated name; don't suffix twice.
        if base_s.ends_with(suffix_s) {
            return base;
        }
        // Match monomorphize_function, which joins with '_'.
        StrId(intern_fmt!(self.context, "{}_{}", base_s, suffix_s))
    }

    pub fn struct_name_of_value(&self, v: Value) -> Option<StrId> {
        match self.current_block_data.value_types.get(&v)? {
            SsaType::User(name, _) => Some(*name),
            SsaType::Pointer(inner) => match inner.as_ref() {
                SsaType::User(name, _) => Some(*name),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn emit_owning_free_call(
        &mut self,
        alloc_val: Value,
        alloc_cls_name: StrId,
        pointee_ty: &HirType,
        ptr_val: Value,
    ) {
        let free_fn = self.monomorphized_method_name(alloc_cls_name, "free", pointee_ty);
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(free_fn),
            args: SmallVec::from_slice_copy(&[Operand::Value(alloc_val), Operand::Value(ptr_val)]),
        });
    }

    pub fn emit_free_raw_call(
        &mut self,
        alloc_val: Value,
        alloc_cls_name: StrId,
        pointee_ty: &HirType,
        ptr_val: Value,
    ) {
        let (data_ptr, size_v, align_v) = if let HirType::Slice(inner) = pointee_ty {
            let elem_ssa = lower_type_hir(inner, self.enums);
            let Layout {
                size: elem_size,
                align: elem_align,
            } = layout_of_ssa(&elem_ssa, TargetInfo { ptr_bytes: 8 })
                .expect("[emit_free_raw_call] layout_of_ssa failed on slice element");

            let data_ptr = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: data_ptr,
                base: Operand::Value(ptr_val),
                offset: 0,
            });
            self.current_block_data
                .value_types
                .insert(data_ptr, SsaType::Pointer(Box::new(elem_ssa)));

            let cap_v = self.current_block_data.fresh_value();
            self.emit(Instruction::LoadField {
                dest: cap_v,
                base: Operand::Value(ptr_val),
                offset: 16,
            });
            self.current_block_data
                .value_types
                .insert(cap_v, SsaType::Usize);

            let elem_size_v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: elem_size_v,
                ty: SsaType::Usize,
                value: Operand::ConstInt(elem_size as i64),
            });
            self.current_block_data
                .value_types
                .insert(elem_size_v, SsaType::Usize);

            let total_size_v = self.current_block_data.fresh_value();
            self.emit(Instruction::Binary {
                dest: total_size_v,
                op: BinOp::Mul,
                left: Operand::Value(cap_v),
                right: Operand::Value(elem_size_v),
            });
            self.current_block_data
                .value_types
                .insert(total_size_v, SsaType::Usize);

            let align_v = self.current_block_data.fresh_value();
            self.emit(Instruction::Const {
                dest: align_v,
                ty: SsaType::Usize,
                value: Operand::ConstInt(elem_align as i64),
            });
            self.current_block_data
                .value_types
                .insert(align_v, SsaType::Usize);

            (data_ptr, total_size_v, align_v)
        } else {
            let query_ty = lower_type_hir(pointee_ty, self.enums);
            let size_v = self.current_block_data.fresh_value();
            self.emit(Instruction::Intrinsic {
                dest: Some(size_v),
                op: IntrinsicOp::SizeOf,
                query_ty: Some(query_ty.clone()),
                args: SmallVec::new(),
            });
            self.current_block_data
                .value_types
                .insert(size_v, SsaType::Usize);

            let align_v = self.current_block_data.fresh_value();
            self.emit(Instruction::Intrinsic {
                dest: Some(align_v),
                op: IntrinsicOp::AlignOf,
                query_ty: Some(query_ty),
                args: SmallVec::new(),
            });
            self.current_block_data
                .value_types
                .insert(align_v, SsaType::Usize);

            (ptr_val, size_v, align_v)
        };

        let free_raw_fn = self.mangled_method_name(alloc_cls_name, "free_raw");
        self.emit(Instruction::Call {
            dest: None,
            func: Operand::FunctionRef(free_raw_fn),
            args: SmallVec::from_slice_copy(&[
                Operand::Value(alloc_val),
                Operand::Value(data_ptr),
                Operand::Value(size_v),
                Operand::Value(align_v),
            ]),
        });
    }

    pub fn emit_owned_pointer_drop_with_known_allocator_ty<R: AllocatorResolver>(
        &mut self,
        owner: Option<StrId>,
        pointee: &DropKind<'a, 'bump>,
        pointee_ty: &HirType<'a, 'bump>,
        allocator: &ProvenanceAnnotation<'bump>,
        known_allocator_field_ty: Option<&HirType<'a, 'bump>>,
        ptr_val: Value,
        track_partial_moves: bool,
        drop_state: Option<&DropMoveState<'a, 'bump>>,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        let alloc_val = match (allocator.root, allocator.path, known_allocator_field_ty) {
            (
                hir::ProvenanceRoot::ThisRoot,
                [hir::ProvenancePathSegment::Field(field)],
                Some(field_ty),
            ) => {
                let this_val = resolver.resolve_root(&hir::ProvenanceRoot::ThisRoot);
                self.field_addr_typed(this_val, *field, field_ty).0
            }
            _ => self.resolve_allocator_value(allocator, resolver),
        };
        self.emit_owned_pointer_drop_from_alloc(
            owner,
            pointee,
            pointee_ty,
            allocator,
            alloc_val,
            ptr_val,
            track_partial_moves,
            drop_state,
            resolver,
            span,
        );
    }

    pub fn emit_owned_pointer_drop<R: AllocatorResolver>(
        &mut self,
        owner: Option<StrId>,
        pointee: &DropKind<'a, 'bump>,
        pointee_ty: &HirType<'a, 'bump>,
        allocator: &ProvenanceAnnotation<'bump>,
        ptr_val: Value,
        track_partial_moves: bool,
        drop_state: Option<&DropMoveState<'a, 'bump>>,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        let (alloc_val, known_name) = self.resolve_allocator_value_named(allocator, resolver);
        let alloc_cls_name = known_name
            .or_else(|| self.struct_name_of_value(alloc_val))
            .unwrap_or_else(|| {
                panic!(
                    "emit_owned_pointer_drop: could not determine allocator's struct type \
                     (alloc_val={:?}, resolved SsaType={:?}, allocator={})",
                    alloc_val,
                    self.current_block_data.value_types.get(&alloc_val),
                    allocator,
                )
            });

        let kind = self
            .allocator_kind
            .get(&alloc_cls_name)
            .copied()
            .unwrap_or_else(|| {
                eprintln!(
                    "[alloc-drop] allocator={:?}\n  alloc_val={:?} ty={:?}\n  alloc_cls_name={}",
                    allocator,
                    alloc_val,
                    self.current_block_data.value_types.get(&alloc_val),
                    alloc_cls_name
                );
                unreachable!(
                    "struct `{}` used as an allocator but has no AllocatorKind entry at {span}",
                    alloc_cls_name
                )
            });

        let partial_move = track_partial_moves
            && owner.is_some()
            && drop_state.map_or(false, |ds| ds.has_any_field_moves(owner.unwrap()));

        match pointee {
            DropKind::Type(struct_name) => match (kind, partial_move) {
                (AllocatorKind::Owning, false) => {
                    self.emit_owning_free_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
                (AllocatorKind::Owning, true) | (AllocatorKind::RawOnly, true) => {
                    if let (Some(o), Some(ds)) = (owner, drop_state) {
                        self.emit_partial_struct_field_drops(o, *struct_name, ptr_val, ds);
                    }
                    self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
                (AllocatorKind::RawOnly, false) => {
                    if let Some(glue) = self.glue_registry.glue_name_for(*struct_name) {
                        self.emit(Instruction::Call {
                            dest: None,
                            func: Operand::FunctionRef(glue),
                            args: SmallVec::from_slice_copy(&[Operand::Value(ptr_val)]),
                        });
                    }
                    self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
            },

            DropKind::Slice {
                element,
                element_ty,
            } => {
                self.emit_slice_loop_drop(element, element_ty, ptr_val, resolver, span);
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }

            DropKind::Undroppable => {
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }

            DropKind::OwnedPointer {
                pointee: inner_pointee,
                pointee_ty: inner_pointee_ty,
                allocator: inner_allocator,
            } => {
                let inner_val = if matches!(inner_pointee_ty, HirType::Slice(_)) {
                    ptr_val
                } else {
                    let loaded = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest: loaded,
                        ptr: Operand::Value(ptr_val),
                    });
                    self.current_block_data
                        .value_types
                        .insert(loaded, lower_type_hir(pointee_ty, self.enums));
                    loaded
                };
                self.emit_owned_pointer_drop(
                    owner,
                    inner_pointee,
                    inner_pointee_ty,
                    inner_allocator,
                    inner_val,
                    false,
                    drop_state,
                    resolver,
                    span,
                );
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }
        }
    }

    fn emit_owned_pointer_drop_from_alloc<R: AllocatorResolver>(
        &mut self,
        owner: Option<StrId>,
        pointee: &DropKind<'a, 'bump>,
        pointee_ty: &HirType<'a, 'bump>,
        allocator: &ProvenanceAnnotation<'bump>,
        alloc_val: Value,
        ptr_val: Value,
        track_partial_moves: bool,
        drop_state: Option<&DropMoveState<'a, 'bump>>,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        let alloc_cls_name = self.struct_name_of_value(alloc_val).unwrap_or_else(|| {
            panic!(
                "emit_owned_pointer_drop: could not determine allocator's struct type \
                 (alloc_val={:?}, resolved SsaType={:?}, allocator={})",
                alloc_val,
                self.current_block_data.value_types.get(&alloc_val),
                allocator,
            )
        });

        let kind = self
            .allocator_kind
            .get(&alloc_cls_name)
            .copied()
            .unwrap_or_else(|| {
                eprintln!(
                    "[alloc-drop] allocator={:?}\n  alloc_val={:?} ty={:?}\n  alloc_cls_name={}",
                    allocator,
                    alloc_val,
                    self.current_block_data.value_types.get(&alloc_val),
                    alloc_cls_name
                );
                unreachable!(
                    "struct `{}` used as an allocator but has no AllocatorKind entry at {span}",
                    alloc_cls_name
                )
            });

        let partial_move = track_partial_moves
            && owner.is_some()
            && drop_state.map_or(false, |ds| ds.has_any_field_moves(owner.unwrap()));

        match pointee {
            DropKind::Type(struct_name) => match (kind, partial_move) {
                (AllocatorKind::Owning, false) => {
                    self.emit_owning_free_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
                (AllocatorKind::Owning, true) | (AllocatorKind::RawOnly, true) => {
                    if let (Some(o), Some(ds)) = (owner, drop_state) {
                        self.emit_partial_struct_field_drops(o, *struct_name, ptr_val, ds);
                    }
                    self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
                (AllocatorKind::RawOnly, false) => {
                    if let Some(glue) = self.glue_registry.glue_name_for(*struct_name) {
                        self.emit(Instruction::Call {
                            dest: None,
                            func: Operand::FunctionRef(glue),
                            args: SmallVec::from_slice_copy(&[Operand::Value(ptr_val)]),
                        });
                    }
                    self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
                }
            },

            DropKind::Slice {
                element,
                element_ty,
            } => {
                self.emit_slice_loop_drop(element, element_ty, ptr_val, resolver, span);
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }

            DropKind::Undroppable => {
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }

            DropKind::OwnedPointer {
                pointee: inner_pointee,
                pointee_ty: inner_pointee_ty,
                allocator: inner_allocator,
            } => {
                let inner_val = if matches!(inner_pointee_ty, HirType::Slice(_)) {
                    ptr_val
                } else {
                    let loaded = self.current_block_data.fresh_value();
                    self.emit(Instruction::Load {
                        dest: loaded,
                        ptr: Operand::Value(ptr_val),
                    });
                    self.current_block_data
                        .value_types
                        .insert(loaded, lower_type_hir(pointee_ty, self.enums));
                    loaded
                };
                self.emit_owned_pointer_drop(
                    owner,
                    inner_pointee,
                    inner_pointee_ty,
                    inner_allocator,
                    inner_val,
                    false,
                    drop_state,
                    resolver,
                    span,
                );
                self.emit_free_raw_call(alloc_val, alloc_cls_name, pointee_ty, ptr_val);
            }
        }
    }

    fn field_addr_typed(
        &mut self,
        obj_val: Value,
        field: StrId,
        known_field_ty: &HirType<'a, 'bump>,
    ) -> (Value, SsaType) {
        let cls_name = match self.current_block_data.value_types.get(&obj_val) {
            Some(SsaType::User(name, _)) => *name,
            Some(SsaType::Pointer(inner)) => match inner.as_ref() {
                SsaType::User(name, _) => *name,
                other => panic!("[field_addr_typed] pointer to non-User type: {:?}", other),
            },
            other => panic!(
                "[field_addr_typed] could not determine struct type: {:?}",
                other
            ),
        };

        let offsets = self
            .struct_field_offsets
            .get(&cls_name)
            .unwrap_or_else(|| panic!("Unknown struct {} in provenance path", cls_name));
        let offset = *offsets
            .get(&field)
            .unwrap_or_else(|| panic!("Unknown field {} on struct {}", field, cls_name));

        if let HirType::Generic(param) = known_field_ty {
            panic!(
                "[field_addr_typed] field `{}` on struct `{}` has unresolved generic type \
                 parameter `{}`.",
                field, cls_name, param
            );
        }
        let field_ty = lower_type_hir(known_field_ty, self.enums);

        let addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: addr,
            base: Operand::Value(obj_val),
            offset,
        });
        self.current_block_data
            .value_types
            .insert(addr, SsaType::Pointer(Box::new(field_ty.clone())));

        (addr, field_ty)
    }

    pub fn emit_element_drop<R: AllocatorResolver>(
        &mut self,
        kind: &DropKind<'a, 'bump>,
        elem_addr: Value,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        match kind {
            DropKind::Type(struct_name) => {
                if let Some(glue) = self.glue_registry.glue_name_for(*struct_name) {
                    self.emit(Instruction::Call {
                        dest: None,
                        func: Operand::FunctionRef(glue),
                        args: SmallVec::from_slice_copy(&[Operand::Value(elem_addr)]),
                    });
                }
                self.emit_owned_chain_field_drops(
                    *struct_name,
                    elem_addr,
                    None,
                    None,
                    resolver,
                    span,
                );
            }
            DropKind::OwnedPointer {
                pointee,
                pointee_ty,
                allocator,
            } => {
                let loaded = self.current_block_data.fresh_value();
                self.emit(Instruction::Load {
                    dest: loaded,
                    ptr: Operand::Value(elem_addr),
                });
                self.current_block_data
                    .value_types
                    .insert(loaded, lower_type_hir(pointee_ty, self.enums));
                self.emit_owned_pointer_drop(
                    None, pointee, pointee_ty, allocator, loaded, false, None, resolver, span,
                );
            }
            DropKind::Slice {
                element,
                element_ty,
            } => {
                self.emit_slice_loop_drop(element, element_ty, elem_addr, resolver, span);
            }
            DropKind::Undroppable => {}
        }
    }

    /// Drops every owned chain hanging off `base` (an address / `User` value of
    /// `struct_name`). This is also the entry point `DropGlueBuilder` should call
    /// for each struct, so nested lists and lists inside fields get dropped.
    ///
    /// `owner`/`drop_state` are only used to skip chain heads that were moved out.
    pub fn emit_owned_chain_field_drops<R: AllocatorResolver>(
        &mut self,
        struct_name: StrId,
        base: Value,
        owner: Option<StrId>,
        drop_state: Option<&DropMoveState<'a, 'bump>>,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        let structs = self.structs;
        let offsets = self.struct_field_offsets;
        for chain in owned_chains_of(structs, offsets, self.allocator_kind, struct_name) {
            if let (Some(o), Some(ds)) = (owner, drop_state) {
                if ds.is_field_moved(o, chain.head_field) {
                    continue;
                }
            }
            let mut this_resolver = ThisRebased {
                this_val: base,
                inner: &mut *resolver,
            };
            self.emit_owned_chain_drop(&chain, base, &mut this_resolver, span);
        }
    }

    /// Emits:
    ///
    /// ```text
    /// pre:   alloc = <resolve once>; head = load base.head; jump cond
    /// cond:  cur = phi [pre: head, body: next]
    ///        if cur == null goto after else body
    /// body:  node = cur; next = node.link      // read before free
    ///        drop node's non-link fields
    ///        free_raw(alloc, node)
    ///        jump cond
    /// after: ...
    /// ```
    fn emit_owned_chain_drop<R: AllocatorResolver>(
        &mut self,
        chain: &OwnedChain<'a, 'bump>,
        base: Value,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        let head_ssa = lower_type_hir(&chain.head_field_ty, self.enums);
        if head_ssa.is_tagged_nullable() {
            return; // only pointer-optimized nullables / plain owned pointers
        }
        let link_ssa = lower_type_hir(&chain.link_field_ty, self.enums);
        let node_ssa = lower_type_hir(&chain.node_ty, self.enums);
        let node_ptr_ty = SsaType::Pointer(Box::new(node_ssa));

        // Resolved once, outside the loop: every node came from the same allocator.
        let (alloc_val, known_name) = match &chain.allocator {
            ChainAlloc::Annotated(a) => self.resolve_allocator_value_named(a, resolver),
            ChainAlloc::Field(f) => {
                let (addr, _ty, name) = self.field_addr_on_value(base, *f);
                (addr, name)
            }
        };
        let alloc_cls_name = known_name
            .or_else(|| self.struct_name_of_value(alloc_val))
            .unwrap_or_else(|| {
                panic!(
                    "emit_owned_chain_drop: could not determine allocator's struct type \
                     (alloc_val={:?}, resolved SsaType={:?}, allocator={:?})",
                    alloc_val,
                    self.current_block_data.value_types.get(&alloc_val),
                    chain.allocator,
                )
            });

        let head_addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: head_addr,
            base: Operand::Value(base),
            offset: chain.head_offset,
        });
        self.current_block_data
            .value_types
            .insert(head_addr, SsaType::Pointer(Box::new(head_ssa.clone())));
        let head = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: head,
            ptr: Operand::Value(head_addr),
        });
        self.current_block_data
            .value_types
            .insert(head, head_ssa.clone());

        let pre_bb = self.current_block_data.current_block;
        let cond_bb = self.current_block_data.new_block();
        let body_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();
        self.emit(Instruction::Jump { target: cond_bb });

        // cond
        self.current_block_data.switch_to(cond_bb);
        let cur = self.current_block_data.fresh_value();
        self.current_block_data
            .value_types
            .insert(cur, head_ssa.clone());
        let phi_idx = self.current_block_data.bb().instructions.len();
        self.emit(Instruction::Phi {
            dest: cur,
            incoming: SmallVec::new(),
        });
        self.contribute_phi_edge(cond_bb, phi_idx, pre_bb, head);

        let zero = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: zero,
            ty: node_ptr_ty.clone(),
            value: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(zero, node_ptr_ty.clone());
        let is_null = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: is_null,
            op: BinOp::Eq,
            left: Operand::Value(cur),
            right: Operand::Value(zero),
        });
        self.current_block_data
            .value_types
            .insert(is_null, SsaType::Bool);
        self.emit(Instruction::Branch {
            cond: Operand::Value(is_null),
            then_bb: after_bb,
            else_bb: body_bb,
        });

        // body
        self.current_block_data.switch_to(body_bb);
        let node = self.current_block_data.fresh_value();
        self.emit(Instruction::Cast {
            dest: node,
            value: Operand::Value(cur),
            kind: cast_kind(&node_ptr_ty, &node_ptr_ty),
        });
        self.current_block_data
            .value_types
            .insert(node, node_ptr_ty.clone());

        // next must be read before the node is freed
        let link_addr = self.current_block_data.fresh_value();
        self.emit(Instruction::FieldAddr {
            dest: link_addr,
            base: Operand::Value(node),
            offset: chain.link_offset,
        });
        self.current_block_data
            .value_types
            .insert(link_addr, SsaType::Pointer(Box::new(link_ssa.clone())));
        let next_raw = self.current_block_data.fresh_value();
        self.emit(Instruction::Load {
            dest: next_raw,
            ptr: Operand::Value(link_addr),
        });
        self.current_block_data
            .value_types
            .insert(next_raw, link_ssa.clone());
        let next = if link_ssa != head_ssa {
            let retyped = self.current_block_data.fresh_value();
            self.emit(Instruction::Cast {
                dest: retyped,
                value: Operand::Value(next_raw),
                kind: cast_kind(&head_ssa, &head_ssa),
            });
            self.current_block_data
                .value_types
                .insert(retyped, head_ssa.clone());
            retyped
        } else {
            next_raw
        };

        // drop everything in the node except the link (aliases are Undroppable)
        let structs = self.structs;
        let enums = self.enums;
        if let (Some(node_struct), Some(node_offsets)) = (
            structs.get(&chain.node_struct),
            self.struct_field_offsets.get(&chain.node_struct),
        ) {
            let mut node_resolver = ThisRebased {
                this_val: node,
                inner: &mut *resolver,
            };
            for f in node_struct.fields.iter() {
                if f.name == chain.link_field {
                    continue;
                }
                let kind = f.field_type.drop_kind();
                if !kind.is_droppable() {
                    continue;
                }
                let Some(&off) = node_offsets.get(&f.name) else {
                    continue;
                };
                let fty = lower_type_hir(&f.field_type, enums);
                let faddr = self.current_block_data.fresh_value();
                self.emit(Instruction::FieldAddr {
                    dest: faddr,
                    base: Operand::Value(node),
                    offset: off,
                });
                self.current_block_data
                    .value_types
                    .insert(faddr, SsaType::Pointer(Box::new(fty)));
                self.emit_element_drop(&kind, faddr, &mut node_resolver, span);
            }
        }

        self.emit_free_raw_call(alloc_val, alloc_cls_name, &chain.node_ty, node);

        let body_tail = self.current_block_data.current_block;
        self.contribute_phi_edge(cond_bb, phi_idx, body_tail, next);
        self.emit(Instruction::Jump { target: cond_bb });

        self.current_block_data.switch_to(after_bb);
    }

    pub fn contribute_phi_edge(
        &mut self,
        block_id: BlockId,
        phi_idx: usize,
        from_bb: BlockId,
        val: Value,
    ) {
        let block = self
            .current_block_data
            .func
            .blocks
            .iter_mut()
            .find(|b| b.id == block_id)
            .expect("phi block missing");
        if let Instruction::Phi { incoming, .. } = &mut block.instructions[phi_idx] {
            incoming.push((from_bb, val));
        }
    }

    pub fn emit_slice_loop_drop<R: AllocatorResolver>(
        &mut self,
        element_kind: &DropKind<'a, 'bump>,
        element_ty: &HirType<'a, 'bump>,
        fat_ptr_addr: Value,
        resolver: &mut R,
        span: SourceSpan<'a>,
    ) {
        if !element_kind.is_droppable() {
            return;
        }

        let elem_ssa = lower_type_hir(element_ty, self.enums);
        let elem_size = sizeof_ssa(&elem_ssa, TargetInfo { ptr_bytes: 8 })
            .expect("slice element type has no known size");

        let data_ptr = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: data_ptr,
            base: Operand::Value(fat_ptr_addr),
            offset: 0,
        });
        self.current_block_data
            .value_types
            .insert(data_ptr, SsaType::Pointer(Box::new(elem_ssa.clone())));

        let len_v = self.current_block_data.fresh_value();
        self.emit(Instruction::LoadField {
            dest: len_v,
            base: Operand::Value(fat_ptr_addr),
            offset: 8,
        });
        self.current_block_data
            .value_types
            .insert(len_v, SsaType::Usize);

        let i_init = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: i_init,
            ty: SsaType::Usize,
            value: Operand::ConstInt(0),
        });
        self.current_block_data
            .value_types
            .insert(i_init, SsaType::Usize);

        let elem_size_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: elem_size_v,
            ty: SsaType::Usize,
            value: Operand::ConstInt(elem_size as i64),
        });
        self.current_block_data
            .value_types
            .insert(elem_size_v, SsaType::Usize);

        let pre_loop_bb = self.current_block_data.current_block;
        let cond_bb = self.current_block_data.new_block();
        let body_bb = self.current_block_data.new_block();
        let after_bb = self.current_block_data.new_block();
        self.emit(Instruction::Jump { target: cond_bb });

        self.current_block_data.switch_to(cond_bb);
        let i_phi = self.current_block_data.fresh_value();
        self.current_block_data
            .value_types
            .insert(i_phi, SsaType::Usize);
        let phi_idx = self.current_block_data.bb().instructions.len();
        self.emit(Instruction::Phi {
            dest: i_phi,
            incoming: SmallVec::new(),
        });
        self.contribute_phi_edge(cond_bb, phi_idx, pre_loop_bb, i_init);

        let cond = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: cond,
            op: BinOp::Lt,
            left: Operand::Value(i_phi),
            right: Operand::Value(len_v),
        });
        self.current_block_data
            .value_types
            .insert(cond, SsaType::Bool);
        self.emit(Instruction::Branch {
            cond: Operand::Value(cond),
            then_bb: body_bb,
            else_bb: after_bb,
        });

        self.current_block_data.switch_to(body_bb);
        let byte_off = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: byte_off,
            op: BinOp::Mul,
            left: Operand::Value(i_phi),
            right: Operand::Value(elem_size_v),
        });
        self.current_block_data
            .value_types
            .insert(byte_off, SsaType::Usize);

        let elem_addr = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: elem_addr,
            op: BinOp::Add,
            left: Operand::Value(data_ptr),
            right: Operand::Value(byte_off),
        });
        self.current_block_data
            .value_types
            .insert(elem_addr, SsaType::Pointer(Box::new(elem_ssa)));

        self.emit_element_drop(element_kind, elem_addr, resolver, span);

        let i_next = self.current_block_data.fresh_value();
        self.emit(Instruction::Binary {
            dest: i_next,
            op: BinOp::Add,
            left: Operand::Value(i_phi),
            right: Operand::ConstInt(1),
        });
        self.current_block_data
            .value_types
            .insert(i_next, SsaType::Usize);

        let body_tail = self.current_block_data.current_block;
        self.contribute_phi_edge(cond_bb, phi_idx, body_tail, i_next);
        self.emit(Instruction::Jump { target: cond_bb });

        self.current_block_data.switch_to(after_bb);
    }

    pub fn emit_partial_struct_field_drops(
        &mut self,
        owner: StrId,
        struct_name: StrId,
        base: Value,
        drop_state: &DropMoveState<'a, 'bump>,
    ) {
        let Some(hir_struct) = self.structs.get(&struct_name) else {
            return;
        };
        let Some(offsets) = self.struct_field_offsets.get(&struct_name) else {
            return;
        };

        for field in hir_struct.fields.iter().rev() {
            if drop_state.is_field_moved(owner, field.name) {
                continue;
            }
            let HirType::Struct {
                name: field_struct_name,
                ..
            } = &field.field_type
            else {
                continue;
            };
            let Some(field_glue) = self.glue_registry.glue_name_for(*field_struct_name) else {
                continue;
            };
            let Some(&offset) = offsets.get(&field.name) else {
                continue;
            };
            let field_ptr = self.current_block_data.fresh_value();
            self.current_block_data.value_types.insert(
                field_ptr,
                SsaType::Pointer(Box::new(SsaType::User(*field_struct_name, vec![]))),
            );
            self.emit(Instruction::FieldAddr {
                dest: field_ptr,
                base: Operand::Value(base),
                offset,
            });
            self.emit(Instruction::Call {
                dest: None,
                func: Operand::FunctionRef(field_glue),
                args: SmallVec::from_slice_copy(&[Operand::Value(field_ptr)]),
            });
        }
    }
}
