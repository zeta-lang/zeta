use std::fmt;

use crate::hir::{
    HirEnum, HirFunc, HirInterface, HirParam, HirStruct, HirType, StrId, ThisPassingKind,
};
use crate::ir_conversion::lower_type_hir;
use crate::ir_hasher::HashMap;

impl SsaType {
    pub fn is_stored_inline(&self) -> bool {
        match self {
            SsaType::User(..)
            | SsaType::Enum { .. }
            | SsaType::Tuple(_)
            | SsaType::Array(..)
            | SsaType::Slice(_) => true,
            SsaType::Owned(i) => matches!(i.as_ref(), SsaType::Slice(_)),
            t => t.is_tagged_nullable(),
        }
    }

    pub fn ptr_to(inner: SsaType) -> Self {
        SsaType::Pointer(SsaPointerKind::UnsafeMut, Box::new(inner))
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, SsaType::Pointer(..) | SsaType::Owned(_))
    }

    /// Anything that lives in a pointer-sized register.
    pub fn is_pointer_like(&self) -> bool {
        self.is_pointer() || matches!(self, SsaType::FuncPointer { .. })
    }

    pub fn as_pointer(&self) -> Option<&SsaType> {
        match self {
            SsaType::Pointer(_, inner) | SsaType::Owned(inner) => Some(inner),
            _ => None,
        }
    }

    /// True if this nullable can be represented with zero-cost null-pointer
    /// encoding (no tag byte needed), only applies when the inner type is
    /// itself a pointer.
    pub fn nullable_is_pointer_optimized(&self) -> bool {
        matches!(self, SsaType::Nullable(inner) if inner.is_pointer())
    }

    /// If this is `Nullable(Pointer(_))`, returns the underlying pointer
    /// type directly, the nullable collapses to the pointer with 0 = null.
    pub fn nullable_pointer_repr(&self) -> Option<&SsaType> {
        match self {
            SsaType::Nullable(inner) if inner.is_pointer() => Some(inner),
            _ => None,
        }
    }

    /// True if this is a tagged-union nullable (inner type is NOT a pointer,
    /// so it needs an explicit discriminant byte).
    pub fn is_tagged_nullable(&self) -> bool {
        matches!(self, SsaType::Nullable(inner) if !inner.is_pointer())
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            SsaType::I8
                | SsaType::I16
                | SsaType::I32
                | SsaType::I64
                | SsaType::U8
                | SsaType::U16
                | SsaType::U32
                | SsaType::U64
                | SsaType::Isize
                | SsaType::Usize
                | SsaType::Char
        )
    }

    pub fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            SsaType::I8 | SsaType::I16 | SsaType::I32 | SsaType::I64 | SsaType::Isize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, SsaType::F32 | SsaType::F64)
    }

    pub fn bit_width(&self) -> Option<u16> {
        Some(match self {
            SsaType::Bool => 1,

            SsaType::I8 | SsaType::U8 => 8,
            SsaType::I16 | SsaType::U16 => 16,
            SsaType::I32 | SsaType::U32 | SsaType::F32 => 32,
            SsaType::I64 | SsaType::U64 | SsaType::F64 => 64,
            SsaType::I128 | SsaType::U128 => 128,

            SsaType::Isize | SsaType::Usize => 64, // target for now

            SsaType::Char => 32,

            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operand {
    Value(Value),
    ConstInt(i64),
    ConstFloat(f64),
    ConstBool(bool),
    ConstString(StrId),
    FunctionRef(StrId),
    GlobalRef(StrId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicOp {
    SizeOf,
    AlignOf,
    AssertAlign,
    TypeName,
    CpuRelax,
    AtomicLoad {
        ordering: AtomicOrdering,
    },
    AtomicStore {
        ordering: AtomicOrdering,
    },
    AtomicSwap {
        ordering: AtomicOrdering,
    },
    AtomicCas {
        success: AtomicOrdering,
        failure: AtomicOrdering,
    },
    AtomicFetchAdd {
        ordering: AtomicOrdering,
    },
    AtomicFetchSub {
        ordering: AtomicOrdering,
    },
    AtomicFetchAnd {
        ordering: AtomicOrdering,
    },
    AtomicFetchOr {
        ordering: AtomicOrdering,
    },
    AtomicFetchXor {
        ordering: AtomicOrdering,
    },
    AtomicFence {
        ordering: AtomicOrdering,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    F32,
    F64,
    Isize, // Signed pointer-sized integer
    Usize, // Unsigned pointer-sized integer
    Null,

    // Special types
    Bool,
    String, // Fat pointer to string data
    Void,   // Unit type, zero size

    // Composite types
    User(StrId, StrId, Vec<SsaType>), // mangled_name, unmangled_name, fields
    Interface(StrId, StrId),          // mangled_name, unmangled_name
    Enum {
        name: StrId,
        unmangled_name: StrId,
        variants: Vec<Vec<SsaType>>,
    }, // Tagged union of types
    Tuple(Vec<SsaType>),              // Fixed-size collection of heterogeneous types

    // Pointer types
    Pointer(SsaPointerKind, Box<SsaType>), // Pointer to another type
    Owned(Box<SsaType>),

    // Dynamically sized types
    Dyn,                        // Trait object (fat pointer)
    Slice(Box<SsaType>),        // Slice (fat pointer)
    Array(Box<SsaType>, usize), // Array
    Char,

    Nullable(Box<SsaType>),
    FuncPointer {
        params: Vec<SsaType>,
        return_type: Box<SsaType>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SsaPointerKind {
    SafeConst,
    SafeMut,
    UnsafeConst,
    UnsafeMut,
    RefShared,
    RefMut,
    RefAlias,
}

impl SsaPointerKind {
    pub fn is_mut(&self) -> bool {
        matches!(
            self,
            SsaPointerKind::SafeMut | SsaPointerKind::UnsafeMut | SsaPointerKind::RefMut
        )
    }
}

impl fmt::Display for SsaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaType::I8 => write!(f, "i8"),
            SsaType::U8 => write!(f, "u8"),
            SsaType::I16 => write!(f, "i16"),
            SsaType::U16 => write!(f, "u16"),
            SsaType::I32 => write!(f, "i32"),
            SsaType::U32 => write!(f, "u32"),
            SsaType::I64 => write!(f, "i64"),
            SsaType::U64 => write!(f, "u64"),
            SsaType::I128 => write!(f, "i128"),
            SsaType::U128 => write!(f, "u128"),
            SsaType::F32 => write!(f, "f32"),
            SsaType::F64 => write!(f, "f64"),
            SsaType::Isize => write!(f, "isize"),
            SsaType::Usize => write!(f, "usize"),
            SsaType::Null => write!(f, "null"),
            SsaType::Bool => write!(f, "bool"),
            SsaType::String => write!(f, "str"),
            SsaType::Void => write!(f, "void"),
            SsaType::Char => write!(f, "char"),
            SsaType::Dyn => write!(f, "dyn"),
            SsaType::User(_, unmangled, _) => write!(f, "{}", unmangled),
            SsaType::Interface(_, unmangled) => write!(f, "{}", unmangled),
            SsaType::Enum { unmangled_name, .. } => write!(f, "{}", unmangled_name),
            SsaType::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            SsaType::Pointer(kind, inner) => match kind {
                SsaPointerKind::SafeConst => write!(f, "*const {}", inner),
                SsaPointerKind::SafeMut => write!(f, "*mut {}", inner),
                SsaPointerKind::UnsafeConst => write!(f, "[*]const {}", inner),
                SsaPointerKind::UnsafeMut => write!(f, "[*]mut {}", inner),
                SsaPointerKind::RefShared => write!(f, "&{}", inner),
                SsaPointerKind::RefMut => write!(f, "&mut {}", inner),
                SsaPointerKind::RefAlias => write!(f, "&alias {}", inner),
            },
            SsaType::Owned(inner) => write!(f, "^{}", inner),
            SsaType::Slice(inner) => write!(f, "[]{}", inner),
            SsaType::Array(inner, len) => write!(f, "[{}]{}", len, inner),
            SsaType::Nullable(inner) => write!(f, "{}?", inner),
            SsaType::FuncPointer {
                params,
                return_type,
            } => {
                write!(f, "func(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "): {}", return_type)
            }
        }
    }
}

use crate::ast::FuncModifiers;
use crate::registry::global_registry::StaticDef;
use smallvec::SmallVec;
use zetaruntime::string_pool::StringPool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    // integer
    Truncate,
    SignExtend,
    ZeroExtend,

    // integer <-> float
    SignedIntToFloat,
    UnsignedIntToFloat,
    FloatToSignedInt,
    FloatToUnsignedInt,

    // float
    FloatExtend,
    FloatTruncate,

    // bit reinterpretation
    Bitcast,

    // pointers
    PtrToInt,
    IntToPtr,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Intrinsic {
        dest: Option<Value>,
        op: IntrinsicOp,
        /// The type being queried, for SizeOf/AlignOf/TypeName. None for AssertAlign.
        query_ty: Option<SsaType>,
        /// Runtime args: [ptr, align] for AssertAlign, empty otherwise.
        args: SmallVec<Operand, 4>,
    },

    /// Binary operation: dest = left OP right
    Binary {
        dest: Value,
        op: BinOp,
        left: Operand,
        right: Operand,
    },

    Cast {
        dest: Value,
        value: Operand,
        kind: CastKind,
    },

    Undef {
        dest: Value,
        ty: SsaType,
    },

    /// Unary operation: dest = OP operand
    Unary {
        dest: Value,
        op: UnOp,
        operand: Operand,
    },

    /// Phi node for SSA
    Phi {
        dest: Value,
        incoming: SmallVec<(BlockId, Value), 4>, // usually <=4 predecessors
    },

    /// Function call
    Call {
        dest: Option<Value>,
        func: Operand,
        args: SmallVec<Operand, 8>, // most calls <8 args
    },

    /// Virtual/interface call through vtable
    InterfaceDispatch {
        dest: Option<Value>,
        object: Value,      // interface reference
        method_slot: usize, // vtable slot
        args: SmallVec<Operand, 8>,
    },

    /// Cast struct -> interface
    UpcastToInterface {
        dest: Value,
        object: Value,
        interface_id: usize,
    },

    /// Stack allocation (SSA-local)
    StackAlloc {
        dest: Value,
        ty: SsaType,
        count: usize, // number of elements if array
    },

    StoreField {
        base: Operand,
        offset: usize,
        value: Operand,
    },
    LoadField {
        dest: Value,
        base: Operand,
        offset: usize,
    },

    Jump {
        target: BlockId,
    },

    Branch {
        cond: Operand,
        then_bb: BlockId,
        else_bb: BlockId,
    },

    Ret {
        value: Option<Operand>,
    },
    Const {
        dest: Value,
        ty: SsaType,
        value: Operand,
    },
    AddressOf {
        dest: Value,
        source: Value,
    },
    Load {
        dest: Value,
        ptr: Operand,
    },
    Store {
        ptr: Operand,
        value: Operand,
    },
    FieldAddr {
        dest: Value,
        base: Operand,
        offset: usize,
    },
    GlobalAddr {
        dest: Value,
        name: StrId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationOperand {
    Literal(StrId),
    Value(Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    LogicalNot,
    BitNot,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: StrId,
    pub params: SmallVec<(Value, SsaType), 8>,
    pub ret_type: SsaType,
    pub blocks: SmallVec<BasicBlock, 3>,
    pub value_types: HashMap<Value, SsaType>,
    pub entry: BlockId,
    pub function_metadata: FuncModifiers,
}

impl Function {
    pub fn from_signature(
        hir_fn: &HirFunc,
        structs: &HashMap<StrId, HirStruct>,
        enums: &HashMap<StrId, HirEnum>,
        interfaces: &HashMap<StrId, HirInterface>,
        context: &StringPool,
    ) -> Function {
        let mut params: SmallVec<(Value, SsaType), 8> = SmallVec::new();
        let mut value_types: HashMap<Value, SsaType> = HashMap::default();
        let mut next_value = 0usize;

        if let Some(hir_params) = hir_fn.params {
            for p in hir_params {
                let v = Value(next_value);
                next_value += 1;

                let ty = match p {
                    HirParam::This { kind, .. } => {
                        let inner = match hir_fn.impl_target {
                            Some(target) => {
                                Self::this_inner_type(target, structs, enums, interfaces, context)
                            }
                            None => unreachable!(),
                        };
                        match kind {
                            ThisPassingKind::Move | ThisPassingKind::MoveMut => inner,
                            _ => SsaType::Pointer(
                                crate::ssa_ir::SsaPointerKind::UnsafeMut,
                                Box::new(inner),
                            ),
                        }
                    }
                    HirParam::Normal { param_type, .. } => {
                        lower_type_hir(param_type, enums, structs)
                    }
                };

                value_types.insert(v, ty.clone());
                params.push((v, ty));
            }
        }

        let ret_type = lower_type_hir(
            hir_fn.return_type.as_ref().unwrap_or(&HirType::Void),
            enums,
            structs,
        );

        Function {
            name: hir_fn.name,
            params,
            ret_type,
            blocks: SmallVec::new(),
            value_types,
            entry: BlockId(0),
            function_metadata: hir_fn.function_metadata,
        }
    }

    fn primitive_ssa_type(name: &str) -> Option<SsaType> {
        Some(match name {
            "i8" => SsaType::I8,
            "i16" => SsaType::I16,
            "i32" => SsaType::I32,
            "i64" => SsaType::I64,
            "i128" => SsaType::I128,
            "u8" => SsaType::U8,
            "u16" => SsaType::U16,
            "u32" => SsaType::U32,
            "u64" => SsaType::U64,
            "u128" => SsaType::U128,
            "isize" => SsaType::Isize,
            "usize" => SsaType::Usize,
            "f32" => SsaType::F32,
            "f64" => SsaType::F64,
            "bool" => SsaType::Bool,
            "str" => SsaType::String,
            "char" => SsaType::Char,
            _ => return None,
        })
    }

    fn this_inner_type(
        target: StrId,
        structs: &HashMap<StrId, HirStruct>,
        enums: &HashMap<StrId, HirEnum>,
        interfaces: &HashMap<StrId, HirInterface>,
        context: &StringPool,
    ) -> SsaType {
        let name = target.to_string();

        if name == "slice" {
            return SsaType::Slice(Box::new(SsaType::U8)); // erased element, ptr+len only
        }

        if let Some(elem_name) = name.strip_prefix("slice_") {
            if let Some(elem_ty) = Self::primitive_ssa_type(elem_name) {
                return SsaType::Slice(Box::new(elem_ty));
            }

            let elem_id = StrId(context.intern(elem_name));
            if let Some(elem_struct) = structs.get(&elem_id) {
                let field_types: Vec<SsaType> = elem_struct
                    .fields
                    .iter()
                    .map(|f| lower_type_hir(&f.field_type, enums, structs))
                    .collect();
                return SsaType::Slice(Box::new(SsaType::User(elem_id, elem_id, field_types)));
            }

            return SsaType::Slice(Box::new(SsaType::Void));
        }

        if let Some(ty) = Self::primitive_ssa_type(&name) {
            return ty;
        }

        if interfaces.contains_key(&target) {
            return SsaType::Interface(target, target);
        }

        SsaType::User(target, target, vec![])
    }
}

#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: StrId,
    pub slot: usize, // index in vtable
    pub is_virtual: bool,
}

#[derive(Debug, Clone, Default)]
pub struct StructLayout {
    pub vtable: SmallVec<MethodInfo, 12>,
}

#[derive(Debug, Clone, Default)]
pub struct InterfaceLayout {
    pub methods: SmallVec<StrId, 12>, // names in declaration order
}

#[derive(Debug, Clone)]
pub struct Module<'a, 'bump>
where
    'bump: 'a,
{
    pub functions: HashMap<StrId, Function>,
    pub structs: HashMap<StrId, HirStruct<'a, 'bump>>,
    pub interfaces: HashMap<StrId, HirInterface<'a, 'bump>>,
    pub enums: HashMap<StrId, HirEnum<'a, 'bump>>,
    pub types: HashMap<StrId, SsaType>,

    pub struct_layouts: HashMap<StrId, StructLayout>,
    pub interface_layouts: HashMap<StrId, InterfaceLayout>,
    pub struct_interface_vtables: HashMap<(StrId, StrId), VTableInfo>,
    pub statics: HashMap<StrId, StaticDef>,
}

impl<'a, 'bump> Module<'a, 'bump> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::default(),
            structs: HashMap::default(),
            interfaces: HashMap::default(),
            enums: HashMap::default(),
            types: HashMap::default(),

            struct_layouts: HashMap::default(),
            interface_layouts: HashMap::default(),
            struct_interface_vtables: HashMap::default(),
            statics: HashMap::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    I64,
    Bool,
    Str,
    Enum(StrId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    LogicalAnd,
    LogicalOr,
}

pub fn inst_is_terminator(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Jump { .. } | Instruction::Branch { .. } | Instruction::Ret { .. }
    )
}

#[derive(Debug, Clone, Default)]
pub struct VTable {
    pub struct_name: StrId,
    pub interface: StrId,
    pub methods: Vec<StrId>,
}

#[derive(Debug, Clone, Default)]
pub struct VTableInfo {
    pub interface: StrId,
    pub methods: Vec<StrId>,
}

pub fn cast_kind(src: &SsaType, dst: &SsaType) -> CastKind {
    use CastKind::*;

    if src == dst {
        return Bitcast;
    }

    if src.is_integer() && dst.is_integer() {
        let sb = src.bit_width().unwrap();
        let db = dst.bit_width().unwrap();

        return if db > sb {
            if src.is_signed_integer() {
                SignExtend
            } else {
                ZeroExtend
            }
        } else if db == sb {
            // Same width, different SsaType variant (e.g. Usize <-> U64,
            // Isize <-> I64): no actual narrowing happens at the machine
            // level, so this must be a Bitcast. `ireduce` requires a
            // strictly narrower destination and will fail the Cranelift
            // verifier if src/dst widths are equal.
            Bitcast
        } else {
            Truncate
        };
    }

    if src.is_integer() && dst.is_float() {
        return if src.is_signed_integer() {
            SignedIntToFloat
        } else {
            UnsignedIntToFloat
        };
    }

    if src.is_float() && dst.is_integer() {
        return if dst.is_signed_integer() {
            FloatToSignedInt
        } else {
            FloatToUnsignedInt
        };
    }

    if src.is_float() && dst.is_float() {
        let sb = src.bit_width().unwrap();
        let db = dst.bit_width().unwrap();

        return if db > sb { FloatExtend } else { FloatTruncate };
    }

    if src.is_pointer_like() && dst.is_pointer_like() {
        return Bitcast;
    }
    if src.is_pointer_like() && dst.is_integer() {
        return PtrToInt;
    }
    if src.is_integer() && dst.is_pointer_like() {
        return IntToPtr;
    }

    if (matches!(src, SsaType::Slice(inner) if **inner == SsaType::U8)
        && matches!(dst, SsaType::String))
        || (matches!(src, SsaType::String)
            && matches!(dst, SsaType::Slice(inner) if **inner == SsaType::U8))
    {
        return Bitcast;
    }

    panic!("unsupported cast {:?} -> {:?}", src, dst);
}

#[derive(Copy, Clone, Debug)]
pub enum AllocatorKind {
    /// Implements `Allocator`: has a `free<T>(&mut this, ^Self T)` that
    /// internally handles dropping T before freeing
    Owning,
    /// Implements only `RawAllocator`: has `free_raw(&mut this, *void, size, align)`,
    /// which knows nothing about T
    RawOnly,
}
