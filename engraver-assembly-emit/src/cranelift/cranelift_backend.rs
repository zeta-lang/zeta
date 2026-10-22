use crate::backend::Backend;
use crate::cranelift::{clif_type, cranelift_intrinsics};
use cranelift_codegen::Context;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::ir::Block as ClifBlock;
use cranelift_codegen::ir::BlockArg;
use cranelift_codegen::ir::Function as ClifFunction;
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::ir::MemFlags;
use cranelift_codegen::ir::Signature;
use cranelift_codegen::ir::Type;
use cranelift_codegen::ir::Value as ClifValue;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::isa;
use cranelift_codegen::settings::Configurable;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module as ClifModule};
use cranelift_object::{ObjectBuilder, ObjectModule, ObjectProduct, object};
use ir::ast::{ExternModifier, InlineModifier};
use ir::hir::StrId;
use ir::ir_hasher::FxHashBuilder;
use ir::layout::TargetInfo;
use ir::layout::sizeof_ssa;
use ir::registry::global_registry::StaticInit;
use ir::ssa_ir::{
    AtomicOrdering, BinOp, BlockId, CastKind, Function, Instruction, Module, Operand, SsaType,
    UnOp, Value, inst_is_terminator,
};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fmt, io};
use zetaruntime::string_pool::{StringPool, VmString};

/// Maps an IR SSA value directly to the Cranelift value that defines it.
/// The IR is already in SSA form, so no `Variable`s (and therefore no
/// on-the-fly SSA reconstruction by `cranelift-frontend`) are needed.
type ValueMap = HashMap<Value, ClifValue, FxHashBuilder>;
type BlockMap = HashMap<BlockId, ClifBlock, FxHashBuilder>;
type BlockIdx = HashMap<BlockId, usize, FxHashBuilder>;
type PhiMap = HashMap<(BlockId, Value), usize, FxHashBuilder>;

/// Resolves `__zeta_atomic_<op>_<suffix>` to a `&'static str` for a given suffix.
macro_rules! atomic_sym {
    ($op:literal, $suffix:expr) => {
        match $suffix {
            "u8" => concat!("__zeta_atomic_", $op, "_u8"),
            "u16" => concat!("__zeta_atomic_", $op, "_u16"),
            "u32" => concat!("__zeta_atomic_", $op, "_u32"),
            "u64" => concat!("__zeta_atomic_", $op, "_u64"),
            _ => unreachable!(),
        }
    };
}

pub struct CraneliftBackend {
    module: ObjectModule,
    string_data: HashMap<VmString, ZetaDataId, FxHashBuilder>,
    func_ids: HashMap<StrId, FuncId, FxHashBuilder>,
    /// Cranelift types of each function's (non-sret) parameters.
    func_param_types: HashMap<StrId, Vec<Type>, FxHashBuilder>,
    context: Arc<StringPool>,
    target: TargetInfo,
    emit_asm: bool,
    main_emitted: bool,
    func_ret_types: HashMap<StrId, SsaType, FxHashBuilder>,
    /// The hidden sret pointer of the function currently being emitted
    /// (an entry-block parameter, so it dominates everything).
    current_sret_ptr: Option<ClifValue>,
    cpu_relax_func: FuncId,
    global_data: HashMap<StrId, DataId, FxHashBuilder>,
    static_data: HashMap<StrId, DataId, FxHashBuilder>,
    string_bytes: HashMap<StrId, DataId, FxHashBuilder>,
    /// Reused between functions to keep allocations alive.
    clif_ctx: Option<Context>,
    fb_ctx: Option<FunctionBuilderContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ZetaDataId(DataId);

impl CraneliftBackend {
    pub fn new(context: Arc<StringPool>, optimize: bool, verbose: bool) -> Self {
        let mut cranelift_builder: cranelift_codegen::settings::Builder =
            cranelift_codegen::settings::builder();

        cranelift_builder
            .set(
                "opt_level",
                if optimize { "speed_and_size" } else { "none" },
            )
            .unwrap();

        cranelift_builder.set("is_pic", "true").unwrap();

        let verify = cfg!(debug_assertions) || std::env::var_os("ZETA_VERIFY_CLIF").is_some();
        cranelift_builder
            .set("enable_verifier", if verify { "true" } else { "false" })
            .unwrap();

        // saves around 40% compile time on 7K LOC
        cranelift_builder
            .set("regalloc_algorithm", "single_pass")
            .unwrap();

        let flags = cranelift_codegen::settings::Flags::new(cranelift_builder);
        if verbose {
            flags.regalloc_verbose_logs();
        }
        let isa = isa::lookup_by_name("x86_64").unwrap().finish(flags);
        let builder = ObjectBuilder::new(
            isa.unwrap(),
            "zetamir",
            cranelift_module::default_libcall_names(),
        )
        .unwrap();
        let mut module = ObjectModule::new(builder);

        let sig_cpu_relax = Signature::new(module.isa().default_call_conv());
        let cpu_relax_func = module
            .declare_function("__zeta_cpu_relax", Linkage::Import, &sig_cpu_relax)
            .unwrap();

        CraneliftBackend {
            module,
            string_data: HashMap::with_hasher(FxHashBuilder),
            func_ids: HashMap::with_hasher(FxHashBuilder),
            func_param_types: HashMap::with_hasher(FxHashBuilder),
            context,
            target: TargetInfo { ptr_bytes: 8 },
            emit_asm: false,
            main_emitted: false,
            func_ret_types: HashMap::with_hasher(FxHashBuilder),
            current_sret_ptr: None,
            cpu_relax_func,
            global_data: HashMap::with_hasher(FxHashBuilder),
            static_data: HashMap::with_hasher(FxHashBuilder),
            string_bytes: HashMap::with_hasher(FxHashBuilder),
            clif_ctx: None,
            fb_ctx: None,
        }
    }

    fn declare_statics(&mut self, module: &Module) {
        for (name, def) in &module.statics {
            if self.static_data.contains_key(name) {
                continue;
            }
            let sym = format!("__zeta_static_{}", self.context.resolve_string(name));
            let l = ir::layout::layout_of_ssa(&def.ty, self.target)
                .unwrap_or_else(|e| panic!("static `{}`: no layout: {:?}", sym, e));
            let id = self
                .module
                .declare_data(&sym, Linkage::Local, true, false)
                .unwrap_or_else(|e| panic!("declare static {}: {:?}", sym, e));
            let mut d = DataDescription::new();
            let size = l.size.max(1);
            match &def.init {
                StaticInit::Zero => d.define_zeroinit(size),
                StaticInit::Bytes(b) => {
                    let mut b = b.clone();
                    b.resize(size, 0);
                    d.define(b.into_boxed_slice());
                }
                StaticInit::Relocated { bytes, relocs } => {
                    let mut b = bytes.clone();
                    b.resize(size, 0);
                    d.define(b.into_boxed_slice());
                    for r in relocs {
                        let bytes_id = self.get_or_create_string_bytes(&r.string);
                        let gv = self.module.declare_data_in_data(bytes_id, &mut d);
                        d.write_data_addr(r.offset as u32, gv, 0);
                    }
                }
            }
            d.set_align(l.align.max(1) as u64);
            self.module.define_data(id, &d).unwrap();
            self.static_data.insert(*name, id);
        }
    }

    fn atomic_ordering_value(
        &self,
        builder: &mut FunctionBuilder,
        ordering: AtomicOrdering,
    ) -> ClifValue {
        let value = match ordering {
            AtomicOrdering::Relaxed => 0,
            AtomicOrdering::Acquire => 1,
            AtomicOrdering::Release => 2,
            AtomicOrdering::AcqRel => 3,
            AtomicOrdering::SeqCst => 4,
        };

        builder.ins().iconst(types::I32, value)
    }

    fn declare_c_import(
        &mut self,
        name: &'static str,
        params: &[types::Type],
        ret: Option<types::Type>,
    ) -> FuncId {
        let key = StrId::from_static(name);

        if let Some(f) = self.func_ids.get(&key) {
            return *f;
        }

        let mut sig = Signature::new(self.module.isa().default_call_conv());

        for ty in params {
            sig.params.push(AbiParam::new(*ty));
        }

        if let Some(ty) = ret {
            sig.returns.push(AbiParam::new(ty));
        }

        let fid = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .unwrap_or_else(|e| panic!("failed to declare {name}: {:?}", e));

        self.func_ids.insert(key, fid);
        fid
    }

    /// Declares (if needed) and calls one of the C atomic helpers, returning its
    /// first result, if any.
    fn call_c_helper(
        &mut self,
        builder: &mut FunctionBuilder,
        symbol: &'static str,
        params: &[types::Type],
        ret: Option<types::Type>,
        call_args: &[ClifValue],
    ) -> Option<ClifValue> {
        let fid = self.declare_c_import(symbol, params, ret);
        let fref = self.module.declare_func_in_func(fid, &mut builder.func);
        let call = builder.ins().call(fref, call_args);
        builder.inst_results(call).first().copied()
    }

    fn panic_func(&mut self) -> FuncId {
        let name = StrId::from_static("zeta_debug_debug_panic");
        if let Some(f) = self.func_ids.get(&name) {
            return *f;
        }
        let mut sig = Signature::new(self.module.isa().default_call_conv());
        sig.params.push(AbiParam::new(types::I64));
        let fid = self
            .module
            .declare_function("zeta_debug_debug_panic", Linkage::Import, &sig)
            .unwrap_or_else(|e| panic!("failed to declare zeta_debug_debug_panic: {:?}", e));
        self.func_ids.insert(name, fid);
        fid
    }

    fn emit_zero_return(&mut self, builder: &mut FunctionBuilder, func: &Function) {
        if func.ret_type != SsaType::Void && Self::is_aggregate_ty(&func.ret_type) {
            let p = self.current_sret_ptr.expect("sret pointer missing");
            builder.ins().return_(&[p]);
            return;
        }
        match &func.ret_type {
            SsaType::Void => {
                builder.ins().return_(&[]);
            }
            SsaType::F32 => {
                let z = builder.ins().f32const(0.0);
                builder.ins().return_(&[z]);
            }
            SsaType::F64 => {
                let z = builder.ins().f64const(0.0);
                builder.ins().return_(&[z]);
            }
            other => {
                let z = builder.ins().iconst(clif_type(other), 0);
                builder.ins().return_(&[z]);
            }
        }
    }

    fn guard_or_panic(
        &mut self,
        builder: &mut FunctionBuilder,
        cond: ClifValue,
        msg: &str,
        func: &Function,
    ) {
        let fail = builder.create_block();
        let cont = builder.create_block();
        builder.set_cold_block(fail);
        builder.ins().brif(cond, fail, &[], cont, &[]);

        builder.switch_to_block(fail);
        let did = self.get_or_create_string(&StrId(self.context.intern(msg)));
        let gv = self.module.declare_data_in_func(did, &mut builder.func);
        let msg_ptr = builder.ins().global_value(types::I64, gv);
        let pf = self.panic_func();
        let fref = self.module.declare_func_in_func(pf, &mut builder.func);
        builder.ins().call(fref, &[msg_ptr]);
        self.emit_zero_return(builder, func);

        builder.switch_to_block(cont);
    }

    fn emit_checked_int_div(
        &mut self,
        builder: &mut FunctionBuilder,
        op: &BinOp,
        l: ClifValue,
        r: ClifValue,
        unsigned: bool,
        func: &Function,
    ) -> ClifValue {
        let ty = builder.func.dfg.value_type(r);
        let zero = builder.ins().iconst(ty, 0);
        let is_zero = builder.ins().icmp(IntCC::Equal, r, zero);
        let what = if matches!(op, BinOp::Div) {
            "division"
        } else {
            "remainder"
        };
        self.guard_or_panic(
            builder,
            is_zero,
            &format!("{what} by zero in `{}`", func.name),
            func,
        );

        if !unsigned && matches!(op, BinOp::Div) {
            let min_val = if ty.bits() >= 64 {
                i64::MIN
            } else {
                -(1i64 << (ty.bits() - 1))
            };
            let min = builder.ins().iconst(ty, min_val);
            let neg1 = builder.ins().iconst(ty, -1);
            let is_min = builder.ins().icmp(IntCC::Equal, l, min);
            let is_m1 = builder.ins().icmp(IntCC::Equal, r, neg1);
            let overflow = builder.ins().band(is_min, is_m1);
            self.guard_or_panic(
                builder,
                overflow,
                &format!("signed division overflow in `{}`", func.name),
                func,
            );
        }

        match (op, unsigned) {
            (BinOp::Div, true) => builder.ins().udiv(l, r),
            (BinOp::Div, false) => builder.ins().sdiv(l, r),
            (BinOp::Mod, true) => builder.ins().urem(l, r),
            (_, false) => builder.ins().srem(l, r),
            _ => unreachable!(),
        }
    }

    fn get_or_declare_func_ref(&mut self, name_id: StrId) -> FuncId {
        if let Some(fid) = self.func_ids.get(&name_id) {
            return *fid;
        }

        let func_name = self.context.resolve_string(&name_id);

        let mut sig = Signature::new(self.module.isa().default_call_conv());
        if let Some(param_tys) = self.func_param_types.get(&name_id) {
            for ty in param_tys {
                sig.params.push(AbiParam::new(*ty));
            }
        }
        match self.func_ret_types.get(&name_id) {
            Some(SsaType::Void) => {}
            Some(ret_ty) => sig.returns.push(AbiParam::new(clif_type(ret_ty))),
            None => sig.returns.push(AbiParam::new(types::I64)),
        }

        let fid = self
            .module
            .declare_function(func_name, Linkage::Import, &sig)
            .unwrap_or_else(|e| panic!("failed to declare function ref {}: {:?}", func_name, e));

        self.func_ids.insert(name_id, fid);
        fid
    }

    fn get_or_create_global_data(&mut self, name_id: StrId) -> DataId {
        if let Some(id) = self.global_data.get(&name_id) {
            return *id;
        }

        let name = self.context.resolve_string(&name_id);
        let id = self
            .module
            .declare_data(name, Linkage::Import, true, false)
            .unwrap_or_else(|e| panic!("failed to declare global {}: {:?}", name, e));

        self.global_data.insert(name_id, id);
        id
    }

    fn addr_type(&self) -> Type {
        self.module.isa().pointer_type()
    }

    /// Makes sure `value` is a pointer-typed Cranelift value.
    fn coerce_addr(
        &self,
        builder: &mut FunctionBuilder,
        value: ClifValue,
        site: &str,
    ) -> ClifValue {
        let want = self.addr_type();
        let have = builder.func.dfg.value_type(value);
        if have == want {
            return value;
        }
        eprintln!(
            "WARNING [{site}]: address value has clif type {:?}, expected pointer type {:?}; \
                 coercing. This indicates a bug upstream. Emitted a coercion for the compilation to succeed",
            have, want
        );
        if have.bits() < want.bits() {
            builder.ins().uextend(want, value)
        } else if have.bits() > want.bits() {
            builder.ins().ireduce(want, value)
        } else {
            builder.ins().bitcast(want, MemFlags::new(), value)
        }
    }

    fn is_aggregate_ty(ty: &SsaType) -> bool {
        match ty {
            SsaType::User(_, _, _)
            | SsaType::Array(_, _)
            | SsaType::Tuple(_)
            | SsaType::Enum { .. }
            | SsaType::Slice(_)
            | SsaType::String => true,
            SsaType::Owned(inner) => matches!(inner.as_ref(), SsaType::Slice(_)),
            SsaType::Nullable(_) => ty.is_tagged_nullable(),
            _ => false,
        }
    }

    /// Copies `size` bytes from `src` to `dst` with the widest loads/stores possible.
    fn emit_aggregate_copy(
        builder: &mut FunctionBuilder,
        dst: ClifValue,
        src: ClifValue,
        size: usize,
    ) {
        let flags = MemFlags::new();
        let mut copied = 0usize;
        for (chunk_ty, chunk_size) in [(types::I64, 8usize), (types::I32, 4), (types::I8, 1)] {
            while copied + chunk_size <= size {
                let chunk = builder.ins().load(chunk_ty, flags, src, copied as i32);
                builder.ins().store(flags, chunk, dst, copied as i32);
                copied += chunk_size;
            }
        }
    }

    /// Integer width coercion of a call argument to the callee's parameter type.
    fn coerce_int_arg(
        builder: &mut FunctionBuilder,
        raw: ClifValue,
        expected: Type,
        signed: bool,
    ) -> ClifValue {
        let raw_ty = builder.func.dfg.value_type(raw);
        if raw_ty == expected {
            raw
        } else if raw_ty.bits() < expected.bits() {
            if signed {
                builder.ins().sextend(expected, raw)
            } else {
                builder.ins().uextend(expected, raw)
            }
        } else if raw_ty.bits() > expected.bits() {
            builder.ins().ireduce(expected, raw)
        } else {
            raw
        }
    }

    /// Collects the block arguments a branch from `curr_bb` to `target` must pass
    /// for the phis at the top of `target`.
    fn phi_args(
        &self,
        builder: &mut FunctionBuilder,
        var_map: &ValueMap,
        func: &Function,
        block_idx: &BlockIdx,
        curr_bb: BlockId,
        target: &BlockId,
        site: &str,
    ) -> Vec<BlockArg> {
        let mut args: Vec<BlockArg> = Vec::new();
        for inst in &func.blocks[block_idx[target]].instructions {
            let Instruction::Phi { dest, incoming } = inst else {
                break;
            };
            let Some(entry) = incoming.iter().find(|p| p.0 == curr_bb) else {
                panic!(
                    "phi {:?} in {:?} has no incoming for pred {:?} (jumping from {:?})",
                    dest, target, curr_bb, curr_bb
                );
            };
            let v = entry.1;
            let valv = *var_map.get(&v).unwrap_or_else(|| {
                panic!(
                    "phi incoming value not lowered ({site}): curr_bb={:?} -> target={:?}, \
                     phi_dest={:?}, missing_val={:?}",
                    curr_bb, target, dest, v
                )
            });
            let phi_ty = func
                .value_types
                .get(dest)
                .expect("phi dest missing a value type");
            let clif_ty = if *phi_ty == SsaType::Void {
                types::I8
            } else {
                clif_type(phi_ty)
            };
            let valv = self.coerce_for_block_arg(builder, valv, clif_ty, site);
            args.push(BlockArg::Value(valv));
        }
        args
    }

    fn ptr_operand(
        &self,
        builder: &mut FunctionBuilder,
        var_map: &ValueMap,
        operand: &Operand,
        name: &str,
    ) -> ClifValue {
        match operand {
            Operand::Value(v) => get_val(var_map, v, name),
            Operand::ConstInt(i) => builder.ins().iconst(self.addr_type(), *i),
            _ => panic!("{name}: invalid pointer operand {:?}", operand),
        }
    }

    fn resolve_atomic_ty(
        func: &Function,
        query_ty: &Option<SsaType>,
        value_operand: &Operand,
        dest: Option<Value>,
        name: &str,
    ) -> Type {
        if let Some(ty) = query_ty.as_ref() {
            return atomic_type_from_ssa(ty, name);
        }
        match value_operand {
            Operand::Value(v) => {
                let ty = func
                    .value_types
                    .get(v)
                    .unwrap_or_else(|| panic!("{name}: value type missing"));
                atomic_type_from_ssa(ty, name)
            }
            Operand::ConstInt(_) => match dest {
                Some(d) => {
                    let ty = func
                        .value_types
                        .get(&d)
                        .unwrap_or_else(|| panic!("{name}: destination type missing"));
                    atomic_type_from_ssa(ty, name)
                }
                None => panic!("{name}: query_ty is required when storing a constant"),
            },
            _ => panic!("{name}: invalid value operand {:?}", value_operand),
        }
    }

    fn lower_basic_inst(
        &mut self,
        inst: &Instruction,
        curr_bb: BlockId,
        builder: &mut FunctionBuilder,
        var_map: &mut ValueMap,
        block_map: &BlockMap,
        phi_param_map: &PhiMap,
        func: &Function,
        block_idx: &BlockIdx,
    ) {
        match inst {
            Instruction::Undef { dest, ty } => {
                let clif_ty = if *ty == SsaType::Void {
                    types::I8
                } else {
                    clif_type(ty)
                };

                let val = if clif_ty == types::F32 {
                    builder.ins().f32const(0.0)
                } else if clif_ty == types::F64 {
                    builder.ins().f64const(0.0)
                } else {
                    builder.ins().iconst(clif_ty, 0)
                };

                define(var_map, builder, *dest, val, Some(clif_ty), "Undef");
            }
            Instruction::Const { dest, ty, value } => {
                let clif_ty = clif_type(ty);
                let val = match value {
                    Operand::ConstInt(i) => {
                        if clif_ty == types::F32 {
                            builder.ins().f32const(*i as f32)
                        } else if clif_ty == types::F64 {
                            builder.ins().f64const(*i as f64)
                        } else {
                            builder.ins().iconst(clif_ty, *i)
                        }
                    }
                    Operand::ConstBool(b) => {
                        builder.ins().iconst(types::I8, if *b { 1 } else { 0 })
                    }
                    Operand::ConstString(s) => {
                        let did = self.get_or_create_string(s);
                        let gv = self.module.declare_data_in_func(did, &mut builder.func);
                        builder.ins().global_value(types::I64, gv)
                    }
                    Operand::Value(v) => get_val(var_map, v, "Const operand"),
                    Operand::ConstFloat(i) => {
                        if clif_ty == types::F32 {
                            builder.ins().f32const(*i as f32)
                        } else if clif_ty == types::F64 {
                            builder.ins().f64const(*i)
                        } else {
                            unreachable!("ConstFloat with non-float type")
                        }
                    }
                    Operand::FunctionRef(str_id) => {
                        let func_id = self.get_or_declare_func_ref(*str_id);
                        let funcref = self.module.declare_func_in_func(func_id, &mut builder.func);
                        let addr_ty = self.addr_type();
                        builder.ins().func_addr(addr_ty, funcref)
                    }
                    Operand::GlobalRef(str_id) => {
                        let data_id = self.get_or_create_global_data(*str_id);
                        let gv = self.module.declare_data_in_func(data_id, &mut builder.func);
                        let addr_ty = self.addr_type();
                        builder.ins().global_value(addr_ty, gv)
                    }
                };
                define(var_map, builder, *dest, val, Some(clif_ty), "Const");
            }

            Instruction::Binary {
                dest,
                op,
                left,
                right,
            } => {
                let ssa_of = |operand: &Operand| -> Option<&SsaType> {
                    match operand {
                        Operand::Value(v) => func.value_types.get(v),
                        _ => None,
                    }
                };

                let l_val = match left {
                    Operand::Value(v) => Some(get_val(var_map, v, "Binary(left)")),
                    _ => None,
                };
                let r_val = match right {
                    Operand::Value(v) => Some(get_val(var_map, v, "Binary(right)")),
                    _ => None,
                };

                // Float type from the actual values, not from func.value_types.
                let float_ty = [l_val, r_val]
                    .iter()
                    .flatten()
                    .map(|v| builder.func.dfg.value_type(*v))
                    .find(|t| t.is_float());

                // Materialise constants, matching the float type when there is one.
                let make_const = |operand: &Operand, builder: &mut FunctionBuilder| match operand {
                    Operand::ConstInt(i) => match float_ty {
                        Some(types::F32) => builder.ins().f32const(*i as f32),
                        Some(_) => builder.ins().f64const(*i as f64),
                        None => builder.ins().iconst(types::I64, *i),
                    },
                    Operand::ConstFloat(f) => match float_ty {
                        Some(types::F32) => builder.ins().f32const(*f as f32),
                        _ => builder.ins().f64const(*f),
                    },
                    _ => unimplemented!("binary operand {:?}", operand),
                };
                let l = match l_val {
                    Some(v) => v,
                    None => make_const(left, builder),
                };
                let r = match r_val {
                    Some(v) => v,
                    None => make_const(right, builder),
                };

                let lt = builder.func.dfg.value_type(l);
                let rt = builder.func.dfg.value_type(r);

                let signed_of = |o: &Operand| ssa_of(o).map_or(true, |t| t.is_signed_integer());

                let (l, r) = if lt.is_float() && rt.is_int() {
                    (l, builder.ins().fcvt_from_sint(lt, r))
                } else if lt.is_int() && rt.is_float() {
                    (builder.ins().fcvt_from_sint(rt, l), r)
                } else if lt.is_int() && rt.is_int() && lt != rt {
                    if lt.bits() < rt.bits() {
                        let l2 = if signed_of(left) {
                            builder.ins().sextend(rt, l)
                        } else {
                            builder.ins().uextend(rt, l)
                        };
                        (l2, r)
                    } else {
                        let r2 = if signed_of(right) {
                            builder.ins().sextend(lt, r)
                        } else {
                            builder.ins().uextend(lt, r)
                        };
                        (l, r2)
                    }
                } else {
                    (l, r)
                };

                let is_float = builder.func.dfg.value_type(l).is_float();
                let unsigned = ssa_of(left)
                    .or_else(|| ssa_of(right))
                    .map(is_unsigned_ssa_type)
                    .unwrap_or(false);

                let res = if is_float {
                    use cranelift_codegen::ir::condcodes::FloatCC;
                    match op {
                        BinOp::Add => builder.ins().fadd(l, r),
                        BinOp::Sub => builder.ins().fsub(l, r),
                        BinOp::Mul => builder.ins().fmul(l, r),
                        BinOp::Div => builder.ins().fdiv(l, r),
                        BinOp::Eq => builder.ins().fcmp(FloatCC::Equal, l, r),
                        BinOp::Ne => builder.ins().fcmp(FloatCC::NotEqual, l, r),
                        BinOp::Lt => builder.ins().fcmp(FloatCC::LessThan, l, r),
                        BinOp::Le => builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r),
                        BinOp::Gt => builder.ins().fcmp(FloatCC::GreaterThan, l, r),
                        BinOp::Ge => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r),
                        other => {
                            panic!("operator {other:?} is not valid on floating-point operands")
                        }
                    }
                } else {
                    match op {
                        BinOp::Add => builder.ins().iadd(l, r),
                        BinOp::Sub => builder.ins().isub(l, r),
                        BinOp::Mul => builder.ins().imul(l, r),
                        BinOp::Div | BinOp::Mod => {
                            self.emit_checked_int_div(builder, op, l, r, unsigned, func)
                        }
                        BinOp::Eq => builder.ins().icmp(IntCC::Equal, l, r),
                        BinOp::Ne => builder.ins().icmp(IntCC::NotEqual, l, r),
                        BinOp::Lt => builder.ins().icmp(
                            if unsigned {
                                IntCC::UnsignedLessThan
                            } else {
                                IntCC::SignedLessThan
                            },
                            l,
                            r,
                        ),
                        BinOp::Le => builder.ins().icmp(
                            if unsigned {
                                IntCC::UnsignedLessThanOrEqual
                            } else {
                                IntCC::SignedLessThanOrEqual
                            },
                            l,
                            r,
                        ),
                        BinOp::Gt => builder.ins().icmp(
                            if unsigned {
                                IntCC::UnsignedGreaterThan
                            } else {
                                IntCC::SignedGreaterThan
                            },
                            l,
                            r,
                        ),
                        BinOp::Ge => builder.ins().icmp(
                            if unsigned {
                                IntCC::UnsignedGreaterThanOrEqual
                            } else {
                                IntCC::SignedGreaterThanOrEqual
                            },
                            l,
                            r,
                        ),
                        BinOp::BitAnd => builder.ins().band(l, r),
                        BinOp::BitOr => builder.ins().bor(l, r),
                        BinOp::BitXor => builder.ins().bxor(l, r),
                        BinOp::ShiftLeft => builder.ins().ishl(l, r),
                        BinOp::ShiftRight => {
                            if unsigned {
                                builder.ins().ushr(l, r)
                            } else {
                                builder.ins().sshr(l, r)
                            }
                        }
                        BinOp::LogicalAnd => unreachable!(),
                        BinOp::LogicalOr => unreachable!(),
                    }
                };
                let res_ty = builder.func.dfg.value_type(res);

                // Binary instructions can temporarily lack an entry in func.value_types
                // (notably comparison results produced by lowering). The Cranelift result
                // already carries the authoritative machine type, so use it as a fallback
                // instead of crashing after successfully emitting the instruction.
                let clif_ty = match func.value_types.get(dest) {
                    Some(ty) => clif_type(ty),
                    None => {
                        eprintln!(
                            "WARNING [Binary]: missing SSA value type for {dest:?} \
                            ({op:?}); using actual Cranelift result type {:?}. \
                            This indicates an upstream IR type-map bug.",
                            res_ty,
                        );
                        res_ty
                    }
                };

                let res = if res_ty == clif_ty || res_ty.is_float() || clif_ty.is_float() {
                    res
                } else {
                    match (res_ty, clif_ty) {
                        // Integer widening
                        _ if res_ty.is_int()
                            && clif_ty.is_int()
                            && res_ty.bits() < clif_ty.bits() =>
                        {
                            builder.ins().uextend(clif_ty, res)
                        }

                        // Integer narrowing
                        _ if res_ty.is_int()
                            && clif_ty.is_int()
                            && res_ty.bits() > clif_ty.bits() =>
                        {
                            builder.ins().ireduce(clif_ty, res)
                        }

                        _ => panic!(
                            "cannot assign value of type {:?} to variable of type {:?}",
                            res_ty, clif_ty
                        ),
                    }
                };

                define(var_map, builder, *dest, res, Some(clif_ty), "Binary");
            }

            Instruction::Phi { dest, .. } => {
                let clif_bb = block_map
                    .get(&curr_bb)
                    .expect("current block missing for Phi");

                let idx = phi_param_map
                    .get(&(curr_bb, *dest))
                    .expect("phi param missing");

                let param = builder.block_params(*clif_bb)[*idx];

                define(var_map, builder, *dest, param, None, "Phi");
            }

            Instruction::Jump { target } => {
                let args = self.phi_args(
                    builder,
                    var_map,
                    func,
                    block_idx,
                    curr_bb,
                    target,
                    "Jump(phi arg)",
                );
                builder.ins().jump(block_map[target], &args);
            }

            Instruction::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let c = match cond {
                    Operand::Value(v) => get_val(var_map, v, "Branch cond"),
                    _ => unimplemented!(),
                };

                let c_type = builder.func.dfg.value_type(c);
                let cond_bool = if c_type != types::I8 {
                    let zero = builder.ins().iconst(c_type, 0);
                    builder.ins().icmp(IntCC::NotEqual, c, zero)
                } else {
                    c
                };

                let then_args = self.phi_args(
                    builder,
                    var_map,
                    func,
                    block_idx,
                    curr_bb,
                    then_bb,
                    "Branch(then phi arg)",
                );
                let else_args = self.phi_args(
                    builder,
                    var_map,
                    func,
                    block_idx,
                    curr_bb,
                    else_bb,
                    "Branch(else phi arg)",
                );

                builder.ins().brif(
                    cond_bool,
                    block_map[then_bb],
                    &then_args,
                    block_map[else_bb],
                    &else_args,
                );
            }

            Instruction::Ret { value } => {
                if func.ret_type == SsaType::Void {
                    builder.ins().return_(&[]);
                } else if Self::is_aggregate_ty(&func.ret_type) {
                    let sret_ptr = self
                        .current_sret_ptr
                        .expect("Ret: aggregate return but no sret pointer bound");

                    match value {
                        Some(Operand::Value(v)) => {
                            let src_addr = get_val(var_map, v, "Ret");
                            let size = ir::layout::sizeof_ssa(&func.ret_type, self.target)
                                .expect("Ret: aggregate return type has unknown size");
                            Self::emit_aggregate_copy(builder, sret_ptr, src_addr, size);
                        }
                        None => {}
                        Some(_) => panic!(
                            "Ret: aggregate return must be a Value (address) in function `{}` (ret_type: {:?}, value: {:?})",
                            func.name, func.ret_type, value
                        ),
                    }

                    builder.ins().return_(&[sret_ptr]);
                } else {
                    let ret_ty = clif_type(&func.ret_type);

                    let rv = match value {
                        Some(Operand::Value(v)) => {
                            let val = get_val(var_map, v, "Ret");
                            let actual_ty = builder.func.dfg.value_type(val);

                            if actual_ty == ret_ty {
                                val
                            } else if actual_ty.bits() < ret_ty.bits() {
                                builder.ins().uextend(ret_ty, val)
                            } else if actual_ty.bits() > ret_ty.bits() {
                                builder.ins().ireduce(ret_ty, val)
                            } else {
                                self.coerce_bitcast(builder, val, ret_ty)
                            }
                        }

                        Some(Operand::ConstInt(i)) => builder.ins().iconst(ret_ty, *i),

                        _ => builder.ins().iconst(ret_ty, 0),
                    };

                    builder.ins().return_(&[rv]);
                }
            }
            Instruction::StackAlloc { dest, ty, count } => {
                let elem_size = sizeof_ssa(ty, self.target)
                    .unwrap_or_else(|_| panic!("StackAlloc: unknown size for {:?}", ty));

                let size_bytes = elem_size * (*count).max(1);

                let ptr = cranelift_intrinsics::stack_alloc(builder, &mut self.module, size_bytes);

                let ptr = self.coerce_addr(builder, ptr, "StackAlloc");
                var_map.insert(*dest, ptr);
            }
            Instruction::LoadField { dest, base, offset } => {
                let field_ty = func
                    .value_types
                    .get(dest)
                    .expect("LoadField: dest type unknown");
                if self.is_zst(field_ty) {
                    let z = builder.ins().iconst(types::I8, 0);
                    var_map.insert(*dest, z);
                    return;
                }

                let base_val = match base {
                    Operand::Value(bv) => get_val(var_map, bv, "LoadField base"),
                    _ => panic!("LoadField base must be a Value"),
                };

                if Self::is_aggregate_ty(field_ty) {
                    let addr = builder.ins().iadd_imm(base_val, *offset as i64);
                    let addr = self.coerce_addr(builder, addr, "LoadField(aggregate)");
                    var_map.insert(*dest, addr);
                } else {
                    let clif_ty = clif_type(field_ty);
                    let loaded =
                        builder
                            .ins()
                            .load(clif_ty, MemFlags::new(), base_val, *offset as i32);
                    var_map.insert(*dest, loaded);
                }
            }
            Instruction::StoreField {
                base,
                offset,
                value,
            } => {
                let value_ssa_ty = match value {
                    Operand::Value(v) => func.value_types.get(v),
                    _ => None,
                };
                if value_ssa_ty.map_or(false, |t| self.is_zst(t)) {
                    return;
                }

                let base_val = match base {
                    Operand::Value(bv) => get_val(var_map, bv, "StoreField base"),
                    _ => panic!("StoreField base must be a Value"),
                };

                let is_aggregate = value_ssa_ty.map_or(false, Self::is_aggregate_ty);

                if is_aggregate {
                    let src_addr = match value {
                        Operand::Value(vv) => get_val(var_map, vv, "StoreField aggregate"),
                        _ => panic!("StoreField: aggregate value must be a Value (address)"),
                    };
                    let size = ir::layout::sizeof_ssa(value_ssa_ty.unwrap(), self.target)
                        .expect("StoreField: aggregate has unknown size");
                    let addr = builder.ins().iadd_imm(base_val, *offset as i64);
                    Self::emit_aggregate_copy(builder, addr, src_addr, size);
                } else {
                    let value_val = match value {
                        Operand::Value(vv) => get_val(var_map, vv, "StoreField value"),
                        Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),
                        _ => unimplemented!("StoreField: unsupported value operand"),
                    };

                    builder
                        .ins()
                        .store(MemFlags::new(), value_val, base_val, *offset as i32);
                }
            }
            Instruction::Call {
                dest,
                func: called_func,
                args,
            } if matches!(called_func, Operand::Value(_)) => {
                let fn_val = match called_func {
                    Operand::Value(v) => v,
                    _ => unreachable!(),
                };

                let (param_tys, ret_ty) = match func.value_types.get(fn_val) {
                    Some(SsaType::FuncPointer {
                        params,
                        return_type,
                    }) => (params.clone(), (**return_type).clone()),
                    other => panic!(
                        "Call: indirect call target `{:?}` has non-function-pointer type: {:?}",
                        fn_val, other
                    ),
                };

                let fn_ptr = get_val(var_map, fn_val, "Call: function pointer");

                let mut sig = self.module.make_signature();
                for pty in &param_tys {
                    sig.params.push(AbiParam::new(clif_type(pty)));
                }
                let ret_is_void = ret_ty == SsaType::Void;
                if !ret_is_void {
                    sig.returns.push(AbiParam::new(clif_type(&ret_ty)));
                }
                let sig_ref = builder.func.import_signature(sig);

                let mut arg_vals = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    let raw_val = match a {
                        Operand::Value(v) => get_val(var_map, v, "indirect call arg"),
                        Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),
                        _ => unimplemented!("indirect call arg type not supported"),
                    };
                    let expected_ty = param_tys.get(i).map(clif_type).unwrap_or(types::I64);
                    arg_vals.push(Self::coerce_int_arg(builder, raw_val, expected_ty, false));
                }

                let call_inst = builder.ins().call_indirect(sig_ref, fn_ptr, &arg_vals);
                let res0 = builder.inst_results(call_inst).first().copied();

                if let Some(d) = dest {
                    match res0 {
                        Some(res_val) => {
                            let expected_ty = func.value_types.get(d).map(clif_type);
                            define(
                                var_map,
                                builder,
                                *d,
                                res_val,
                                expected_ty,
                                "Call(indirect result)",
                            );
                        }
                        None => {
                            let dummy_val = builder.ins().iconst(types::I64, 0);
                            var_map.insert(*d, dummy_val);
                        }
                    }
                }
            }

            Instruction::Call {
                dest,
                func: called_func,
                args,
            } => {
                let func_name_id = match called_func {
                    Operand::FunctionRef(s) => s,
                    _ => panic!(
                        "Call target must be a FunctionRef in current lowering but got {called_func:?}"
                    ),
                };
                let func_name = self.context.resolve_string(func_name_id);

                let (ret_is_aggregate, ret_size) = match self.func_ret_types.get(func_name_id) {
                    Some(t) if Self::is_aggregate_ty(t) => (
                        true,
                        sizeof_ssa(t, self.target)
                            .expect("[Call] aggregate return type has unknown size"),
                    ),
                    _ => (false, 0),
                };
                let param_tys = self.func_param_types.get(func_name_id);

                let mut arg_vals: Vec<ClifValue> = Vec::with_capacity(args.len() + 1);

                let sret_alloc = if ret_is_aggregate {
                    let addr =
                        cranelift_intrinsics::stack_alloc(builder, &mut self.module, ret_size);
                    arg_vals.push(addr);
                    Some(addr)
                } else {
                    None
                };

                for (i, a) in args.iter().enumerate() {
                    let raw_val = match a {
                        Operand::Value(v) => get_val(var_map, v, "call arg"),
                        Operand::ConstInt(c) => builder.ins().iconst(types::I64, *c),
                        _ => unimplemented!("call arg type not supported"),
                    };
                    let expected_ty = param_tys
                        .and_then(|p| p.get(i))
                        .copied()
                        .unwrap_or(types::I64);
                    let src_signed = match a {
                        Operand::Value(v) => func
                            .value_types
                            .get(v)
                            .map_or(true, |t| t.is_signed_integer()),
                        _ => true,
                    };
                    arg_vals.push(Self::coerce_int_arg(
                        builder,
                        raw_val,
                        expected_ty,
                        src_signed,
                    ));
                }

                let func_id = if let Some(fid) = self.func_ids.get(func_name_id) {
                    *fid
                } else {
                    let mut sig = Signature::new(self.module.isa().default_call_conv());
                    for v in &arg_vals {
                        sig.params
                            .push(AbiParam::new(builder.func.dfg.value_type(*v)));
                    }
                    if dest.is_some() {
                        sig.returns.push(AbiParam::new(types::I64));
                    }
                    let fid = ObjectModule::declare_function(
                        &mut self.module,
                        func_name,
                        Linkage::Import,
                        &sig,
                    )
                    .unwrap_or_else(|e| panic!("failed to declare import {}: {:?}", func_name, e));
                    self.func_ids.insert(*func_name_id, fid);
                    fid
                };

                let callee = self.module.declare_func_in_func(func_id, &mut builder.func);
                let call_inst = builder.ins().call(callee, &arg_vals);
                let res0 = builder.inst_results(call_inst).first().copied();

                if let Some(d) = dest {
                    if ret_is_aggregate {
                        let addr = self.coerce_addr(builder, sret_alloc.unwrap(), "Call(sret)");
                        var_map.insert(*d, addr);
                    } else if let Some(res_val) = res0 {
                        let expected_ty = func.value_types.get(d).map(clif_type);
                        define(var_map, builder, *d, res_val, expected_ty, "Call(result)");
                    } else {
                        let dummy_val = builder.ins().iconst(types::I64, 0);
                        var_map.insert(*d, dummy_val);
                    }
                }
            }

            Instruction::Cast { dest, value, kind } => {
                let src = match value {
                    Operand::Value(v) => get_val(var_map, v, "Cast"),
                    _ => unreachable!(),
                };

                let dst_ty = clif_type(func.value_types.get(dest).unwrap());

                let result = match kind {
                    CastKind::Truncate => builder.ins().ireduce(dst_ty, src),

                    CastKind::SignExtend => builder.ins().sextend(dst_ty, src),

                    CastKind::ZeroExtend => builder.ins().uextend(dst_ty, src),

                    CastKind::SignedIntToFloat => builder.ins().fcvt_from_sint(dst_ty, src),

                    CastKind::UnsignedIntToFloat => builder.ins().fcvt_from_uint(dst_ty, src),

                    CastKind::FloatToSignedInt => builder.ins().fcvt_to_sint_sat(dst_ty, src),
                    CastKind::FloatToUnsignedInt => builder.ins().fcvt_to_uint_sat(dst_ty, src),

                    CastKind::FloatExtend => builder.ins().fpromote(dst_ty, src),

                    CastKind::FloatTruncate => builder.ins().fdemote(dst_ty, src),

                    CastKind::Bitcast => self.coerce_bitcast(builder, src, dst_ty),

                    CastKind::PtrToInt => self.coerce_bitcast(builder, src, dst_ty),

                    CastKind::IntToPtr => self.coerce_bitcast(builder, src, dst_ty),
                };

                define(var_map, builder, *dest, result, Some(dst_ty), "Cast");
            }

            Instruction::Unary { dest, op, operand } => {
                let val = match operand {
                    Operand::Value(v) => get_val(var_map, v, "Unary operand"),
                    Operand::ConstInt(i) => {
                        let ty = func.value_types.get(dest).unwrap();
                        builder.ins().iconst(clif_type(ty), *i)
                    }
                    _ => unimplemented!(),
                };

                let res = match op {
                    UnOp::Neg => {
                        if builder.func.dfg.value_type(val).is_float() {
                            builder.ins().fneg(val)
                        } else {
                            builder.ins().ineg(val)
                        }
                    }

                    UnOp::BitNot => builder.ins().bnot(val),

                    UnOp::LogicalNot => {
                        let value_type = builder.func.dfg.value_type(val);
                        match value_type {
                            types::I8 => {
                                let zero = builder.ins().iconst(value_type, 0);
                                builder.ins().icmp(IntCC::Equal, val, zero)
                            }
                            ty => panic!("LogicalNot not supported for {:?}", ty),
                        }
                    }
                };

                let ty = func.value_types.get(dest).unwrap();
                let clif_ty = clif_type(ty);

                debug_assert!(
                    builder.func.dfg.value_type(res) == clif_ty,
                    "Unary {:?} produced {:?}, expected {:?}",
                    op,
                    builder.func.dfg.value_type(res),
                    clif_ty,
                );

                define(var_map, builder, *dest, res, Some(clif_ty), "Unary");
            }

            Instruction::AddressOf { dest, source } => {
                let src_val = get_val(var_map, source, "AddressOf source");
                let addr = self.coerce_addr(builder, src_val, "AddressOf");
                var_map.insert(*dest, addr);
            }

            Instruction::Load { dest, ptr } => {
                let ty = func
                    .value_types
                    .get(dest)
                    .expect("Load: destination type missing");

                if self.is_zst(ty) {
                    let val = builder.ins().iconst(types::I8, 0);
                    define(var_map, builder, *dest, val, Some(types::I8), "Load(zst)");
                    return;
                }
                let ptr_val = match ptr {
                    Operand::Value(v) => get_val(var_map, v, "Load pointer"),
                    Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),
                    _ => panic!("Load: invalid pointer operand"),
                };

                if Self::is_aggregate_ty(ty) {
                    let addr = self.coerce_addr(builder, ptr_val, "Load(aggregate)");
                    var_map.insert(*dest, addr);
                    return;
                }

                let clif_ty = clif_type(ty);

                let loaded = builder.ins().load(clif_ty, MemFlags::new(), ptr_val, 0);
                define(
                    var_map,
                    builder,
                    *dest,
                    loaded,
                    Some(clif_ty),
                    "Load(scalar)",
                );
            }

            Instruction::Store { ptr, value } => {
                let value_ssa_ty = match value {
                    Operand::Value(v) => func.value_types.get(v),
                    _ => None,
                };
                if value_ssa_ty.map_or(false, |t| self.is_zst(t)) {
                    return;
                }

                let ptr_val = match ptr {
                    Operand::Value(v) => get_val(var_map, v, "Store pointer"),
                    Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),
                    _ => panic!("Store: invalid pointer operand"),
                };

                let is_aggregate = value_ssa_ty.map_or(false, Self::is_aggregate_ty);

                if is_aggregate {
                    let src_addr = match value {
                        Operand::Value(v) => get_val(var_map, v, "Store aggregate"),
                        _ => panic!("Store: aggregate value must be a Value (address)"),
                    };
                    let size = ir::layout::sizeof_ssa(value_ssa_ty.unwrap(), self.target)
                        .expect("Store: aggregate has unknown size");
                    Self::emit_aggregate_copy(builder, ptr_val, src_addr, size);
                    return;
                }

                let value_val = match value {
                    Operand::Value(v) => get_val(var_map, v, "Store value"),

                    Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),

                    Operand::ConstBool(b) => {
                        builder.ins().iconst(types::I8, if *b { 1 } else { 0 })
                    }

                    Operand::ConstFloat(f) => builder.ins().f64const(*f),

                    _ => unimplemented!("Store operand type"),
                };

                builder.ins().store(MemFlags::new(), value_val, ptr_val, 0);
            }

            Instruction::InterfaceDispatch {
                dest,
                object,
                method_slot,
                args,
            } => {
                let obj = get_val(var_map, object, "InterfaceDispatch object");

                let flags = MemFlags::new();
                // data_ptr   = load object[0]
                let data_ptr = builder.ins().load(types::I64, flags, obj, 0);

                // vtable_ptr = load object[PTR_SIZE]
                let ptr_size: i64 = self.target.ptr_bytes as i64;
                let vtable_addr = builder.ins().iadd_imm(obj, ptr_size);
                let vtable_ptr = builder.ins().load(types::I64, flags, vtable_addr, 0);

                // fnptr = load vtable_ptr[method_slot * PTR_SIZE]
                let slot_index = *method_slot as i64;
                let slot_offset = slot_index * ptr_size;
                let fnptr_idx = builder.ins().iadd_imm(vtable_ptr, slot_offset);
                let fnptr_addr = builder.ins().load(types::I64, flags, fnptr_idx, 0);
                let fn_ptr = builder.ins().load(types::I64, flags, fnptr_addr, 0);

                let mut call_args = Vec::with_capacity(args.len() + 1);
                call_args.push(data_ptr);

                for arg in args {
                    let val = match arg {
                        Operand::Value(v) => get_val(var_map, v, "InterfaceDispatch arg"),
                        Operand::ConstInt(i) => builder.ins().iconst(types::I64, *i),
                        Operand::ConstFloat(f) => builder.ins().f64const(*f),
                        Operand::ConstBool(b) => {
                            builder.ins().iconst(types::I8, if *b { 1 } else { 0 })
                        }
                        Operand::ConstString(_str_id) => unimplemented!(),
                        Operand::FunctionRef(_str_id) => unimplemented!(),
                        Operand::GlobalRef(_str_id) => unimplemented!(),
                    };
                    call_args.push(val);
                }

                let mut sig = self.module.make_signature();
                sig.params.push(AbiParam::new(types::I64)); // object + args
                for _ in args {
                    sig.params.push(AbiParam::new(types::I64));
                }

                if dest.is_some() {
                    sig.returns.push(AbiParam::new(types::I64));
                }

                let sig_ref = builder.func.import_signature(sig);

                // call_indirect fnptr(data_ptr, args...)
                let call = builder.ins().call_indirect(sig_ref, fn_ptr, &call_args);

                if let Some(d) = dest {
                    let res = builder.inst_results(call).first().copied();

                    let value = match res {
                        Some(v) => v,
                        None => builder.ins().iconst(types::I64, 0),
                    };

                    var_map.insert(*d, value);
                }
            }
            Instruction::UpcastToInterface { .. } => todo!(),
            Instruction::FieldAddr { dest, base, offset } => {
                let base_val = match base {
                    Operand::Value(bv) => get_val(var_map, bv, "FieldAddr base"),
                    _ => panic!("FieldAddr base must be a Value"),
                };

                let addr = builder.ins().iadd_imm(base_val, *offset as i64);

                let addr = self.coerce_addr(builder, addr, "FieldAddr");
                var_map.insert(*dest, addr);
            }
            Instruction::Intrinsic {
                dest,
                op,
                query_ty,
                args,
            } => {
                use ir::ssa_ir::IntrinsicOp;
                match op {
                    IntrinsicOp::SizeOf | IntrinsicOp::AlignOf => {
                        let ty = query_ty.as_ref().expect("SizeOf/AlignOf requires query_ty");
                        let layout =
                            ir::layout::layout_of_ssa(ty, self.target).unwrap_or_else(|e| {
                                panic!("{:?}: failed to compute layout for {:?}: {:?}", op, ty, e)
                            });
                        let n = match op {
                            IntrinsicOp::SizeOf => layout.size as i64,
                            IntrinsicOp::AlignOf => layout.align as i64,
                            _ => unreachable!(),
                        };
                        let val = builder.ins().iconst(types::I64, n);
                        if let Some(d) = dest {
                            var_map.insert(*d, val);
                        }
                    }

                    IntrinsicOp::TypeName => {
                        let ty: &SsaType = query_ty.as_ref().expect("TypeName requires query_ty");
                        let name = format!("{}", ty);
                        let interned = self.context.intern(&name);
                        let did = self.get_or_create_string(&StrId(interned));
                        let gv = self.module.declare_data_in_func(did, &mut builder.func);
                        let val = builder.ins().global_value(types::I64, gv);
                        if let Some(d) = dest {
                            var_map.insert(*d, val);
                        }
                    }

                    IntrinsicOp::AssertAlign => {} // Lowered differently
                    IntrinsicOp::AtomicLoad { ordering } => {
                        let dest = dest.expect("AtomicLoad requires a destination");

                        if args.len() != 1 {
                            panic!("AtomicLoad expects 1 argument, got {}", args.len());
                        }

                        let result_ssa_ty = func
                            .value_types
                            .get(&dest)
                            .unwrap_or_else(|| panic!("AtomicLoad: destination type missing"));

                        let atomic_ty = atomic_type_from_ssa(result_ssa_ty, "AtomicLoad");
                        let ptr = self.ptr_operand(builder, var_map, &args[0], "AtomicLoad");
                        let symbol = atomic_sym!("load", atomic_suffix(atomic_ty));
                        let order = self.atomic_ordering_value(builder, *ordering);
                        let addr_ty = self.addr_type();

                        let val = self
                            .call_c_helper(
                                builder,
                                symbol,
                                &[addr_ty, types::I32],
                                Some(atomic_ty),
                                &[ptr, order],
                            )
                            .expect("AtomicLoad: helper returned no value");

                        define(var_map, builder, dest, val, Some(atomic_ty), "AtomicLoad");
                    }

                    IntrinsicOp::AtomicStore { ordering } => {
                        if args.len() != 2 {
                            panic!("AtomicStore expects 2 arguments, got {}", args.len());
                        }

                        let atomic_ty =
                            Self::resolve_atomic_ty(func, query_ty, &args[1], None, "AtomicStore");
                        let ptr = self.ptr_operand(builder, var_map, &args[0], "AtomicStore");
                        let value = self.resolve_atomic_operand(
                            builder,
                            var_map,
                            &args[1],
                            atomic_ty,
                            "AtomicStore",
                        );
                        let symbol = atomic_sym!("store", atomic_suffix(atomic_ty));
                        let order = self.atomic_ordering_value(builder, *ordering);
                        let addr_ty = self.addr_type();

                        self.call_c_helper(
                            builder,
                            symbol,
                            &[addr_ty, atomic_ty, types::I32],
                            None,
                            &[ptr, value, order],
                        );
                    }

                    IntrinsicOp::AtomicSwap { ordering } => {
                        let dest = dest.expect("AtomicSwap requires a destination");

                        if args.len() != 2 {
                            panic!("AtomicSwap expects 2 arguments, got {}", args.len());
                        }

                        let atomic_ty = Self::resolve_atomic_ty(
                            func,
                            query_ty,
                            &args[1],
                            Some(dest),
                            "AtomicSwap",
                        );
                        let ptr = self.ptr_operand(builder, var_map, &args[0], "AtomicSwap");
                        let value = self.resolve_atomic_operand(
                            builder,
                            var_map,
                            &args[1],
                            atomic_ty,
                            "AtomicSwap",
                        );
                        let symbol = atomic_sym!("swap", atomic_suffix(atomic_ty));
                        let order = self.atomic_ordering_value(builder, *ordering);
                        let addr_ty = self.addr_type();

                        let old = self
                            .call_c_helper(
                                builder,
                                symbol,
                                &[addr_ty, atomic_ty, types::I32],
                                Some(atomic_ty),
                                &[ptr, value, order],
                            )
                            .expect("AtomicSwap: helper returned no value");

                        define(var_map, builder, dest, old, Some(atomic_ty), "AtomicSwap");
                    }

                    IntrinsicOp::AtomicCas { success, failure } => {
                        let dest = dest.expect("AtomicCas requires a destination");

                        if args.len() != 3 {
                            panic!("AtomicCas expects 3 arguments, got {}", args.len());
                        }

                        let atomic_ty = Self::resolve_atomic_ty(
                            func,
                            query_ty,
                            &args[2],
                            Some(dest),
                            "AtomicCas",
                        );
                        let ptr = self.ptr_operand(builder, var_map, &args[0], "AtomicCas");
                        let expected = self.resolve_atomic_operand(
                            builder,
                            var_map,
                            &args[1],
                            atomic_ty,
                            "AtomicCas(expected)",
                        );
                        let new = self.resolve_atomic_operand(
                            builder,
                            var_map,
                            &args[2],
                            atomic_ty,
                            "AtomicCas(new)",
                        );
                        let symbol = atomic_sym!("cas", atomic_suffix(atomic_ty));
                        let success_order = self.atomic_ordering_value(builder, *success);
                        let failure_order = self.atomic_ordering_value(builder, *failure);
                        let addr_ty = self.addr_type();

                        let old = self
                            .call_c_helper(
                                builder,
                                symbol,
                                &[addr_ty, atomic_ty, atomic_ty, types::I32, types::I32],
                                Some(atomic_ty),
                                &[ptr, expected, new, success_order, failure_order],
                            )
                            .expect("AtomicCas: helper returned no value");

                        define(var_map, builder, dest, old, Some(atomic_ty), "AtomicCas");
                    }

                    IntrinsicOp::AtomicFetchAdd { ordering }
                    | IntrinsicOp::AtomicFetchSub { ordering }
                    | IntrinsicOp::AtomicFetchAnd { ordering }
                    | IntrinsicOp::AtomicFetchOr { ordering }
                    | IntrinsicOp::AtomicFetchXor { ordering } => {
                        let dest = dest.expect("AtomicFetch* requires a destination");

                        if args.len() != 2 {
                            panic!("{op:?} expects 2 arguments, got {}", args.len());
                        }

                        let atomic_ty = Self::resolve_atomic_ty(
                            func,
                            query_ty,
                            &args[1],
                            Some(dest),
                            "AtomicFetch",
                        );
                        let ptr = self.ptr_operand(builder, var_map, &args[0], "AtomicFetch");
                        let value = self.resolve_atomic_operand(
                            builder,
                            var_map,
                            &args[1],
                            atomic_ty,
                            "AtomicFetch",
                        );

                        let suffix = atomic_suffix(atomic_ty);
                        let symbol = match op {
                            IntrinsicOp::AtomicFetchAdd { .. } => atomic_sym!("add", suffix),
                            IntrinsicOp::AtomicFetchSub { .. } => atomic_sym!("sub", suffix),
                            IntrinsicOp::AtomicFetchAnd { .. } => atomic_sym!("and", suffix),
                            IntrinsicOp::AtomicFetchOr { .. } => atomic_sym!("or", suffix),
                            IntrinsicOp::AtomicFetchXor { .. } => atomic_sym!("xor", suffix),
                            _ => unreachable!(),
                        };

                        let order = self.atomic_ordering_value(builder, *ordering);
                        let addr_ty = self.addr_type();

                        /*
                         * The C helper uses __atomic_fetch_*,
                         * so the returned value is the value from BEFORE the RMW.
                         */
                        let old = self
                            .call_c_helper(
                                builder,
                                symbol,
                                &[addr_ty, atomic_ty, types::I32],
                                Some(atomic_ty),
                                &[ptr, value, order],
                            )
                            .expect("AtomicFetch*: helper returned no value");

                        define(var_map, builder, dest, old, Some(atomic_ty), "AtomicFetch");
                    }

                    IntrinsicOp::AtomicFence { ordering } => {
                        if !args.is_empty() {
                            panic!("AtomicFence expects no arguments, got {}", args.len());
                        }

                        let order = self.atomic_ordering_value(builder, *ordering);
                        self.call_c_helper(
                            builder,
                            "__zeta_atomic_fence",
                            &[types::I32],
                            None,
                            &[order],
                        );
                    }

                    IntrinsicOp::CpuRelax => {
                        let relax_ref = self
                            .module
                            .declare_func_in_func(self.cpu_relax_func, &mut builder.func);
                        builder.ins().call(relax_ref, &[]);
                    }
                }
            }
            Instruction::GlobalAddr { dest, name } => {
                let id = *self
                    .static_data
                    .get(name)
                    .unwrap_or_else(|| panic!("GlobalAddr: undeclared static {:?}", name));
                let gv = self.module.declare_data_in_func(id, &mut builder.func);
                let addr = builder.ins().global_value(self.addr_type(), gv);
                let addr = self.coerce_addr(builder, addr, "GlobalAddr");
                var_map.insert(*dest, addr);
            }
        }
    }

    fn get_or_create_string(&mut self, s: &StrId) -> DataId {
        let vm_str = **s;
        if let Some(id) = self.string_data.get(&vm_str) {
            return id.0;
        }
        let n = self.string_data.len();
        let bytes = self.context.resolve_bytes(s);

        // raw bytes
        let bytes_id = self
            .module
            .declare_data(&format!("t_strb_{n}"), Linkage::Local, false, false)
            .unwrap();
        let mut bd = DataDescription::new();
        bd.define(bytes.to_vec().into_boxed_slice());
        self.module.define_data(bytes_id, &bd).unwrap();

        // descriptor { ptr, len }
        let desc_id = self
            .module
            .declare_data(&format!("t_str_{n}"), Linkage::Local, false, false)
            .unwrap();
        let mut dd = DataDescription::new();
        let mut blob = vec![0u8; 16];
        blob[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        dd.define(blob.into_boxed_slice());
        dd.set_align(8);
        let gv = self.module.declare_data_in_data(bytes_id, &mut dd);
        dd.write_data_addr(0, gv, 0);
        self.module.define_data(desc_id, &dd).unwrap();

        self.string_data.insert(vm_str, ZetaDataId(desc_id));
        desc_id
    }

    fn get_or_create_string_bytes(&mut self, s: &StrId) -> DataId {
        let key = *s;
        if let Some(id) = self.string_bytes.get(&key) {
            return *id;
        }
        let n = self.string_bytes.len();
        let bytes = self.context.resolve_bytes(s);
        let id = self
            .module
            .declare_data(&format!("t_sstrb_{n}"), Linkage::Local, false, false)
            .unwrap();
        let mut bd = DataDescription::new();
        bd.define(bytes.to_vec().into_boxed_slice());
        self.module.define_data(id, &bd).unwrap();
        self.string_bytes.insert(key, id);
        id
    }

    fn emit_main_wrapper(&mut self, zeta_main_fid: FuncId, args_init_fid: FuncId) {
        let mut sig = Signature::new(self.module.isa().default_call_conv());
        sig.params.push(AbiParam::new(types::I32)); // argc
        sig.params.push(AbiParam::new(self.addr_type())); // argv
        sig.returns.push(AbiParam::new(types::I32));

        let main_fid = self
            .module
            .declare_function("main", Linkage::Export, &sig)
            .expect("failed to declare main wrapper");

        let mut ctx = self.module.make_context();
        ctx.func = ClifFunction::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, main_fid.as_u32()),
            sig,
        );

        let mut fbctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let argc = builder.block_params(entry)[0];
        let argv = builder.block_params(entry)[1];

        let argc_usize = builder.ins().uextend(self.addr_type(), argc);

        // Call zeta::sys::args::init(argc, argv)
        let init_ref = self
            .module
            .declare_func_in_func(args_init_fid, &mut builder.func);

        builder.ins().call(init_ref, &[argc_usize, argv]);

        let zeta_main_ref = self
            .module
            .declare_func_in_func(zeta_main_fid, &mut builder.func);

        let call = builder.ins().call(zeta_main_ref, &[]);

        let first_result = builder.inst_results(call).first().copied();

        let ret_val = if let Some(result) = first_result {
            builder.ins().ireduce(types::I32, result)
        } else {
            builder.ins().iconst(types::I32, 0)
        };

        builder.ins().return_(&[ret_val]);

        builder.seal_all_blocks();
        builder.finalize();

        self.module
            .define_function(main_fid, &mut ctx)
            .expect("failed to define main wrapper");

        self.module.clear_context(&mut ctx);
    }

    fn coerce_for_block_arg(
        &self,
        builder: &mut FunctionBuilder,
        value: ClifValue,
        want: Type,
        site: &str,
    ) -> ClifValue {
        let have = builder.func.dfg.value_type(value);
        if have == want {
            return value;
        }
        eprintln!(
            "WARNING [{site}]: phi incoming value has clif type {:?} but the phi's \
             declared type is {:?}.",
            have, want
        );
        if have.is_int() && want.is_int() {
            if have.bits() < want.bits() {
                builder.ins().uextend(want, value)
            } else {
                builder.ins().ireduce(want, value)
            }
        } else {
            builder.ins().bitcast(want, MemFlags::new(), value)
        }
    }

    fn coerce_bitcast(
        &self,
        builder: &mut FunctionBuilder,
        src: ClifValue,
        dst_ty: Type,
    ) -> ClifValue {
        let src_ty = builder.func.dfg.value_type(src);
        if src_ty == dst_ty {
            return src;
        }
        if src_ty.bits() == dst_ty.bits() {
            return builder.ins().bitcast(dst_ty, MemFlags::new(), src);
        }
        if src_ty.is_int() && dst_ty.is_int() {
            return if src_ty.bits() < dst_ty.bits() {
                builder.ins().uextend(dst_ty, src)
            } else {
                builder.ins().ireduce(dst_ty, src)
            };
        }
        panic!(
            "coerce_bitcast: cannot reconcile {:?} ({} bits) with {:?} ({} bits)",
            src_ty,
            src_ty.bits(),
            dst_ty,
            dst_ty.bits()
        );
    }

    #[inline]
    fn is_zst(&self, ty: &SsaType) -> bool {
        sizeof_ssa(ty, self.target).map(|s| s == 0).unwrap_or(false)
    }

    fn resolve_atomic_operand(
        &self,
        builder: &mut FunctionBuilder,
        var_map: &ValueMap,
        operand: &Operand,
        expected_ty: Type,
        name: &str,
    ) -> ClifValue {
        match operand {
            Operand::Value(v) => {
                let value = get_val(var_map, v, name);
                let actual_ty = value_type(builder, value);

                if actual_ty != expected_ty {
                    panic!(
                        "{name}: operand has Cranelift type {:?}, expected {:?}",
                        actual_ty, expected_ty
                    );
                }

                value
            }

            Operand::ConstInt(i) => builder.ins().iconst(expected_ty, *i),

            _ => unimplemented!("{name}: unsupported operand {:?}", operand),
        }
    }
}

impl Backend for CraneliftBackend {
    fn emit_module(&mut self, module: &Module) {
        self.declare_statics(module);
        let mut zeta_main_fid = None;

        for (name, func) in &module.functions {
            let mut sig = Signature::new(self.module.isa().default_call_conv());
            let ret_is_aggregate =
                func.ret_type != SsaType::Void && Self::is_aggregate_ty(&func.ret_type);

            if ret_is_aggregate {
                sig.params.push(AbiParam::new(types::I64)); // hidden sret pointer, first
            }
            for param in &func.params {
                sig.params.push(AbiParam::new(clif_type(&param.1)));
            }
            if func.ret_type != SsaType::Void {
                let ret_clif_ty = if ret_is_aggregate {
                    types::I64
                } else {
                    clif_type(&func.ret_type)
                };
                sig.returns.push(AbiParam::new(ret_clif_ty));
            }

            let resolved_name = self.context.resolve_string(&*name);

            let has_body = !func.blocks.is_empty()
                && !(func.blocks.len() == 1 && func.blocks[0].instructions.is_empty());
            let is_extern_abi = matches!(
                func.function_metadata.extern_modifier,
                ExternModifier::Abi(_)
            );
            let linkage = match (is_extern_abi, has_body) {
                (true, false) => Linkage::Import,
                (true, true) => Linkage::Export,
                (false, _) => Linkage::Local,
            };

            let (actual_name, linkage) = if resolved_name == "main" && !self.main_emitted {
                zeta_main_fid = Some(name.clone());
                ("_zeta_main", Linkage::Local)
            } else if resolved_name == "main" && self.main_emitted {
                continue;
            } else {
                (resolved_name, linkage)
            };

            let fid = self
                .module
                .declare_function(actual_name, linkage, &sig)
                .unwrap_or_else(|e| {
                    panic!("failed to declare function {} due to {}", actual_name, e)
                });
            self.func_ids.insert(name.clone(), fid);
            self.func_param_types.insert(
                name.clone(),
                func.params.iter().map(|(_, ty)| clif_type(ty)).collect(),
            );
            self.func_ret_types
                .insert(name.clone(), func.ret_type.clone());
        }

        for func in module.functions.values() {
            if func.name.eq("main") && !self.func_ids.contains_key(&func.name) {
                continue;
            }
            let has_body = !func.blocks.is_empty()
                && !(func.blocks.len() == 1 && func.blocks[0].instructions.is_empty());
            let is_true_extern = matches!(
                func.function_metadata.extern_modifier,
                ExternModifier::Abi(_)
            ) && !has_body;
            if is_true_extern {
                self.emit_extern(func);
            } else if !func.name.eq("main") {
                self.emit_function(func);
            }
        }

        if let Some(zeta_main_name) = zeta_main_fid {
            if !self.main_emitted {
                let main_func = module
                    .functions
                    .values()
                    .find(|f| f.name.eq("main"))
                    .expect("main function missing from module");
                self.emit_function(main_func);

                let args_init_name = StrId::from_static("zeta_sys_args_init");
                let args_init_fid = *self
                    .func_ids
                    .get(&args_init_name)
                    .expect("zeta_sys_args_init was not declared");

                let fid = self.func_ids[&zeta_main_name];
                self.emit_main_wrapper(fid, args_init_fid);
                self.main_emitted = true;
            }
        }
    }

    fn emit_function(&mut self, func: &Function) {
        let ret_is_aggregate =
            func.ret_type != SsaType::Void && Self::is_aggregate_ty(&func.ret_type);

        let mut sig = Signature::new(self.module.isa().default_call_conv());
        if ret_is_aggregate {
            sig.params.push(AbiParam::new(types::I64));
        }
        for param in &func.params {
            sig.params.push(AbiParam::new(clif_type(&param.1)));
        }
        if func.ret_type != SsaType::Void {
            let ret_clif_ty = if ret_is_aggregate {
                types::I64
            } else {
                clif_type(&func.ret_type)
            };
            sig.returns.push(AbiParam::new(ret_clif_ty));
        }

        let fid = self.func_ids[&func.name];
        let mut ctx = match self.clif_ctx.take() {
            Some(c) => c,
            None => self.module.make_context(),
        };
        ctx.func = ClifFunction::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, fid.as_u32()),
            sig,
        );

        if let InlineModifier::Noinline = func.function_metadata.inline_modifier {
            // Cranelift hint for noinline
        }

        let mut fbctx = self
            .fb_ctx
            .take()
            .unwrap_or_else(FunctionBuilderContext::new);

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

            let mut block_idx: BlockIdx =
                HashMap::with_capacity_and_hasher(func.blocks.len(), FxHashBuilder);
            for (i, b) in func.blocks.iter().enumerate() {
                block_idx.insert(b.id, i);
            }

            let mut block_map: BlockMap =
                HashMap::with_capacity_and_hasher(func.blocks.len(), FxHashBuilder);
            for bb in &func.blocks {
                block_map.insert(bb.id, builder.create_block());
            }

            let entry = block_map[&func.entry];
            // Function parameters first, so they occupy the leading entry-block params.
            builder.append_block_params_for_function_params(entry);

            let mut phi_param_map: PhiMap = HashMap::with_hasher(FxHashBuilder);
            for bb in &func.blocks {
                let clif_bb = block_map[&bb.id];
                let mut phi_index = builder.block_params(clif_bb).len();

                for inst in &bb.instructions {
                    if let Instruction::Phi { dest, .. } = inst {
                        let ty = func.value_types.get(dest).unwrap();
                        let clif_ty = clif_type(ty);

                        builder.append_block_param(clif_bb, clif_ty);

                        phi_param_map.insert((bb.id, *dest), phi_index);
                        phi_index += 1;
                    } else {
                        break;
                    }
                }
            }

            let inst_count: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
            let mut var_map: ValueMap =
                HashMap::with_capacity_and_hasher(inst_count + func.params.len(), FxHashBuilder);

            // Bind the IR parameters (and the sret pointer) directly to the entry
            // block parameters. They dominate every other block.
            self.current_sret_ptr = None;
            let param_offset = if ret_is_aggregate {
                self.current_sret_ptr = Some(builder.block_params(entry)[0]);
                1
            } else {
                0
            };
            for (i, &(v, _)) in func.params.iter().enumerate() {
                var_map.insert(v, builder.block_params(entry)[i + param_offset]);
            }

            for bb in &func.blocks {
                let clif_bb = block_map[&bb.id];
                builder.switch_to_block(clif_bb);

                for inst in &bb.instructions {
                    self.lower_basic_inst(
                        inst,
                        bb.id,
                        &mut builder,
                        &mut var_map,
                        &block_map,
                        &phi_param_map,
                        func,
                        &block_idx,
                    );
                }

                let last_was_terminator = bb
                    .instructions
                    .last()
                    .map_or(false, |i| inst_is_terminator(i));
                if !last_was_terminator {
                    self.emit_zero_return(&mut builder, func);
                }
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        if let Err(e) = self.module.define_function(fid, &mut ctx) {
            eprintln!(
                "=== failed to compile `{}` ===\n{}\n--- CLIF ---\n{}",
                func.name,
                e,
                ctx.func.display()
            );
            panic!("define_function failed for `{}`", func.name);
        }
        self.current_sret_ptr = None;

        if self.emit_asm {
            println!("==========================");
            println!("Assembly for function `{}`:", &func.name);
            println!("==========================");
            println!("--- Cranelift IR ---");
            println!("{}", ctx.func.display());
        }

        self.module.clear_context(&mut ctx);
        self.clif_ctx = Some(ctx);
        self.fb_ctx = Some(fbctx);
    }

    fn emit_extern(&mut self, func: &Function) {
        let mut sig = Signature::new(self.module.isa().default_call_conv());
        for param in &func.params {
            sig.params.push(AbiParam::new(clif_type(&param.1)));
        }

        if func.ret_type != SsaType::Void {
            sig.returns.push(AbiParam::new(clif_type(&func.ret_type)));
        }

        self.module
            .declare_function(
                self.context.resolve_string(&*func.name),
                Linkage::Import,
                &sig,
            )
            .unwrap();
    }

    fn finish(self, out_dir: &PathBuf) -> Result<PathBuf, EmitError> {
        let obj: ObjectProduct = self.module.finish();

        let data: Vec<u8> = obj.emit().map_err(|e| EmitError::Emit(e))?;

        std::fs::create_dir_all(&out_dir).map_err(|e| EmitError::Io(e))?;

        let out_path = out_dir.join("out.o");

        File::create(&out_path)
            .and_then(|mut file| file.write_all(&data))
            .map_err(|e| EmitError::Io(e))?;

        Ok(out_path)
    }
}

#[derive(Debug)]
pub enum EmitError {
    Io(io::Error),
    Emit(object::write::Error),
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::Io(e) => write!(f, "IO error: {}", e),
            EmitError::Emit(e) => write!(f, "Emit error: {}", e),
        }
    }
}

impl Error for EmitError {}

/// Looks up the Cranelift value that defines an IR value.
#[inline]
fn get_val(var_map: &ValueMap, v: &Value, what: &str) -> ClifValue {
    match var_map.get(v) {
        Some(x) => *x,
        None => panic!(
            "{what}: use of undefined value {:?} (not lowered yet, or defined in a block \
             that does not come before this use)",
            v
        ),
    }
}

/// Records the Cranelift value that defines an IR value, warning if the Cranelift
/// type disagrees with the type predicted from `func.value_types`.
#[inline]
fn define(
    var_map: &mut ValueMap,
    builder: &FunctionBuilder,
    dest: Value,
    value: ClifValue,
    expected: Option<Type>,
    site: &str,
) {
    if let Some(exp) = expected {
        let actual = builder.func.dfg.value_type(value);
        if exp != actual {
            eprintln!(
                "WARNING [{site}]: SsaType predicted clif type {:?} but the actual \
                 value produced is {:?}. This mismatch means func.value_types \
                 disagrees with codegen here.",
                exp, actual
            );
        }
    }
    var_map.insert(dest, value);
}

#[inline]
fn value_type(builder: &FunctionBuilder, value: ClifValue) -> Type {
    builder.func.dfg.value_type(value)
}

fn atomic_type_from_ssa(ty: &SsaType, name: &str) -> Type {
    let clif_ty = clif_type(ty);

    if !clif_ty.is_int() {
        panic!(
            "{name}: atomic operations require an integer type, got {:?} -> {:?}",
            ty, clif_ty
        );
    }

    clif_ty
}

fn is_unsigned_ssa_type(ty: &SsaType) -> bool {
    matches!(
        ty,
        SsaType::U8 | SsaType::U16 | SsaType::U32 | SsaType::U64 | SsaType::U128 | SsaType::Usize
    )
}

fn atomic_suffix(ty: types::Type) -> &'static str {
    match ty {
        types::I8 => "u8",
        types::I16 => "u16",
        types::I32 => "u32",
        types::I64 => "u64",
        _ => panic!("unsupported atomic Cranelift type: {ty}"),
    }
}
