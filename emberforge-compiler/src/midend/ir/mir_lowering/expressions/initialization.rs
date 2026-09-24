use ir::{
    hir::{HirEnum, HirExpr, HirFieldInit, StrId},
    ir_conversion::lower_type_hir,
    layout::TargetInfo,
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};

use crate::{
    midend::ir::mir_lowering::{FunctionLowerer, lowerer::FieldInitVal},
    optimized_string_buffering,
};

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn try_lower_bare_enum_variant(
        &mut self,
        enum_name: &StrId,
        variant: &StrId,
    ) -> Option<Value> {
        let hir_enum = self.enums.get(enum_name).or_else(|| {
            let base_str = self.context.resolve_string(enum_name);
            let candidates: Vec<&HirEnum> = self
                .enums
                .values()
                .filter(|e| {
                    e.name.as_str() == base_str && e.variants.iter().any(|v| v.name == *variant)
                })
                .collect();
            if candidates.len() == 1 {
                Some(candidates[0])
            } else {
                candidates.into_iter().find(|e| e.name == *enum_name)
            }
        })?;

        let resolved_enum_name = hir_enum.name;
        hir_enum
            .variants
            .iter()
            .any(|v| v.name == *variant)
            .then(|| self.lower_enum_init(&resolved_enum_name, variant, &[], &Default::default()))
    }

    pub(crate) fn lower_enum_init(
        &mut self,
        enum_name: &StrId,
        variant: &StrId,
        args: &[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
    ) -> Value {
        let enums = self.enums;
        let hir_enum = enums
            .get(enum_name)
            .unwrap_or_else(|| panic!("[lower_enum_init] unknown enum `{}` in {span}.", enum_name));
        let resolved_enum_name = hir_enum.name;

        let lowered_variants: Vec<Vec<SsaType>> = hir_enum
            .variants
            .iter()
            .map(|v| {
                v.fields
                    .iter()
                    .map(|f| lower_type_hir(&f.field_type, enums))
                    .collect()
            })
            .collect();

        let tag = hir_enum
            .variants
            .iter()
            .position(|v| v.name == *variant)
            .unwrap_or_else(|| {
                panic!(
                    "[lower_enum_init] enum `{}` has no variant `{}` in {span}",
                    resolved_enum_name, variant
                )
            });

        let field_tys = lowered_variants[tag].clone();
        let (offsets, _) = Self::payload_layout(&field_tys);
        let max_payload = lowered_variants
            .iter()
            .map(|tys| Self::payload_layout(tys).1)
            .max()
            .unwrap_or(0);

        let mut inits = Vec::with_capacity(args.len());
        for (arg, fty) in args.iter().zip(field_tys.iter()) {
            inits.push(self.lower_init_operand(arg, fty));
        }

        let tag_v = self.current_block_data.fresh_value();
        self.emit(Instruction::Const {
            dest: tag_v,
            ty: SsaType::I64,
            value: Operand::ConstInt(tag as i64),
        });
        self.current_block_data
            .value_types
            .insert(tag_v, SsaType::I64);

        let obj = self.current_block_data.fresh_value();
        self.emit(Instruction::StackAlloc {
            dest: obj,
            ty: SsaType::Array(Box::new(SsaType::I8), 8 + max_payload),
            count: 1,
        });
        self.current_block_data.value_types.insert(
            obj,
            SsaType::Enum {
                name: resolved_enum_name,
                variants: lowered_variants,
            },
        );

        self.emit(Instruction::StoreField {
            base: Operand::Value(obj),
            offset: 0,
            value: Operand::Value(tag_v),
        });

        for ((init, fty), off) in inits.into_iter().zip(field_tys.iter()).zip(offsets.iter()) {
            self.store_init(obj, 8 + off, fty, init);
        }

        obj
    }

    pub(crate) fn store_vtable_if_any(&mut self, obj: Value, struct_name: StrId) {
        let Some(vslots) = self.struct_vtable_slots.get(&struct_name) else {
            return;
        };
        if vslots.is_empty() {
            return;
        }

        let vtable_name =
            optimized_string_buffering::make_vtable_name(struct_name, self.context.clone());
        self.emit(Instruction::StoreField {
            base: Operand::Value(obj),
            offset: 0usize,
            value: Operand::GlobalRef(vtable_name),
        });
    }

    pub(crate) fn lower_struct_init(
        &mut self,
        name: &HirExpr,
        args: &[HirFieldInit<'a, 'bump>],
        span: SourceSpan<'a>,
    ) -> Value {
        let struct_name = match name {
            HirExpr::Ident(n, _) => *n,
            other => panic!("StructInit name must be identifier; got {:?}", other),
        };

        let structs = self.structs;
        let hir_struct = structs
            .get(&struct_name)
            .unwrap_or_else(|| panic!("Struct {} not found at {span}", struct_name));
        let field_types: Vec<SsaType> = hir_struct
            .fields
            .iter()
            .map(|f| lower_type_hir(&f.field_type, self.enums))
            .collect();

        let mut inits: Vec<(StrId, SsaType, FieldInitVal)> = Vec::with_capacity(args.len());
        for arg in args {
            let idx = hir_struct
                .fields
                .iter()
                .position(|f| f.name == arg.name)
                .unwrap_or_else(|| {
                    panic!("Struct {} has no field {} at {span}", struct_name, arg.name)
                });
            let field_ty = field_types[idx].clone();
            if Self::is_move_by_value(&field_ty) {
                self.record_arg_move(&arg.value);
            }
            let init = self.lower_init_operand(&arg.value, &field_ty);
            inits.push((arg.name, field_ty, init));
        }

        let alloc_ty = SsaType::User(struct_name, field_types);
        let obj = self.new_value();
        self.emit(Instruction::StackAlloc {
            dest: obj,
            ty: alloc_ty.clone(),
            count: 0,
        });
        self.current_block_data.value_types.insert(obj, alloc_ty);

        let offsets_map = self.struct_field_offsets;
        let offsets = offsets_map
            .get(&struct_name)
            .unwrap_or_else(|| panic!("Unknown struct {} when initializing", struct_name));
        for (fname, fty, init) in inits {
            let offset = *offsets
                .get(&fname)
                .unwrap_or_else(|| panic!("Unknown field {} on struct {}", fname, struct_name));
            self.store_init(obj, offset, &fty, init);
        }

        self.store_vtable_if_any(obj, struct_name);
        obj
    }

    pub(crate) fn payload_layout(field_tys: &[SsaType]) -> (Vec<usize>, usize) {
        let target = TargetInfo { ptr_bytes: 8 };
        let mut cursor = 0usize;
        let mut offsets = Vec::with_capacity(field_tys.len());
        for ty in field_tys {
            let (size, align) = ir::layout::layout_of_ssa(ty, target)
                .map(|l| (l.size, l.align))
                .unwrap_or((8, 8));
            cursor = Self::align_up(cursor, align);
            offsets.push(cursor);
            cursor += size;
        }
        (offsets, cursor)
    }
}
