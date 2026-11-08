use cranelift_codegen::ir::{InstBuilder, StackSlotData, StackSlotKind, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;
use cranelift_object::ObjectModule;

const ZST_SENTINEL_ADDR: i64 = 0x1;

pub fn stack_alloc(
    builder: &mut FunctionBuilder,
    module: &ObjectModule,
    size_bytes: usize,
) -> Value {
    let ptr_ty = module.isa().pointer_type();

    if size_bytes == 0 {
        return builder.ins().iconst(ptr_ty, ZST_SENTINEL_ADDR);
    }

    let aligned_size = size_bytes.div_ceil(8);
    let slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        aligned_size as u32,
        3,
    ));
    builder.ins().stack_addr(ptr_ty, slot, 0)
}
