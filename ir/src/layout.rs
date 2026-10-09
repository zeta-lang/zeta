use crate::ssa_ir::SsaType;

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub size: usize,
    pub align: usize,
}

#[derive(Debug)]
pub enum LayoutError {
    Unsized(&'static str),
    Recursive,
    Unknown,
}

#[derive(Clone, Copy)]
pub struct TargetInfo {
    pub ptr_bytes: u64,
}

#[inline(always)]
const fn round_up(x: usize, align: usize) -> usize {
    if align <= 1 {
        return x;
    }
    let m = align - 1;
    (x + m) & !m
}

pub struct EnumLayout {
    pub layout: Layout,
    pub tag_offset: usize,
    pub tag_size: usize,
    pub payload_offset: usize,
}

pub fn enum_layout_of_ssa(
    variants: &[Vec<SsaType>],
    target: TargetInfo,
) -> Result<EnumLayout, LayoutError> {
    if variants.is_empty() {
        return Ok(EnumLayout {
            layout: Layout { size: 0, align: 1 },
            tag_offset: 0,
            tag_size: 0,
            payload_offset: 0,
        });
    }

    let mut max_payload = Layout { size: 0, align: 1 };
    for fields in variants {
        let l = layout_of_ssa(&SsaType::Tuple(fields.clone()), target)?;
        max_payload.size = max_payload.size.max(l.size);
        max_payload.align = max_payload.align.max(l.align);
    }

    let tag_size = 8usize;
    let tag_offset = 0usize;
    let payload_offset = round_up_to_align(tag_size, max_payload.align.max(1));
    let struct_align = max_payload.align.max(tag_size);
    let size = round_up_to_align(payload_offset + max_payload.size, struct_align);

    Ok(EnumLayout {
        layout: Layout {
            size,
            align: struct_align,
        },
        tag_offset,
        tag_size,
        payload_offset,
    })
}

pub fn sizeof_ssa(ty: &SsaType, target: TargetInfo) -> Result<usize, LayoutError> {
    Ok(layout_of_ssa(ty, target)?.size)
}

pub fn alignof_ssa(ty: &SsaType, target: TargetInfo) -> Result<usize, LayoutError> {
    Ok(layout_of_ssa(ty, target)?.align)
}

pub fn layout_of_ssa(ty: &SsaType, target: TargetInfo) -> Result<Layout, LayoutError> {
    match ty {
        SsaType::Void => Ok(Layout { size: 0, align: 1 }),
        SsaType::Bool | SsaType::I8 | SsaType::U8 => Ok(Layout { size: 1, align: 1 }),
        SsaType::I16 | SsaType::U16 => Ok(Layout { size: 2, align: 2 }),
        SsaType::I32 | SsaType::U32 | SsaType::F32 => Ok(Layout { size: 4, align: 4 }),
        SsaType::I64 | SsaType::U64 | SsaType::F64 => Ok(Layout { size: 8, align: 8 }),

        SsaType::Slice(_) => Ok(Layout { size: 16, align: 8 }),

        SsaType::Dyn => Ok(Layout { size: 8, align: 8 }),

        // Tuples/structs: sequential fields with padding between and at end to struct align.
        SsaType::Tuple(fields) | SsaType::User(_, _, fields) => {
            let mut off = 0usize;
            let mut max_align = 1usize;
            for fty in fields {
                let f: Layout = layout_of_ssa(fty, target)?;
                max_align = max_align.max(f.align);
                off = round_up(off, f.align);
                off = off.checked_add(f.size).ok_or(LayoutError::Unknown)?;
            }
            let size = round_up(off, max_align);
            Ok(Layout {
                size,
                align: max_align,
            })
        }

        // Enums/sum types: Simple tagged union:
        SsaType::Enum { variants, .. } => Ok(enum_layout_of_ssa(variants, target)?.layout),

        SsaType::I128 => Ok(Layout { size: 16, align: 8 }),
        SsaType::Isize => Ok(Layout { size: 8, align: 8 }),
        SsaType::Usize => Ok(Layout { size: 8, align: 8 }),
        SsaType::String => Ok(Layout { size: 16, align: 8 }),
        SsaType::U128 => Ok(Layout { size: 16, align: 8 }),
        SsaType::Pointer(_, _) => Ok(Layout {
            size: target.ptr_bytes as usize,
            align: target.ptr_bytes as usize,
        }),
        SsaType::Null => Ok(Layout { size: 0, align: 1 }),
        SsaType::Char => Ok(Layout { size: 4, align: 4 }),
        SsaType::Interface(_str_id, _) => todo!(),
        SsaType::Nullable(inner) => {
            if inner.is_pointer() {
                layout_of_ssa(inner, target)
            } else {
                let p = layout_of_ssa(inner, target)?;
                let off = round_up_to_align(1, p.align.max(1));
                let align = p.align.max(8);
                Ok(Layout {
                    size: round_up_to_align(off + p.size, align),
                    align,
                })
            }
        }

        SsaType::Array(ssa_type, length) => Ok(Layout {
            size: sizeof_ssa(ssa_type, TargetInfo { ptr_bytes: 8 })? * length,
            align: 8,
        }),

        SsaType::Owned(inner) => match inner.as_ref() {
            SsaType::Slice(_) => Ok(Layout { size: 24, align: 8 }),
            _ => Ok(Layout {
                size: target.ptr_bytes as usize,
                align: target.ptr_bytes as usize,
            }),
        },
        SsaType::FuncPointer { .. } => Ok(Layout {
            size: target.ptr_bytes as usize,
            align: target.ptr_bytes as usize,
        }),
    }
}

pub fn round_up_to_align(offset: usize, align: usize) -> usize {
    debug_assert!(
        align.is_power_of_two(),
        "alignment must be a power of two, got {align}"
    );
    (offset + align - 1) & !(align - 1)
}

pub fn nullable_payload_offset(inner: &SsaType, target: TargetInfo) -> Result<usize, LayoutError> {
    Ok(round_up_to_align(1, alignof_ssa(inner, target)?.max(1)))
}
