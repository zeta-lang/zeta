use crate::arena::GrowableAtomicBump;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry as DashEntry;
use smallvec::SmallVec;

use std::alloc::AllocError;
use std::collections::HashMap;
use std::fmt::{self, Formatter, Write};
use std::hash::{BuildHasher, Hash, Hasher};
use std::ptr;
use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;
use std::str::from_utf8_unchecked;

const LANES: usize = 32;
const FX_INIT: u64 = 0xcbf29ce484222325;
const FX_PRIME: u64 = 0x100000001b3;
const MEGABYTE: usize = 1024 * 1024;

struct SmallVecWriter<const N: usize> {
    buf: SmallVec<u8, N>,
}

impl<const N: usize> SmallVecWriter<N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            buf: SmallVec::new(),
        }
    }
}

impl<const N: usize> Write for SmallVecWriter<N> {
    #[inline(always)]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

#[derive(Default, Clone, Copy)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        let mut acc = 0u64;

        for &b in bytes {
            acc = acc.wrapping_mul(0x100).wrapping_add(b as u64);
        }

        self.0 = acc;
    }
}

#[derive(Default, Clone, Copy)]
struct IdentityBuild;

impl BuildHasher for IdentityBuild {
    type Hasher = IdentityHasher;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher::default()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VmString {
    pub offset: *const u8,
    pub length: usize,
}

impl fmt::Debug for VmString {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for VmString {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl VmString {
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        unsafe { from_utf8_unchecked(std::slice::from_raw_parts(self.offset, self.length)) }
    }
}

impl PartialEq for VmString {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        if self.offset.is_null() || other.offset.is_null() {
            return self.offset == other.offset;
        }

        let (a, b): (&[u8], &[u8]) = unsafe {
            (
                std::slice::from_raw_parts(self.offset, self.length),
                std::slice::from_raw_parts(other.offset, other.length),
            )
        };

        a == b
    }
}

impl Eq for VmString {}

impl Default for VmString {
    #[inline]
    fn default() -> Self {
        Self {
            offset: ptr::null(),
            length: 0,
        }
    }
}

impl Hash for VmString {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            std::slice::from_raw_parts(self.offset, self.length).hash(state);
        }
    }
}

unsafe impl Send for VmString {}
unsafe impl Sync for VmString {}

type StringBucket = SmallVec<VmString, 2>;

#[derive(Debug)]
pub struct StringPool {
    data_buffer: GrowableAtomicBump<'static>,

    interned_strings: DashMap<u64, StringBucket, IdentityBuild>,
}

#[derive(Debug)]
pub struct ThreadLocalStringPool<'a> {
    global: &'a StringPool,
    local_strings: HashMap<u64, StringBucket, IdentityBuild>,
}

impl StringPool {
    #[inline]
    pub fn new() -> Result<Self, AllocError> {
        Ok(Self {
            data_buffer: GrowableAtomicBump::with_capacity_and_aligned(2 * MEGABYTE, 32)?,

            interned_strings: DashMap::with_hasher(IdentityBuild),
        })
    }

    /// Create a per-worker interner.
    ///
    /// Keep one of these around for the lifetime of a compiler worker
    /// instead of constructing one for every string operation.
    #[inline]
    pub fn thread_local(&self) -> ThreadLocalStringPool<'_> {
        ThreadLocalStringPool {
            global: self,
            local_strings: HashMap::with_hasher(IdentityBuild),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data_buffer.len()
    }

    #[inline]
    pub fn intern(&self, s: &str) -> VmString {
        self.intern_bytes(s.as_bytes())
    }

    #[inline]
    pub fn intern_bytes(&self, bytes: &[u8]) -> VmString {
        let hash = Self::hash_bytes(bytes);
        self.intern_global(bytes, hash)
    }

    #[inline]
    pub fn intern_fmt(&self, args: fmt::Arguments<'_>) -> VmString {
        let mut writer = SmallVecWriter::<128>::new();

        fmt::write(&mut writer, args).expect("formatting failed");

        let hash = Self::hash_bytes(&writer.buf);

        self.intern_global(&writer.buf, hash)
    }

    #[inline]
    fn intern_global(&self, bytes: &[u8], hash: u64) -> VmString {
        match self.interned_strings.entry(hash) {
            DashEntry::Occupied(mut entry) => {
                let collision_list = entry.get();

                if let Some(vm_string) = Self::find_simd(self, collision_list, bytes) {
                    return vm_string;
                }

                let vm_string = self.allocate_global(bytes);

                entry.get_mut().push(vm_string);

                vm_string
            }

            DashEntry::Vacant(entry) => {
                let vm_string = self.allocate_global(bytes);

                let mut bucket = StringBucket::new();
                bucket.push(vm_string);

                entry.insert(bucket);

                vm_string
            }
        }
    }

    #[inline(always)]
    fn allocate_global(&self, bytes: &[u8]) -> VmString {
        let slice = self
            .data_buffer
            .alloc_many(bytes)
            .expect("Failed to allocate string slice");

        VmString {
            offset: slice.as_ptr(),
            length: bytes.len(),
        }
    }

    #[inline(always)]
    pub fn resolve_bytes(&self, vm_string: &VmString) -> &[u8] {
        unsafe { std::slice::from_raw_parts(vm_string.offset, vm_string.length) }
    }

    #[inline(always)]
    pub fn resolve_string(&self, vm_string: &VmString) -> &str {
        unsafe { from_utf8_unchecked(self.resolve_bytes(vm_string)) }
    }

    #[inline(always)]
    fn eq_simd(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        if a.len() <= 16 {
            return a == b;
        }

        let mut i = 0;

        while i + LANES <= a.len() {
            let va = Simd::<u8, LANES>::from_slice(&a[i..]);
            let vb = Simd::<u8, LANES>::from_slice(&b[i..]);

            if !va.simd_eq(vb).all() {
                return false;
            }

            i += LANES;
        }

        a[i..] == b[i..]
    }

    fn find_simd(&self, collision_list: &StringBucket, bytes: &[u8]) -> Option<VmString> {
        for &vm_string in collision_list.iter() {
            if vm_string.length != bytes.len() {
                continue;
            }

            let stored = self.resolve_bytes(&vm_string);

            if Self::eq_simd(stored, bytes) {
                return Some(vm_string);
            }
        }

        None
    }

    #[inline(always)]
    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hash = FX_INIT;

        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FX_PRIME);
        }

        hash
    }
}

impl ThreadLocalStringPool<'_> {
    pub fn intern(&mut self, s: &str) -> VmString {
        self.intern_bytes(s.as_bytes())
    }

    pub fn intern_bytes(&mut self, bytes: &[u8]) -> VmString {
        let hash = StringPool::hash_bytes(bytes);

        if let Some(collision_list) = self.local_strings.get(&hash) {
            if let Some(vm_string) = StringPool::find_simd(self.global, collision_list, bytes) {
                return vm_string;
            }
        }

        let vm_string = self.global.intern_global(bytes, hash);

        self.local_strings.entry(hash).or_default().push(vm_string);

        vm_string
    }

    pub fn intern_fmt(&mut self, args: fmt::Arguments<'_>) -> VmString {
        let mut writer = SmallVecWriter::<128>::new();

        fmt::write(&mut writer, args).expect("formatting failed");

        self.intern_bytes(&writer.buf)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.local_strings.reserve(additional);
    }

    pub fn clear_local_cache(&mut self) {
        self.local_strings.clear();
    }
}

#[macro_export]
macro_rules! intern_fmt {
    ($pool:expr, $($arg:tt)*) => {
        $pool.intern_fmt(format_args!($($arg)*))
    };
}
