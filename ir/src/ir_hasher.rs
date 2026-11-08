pub use fxhash::FxHashMap;
pub use stable_hash_proc_macro::StableHash;
use std::{hash::BuildHasher, sync::LazyLock};
use xxhash_rust::xxh3::Xxh3;
use zetaruntime::string_pool::StringPool;

use crate::{hir::StrId, span::SourceSpan};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FxHashBuilder;

pub type HashMap<K, V> = FxHashMap<K, V>;
pub type HashSet<T> = std::collections::HashSet<T, FxHashBuilder>;

impl BuildHasher for FxHashBuilder {
    type Hasher = fxhash::FxHasher;
    fn build_hasher(&self) -> Self::Hasher {
        fxhash::FxHasher::default()
    }
}

static FINGERPRINT_VERSION: LazyLock<u128> = LazyLock::new(|| compiler_identity());

fn compiler_identity() -> u128 {
    let exe = std::env::current_exe().expect("current_exe");
    let bytes = std::fs::read(exe).expect("read compiler binary");
    xxhash_rust::xxh3::xxh3_128(&bytes)
}

pub struct FpHasher(xxhash_rust::xxh3::Xxh3);

impl FpHasher {
    pub fn new() -> Self {
        let mut h = Self(Xxh3::new());
        h.u128(*LazyLock::force(&FINGERPRINT_VERSION));
        h
    }

    pub fn u8(&mut self, v: u8) {
        self.0.update(&[v]);
    }

    pub fn u32(&mut self, v: u32) {
        self.0.update(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.0.update(&v.to_le_bytes());
    }

    pub fn u128(&mut self, v: u128) {
        self.0.update(&v.to_le_bytes());
    }

    pub fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.update(s.as_bytes());
    }

    pub fn finish(self) -> u128 {
        self.0.digest128()
    }
}

/// Everything a node needs besides itself in order to be hashed.
pub struct HashCtx<'p> {
    pub pool: &'p StringPool,
}

/// Hashes the *meaning* of a value into an `FpHasher`.
///
/// Never hash anything that's not persistent per session directly such as `usize`, interned ids, pointers or spans.
/// Write enum tags explicitly and length-prefix anything variable-sized.
pub trait StableHash {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher);
}

impl StableHash for StrId {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.str(self.as_str());
    }
}

impl StableHash for SourceSpan<'_> {
    fn stable_hash(&self, _: &HashCtx<'_>, _: &mut FpHasher) {}
}

macro_rules! impl_int {
    ($($t:ty => $m:ident),+ $(,)?) => {$(
        impl StableHash for $t {
            #[inline]
            fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
                h.$m(*self as _);
            }
        }
    )+};
}
impl_int!(
    u8 => u8, i8 => u8,
    u16 => u32, i16 => u32, u32 => u32, i32 => u32,
    u64 => u64, i64 => u64, usize => u64, isize => u64,
    u128 => u128, i128 => u128,
);

impl StableHash for bool {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.u8(*self as u8);
    }
}

impl StableHash for char {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.u32(*self as u32);
    }
}

impl StableHash for f32 {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.u32(self.to_bits());
    }
}

impl StableHash for f64 {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.u64(self.to_bits());
    }
}

impl StableHash for () {
    fn stable_hash(&self, _: &HashCtx<'_>, _: &mut FpHasher) {}
}

impl StableHash for str {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.str(self);
    }
}

impl StableHash for String {
    fn stable_hash(&self, _: &HashCtx<'_>, h: &mut FpHasher) {
        h.str(self);
    }
}

impl<T: StableHash + ?Sized> StableHash for &T {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        (**self).stable_hash(cx, h);
    }
}

impl<T: StableHash + ?Sized> StableHash for Box<T> {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        (**self).stable_hash(cx, h);
    }
}

impl<T: StableHash + ?Sized> StableHash for std::sync::Arc<T> {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        (**self).stable_hash(cx, h);
    }
}

impl<T: StableHash> StableHash for Option<T> {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        match self {
            None => h.u8(0),
            Some(v) => {
                h.u8(1);
                v.stable_hash(cx, h);
            }
        }
    }
}

impl<T: StableHash> StableHash for [T] {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        h.u64(self.len() as u64);
        for x in self {
            x.stable_hash(cx, h);
        }
    }
}
impl<T: StableHash, const N: usize> StableHash for [T; N] {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        self.as_slice().stable_hash(cx, h);
    }
}
impl<T: StableHash> StableHash for Vec<T> {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        self.as_slice().stable_hash(cx, h);
    }
}

impl<T: StableHash, const N: usize> StableHash for smallvec::SmallVec<T, N> {
    fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
        self.as_slice().stable_hash(cx, h);
    }
}

macro_rules! impl_tuple {
    ($($n:ident),+) => {
        impl<$($n: StableHash),+> StableHash for ($($n,)+) {
            #[allow(non_snake_case)]
            fn stable_hash(&self, cx: &HashCtx<'_>, h: &mut FpHasher) {
                let ($($n,)+) = self;
                $( $n.stable_hash(cx, h); )+
            }
        }
    };
}
impl_tuple!(A);
impl_tuple!(A, B);
impl_tuple!(A, B, C);
impl_tuple!(A, B, C, D);
