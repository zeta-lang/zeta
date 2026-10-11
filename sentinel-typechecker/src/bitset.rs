use smallvec::SmallVec;

/// Contiguous identifier for a local variable within a function (0..N).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct LocalBitSet {
    words: SmallVec<u64, 8>,
}

impl LocalBitSet {
    pub fn same_bits(&self, other: &Self) -> bool {
        let n = self.words.len().max(other.words.len());
        (0..n).all(|i| {
            self.words.get(i).copied().unwrap_or(0) == other.words.get(i).copied().unwrap_or(0)
        })
    }

    #[inline(always)]
    pub fn with_capacity(num_locals: usize) -> Self {
        let words_needed = (num_locals + 63) / 64;
        let mut words = SmallVec::new();
        words.resize(words_needed.max(1), 0);
        Self { words }
    }

    #[inline(always)]
    pub fn test(&self, id: LocalId) -> bool {
        let idx = id.0 as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        self.words
            .get(word_idx)
            .map_or(false, |&w| (w & (1 << bit_idx)) != 0)
    }

    #[inline(always)]
    pub fn set(&mut self, id: LocalId) {
        let idx = id.0 as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        self.words[word_idx] |= 1 << bit_idx;
    }

    #[inline(always)]
    pub fn clear(&mut self, id: LocalId) {
        let idx = id.0 as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        if word_idx < self.words.len() {
            self.words[word_idx] &= !(1 << bit_idx);
        }
    }

    #[inline(always)]
    pub fn union_with(&mut self, other: &Self) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (w, &ow) in self.words.iter_mut().zip(&other.words) {
            *w |= ow;
        }
    }

    #[inline(always)]
    pub fn intersect_with(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }
        for i in min_len..self.words.len() {
            self.words[i] = 0;
        }
    }

    /// Fast fixpoint check: returns true if self is a superset of other
    /// (i.e., other contains no bits that self does not have).
    #[inline(always)]
    pub fn is_superset_of(&self, other: &Self) -> bool {
        for (i, &ow) in other.words.iter().enumerate() {
            let sw = self.words.get(i).copied().unwrap_or(0);
            if (ow & !sw) != 0 {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
}
