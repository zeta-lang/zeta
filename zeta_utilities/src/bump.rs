use std::alloc::{AllocError, Allocator, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;

#[inline(always)]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

#[derive(Debug)]
struct Chunk {
    ptr: NonNull<u8>,
    capacity: usize,
    offset: usize,
    align: usize,
}

impl Chunk {
    fn new(capacity: usize, align: usize) -> Self {
        let layout = unsafe { Layout::from_size_align_unchecked(capacity, align) };
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            capacity,
            offset: 0,
            align,
        }
    }

    fn try_alloc(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let ptr = self.ptr.as_ptr() as usize;
        let base = ptr + self.offset;
        let aligned = align_up(base, layout.align());
        let end = aligned + layout.size();
        let new_offset = end - ptr;

        if new_offset > self.capacity {
            return None;
        }

        self.offset = new_offset;
        Some(unsafe { NonNull::new_unchecked(aligned as *mut u8) })
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = unsafe { Layout::from_size_align_unchecked(self.capacity, self.align) };
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) }
    }
}

#[derive(Debug)]
pub struct GrowableBump {
    chunks: UnsafeCell<Vec<Chunk>>,
    align: usize,
}

impl Clone for GrowableBump {
    fn clone(&self) -> Self {
        let chunks = unsafe { &*self.chunks.get() };

        let initial_capacity = chunks
            .first()
            .expect("GrowableBump always has at least one chunk")
            .capacity;

        Self::new(initial_capacity, self.align)
    }
}

impl GrowableBump {
    pub fn new(initial_capacity: usize, align: usize) -> Self {
        Self {
            chunks: UnsafeCell::new(vec![Chunk::new(initial_capacity, align)]),
            align,
        }
    }

    pub fn offset(&self) -> usize {
        let mut len = 0;
        for chunk in unsafe { self.chunks.as_ref_unchecked() } {
            len += chunk.offset;
        }
        len
    }

    pub fn alloc_value<T>(&self, val: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = self.allocate(layout).unwrap().as_non_null_ptr().cast::<T>();
        unsafe {
            ptr.as_ptr().write(val);
            &mut *ptr.as_ptr()
        }
    }

    pub fn alloc_value_immutable<T>(&self, val: T) -> &T {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = self.allocate(layout).unwrap().as_non_null_ptr().cast::<T>();
        unsafe {
            ptr.as_ptr().write(val);
            &*ptr.as_ptr()
        }
    }

    pub fn alloc_bytes(&self, len: usize) -> &mut [u8] {
        if len == 0 {
            return &mut [];
        }

        let layout = Layout::array::<u8>(len).expect("invalid layout for bytes");
        let ptr = self
            .allocate(layout)
            .unwrap_or_else(|_| panic!("bump allocator out of memory for {len} bytes"))
            .as_non_null_ptr()
            .cast::<u8>();

        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
    }

    pub fn alloc_bytes_copy(&self, src: &[u8]) -> &mut [u8] {
        let dst = self.alloc_bytes(src.len());
        dst.copy_from_slice(src);
        dst
    }

    pub fn reset(&mut self) {
        let chunks = self.chunks.get_mut();
        if let Some(first) = chunks.first_mut() {
            first.offset = 0;
        }
        chunks.truncate(1);
    }

    pub fn alloc_slice_copy<T: Copy>(&self, slice: &[T]) -> &[T] {
        if slice.is_empty() {
            return &[];
        }

        let layout = Layout::array::<T>(slice.len()).expect("Failed to create layout for slice");

        let ptr = self
            .allocate(layout)
            .expect("Failed to allocate slice in bump allocator")
            .as_ptr() as *mut T;

        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            std::slice::from_raw_parts(ptr, slice.len())
        }
    }

    pub fn alloc_slice_mut<T>(&self, slice: &[T]) -> &[T] {
        if slice.is_empty() {
            return &[];
        }

        let layout = Layout::array::<T>(slice.len()).expect("Failed to create layout for slice");

        let ptr = self
            .allocate(layout)
            .expect("Failed to allocate slice in bump allocator")
            .as_ptr() as *mut T;

        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            std::slice::from_raw_parts_mut(ptr, slice.len())
        }
    }

    pub fn alloc_slice<T>(&self, slice: &[T]) -> &[T] {
        if slice.is_empty() {
            return &[];
        }

        let layout = Layout::array::<T>(slice.len()).expect("Failed to create layout for slice");

        let ptr = self
            .allocate(layout)
            .expect("Failed to allocate slice in bump allocator")
            .as_ptr() as *mut T;

        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            std::slice::from_raw_parts(ptr, slice.len())
        }
    }

    #[cold]
    fn grow_allocator(
        &self,
        layout: Layout,
        chunks: *mut Vec<Chunk>,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_cap = unsafe { &mut *chunks }
            .last()
            .map(|c| c.capacity.max(layout.size()) * 2)
            .unwrap_or(layout.size().max(1024));

        unsafe { &mut *chunks }.push(Chunk::new(new_cap, self.align));
        let last = unsafe { &mut *chunks }.last_mut().unwrap();
        let ptr = last.try_alloc(layout).ok_or(AllocError)?; // must succeed after growing

        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }
}

unsafe impl Allocator for GrowableBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let dangling = NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }

        let chunks = self.chunks.get();

        if let Some(last) = unsafe { &mut *chunks }.last_mut()
            && let Some(ptr) = last.try_alloc(layout)
        {
            return Ok(NonNull::slice_from_raw_parts(ptr, layout.size()));
        }

        self.grow_allocator(layout, chunks)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // bump allocator does not support freeing individual allocations
    }
}
