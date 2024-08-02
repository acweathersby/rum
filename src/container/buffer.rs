//! Common sequential storage containers

use std::{alloc::Layout, fmt::Debug, mem::size_of};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

/// Operations for immutable byte buffers
pub trait ImmutableBufferOps {
  /// Returns the buffer transposed as a slice of T.
  ///
  /// Returns None if T is larger than the buffer itself, otherwise returns
  /// a slice whose byte length is no longer than the underlying buffer. This
  /// may lead to a slice with fewer elements of T than expected.
  ///
  /// # Safety
  ///
  /// This method makes no effort to ensure the underlying data is T, thus
  /// it is important that T is a primitive value or aggregate that DOES NOT
  /// contain reference types, such as Box, String, Vec, etc.
  fn as_slice<T: Sized + Copy>(&self) -> &[T];

  fn as_ptr<T: Sized + Copy>(&self) -> *const T;

  /// Size of the buffer in bytes
  fn len(&self) -> usize;

  /// Retrieve a null terminated string from this buffer. Return None if there
  /// are no null items in the input
  fn null_terminated_slice<T: Sized + Eq + Copy + Default>(&self, offset: usize) -> Option<&[T]> {
    let default_val = T::default();

    let slice = &self.as_slice::<T>()[offset..];

    if let Some((i, _)) = slice.iter().enumerate().find(|(_, i)| **i == default_val) {
      Some(&slice[0..i + 1])
    } else {
      None
    }
  }

  fn string_from_null_terminated_bytes(&self, offset: usize) -> Option<String> {
    let string_slice = self.null_terminated_slice::<u8>(offset)?;

    let string_slice = &string_slice[0..string_slice.len() - 1];

    if let Ok(string) = String::from_utf8(string_slice.to_vec()) {
      Some(string)
    } else {
      None
    }
  }

  unsafe fn str_from_null_terminated_bytes<'a>(&'a self, offset: usize) -> Option<&str> {
    let string_slice = self.null_terminated_slice::<u8>(offset)?;

    let string_slice = &string_slice[0..string_slice.len() - 1];

    Some(std::str::from_utf8_unchecked(string_slice))
  }

  /// Fills the given pointer with bytes from this buffer. The amount of bytes
  /// written will be the lower of `len` tne the buffers bytesize.
  fn fill_address(&self, ptr: *mut u8, len: usize) {
    if ptr as usize >= self.as_ptr() as *const u8 as usize && ptr as usize <= (self.as_ptr() as *const u8 as usize + self.len()) {
      unsafe { std::intrinsics::copy(self.as_ptr(), ptr, len.min(self.len())) };
    } else {
      unsafe { std::intrinsics::copy_nonoverlapping(self.as_ptr(), ptr, len.min(self.len())) };
    }
  }
}

impl ImmutableBufferOps for [u8] {
  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(ImmutableBufferOps::as_ptr(self), num_of_eles) }
  }

  fn len(&self) -> usize {
    self.len()
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    self as *const _ as *const T
  }
}

impl ImmutableBufferOps for Vec<u8> {
  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(ImmutableBufferOps::as_ptr(self), num_of_eles) }
  }

  fn len(&self) -> usize {
    self.len()
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    self.as_slice() as *const _ as *const T
  }
}

/// Operations for mutable byte buffers
pub trait MutableBufferOps: ImmutableBufferOps {
  /// Returns the buffer transposed as a mutable slice of T.
  ///
  /// Returns None if T is larger than the buffer itself, otherwise returns
  /// a slice whose byte length is no longer than the underlying buffer. This
  /// may lead to a slice with fewer elements than expected.
  ///
  /// # Safety
  ///
  /// This method makes no effort to ensure the underlying data is T, thus
  /// it is important that T is a primitive value or aggregate that DOES NOT
  /// contain reference types, such as Box, String, Vec, or pointers.
  fn as_mut_slice<T: Sized + Copy>(&mut self) -> &mut [T];

  fn as_mut_ptr<T: Sized + Copy>(&mut self) -> *mut T {
    std::ptr::null_mut()
  }

  /// Retrieve a null terminated string from this buffer. Return None if there
  /// are no null items in the input
  fn null_terminated_slice_mut<T: Sized + Eq + Copy + Default>(&mut self) -> Option<&mut [T]> {
    let default_val = T::default();

    let slice = self.as_mut_slice::<T>();

    if let Some((i, _)) = slice.iter().enumerate().find(|(_, i)| **i == default_val) {
      Some(&mut slice[0..i + 1])
    } else {
      None
    }
  }

  /// Fills the given buffer with data from `items`, starting at `offset`, where
  /// `offset` represents an index into an slice of [T].  
  ///
  /// If the buffer is shorter than the length of `items`, only the first n
  /// items that fit are copied.
  ///
  /// Nothing is copied if `offset` is greater then the length of the buffer.
  fn fill<T: Sized + Copy, C: AsRef<[T]>>(&mut self, offset: usize, items: C) {
    let items = items.as_ref();
    let items_len = items.len();

    let slice = &mut self.as_mut_slice::<T>();
    let offset = offset.min(slice.len());
    let slice = &mut slice[offset..];

    let count = slice.len().min(items_len);
    unsafe {
      std::ptr::copy_nonoverlapping(items.as_ptr(), slice.as_mut_ptr(), count);
    };
  }
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/// Simple byte buffer. Similar to a Vec<u8> without a len value.
pub struct Buffer {
  _internal: *mut u8,
  capacity:  usize,
  alignment: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl From<Vec<u8>> for Buffer {
  fn from(vector: Vec<u8>) -> Self {
    let (ptr, _, capacity, _) = vector.into_raw_parts_with_alloc();
    Buffer { _internal: ptr, capacity, alignment: 1 }
  }
}

impl Clone for Buffer {
  fn clone(&self) -> Self {
    Self::from_items(self.as_slice::<u8>()).unwrap()
  }
}

impl Debug for Buffer {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("Buffer");
    s.field("slice", &self.as_slice::<u8>());
    s.finish()
  }
}

impl AsRef<[u8]> for Buffer {
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl<const SIZE: usize> From<StaticBuffer<SIZE>> for Option<Buffer> {
  fn from(value: StaticBuffer<SIZE>) -> Self {
    let Some(mut b) = Buffer::with_capacity(SIZE) else {
      return None;
    };

    b.fill(0, value.as_slice_const::<u8>());

    Some(b)
  }
}

impl Drop for Buffer {
  fn drop(&mut self) {
    unsafe {
      let len = self.len();
      let align = self.alignment;

      // SAFETY: Layout is identical to the layout defined when this object was
      // allocated, thus  `Self::get_layout(len)` could only have returned
      // `Some(..)` and so this identical call to `Self::get_layout(len)` can
      // only be `Some(..)`.
      let layout = Self::get_layout(len, align).unwrap_unchecked();

      std::alloc::dealloc(self._internal, layout);
    }
  }
}

impl ImmutableBufferOps for Buffer {
  fn len(&self) -> usize {
    self.capacity
  }

  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(self.as_ptr(), num_of_eles) }
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    self._internal as usize as *const T
  }
}

impl MutableBufferOps for Buffer {
  fn as_mut_ptr<T: Sized + Copy>(&mut self) -> *mut T {
    self._internal as *mut T
  }

  fn as_mut_slice<T: Sized + Copy>(&mut self) -> &mut [T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), num_of_eles) }
  }
}

impl Buffer {
  fn get_layout(capacity: usize, alignment: usize) -> Option<Layout> {
    match std::alloc::Layout::from_size_align(capacity, alignment) {
      Err(err) => {
        #[cfg(debug_assertions)]
        eprintln!("{}", err);
        None
      }
      Ok(l) => Some(l),
    }
  }

  /// Creates a new buffer with a byte size of `len`.
  ///
  /// Returns None if there was an error allocating memory.
  pub fn with_capacity(capacity: usize) -> Option<Buffer> {
    let Some(layout) = Self::get_layout(capacity, 0x8) else {
      return None;
    };

    unsafe {
      let buffer = std::alloc::alloc_zeroed(layout);

      if buffer.is_null() {
        return None;
      }

      Some(Buffer { _internal: buffer, capacity, alignment: 0x8 })
    }
  }

  /// Creates a new buffer with a byte size of `len`, with alignment `align`.
  ///
  /// Returns None if there was an error allocating memory.
  pub fn with_aligned_capacity(capacity: usize, alignment: usize) -> Option<Buffer> {
    let Some(layout) = Self::get_layout(capacity, alignment) else {
      return None;
    };

    unsafe {
      let buffer = std::alloc::alloc_zeroed(layout);

      if buffer.is_null() {
        return None;
      }

      Some(Buffer { _internal: buffer, capacity, alignment })
    }
  }

  /// Creates a new buffer with a byte size of `len`. Same as
  /// `with_capicty`, except the underlying memory is not zeroed.
  ///
  /// Returns None if there was an error allocating memory.
  pub fn with_capacity_dirty(capacity: usize) -> Option<Buffer> {
    let Some(layout) = Self::get_layout(capacity, 0x8) else {
      return None;
    };

    unsafe {
      let buffer = std::alloc::alloc(layout);

      if buffer.is_null() {
        return None;
      }

      Some(Buffer { _internal: buffer, capacity, alignment: 0x8 })
    }
  }

  /// Create a buffer from an object that can be referenced as a byte slice.
  pub fn from_items<T: Sized + Copy, C: AsRef<[T]>>(items: C) -> Option<Self> {
    let items = items.as_ref();
    let Some(mut b) = Buffer::with_capacity(size_of::<T>() * items.len()) else {
      return None;
    };

    b.fill(0, items);

    Some(b)
  }
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/// Simple byte buffer that stores buffer properties within the allocated
/// buffer space.
pub struct SmallBuffer {
  buffer: *mut u8,
}

impl Clone for SmallBuffer {
  fn clone(&self) -> Self {
    Self::from_items(self.as_slice::<u8>()).unwrap()
  }
}

impl Debug for SmallBuffer {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("Buffer");
    s.field("slice", &self.as_slice::<u8>());
    s.finish()
  }
}

impl AsRef<[u8]> for SmallBuffer {
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl<const SIZE: usize> From<StaticBuffer<SIZE>> for Option<SmallBuffer> {
  fn from(value: StaticBuffer<SIZE>) -> Self {
    let Some(mut b) = SmallBuffer::with_capacity(SIZE) else {
      return None;
    };

    b.fill(0, value.as_slice_const::<u8>());

    Some(b)
  }
}

impl Drop for SmallBuffer {
  fn drop(&mut self) {
    unsafe {
      let len = self.len();

      // SAFETY: Layout is identical to the layout defined when this object was
      // allocated, thus  `Self::get_layout(len)` could only have returned
      // `Some(..)` and so this identical call to `Self::get_layout(len)` can
      // only be `Some(..)`.
      let layout = Self::get_layout(len, 0x8).unwrap_unchecked();

      std::alloc::dealloc(self.buffer, layout);
    }
  }
}

impl ImmutableBufferOps for SmallBuffer {
  fn len(&self) -> usize {
    unsafe { *(self.buffer as *const _ as *const usize) }
  }

  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(self.as_ptr(), num_of_eles) }
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    (self.buffer as usize + Self::PREAMBLE_LEN) as *const T
  }
}

impl MutableBufferOps for SmallBuffer {
  fn as_mut_ptr<T: Sized + Copy>(&mut self) -> *mut T {
    (self.buffer as usize + Self::PREAMBLE_LEN) as *mut T
  }

  fn as_mut_slice<T: Sized + Copy>(&mut self) -> &mut [T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), num_of_eles) }
  }
}

impl SmallBuffer {
  const PREAMBLE_LEN: usize = size_of::<usize>() << 1;

  fn get_layout(len: usize, alignment: usize) -> Option<Layout> {
    match std::alloc::Layout::from_size_align(len + Self::PREAMBLE_LEN, alignment) {
      Err(err) => {
        #[cfg(debug_assertions)]
        eprintln!("{}", err);
        None
      }
      Ok(l) => Some(l),
    }
  }

  /// Creates a new buffer with a byte size of `len`.
  ///
  /// Returns None if there was an error allocating memory.
  pub fn with_capacity(len: usize) -> Option<SmallBuffer> {
    let Some(layout) = Self::get_layout(len, 0x8) else {
      return None;
    };

    unsafe {
      let buffer = std::alloc::alloc_zeroed(layout);

      if buffer.is_null() {
        return None;
      }

      *(buffer as *mut _ as *mut usize) = len;

      Some(SmallBuffer { buffer })
    }
  }

  /// Creates a new buffer with a byte size of `len`. Same as
  /// `with_capacity`, except the underlying memory is not zeroed.
  ///
  /// Returns None if there was an error allocating memory.
  pub fn with_capacity_dirty(len: usize) -> Option<SmallBuffer> {
    let Some(layout) = Self::get_layout(len, 0x8) else {
      return None;
    };

    unsafe {
      let buffer = std::alloc::alloc(layout);

      if buffer.is_null() {
        return None;
      }

      *(buffer as *mut _ as *mut usize) = len;

      Some(SmallBuffer { buffer })
    }
  }

  /// Create a buffer from an object that can be referenced as a byte slice.
  pub fn from_items<T: Sized + Copy, C: AsRef<[T]>>(items: C) -> Option<Self> {
    let items = items.as_ref();
    let Some(mut b) = SmallBuffer::with_capacity(size_of::<T>() * items.len()) else {
      return None;
    };

    b.fill(0, items);

    Some(b)
  }
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

/// A buffer wrapped around an immutable slice of bytes.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedBuffer<'a> {
  buffer: &'a [u8],
}

impl<'a> AsRef<[u8]> for BorrowedBuffer<'a> {
  fn as_ref(&self) -> &[u8] {
    self.buffer
  }
}

impl<'a, T: Sized + Copy> From<&'a [T]> for BorrowedBuffer<'a> {
  fn from(value: &'a [T]) -> Self {
    let byte_size = std::mem::size_of::<T>();
    let buffer = unsafe { std::slice::from_raw_parts::<'a, u8>(value.as_ptr() as *const _, value.len() * byte_size) };
    Self::new(buffer)
  }
}

impl<'a> ImmutableBufferOps for BorrowedBuffer<'a> {
  fn len(&self) -> usize {
    self.buffer.len()
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    self.buffer.as_ptr() as *const _
  }

  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(self.as_ptr(), num_of_eles) }
  }
}

impl<'a> BorrowedBuffer<'a> {
  pub fn new(buffer: &'a [u8]) -> Self {
    Self { buffer }
  }
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

/// A byte buffer allocated on the stack.
#[derive(Clone, Copy)]
#[repr(align(8))]
pub struct StaticBuffer<const SIZE: usize> {
  buffer: [u8; SIZE],
}

impl<const SIZE: usize> Debug for StaticBuffer<SIZE> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("Buffer");
    s.field("slice", &self.as_slice_const::<u8>());
    s.finish()
  }
}

impl<const SIZE: usize> AsRef<[u8]> for StaticBuffer<SIZE> {
  fn as_ref(&self) -> &[u8] {
    self.as_slice_const()
  }
}

impl<const SIZE: usize> ImmutableBufferOps for StaticBuffer<SIZE> {
  fn len(&self) -> usize {
    SIZE
  }

  fn as_slice<T: Sized + Copy>(&self) -> &[T] {
    StaticBuffer::<SIZE>::as_slice_const(self)
  }

  fn as_ptr<T: Sized + Copy>(&self) -> *const T {
    (&self.buffer) as *const u8 as *const T
  }
}

impl<const SIZE: usize> MutableBufferOps for StaticBuffer<SIZE> {
  fn as_mut_slice<T: Sized + Copy>(&mut self) -> &mut [T] {
    let len = self.len();
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts_mut(self.buffer.as_ptr() as *mut _, num_of_eles) }
  }

  fn as_mut_ptr<T: Sized + Copy>(&mut self) -> *mut T {
    (&mut self.buffer) as *mut u8 as *mut T
  }
}

impl<const SIZE: usize> StaticBuffer<SIZE> {
  /// Creates a new zero-initialized buffer with a byte size of `len`.
  pub const fn new() -> Self {
    Self { buffer: [0; SIZE] }
  }

  pub const fn as_slice_const<T: Sized + Copy>(&self) -> &[T] {
    let len = SIZE;
    let ele_len: usize = size_of::<T>();
    let num_of_eles = len / ele_len;
    unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *mut _, num_of_eles) }
  }
}

#[test]
fn fill_with_array() {
  const ITEMS: [u32; 6] = [0 as u32, 1, 2, 3, 4, 5];

  let b = SmallBuffer::from_items(ITEMS).unwrap();
  assert_eq!(b.as_slice::<u32>(), &ITEMS);

  const SIZE: usize = 4 * ITEMS.len();

  let mut sb = StaticBuffer::<SIZE>::new();

  sb.fill(0, ITEMS);

  assert_eq!(sb.as_slice_const::<u32>(), &ITEMS);

  assert_eq!(sb.as_slice_const::<u32>(), b.as_slice::<u32>());
}

#[test]
fn offset_fill_with_truncation() {
  const ITEMS: [u32; 6] = [1, 2, 3, 4, 5, 6];
  const SIZE: usize = 4 * ITEMS.len();

  let mut b = SmallBuffer::with_capacity(SIZE).unwrap();
  b.fill(3, ITEMS);

  assert_eq!(b.as_slice::<u32>(), &[0, 0, 0, 1, 2, 3]);

  let mut sb = StaticBuffer::<SIZE>::new();

  sb.fill(3, ITEMS);

  assert_eq!(sb.as_slice_const::<u32>(), &[0, 0, 0, 1, 2, 3]);
}

#[test]
fn offset_fill_out_of_bounds() {
  const ITEMS: [u32; 6] = [1, 2, 3, 4, 5, 6];
  const SIZE: usize = 4 * ITEMS.len();

  let mut b = SmallBuffer::with_capacity(SIZE).unwrap();
  b.fill(200, ITEMS);

  assert_eq!(b.as_slice::<u32>(), &[0, 0, 0, 0, 0, 0]);

  let mut sb = StaticBuffer::<SIZE>::new();

  sb.fill(200, ITEMS);

  assert_eq!(sb.as_slice_const::<u32>(), &[0, 0, 0, 0, 0, 0]);
}
