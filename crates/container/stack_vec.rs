use core::ops::{Index, IndexMut};
use std::{
  fmt::{Debug, Display},
  mem::ManuallyDrop,
};

const VECTORIZED_MASK: u8 = 1 << 0;
const ORDERED_MASK: u8 = 1 << 1;

union ArrayVecInner<const STACK_SIZE: usize, T: Sized> {
  array: (usize, std::mem::ManuallyDrop<[T; STACK_SIZE]>),
  vec:   std::mem::ManuallyDrop<Vec<T>>,
}

/// A vector which derives its initial capacity in place. The data is
/// moved to a heap location if the number of elements pushed to the vector
/// exceed the initial capacity.
pub struct ArrayVec<const STACK_SIZE: usize, T: Sized> {
  inner: ArrayVecInner<STACK_SIZE, T>,
  flags: u8,
}

impl<const STACK_SIZE: usize, T: Clone> Clone for ArrayVec<STACK_SIZE, T> {
  fn clone(&self) -> Self {
    let mut other = ArrayVec::new();

    other.flags = self.flags;

    unsafe {
      if (self.flags & VECTORIZED_MASK) > 0 {
        let vec = self.inner.vec.clone();
        other.inner.vec = self.inner.vec.clone();
      } else {
        for i in self.as_slice() {
          other.push(i.clone())
        }
      }
    }

    other
  }
}

impl<const STACK_SIZE: usize, T> Default for ArrayVec<STACK_SIZE, T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<const STACK_SIZE: usize, T> Drop for ArrayVec<STACK_SIZE, T> {
  fn drop(&mut self) {
    self.clear()
  }
}
impl<const STACK_SIZE: usize, T: Debug> FromIterator<T> for ArrayVec<STACK_SIZE, T> {
  fn from_iter<Iter: IntoIterator<Item = T>>(iter: Iter) -> Self {
    let mut vec = Self::default();
    for item in iter {
      vec.push(item);
    }
    vec
  }
}

impl<const STACK_SIZE: usize, T: Debug> Debug for ArrayVec<STACK_SIZE, T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_list();

    s.entries(self.as_slice());

    s.finish()
  }
}

impl<const STACK_SIZE: usize, T> Index<usize> for ArrayVec<STACK_SIZE, T> {
  type Output = T;

  fn index(&self, index: usize) -> &Self::Output {
    if index >= self.len() {
      panic!("Index {index} is out of range of 0..{}", self.len());
    }

    &self.as_slice()[index]
  }
}

impl<const STACK_SIZE: usize, T: ToString> ArrayVec<STACK_SIZE, T> {
  pub fn join(&self, sep: &str) -> String {
    self.as_slice().iter().map(|i| i.to_string()).collect::<Vec<_>>().join(sep)
  }
}

impl<const STACK_SIZE: usize, T> IndexMut<usize> for ArrayVec<STACK_SIZE, T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    self.flags &= !ORDERED_MASK;

    if index >= self.len() {
      panic!("Index {index} is out of range of 0..{}", self.len());
    }

    unsafe { &mut self.as_slice_mut()[index] }
  }
}

impl<const STACK_SIZE: usize, T: Ord + PartialOrd> ArrayVec<STACK_SIZE, T> {
  fn binary_search(
    min: usize,
    max: usize,
    insert: &T,
    entries: &[T],
  ) -> (usize, std::cmp::Ordering) {
    let diff = max - min;
    if diff <= 1 {
      match insert.cmp(&entries[min]) {
        cmp @ std::cmp::Ordering::Less | cmp @ std::cmp::Ordering::Equal => (min, cmp),
        cmp @ std::cmp::Ordering::Greater => (max, cmp),
      }
    } else {
      let center = min + (diff >> 1);
      match insert.cmp(&entries[center]) {
        cmp @ std::cmp::Ordering::Equal => (center, cmp),
        std::cmp::Ordering::Greater => Self::binary_search(center, max, insert, entries),
        std::cmp::Ordering::Less => Self::binary_search(min, center, insert, entries),
      }
    }
  }

  pub fn contains(&self, item: &T) -> bool {
    if !self.data_is_ordered() {
      self.as_slice().contains(item)
    } else {
      self.as_slice().binary_search(item).is_ok()
    }
  }

  pub fn sort(&mut self) {
    self.as_mut_slice().sort();
    self.flags |= ORDERED_MASK;
  }

  pub fn find_ordered(&self, entry: &T) -> Option<usize> {
    if self.data_is_ordered() {
      if let Ok(result) = self.as_slice().binary_search(entry) {
        return Some(result);
      }
    }
    return None;
  }

  pub fn insert_ordered(&mut self, mut entry: T) -> Result<usize, &str> {
    self.push_ordered_internal(entry, false)
  }

  pub fn push_unique(&mut self, mut entry: T) -> Result<usize, &str> {
    self.push_ordered_internal(entry, true)
  }

  #[inline(always)]
  fn push_ordered_internal(&mut self, mut entry: T, dedup: bool) -> Result<usize, &str> {
    if !self.data_is_ordered() {
      Err("ArrayVec is not ordered, cannot perform an ordered insert")
    } else {
      if self.data_is_vectorized() {
        let vec = unsafe { &mut *self.inner.vec };
        let entries = vec.as_mut_slice();

        let (pos, ord) = Self::binary_search(0, entries.len(), &entry, &entries);

        if dedup && ord.is_eq() {
          Ok(pos)
        } else {
          vec.insert(pos, entry);
          self.flags |= ORDERED_MASK;
          Ok(pos)
        }
      } else {
        let (len, array) = unsafe { &mut self.inner.array };

        if *len == 0 {
          self.push(entry);
          self.flags |= ORDERED_MASK;
          Ok(0)
        } else if *len < STACK_SIZE {
          let entries = array.as_mut_slice();

          let (pos, ord) = Self::binary_search(0, *len, &entry, &entries);

          if dedup && ord.is_eq() {
            Ok(pos)
          } else {
            unsafe {
              entries
                .as_ptr()
                .offset(pos as isize)
                .copy_to(entries.as_mut_ptr().offset(pos as isize + 1), *len - pos);
            };

            core::mem::swap(&mut entries[pos], &mut entry);
            core::mem::forget(entry);

            *len += 1;
            self.flags |= ORDERED_MASK;
            Ok(pos)
          }
        } else {
          self.convert_to_vector();
          self.insert_ordered(entry)
        }
      }
    }
  }
}

impl<const STACK_SIZE: usize, T> AsRef<[T]> for ArrayVec<STACK_SIZE, T> {
  fn as_ref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<const STACK_SIZE: usize, T> AsMut<[T]> for ArrayVec<STACK_SIZE, T> {
  fn as_mut(&mut self) -> &mut [T] {
    self.as_mut_slice()
  }
}

impl<const STACK_SIZE: usize, T: Sized> ArrayVec<STACK_SIZE, T> {
  #[inline(never)]
  pub fn new() -> Self {
    let inner: [T; STACK_SIZE] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let out = Self {
      inner: ArrayVecInner { array: (0, ManuallyDrop::new(inner)) },
      flags: ORDERED_MASK,
    };

    out
  }

  pub fn reverse(&mut self) {
    self.as_mut_slice().reverse();
    self.flags &= !ORDERED_MASK
  }

  fn convert_to_vector(&mut self) {
    unsafe {
      let (len, array) = unsafe { &mut self.inner.array };
      let mut vec = Vec::<T>::with_capacity(*len);
      core::ptr::copy(array.as_ptr(), vec.as_mut_ptr(), *len);
      vec.set_len(*len);
      let spot = &mut self.inner.vec;
      std::mem::swap(&mut vec, spot);
      self.flags |= VECTORIZED_MASK;

      std::mem::forget(vec);
    }
  }

  /// Converts the stack vec into a regular vector, consuming the stack vec in
  /// the process.
  pub fn to_vec(mut self) -> Vec<T> {
    if self.data_is_vectorized() {
      unsafe { ManuallyDrop::take(&mut self.inner.vec) }
    } else {
      let (len, array) = unsafe { &mut self.inner.array };
      let mut vec = Vec::<T>::with_capacity(*len);

      unsafe {
        core::ptr::copy(array.as_ptr(), vec.as_mut_ptr(), *len);
        vec.set_len(*len);
      };

      *len = 0;

      vec
    }
  }

  pub fn clear(&mut self) {
    if self.data_is_vectorized() {
      unsafe { drop(ManuallyDrop::take(&mut self.inner.vec)) };
      let inner: [T; STACK_SIZE] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
      self.inner.array = (0, ManuallyDrop::new(inner));
    } else {
      let (len, array) = unsafe { &mut self.inner.array };
      // Retrieve items from the manual drop to begin the drop process.
      let items = unsafe { ManuallyDrop::take(array) };

      // We'll iterate through the "valid" items, and manually drop each one.
      let mut iter = items.into_iter();

      // Make sure we only drop the allocated items by limiting the range.
      for _ in 0..*len {
        if let Some(i) = iter.next() {
          drop(i)
        }
      }

      // The rest is garbage data so we'll just forget it.
      std::mem::forget(iter);

      *len = 0
    }
  }

  pub fn len(&self) -> usize {
    if self.data_is_vectorized() {
      unsafe { self.inner.vec.len() }
    } else {
      unsafe { self.inner.array.0 }
    }
  }

  pub fn iter<'stack>(&'stack self) -> ArrayVecIterator<'stack, STACK_SIZE, T> {
    ArrayVecIterator { inner: self, tracker: 0, len: self.len() }
  }

  pub fn iter_mut<'stack>(&'stack mut self) -> ArrayVecIteratorMut<'stack, STACK_SIZE, T> {
    self.flags &= !ORDERED_MASK;
    ArrayVecIteratorMut { len: self.len(), inner: self, tracker: 0 }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn push(&mut self, mut element: T) {
    self.flags &= !ORDERED_MASK;
    if self.data_is_vectorized() {
      unsafe { (*self.inner.vec).push(element) };
    } else {
      let (len, array) = unsafe { &mut self.inner.array };

      if *len < STACK_SIZE {
        core::mem::swap(&mut array[*len], &mut element);
        core::mem::forget(element);
        *len += 1;
      } else {
        unsafe {
          self.convert_to_vector();
          self.push(element)
        }
      }
    }
  }

  pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
    for i in iter {
      self.push(i)
    }
  }

  pub fn pop(&mut self) -> Option<T> {
    if self.data_is_vectorized() {
      unsafe { (*self.inner.vec).pop() }
    } else {
      let (len, array) = unsafe { &mut self.inner.array };
      if *len > 0 {
        *len -= 1;
        Some(unsafe { std::mem::transmute_copy(&array[*len]) })
      } else {
        None
      }
    }
  }

  #[inline(never)]
  pub fn as_slice(&self) -> &[T] {
    if (self.flags & VECTORIZED_MASK) > 0 {
      unsafe { self.inner.vec.as_slice() }
    } else {
      let (len, array) = unsafe { &self.inner.array };
      unsafe { core::slice::from_raw_parts(array.as_ptr(), *len) }
    }
  }

  #[inline(never)]
  pub fn as_slice_mut(&mut self) -> &mut [T] {
    if (self.flags & VECTORIZED_MASK) > 0 {
      unsafe { (*self.inner.vec).as_mut_slice() }
    } else {
      let (len, array) = unsafe { &mut self.inner.array };
      unsafe { core::slice::from_raw_parts_mut(array.as_mut_ptr(), *len) }
    }
  }

  /// The stored data bound within the local backing store.
  pub fn data_is_vectorized(&self) -> bool {
    self.flags & VECTORIZED_MASK > 0
  }

  pub fn data_is_ordered(&self) -> bool {
    self.flags & ORDERED_MASK > 0
  }

  #[inline(never)]
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.flags &= !ORDERED_MASK;

    if self.data_is_vectorized() {
      unsafe { (*self.inner.vec).as_mut_slice() }
    } else {
      let (len, array) = unsafe { &mut self.inner.array };
      unsafe { core::slice::from_raw_parts_mut(array.as_mut_ptr(), *len) }
    }
  }
}

pub struct ArrayVecIterator<'stack, const STACK_SIZE: usize, T> {
  inner:   &'stack ArrayVec<STACK_SIZE, T>,
  tracker: usize,
  len:     usize,
}

impl<'stack, const STACK_SIZE: usize, T> Iterator for ArrayVecIterator<'stack, STACK_SIZE, T> {
  type Item = &'stack T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.tracker < self.len {
      let index = self.tracker;
      self.tracker += 1;
      Some(self.inner.index(index))
    } else {
      None
    }
  }
}

pub struct ArrayVecIteratorMut<'stack, const STACK_SIZE: usize, T> {
  inner:   &'stack mut ArrayVec<STACK_SIZE, T>,
  tracker: usize,
  len:     usize,
}

impl<'stack, const STACK_SIZE: usize, T> Iterator for ArrayVecIteratorMut<'stack, STACK_SIZE, T> {
  type Item = &'stack mut T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.tracker < self.len {
      let index = self.tracker;
      self.tracker += 1;
      let inner: *mut ArrayVec<STACK_SIZE, T> = self.inner as *mut _;
      let inner = (unsafe { &mut *inner });

      Some(&mut inner[index])
    } else {
      None
    }
  }
}

impl<const STACK_SIZE: usize, T, I: Iterator<Item = T>> From<I> for ArrayVec<STACK_SIZE, T> {
  fn from(value: I) -> Self {
    let mut s_vec = Self::new();

    for i in value {
      s_vec.push(i)
    }

    s_vec
  }
}
