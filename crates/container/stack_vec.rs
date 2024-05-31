use core::ops::{Index, IndexMut};
use std::{
  fmt::{Debug, Display},
  mem::ManuallyDrop,
};

/// A vector which derives its initial capacity from its binding. The data is
/// moved to the heap if the number of elements pushed to the vector exceed
/// the stack capacity.
pub struct ArrayVec<const STACK_SIZE: usize, T: Sized> {
  inner:       std::mem::ManuallyDrop<[T; STACK_SIZE]>,
  vec:         Option<Vec<T>>,
  allocations: usize,
  ordered:     bool,
}

impl<const STACK_SIZE: usize, T: Clone> Clone for ArrayVec<STACK_SIZE, T> {
  fn clone(&self) -> Self {
    let mut other = ArrayVec::new();

    if let Some(vec) = &self.vec {
      other.vec = Some(vec.clone());
    } else {
      for i in self.as_slice() {
        other.push(i.clone())
      }

      other.ordered = self.ordered;
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

    if let Some(vec) = &self.vec {
      &vec[index]
    } else {
      &self.inner[index]
    }
  }
}

impl<const STACK_SIZE: usize, T: ToString> ArrayVec<STACK_SIZE, T> {
  pub fn join(&self, sep: &str) -> String {
    self.as_slice().iter().map(|i| i.to_string()).collect::<Vec<_>>().join(sep)
  }
}

impl<const STACK_SIZE: usize, T> IndexMut<usize> for ArrayVec<STACK_SIZE, T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    self.ordered = false;

    if index >= self.len() {
      panic!("Index {index} is out of range of 0..{}", self.len());
    }

    if let Some(vec) = &mut self.vec {
      &mut vec[index]
    } else {
      &mut self.inner[index]
    }
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
    if self.ordered == false {
      self.as_slice().contains(item)
    } else {
      self.as_slice().binary_search(item).is_ok()
    }
  }

  pub fn sort(&mut self) {
    self.as_mut_slice().sort();
    self.ordered = true
  }

  pub fn push_ordered(&mut self, mut entry: T) -> Result<(), &str> {
    self.push_ordered_internal(entry, false)
  }

  pub fn push_unique(&mut self, mut entry: T) -> Result<(), &str> {
    self.push_ordered_internal(entry, true)
  }

  #[inline(always)]
  fn push_ordered_internal(&mut self, mut entry: T, dedup: bool) -> Result<(), &str> {
    if self.ordered == false {
      Err("ArrayVec is not ordered, cannot perform an ordered push")
    } else {
      if let Some(vec) = &mut self.vec {
        vec.push(entry);
        vec.sort();
        if (dedup) {
          vec.dedup();
        }
        Ok(())
      } else {
        if self.allocations == 0 {
          self.push(entry);
          self.ordered = true;
          Ok(())
        } else if self.allocations < STACK_SIZE {
          let entries = self.inner.as_mut_slice();

          let (pos, ord) = Self::binary_search(0, self.allocations, &entry, &entries);

          if dedup && ord.is_eq() {
            Ok(())
          } else {
            unsafe {
              entries
                .as_ptr()
                .offset(pos as isize)
                .copy_to(entries.as_mut_ptr().offset(pos as isize + 1), self.allocations - pos);
            };

            core::mem::swap(&mut entries[pos], &mut entry);
            core::mem::forget(entry);

            self.allocations += 1;
            self.ordered = true;
            Ok(())
          }
        } else {
          unsafe {
            let mut vec = Vec::<T>::with_capacity(self.allocations);
            core::ptr::copy(self.inner.as_ptr(), vec.as_mut_ptr(), self.allocations);
            vec.set_len(self.allocations);
            self.vec = Some(vec);
            self.push_ordered(entry)
          }
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
      inner:       ManuallyDrop::new(inner),
      vec:         None,
      allocations: 0,
      ordered:     true,
    };

    out
  }

  /// Converts the stack vec into a regular vector, consuming the stack vec in
  /// the process.
  pub fn to_vec(mut self) -> Vec<T> {
    let vec = if let Some(vec) = self.vec.take() {
      vec
    } else {
      let data = &mut self.inner;
      let mut vec = Vec::<T>::with_capacity(self.allocations);
      unsafe {
        core::ptr::copy(data.as_ptr(), vec.as_mut_ptr(), self.allocations);
        vec.set_len(self.allocations);
      };

      vec
    };

    self.allocations = 0;
    vec
  }

  pub fn clear(&mut self) {
    if self.data_is_bounded() {
      // Retrieve items from the manual drop to begin the drop process.
      let items = unsafe { ManuallyDrop::take(&mut self.inner) };

      // We'll iterate through the "valid" items, and manually drop each one.
      let mut iter = items.into_iter();

      // Make sure we only drop the allocated items by limiting the range.
      for _ in 0..self.allocations {
        if let Some(i) = iter.next() {
          drop(i)
        }
      }

      // The rest is garbage data so we'll just forget it.
      std::mem::forget(iter);
    } else {
      drop(self.vec.take())
    }

    self.allocations = 0;
  }

  pub fn len(&self) -> usize {
    if let Some(vec) = &self.vec {
      vec.len()
    } else {
      self.allocations
    }
  }

  pub fn iter<'stack>(&'stack self) -> ArrayVecIterator<'stack, STACK_SIZE, T> {
    ArrayVecIterator { inner: self, tracker: 0, len: self.len() }
  }

  pub fn iter_mut<'stack>(&'stack mut self) -> ArrayVecIteratorMut<'stack, STACK_SIZE, T> {
    self.ordered = false;
    ArrayVecIteratorMut { len: self.len(), inner: self, tracker: 0 }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn push(&mut self, mut element: T) {
    self.ordered = false;
    if let Some(vec) = &mut self.vec {
      vec.push(element);
    } else {
      if self.allocations < STACK_SIZE {
        core::mem::swap(&mut self.inner[self.allocations], &mut element);
        core::mem::forget(element);
        self.allocations += 1;
      } else {
        unsafe {
          let mut vec = Vec::<T>::with_capacity(self.allocations);
          core::ptr::copy(self.inner.as_ptr(), vec.as_mut_ptr(), self.allocations);
          vec.set_len(self.allocations);
          vec.push(element);
          self.vec = Some(vec);
        }
      }
    }
  }

  pub fn pop(&mut self) -> Option<T> {
    if let Some(vec) = &mut self.vec {
      vec.pop()
    } else {
      if self.allocations > 0 {
        self.allocations -= 1;
        Some(unsafe { std::mem::transmute_copy(&self.inner[self.allocations]) })
      } else {
        None
      }
    }
  }

  #[inline(never)]
  pub fn as_slice(&self) -> &[T] {
    if let Some(vec) = &self.vec {
      vec.as_slice()
    } else {
      unsafe { core::slice::from_raw_parts(self.inner.as_ptr(), self.allocations) }
    }
  }

  /// The stored data bound within the local backing store.
  pub fn data_is_bounded(&self) -> bool {
    self.vec.is_none()
  }

  pub fn data_is_ordered(&self) -> bool {
    self.ordered
  }

  #[inline(never)]
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.ordered = false;
    if let Some(vec) = &mut self.vec {
      vec.as_mut_slice()
    } else {
      unsafe { core::slice::from_raw_parts_mut(self.inner.as_mut_ptr(), self.allocations) }
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
      if let Some(vec) = &mut (unsafe { &mut *inner }).vec {
        Some(&mut vec[index])
      } else {
        Some(&mut (unsafe { &mut *inner }).inner[index])
      }
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
