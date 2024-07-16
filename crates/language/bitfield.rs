use std::{self, fmt::Debug};

pub(crate) struct BitFieldArena {
  pub(crate) bits:         *mut u128,
  pub(crate) row_ele_size: usize,
  pub(crate) rows:         usize,
  pub(crate) len:          usize,
}

impl Debug for BitFieldArena {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut list = f.debug_list();
    let slice = &*self.slice();
    for i in 0..self.rows {
      let offset = i * self.row_ele_size;
      let slice = &slice[offset..offset + self.row_ele_size];

      list.entry(&slice.iter().map(|i| format!("{i:0128b}")).collect::<Vec<_>>().join("_"));
    }

    list.finish()
  }
}

impl Drop for BitFieldArena {
  fn drop(&mut self) {
    let layout = std::alloc::Layout::array::<u128>(self.len).expect("Could not create bit field");
    unsafe { std::alloc::dealloc(self.bits as *mut _, layout) };
  }
}

impl BitFieldArena {
  pub fn new(rows: usize, columns: usize) -> Self {
    let bit_blocks = (columns >> 7).max(1);
    let len = rows * bit_blocks;

    let layout = std::alloc::Layout::array::<u128>(len).expect("Could not create bit field");

    let bits = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u128;

    if bits.is_null() {
      panic!("Could not allocate bits field");
    }

    Self { bits, row_ele_size: bit_blocks, rows, len }
  }

  fn slice(&self) -> &[u128] {
    unsafe { std::slice::from_raw_parts(self.bits, self.len) }
  }

  fn slice_mut(&mut self) -> &mut [u128] {
    unsafe { std::slice::from_raw_parts_mut(self.bits, self.len) }
  }

  pub fn set_bit(&mut self, row: usize, bit: usize) {
    let bit = bit & 0x7F;
    let col = bit >> 7;
    let row_offset = row * self.row_ele_size;
    let offset = col + row_offset;
    self.slice_mut()[offset] |= 1 << bit;
  }

  pub fn unset_bit(&mut self, row: usize, bit: usize) {
    let bit = bit & 0x7F;
    let col = bit >> 7;
    let row_offset = row * self.row_ele_size;
    let offset = col + row_offset;
    self.slice_mut()[offset] &= !(1 << bit);
  }

  pub fn is_bit_set(&self, row: usize, bit: usize) -> bool {
    let bit = bit & 0x7F;
    let col = bit >> 7;
    let row_offset = row * self.row_ele_size;
    let offset = col + row_offset;
    (self.slice()[offset] & (1 << bit)) > 0
  }

  pub fn and(&mut self, left: usize, right: usize) -> bool {
    let left = left * self.row_ele_size;
    let right = right * self.row_ele_size;
    let len = self.row_ele_size;
    let slice = self.slice_mut();
    let mut diff = false;

    for i in 0..len {
      let r = slice[right + i];
      let l = &mut slice[left + i];
      let val = *l;
      *l &= r;
      diff |= *l != val;
    }
    diff
  }

  pub fn or(&mut self, left: usize, right: usize) -> bool {
    let left = left * self.row_ele_size;
    let right = right * self.row_ele_size;
    let len = self.row_ele_size;
    let slice = self.slice_mut();
    let mut diff = false;

    for i in 0..len {
      let r = slice[right + i];
      let l = &mut slice[left + i];
      let val = *l;
      *l |= r;
      diff |= *l != val;
    }

    diff
  }

  pub fn mov(&mut self, to: usize, from: usize) -> bool {
    let left = to * self.row_ele_size;
    let right = from * self.row_ele_size;
    let len = self.row_ele_size;
    let slice = self.slice_mut();
    let mut diff = false;

    for i in 0..len {
      let r = slice[right + i];
      let d = &mut slice[left + i];
      let val = *d;
      *d = r;
      diff |= *d != val;
    }

    diff
  }

  pub fn not(&mut self, left: usize) {
    let left = left * self.row_ele_size;
    let len = self.row_ele_size;
    let slice = self.slice_mut();

    for i in 0..len {
      let d = &mut slice[left + i];
      *d = !*d;
    }
  }

  pub fn is_empty(&mut self, left: usize) -> bool {
    let left = left * self.row_ele_size;
    let len = self.row_ele_size;
    let slice = self.slice_mut();

    let mut val = 0;

    for i in 0..len {
      val |= slice[left + i];
    }

    val == 0
  }

  pub fn iter_row_set_indices<'a>(&'a self, row: usize) -> impl Iterator<Item = usize> + 'a {
    BitFieldIndiceIterator {
      row_offset: row * self.row_ele_size,
      bit_offset: 0,
      bitfield:   self,
      col_offset: 0,
    }
  }
}

struct BitFieldIndiceIterator<'bitfield> {
  bitfield:   &'bitfield BitFieldArena,
  row_offset: usize,
  bit_offset: usize,
  col_offset: usize,
}

impl<'bitfield> Iterator for BitFieldIndiceIterator<'bitfield> {
  type Item = usize;
  fn next(&mut self) -> Option<Self::Item> {
    loop {
      if self.bit_offset >= 128 {
        self.col_offset += 1;
        self.bit_offset = 0;
        if self.col_offset >= self.bitfield.row_ele_size {
          return None;
        }
      }

      let bit_val = self.bit_offset;
      self.bit_offset += 1;

      let index_offset = self.col_offset << 7;
      let val = self.bitfield.slice()[self.row_offset + self.col_offset];

      if val == 0 {
        self.bit_offset = 128;
      } else if val & 1 << bit_val > 0 {
        return Some(index_offset + bit_val);
      }
    }
  }
}
