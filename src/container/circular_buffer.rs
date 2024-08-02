use std::fmt::Debug;

#[derive(Debug)]
pub struct StackCircularBuffer<D: Default + Sized + Clone + Copy, const SIZE: usize> {
  start:  usize,
  end:    usize,
  buffer: [D; SIZE],
}

impl<Data: Default + Sized + Clone + Copy, const SIZE: usize> StackCircularBuffer<Data, SIZE> {
  pub fn new() -> Self {
    Self { start: 0, end: 0, buffer: [Data::default(); SIZE] }
  }

  pub fn iter(&mut self) -> CircularBufferIter<Data, SIZE> {
    CircularBufferIter { _index: self.start, _data: self }
  }

  pub fn insert(&mut self, d: Data) {
    if self.len() < SIZE {
      self.buffer[self.end] = d;
      self.end = (1 + self.end) % SIZE;
    } else {
      panic!("Not enough space in buffer")
    }
  }

  pub fn is_empty(&self) -> bool {
    self.start == self.end
  }

  pub fn last_mut(&mut self) -> Option<&mut Data> {
    if self.is_empty() {
      None
    } else if self.end == 0 {
      Some(&mut self.buffer[SIZE - 1])
    } else {
      Some(&mut self.buffer[self.end - 1])
    }
  }

  pub fn clear(&mut self) {
    self.start = self.end;
  }

  pub fn len(&self) -> usize {
    if self.end < self.start {
      SIZE - (self.start - self.end)
    } else {
      self.end - self.start
    }
  }

  /// Get the buffer's end marker.
  pub fn get_end(&self) -> usize {
    self.end
  }

  /// Get the buffer's  start marker.
  pub fn get_start(&self) -> usize {
    self.start
  }

  /// Set the end marker of the buffer.
  ///
  /// # Safety
  /// Any modification of the `start` or `end` markers of a circular
  /// buffer could lead to the unintentional loss or the overwriting of
  /// data.
  pub fn set_end(&mut self, end: usize) {
    self.end = end % SIZE;
  }

  /// Set the start marker of the buffer.
  ///
  /// # Safety
  /// Any modification of the `start` or `end` markers of a circular
  /// buffer could lead to the unintentional loss or the overwriting of
  /// data.
  pub fn set_start(&mut self, start: usize) {
    self.start = start % SIZE;
  }
}

pub struct CircularBufferIter<'data, D: Default + Sized + Clone + Copy, const SIZE: usize> {
  _data:  &'data mut StackCircularBuffer<D, SIZE>,
  _index: usize,
}

impl<'data, D: Default + Sized + Clone + Copy, const SIZE: usize> Iterator for CircularBufferIter<'data, D, SIZE> {
  type Item = D;

  fn next(&mut self) -> Option<Self::Item> {
    if self._index == self._data.end {
      None
    } else {
      let val = self._data.buffer[self._index];

      self._index = (self._index + 1) % SIZE;

      Some(val)
    }
  }
}

#[test]

fn container_overwrite() {}
