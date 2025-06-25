use std::fmt::Debug;

use super::OpId;

#[derive(Clone, Copy)]
pub(crate) struct RegisterSet {
  pub(crate) acquired: u64,
  pub(crate) size:     u8,
}

impl Debug for RegisterSet {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("Active Registers [")?;

    for index in 0..64usize {
      if self.register_is_acquired(index) {
        f.write_fmt(format_args!(" r{index:03}"))?;
      }
    }

    f.write_str("]\n")?;

    Ok(())
  }
}

impl RegisterSet {
  pub(crate) fn new(size: u8) -> Self {
    Self { acquired: 0, size }
  }

  pub(crate) fn join(&self, other: &Self) -> Self {
    Self { acquired: self.acquired | other.acquired, size: self.size }
  }

  #[track_caller]
  pub(crate) fn release_register(&mut self, register_id: usize) {
    self.acquired &= !(1 << (63 - register_id));
  }

  pub(crate) fn register_is_acquired(&self, register_index: usize) -> bool {
    self.acquired & (1 << (63 - register_index as u64)) > 0
  }

  pub(crate) fn mask(&mut self, mask: u64) {
    self.acquired |= mask
  }

  /// Returns true if the register has not already been acquired.
  pub(crate) fn acquire_specific_register(&mut self, register_index: usize) -> bool {
    if self.register_is_acquired(register_index) {
      return false;
    }

    self.acquired |= 1 << (63 - register_index);

    true
  }

  pub(crate) fn get_active_reg_indices<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
    (0..self.size as usize).into_iter().filter(|i| self.register_is_acquired(*i))
  }

  pub(crate) fn acquire_random_register(&mut self) -> Option<usize> {
    let reg_lu = &mut self.acquired;

    let base = !(*reg_lu/* | self.reserved */);

    if base.leading_zeros() as usize >= self.size as usize {
      return None;
    }

    let mask = base | base >> 1;
    let mask = mask | (mask >> 2);
    let mask = mask | (mask >> 4);
    let mask = mask | (mask >> 8);
    let mask = mask | (mask >> 16);
    let mask = mask | (mask >> 32);

    let leading_zeros = mask.leading_zeros();

    //println!("\n\nl: {reg_lu:064b}\nr: {base:064b}\nm: {mask:064b}\nb: {leading_zeros:064b}\n {}\n\n", leading_zeros as usize);

    *reg_lu = *reg_lu | 1 << (63 - leading_zeros);

    debug_assert!((leading_zeros as usize) < self.size as usize);

    Some(leading_zeros as usize)
  }
}

#[cfg(test)]
pub(crate) mod test_register_set {
  use crate::targets::{reg::Reg, x86::x86_types::*};

  use super::RegisterSet;

  type X86registers = RegisterSet;

  #[test]
  fn test_get_free_register() {
    const REGISTERS: [Reg; 6] = [RAX, RDX, RBX, R10, R11, R12];

    let mut register_set = X86registers::new(6);

    assert_eq!(register_set.acquire_random_register(), Some(0));
    assert_eq!(register_set.acquire_random_register(), Some(1));
    assert_eq!(register_set.acquire_random_register(), Some(2));
    assert_eq!(register_set.acquire_random_register(), Some(3));
    assert_eq!(register_set.acquire_random_register(), Some(4));

    register_set.release_register(2);

    assert_eq!(register_set.acquire_random_register(), Some(2));
    assert_eq!(register_set.acquire_random_register(), Some(5));

    assert_eq!(register_set.acquire_random_register(), None);
  }
}
