use std::fmt::Debug;

use super::OpId;

#[derive(Clone, Copy)]
pub(crate) struct RegisterSet<'registers, const NUM_OF_REGISTERS: usize, Register: Clone + Copy> {
  pub(crate) registers:   &'registers [Register; NUM_OF_REGISTERS],
  pub(crate) assignments: [OpId; NUM_OF_REGISTERS],
  pub(crate) reserved:    u64,
  pub(crate) acquired:    u64,
}

impl<'registers, const NUM_OF_REGISTERS: usize, Register: Eq + Clone + Copy + Debug> Debug for RegisterSet<'registers, NUM_OF_REGISTERS, Register> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("Active Registers [")?;

    for (index, reg) in self.registers.iter().enumerate() {
      if self.register_is_acquired(index) {
        f.write_fmt(format_args!(" {reg:?}"))?;
      }
    }

    f.write_str("]\n")?;

    Ok(())
  }
}

impl<'registers, const NUM_OF_REGISTERS: usize, Register: Eq + Clone + Copy + Debug> RegisterSet<'registers, NUM_OF_REGISTERS, Register> {
  pub(crate) fn new(registers: &'registers [Register; NUM_OF_REGISTERS], reserved_registers: Option<&[Register]>) -> Self {
    let mut reserved = 0;

    if let Some(reserved_registers) = reserved_registers {
      for reserved_reg in reserved_registers {
        for (index, register) in registers.iter().enumerate() {
          if register == reserved_reg {
            reserved |= 1 << (index as u64);
            break;
          }
        }
      }
    }

    Self { registers, reserved, acquired: 0, assignments: [OpId::default(); NUM_OF_REGISTERS] }
  }

  #[track_caller]
  pub(crate) fn get_register_from_id(&self, register_id: usize) -> Register {
    *self.registers.get(register_id).expect("")
  }

  #[track_caller]
  pub(crate) fn release_register(&mut self, register_id: usize) {
    // debug_assert!(self.register_is_acquired(register_id), "Register {:?} has not be acquired", self.get_register_from_id(register_id));

    self.acquired &= !(1 << (63 - register_id));
  }

  pub(crate) fn register_is_acquired(&self, register_index: usize) -> bool {
    self.acquired & (1 << (63 - register_index as u64)) > 0
  }

  /// Returns true if the register has not already been acquired.
  pub(crate) fn acquire_specific_register(&mut self, register_index: usize) -> bool {
    if self.register_is_acquired(register_index) {
      return false;
    }

    self.acquired |= 1 << (63 - register_index);

    true
  }

  pub(crate) fn acquire_random_register(&mut self) -> Option<usize> {
    let reg_lu = &mut self.acquired;

    let base = !(*reg_lu/* | self.reserved */);

    if base.leading_zeros() as usize >= NUM_OF_REGISTERS {
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

    debug_assert!((leading_zeros as usize) < NUM_OF_REGISTERS);

    Some(leading_zeros as usize)
  }
}

#[cfg(test)]
pub(crate) mod test_register_set {
  use crate::targets::{reg::Reg, x86::x86_types::*};

  use super::RegisterSet;

  type X86registers<'r> = RegisterSet<'r, 6, Reg>;

  #[test]
  fn test_get_free_register() {
    const REGISTERS: [Reg; 6] = [RAX, RDX, RBX, R10, R11, R12];

    let mut register_set = X86registers::new(&REGISTERS, None);

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
