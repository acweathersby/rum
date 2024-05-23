//! # x86 Registers
//!
//!
//! ## Caller / Callee saved registers
//!
//! - Linux:
//!
//! |                | RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 |
//! | :---------     | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
//! | Callee_Saved   |     |     |     |  X  |  X  |  X  |     |     |     |     |     |     |  X  |  X  |  X  |  X  |    
//! | Caller_Saved   |  X  |  X  |  X  |     |     |     |  X  |  X  |  X  |  X  |  X  |  X  |     |     |     |     |
//! | C Calling Arg  |     |  4  |  3  |     |     |     |  2  |  1  |  5  |  6  |     |     |     |     |     |     |
//! | C Return Arg   |  1  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
//! | Syscall Args   |     |     |  3  |     |     |     |  2  |  1  |  5  |  6  |  4  |     |     |     |     |     |
//! | Syscall Return |  1  |     |  2  |     |     |     |     |     |     |     |     |     |     |     |     |     |
//!
//!
//! - Window:
use std::collections::HashMap;

use rum_logger::todo_note;

use crate::compiler::interpreter::ll::{
  types::{BitSize, LLVal, OpArg},
  x86::types::x86Reg,
};

pub(super) struct RegisterAllocator {
  pub registers:  Vec<Register>,
  pub allocation: HashMap<LLVal, usize>,
}

impl RegisterAllocator {
  /// Push currently used registers to the stack
  pub fn save_current() {}

  /// Restore registers that were pushed to through the `save_current` command.
  pub fn restore_current() {}

  pub fn modify_register(&mut self, from: &OpArg<x86Reg>, to: &OpArg<()>) -> OpArg<x86Reg> {
    match from {
      OpArg::REG(register_name, old_val) => {
        let val = to.ll_val();

        if let Some((index, _)) =
          self.registers.iter().enumerate().find(|i| i.1.name() == *register_name)
        {
          self.allocation.insert(val, index);
          self.allocation.remove(old_val);
          OpArg::REG(*register_name, val)
        } else {
          *from
        }
      }
      op => *op,
    }
  }

  pub fn return_register(&mut self, val: LLVal) -> OpArg<x86Reg> {
    if let Some((index, old_val)) =
      self.registers.iter().enumerate().find(|i| i.1.name() == x86Reg::RAX)
    {
      self.allocation.insert(val, index);

      return OpArg::REG(x86Reg::RAX, val);
    } else {
      panic!()
    }
  }

  pub fn set(&mut self, size: BitSize, op: LLVal) -> x86Reg {
    if let Some(index) = self.allocation.get(&op) {
      return self.registers[*index].name();
    }

    for register in &mut self.registers {
      if let None = &register.val {
        register.val = Some(op);
        self.allocation.insert(op, register.index);
        return register.name();
      }
    }

    // Evict the youngest register first
    for register in &mut self.registers {
      todo_note!("Handle register eviction: {:?}", register.name());

      let op = register.val.unwrap();

      self.allocation.remove(&op);

      register.val = Some(op);

      self.allocation.insert(op, register.index);

      return register.name();
    }

    // Resolve jumps

    panic!("Could not acquire register")
  }
}

pub(super) struct Register {
  pub index:  usize,
  pub val:    Option<LLVal>,
  /// The register is active in the current block.
  pub active: bool,
}

impl Register {
  pub fn name(&self) -> x86Reg {
    const NAMES: [x86Reg; 10] = [
      x86Reg::RAX,
      x86Reg::RCX,
      x86Reg::RDX,
      x86Reg::RBX,
      x86Reg::R8,
      x86Reg::R9,
      x86Reg::R10,
      x86Reg::R11,
      //x86Reg::R12,
      //x86Reg::R13,
      x86Reg::R14,
      x86Reg::R15,
    ];

    NAMES[self.index]
  }
}
