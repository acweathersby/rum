mod bitfield;

mod ir {
  pub(crate) mod ir_block_compiler;
  pub(crate) mod ir_block_optimizer;
  pub(crate) mod ir_const_val;
  pub(crate) mod ir_optimizer_induction;
  pub(crate) mod ir_register_allocator;
  pub(crate) mod ir_to_register_machine;
  pub(super) mod ir_types;
  pub use ir_types::*;
}

pub(crate) mod x86;

#[cfg(test)]
mod test;
