mod bitfield;
pub(crate) mod ssa_block_compiler;
pub(crate) mod ssa_block_optimizer;
pub(crate) mod ssa_optimizer_induction;
pub(crate) mod ssa_to_register_machine;

mod types;
//pub(crate) mod x86;

use types::*;
#[cfg(test)]
mod test;
  