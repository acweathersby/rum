use std::path::PathBuf;

use crate::{
  compiler::{interpreter::error::RumResult, script_parser::parse_ll},
  utils::get_source_file,
};

use super::ir::{
  ir_block_compiler::compile_function_blocks,
  ir_block_optimizer::optimize_function_blocks,
};

#[test]
fn construct_function_blocks() -> RumResult<()> {
  let (input, _) = get_source_file("run_ll_script.lang")?;

  let funct = parse_ll(&input)?;

  let blocks = compile_function_blocks(&funct)?;

  dbg!(&blocks);

  let optimized_blocks = optimize_function_blocks(blocks);

  dbg!(&optimized_blocks);

  let x86_fn = super::x86::x86_compiler::compile_from_ssa_fn(&optimized_blocks)?;

  x86_fn.call();

  // let funct = compile_from_ssa_fn(&blocks)?;
  //
  // funct.call();

  Ok(())
}
