use std::path::PathBuf;

use crate::{
  compiler::{interpreter::error::RumResult, script_parser::parse_ll},
  utils::get_source_file,
};

use super::{
  ssa_block_compiler::compile_function_blocks,
  ssa_block_optimizer::optimize_function_blocks,
};

#[test]
fn construct_function_blocks() -> RumResult<()> {
  let (input, _) = get_source_file("run_ll_script.lang")?;

  let funct = parse_ll(&input)?;

  let blocks = compile_function_blocks(&funct)?;

  //dbg!(&blocks);

  let optimized_blocks = optimize_function_blocks(blocks);

  super::x86::compiler::compile_from_ssa_fn(&optimized_blocks);
  // let funct = compile_from_ssa_fn(&blocks)?;
  //
  // funct.call();

  Ok(())
}
