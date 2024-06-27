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
  let input = r##"
  *32 <- table_row_added( test:f32 ) {
  
    table_22:*32 <-{ 5 }  // ptr(*32, #mem, 22)
    table_33:*32 <-<test>
  
    i:i32 
    i = 3
  
    loop {
      match i >= 0 {
        true {
          [table_22 + i] = f32(1130823691)
          i = i-1
          continue
        }
      }
    }
    
    [table_22 + 4] = 0;
  
    <- table_22
  }"##;

  /**
  i = 3;
  table_33 = table_22 + i * 4;
  while table_33 > 0  {
    [table_33] = 0
    table_33 -= 4
  }

  **/
  let funct = parse_ll(&input)?;

  let blocks = compile_function_blocks(&funct)?;

  let optimized_blocks = optimize_function_blocks(blocks);

  let x86_fn = super::x86::x86_compiler::compile_from_ssa_fn(&optimized_blocks)?;

  let ptr = x86_fn.access_as_call::<fn() -> *mut f32>()();

  unsafe {
    assert_eq!(std::slice::from_raw_parts::<f32>(ptr, 4), [
      231.00017, 231.00017, 231.00017, 231.00017
    ])
  }

  // let funct = compile_from_ssa_fn(&blocks)?;
  //
  // funct.call();

  Ok(())
}
