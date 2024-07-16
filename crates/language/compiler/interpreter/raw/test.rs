use super::ir::{
  ir_block_compiler::compile_function_blocks,
  ir_block_optimizer::optimize_function_blocks,
};
use crate::compiler::{interpreter::error::RumResult, script_parser::parse_ll};

#[test]
fn construct_function_blocks() -> RumResult<()> {
  let input = r##"
  *32 <- table_row_added( test:f32 ) {
  
    table_22:*32, 
    table_33:*32 
      = 
    *{ 5 }, 
    *<test> // ptr(*32, #mem, 22)
  
    i:i32 = 4
    
    loop i + 2  
      >= 0 {
        [table_22 + i] = 231.0
        i = i - 1
      } 
      or {
        [table_22 + 4] = 0
        break
      }
  
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

  dbg!(&funct);

  let blocks = compile_function_blocks(&funct)?;
  dbg!(&blocks);

  let optimized_blocks = optimize_function_blocks(blocks);

  rum_profile::ProfileEngine::report();

  let x86_fn = super::x86::x86_compiler::compile_from_ssa_fn(&optimized_blocks)?;

  let ptr = x86_fn.access_as_call::<fn() -> *mut f32>()();

  unsafe {
    dbg!(std::slice::from_raw_parts::<f32>(ptr, 5));
    assert_eq!(std::slice::from_raw_parts::<f32>(ptr, 5), [231.0, 231.0, 231.0, 231.0, 0.0])
  }

  Ok(())
}
