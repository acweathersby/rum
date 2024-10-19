use crate::{
  ir::ir_rvsdg::{lower, type_solve, RSDVGBinding, RVSDGInternalNode, RVSDGNode, RVSDGNodeType},
  istring::CachedString,
  types::TypeDatabase,
};

#[test]
fn test() {
  const BUILD_UP_TEST_STRING: &'static str = "

  add_two_numbers (l: T?, r: R?) => j? {

    l + r


/*     main:u32 = 0
    
    d = l + r - 200.45 + main 

    g = 22 + d(main, d + d)

    if d is < 0 { 
      r = d(main, d)
    } otherwise {
      r = d(d, main)
    } 
    
    d + g + r */
  
  }
  ";

  let mut module = RVSDGNode::new_module();

  let parsed = &crate::parser::script_parser::parse_raw_module(&BUILD_UP_TEST_STRING).expect("Parsing Failed");

  // dbg!(parsed);

  let mut module = lower::lower_ast_to_rvsdg(parsed, RVSDGNode::new_module());

  for RSDVGBinding { name, in_id, out_id, ty, input_index } in module.outputs.to_vec() {
    if let RVSDGInternalNode::Complex(cplx) = &mut module.nodes[in_id] {
      if cplx.ty == RVSDGNodeType::Function {}
    }
  }

  /*   let Some(fn_node) = module.functs.get_mut(&"add_two_numbers".to_token()) else { panic!("Function does not exists for some reason!") };

  println!("{:#}", fn_node); */
}

#[test]
fn test_simple_type_solve_with_binary_expression() {
  const BUILD_UP_TEST_STRING: &'static str = "
  add_two_numbers (l: u8, d: T) => T {
    l + d.test
    d.test = 2 + d.test + d.var
    d.test
  }
  ";

  let parsed = &crate::parser::script_parser::parse_raw_module(&BUILD_UP_TEST_STRING).expect("Parsing Failed");

  let mut module = lower::lower_ast_to_rvsdg(parsed, RVSDGNode::new_module());

  println!("{:#?}", &module);

  /*   let Some(fn_node) = module.functs.get_mut(&"add_two_numbers".to_token()) else { panic!("Function does not exists for some reason!") };

  let constraints = type_solve::solve(fn_node);

  dbg!(&constraints);

  assert!(constraints.is_ok(), "Expression type should be solved for this node \n {:?}", constraints);
  assert!(constraints.unwrap().is_some(), "Expression type should be solved for this node"); */
}

#[test]
fn test_struct_lowered_to_rvsg() {
  const BUILD_UP_TEST_STRING: &'static str = "
  
  string => [
    len: u32, 
    data: *u8
  ]

  bindingname => [
    test: undef, 
    b: u32, 
  ]

  b () => bindingname {
    2 + 2
  }

  ";

  let parsed = &crate::parser::script_parser::parse_raw_module(&BUILD_UP_TEST_STRING).expect("Parsing Failed");

  let module = lower::lower_ast_to_rvsdg(parsed, RVSDGNode::new_module());

  println!("{:#?}", &module);

  /*   let Some(fn_node) = module.structs.get_mut(&"bindingname".to_token()) else { panic!("Function does not exists for some reason!") };

  let constraints = type_solve::solve(fn_node);

  dbg!(&constraints);

  assert!(constraints.is_ok(), "Expression type should be solved for this node"); */
}
