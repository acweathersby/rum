use crate::{
  ir::ir_rvsdg::{lower, type_solver},
  parser::script_parser::*,
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

  let ty_db = TypeDatabase::new();

  let parsed = &crate::parser::script_parser::parse_raw_module(&BUILD_UP_TEST_STRING).expect("Parsing Failed");

  // dbg!(parsed);

  let func = &parsed.members.members[0];

  let module_members_group_Value::AnnotatedModMember(func) = &func else { panic!("") };

  let module_member_Value::RawRoutine(fn_decl) = &func.member else { panic!("") };

  let fn_node = lower::lower_ast_to_rvsdg(fn_decl, ty_db);

  println!("{:#}", fn_node);
}

#[test]
fn test_simple_type_solve_with_binary_expression() {
  const BUILD_UP_TEST_STRING: &'static str = "
  add_two_numbers (l: T?, r: T?, j: u32) => u64 {
    l + j
  }
  ";

  let ty_db = TypeDatabase::new();

  let parsed = &crate::parser::script_parser::parse_raw_module(&BUILD_UP_TEST_STRING).expect("Parsing Failed");

  let func = &parsed.members.members[0];

  let module_members_group_Value::AnnotatedModMember(func) = &func else { panic!("") };

  let module_member_Value::RawRoutine(fn_decl) = &func.member else { panic!("") };

  let mut fn_node = lower::lower_ast_to_rvsdg(fn_decl, ty_db);

  println!("{:#}", &fn_node);

  let ty_expr = type_solver::solve(&mut fn_node);

  assert!(ty_expr.is_some(), "Expression type should be solved for this node");
}
