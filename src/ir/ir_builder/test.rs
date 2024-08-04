use crate::{
  ir::{
    ir_build_module::process_types,
    ir_builder::{IRBuilder, SuccessorMode},
    ir_graph::*,
  },
  istring::CachedString,
  types::*,
};

#[test]
fn variable_contexts() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"
Temp => [
  bf32: #desc8 | name: u8 | id: u8 | mem: u8
  val:u32
]"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut body = RoutineBody::default();
  let mut vars = RoutineVariables::default();
  let mut sm = IRBuilder::new(&mut body, &type_scope, 0, &type_scope);

  // Get the type info of the Temp value.
  let ty = sm.get_type("Temp".to_token()).unwrap().clone();

  sm.push_variable("test".to_token(), ty.into());

  let var = sm.get_variable_from_id(MemberName::IdMember("test".to_token())).expect("Variable \"test\" should exist");

  let var1 = sm.get_variable_member(&var, MemberName::IdMember("val".to_token())).expect("Variable \"test.name\" should exist");

  let var2 = sm.get_variable_member(&var, MemberName::IdMember("val".to_token())).expect("Variable \"test.name\" should exist");

  assert_eq!(var1.store, var2.store);

  dbg!(var);

  dbg!(sm);
}

#[test]
fn stores() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

      Temp => [d:u32]

"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut body = RoutineBody::default();
  let mut vars = RoutineVariables::default();
  let mut sm = IRBuilder::new(&mut body, &type_scope, 0, &type_scope);

  // Get the type info of the Temp value.
  let ty = sm.get_type("Temp".to_token()).unwrap().clone();

  sm.push_variable("test".to_token(), ty.clone());

  let var_id = VarId::new(0);
  assert_eq!(sm.body.vars.entries[0].ty_var.var, var_id);

  /*   sm.push_ssa(IROp::STORE, ty.clone().into(), &[], var_id);
  assert_eq!(sm.body.vars.entries[0].store, IRGraphId::new(1));

  sm.push_ssa(IROp::MEM_STORE, ty.clone().into(), &[], var_id);
  assert_eq!(sm.body.vars.entries[0].store, IRGraphId::new(2)); */

  dbg!(sm);
}

#[test]
fn blocks() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

      Temp => [d:u32]

"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut body = RoutineBody::default();
  let mut vars = RoutineVariables::default();
  let mut sm = IRBuilder::new(&mut body, &type_scope, 0, &type_scope);

  sm.push_variable("Test".to_token(), PrimitiveType::u32.into());

  let block = sm.create_block();
  sm.set_successor(block, SuccessorMode::Default);
  sm.set_active(block);

  sm.push_variable("Test1".to_token(), PrimitiveType::u32.into());

  assert!(sm.get_variable_from_id(MemberName::IdMember("Test".to_token())).is_some());
  assert!(sm.get_variable_from_id(MemberName::IdMember("Test1".to_token())).is_some());

  dbg!(&sm.body.blocks);
}
