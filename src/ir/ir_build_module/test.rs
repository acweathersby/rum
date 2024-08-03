use super::build_module;

#[test]
fn test() {
  let mut scope = crate::types::TypeContext::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

Temp => [
  bf32 
  : #desc8 
  | name  : u8 
  | id    : u8 
  | mem   : u8

  bval:u32
]

Bent => [
  bf32: #desc6 | name: u16 
  val: u32
]

Union => Bent | Temp | u32

main () =| { 

  d = Bent [
    name = 2
  ]

  r = d.name
}
  "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );
}

#[test]
fn test_type_inference() {
  let mut scope = crate::types::TypeContext::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

Temp => [ a:u32 b:u32 ]
TempA => [ a:u32 b:u32 ]

inferred_procedure ( test: &T? dest: &T? ) =| { 
  test.a = 1
  test.b = 3
}

main_procedure ( t: &TempA d: &Temp ) =| {
  inferred_procedure(t, d) 
  inferred_procedure(t, d) 
}
  "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );
}
