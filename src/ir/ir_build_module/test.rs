use crate::{
  istring::CachedString,
  types::{ComplexType, PrimitiveType},
};

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

#[test]
fn test_primitive_resolution() {
  let mut scope = crate::types::TypeContext::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

test_primitive_variables () =| {
  0x: u8 = 0
  1x: u16 = 0
  2x: u32 = 0
  3x: u64 = 0
  4x: i8 = 0
  5x: i16 = 0
  6x: i32 = 0
  7x: i64 = 0
  8x: f32 = 0
  9x: f64 = 0
  10x: f32v2 = 0
  11x: f32v4 = 0
}
  "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );

  if let Some(ComplexType::Routine(r)) = scope.get(0, "test_primitive_variables".to_token()) {
    assert_eq!(*r.variables.entries[0].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u8);
    assert_eq!(*r.variables.entries[1].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u16);
    assert_eq!(*r.variables.entries[2].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u32);
    assert_eq!(*r.variables.entries[3].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u64);
    assert_eq!(*r.variables.entries[4].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i8);
    assert_eq!(*r.variables.entries[5].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i16);
    assert_eq!(*r.variables.entries[6].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i32);
    assert_eq!(*r.variables.entries[7].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i64);
    assert_eq!(*r.variables.entries[8].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32);
    assert_eq!(*r.variables.entries[9].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f64);
    assert_eq!(*r.variables.entries[10].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32v2);
    assert_eq!(*r.variables.entries[11].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32v4);
  } else {
    panic!("Routine not correctly built");
  }
}
