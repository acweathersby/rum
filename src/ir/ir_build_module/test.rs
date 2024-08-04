use crate::{
  istring::CachedString,
  types::{ComplexType, PrimitiveType},
};

use super::build_module;

#[test]
fn test_bitfield_structs() {
  let mut scope = crate::types::TypeScope::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

Bent => [
  bf32: #desc6 | name: u16,
  val: u32,
]

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
  let mut scope = crate::types::TypeScope::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"



temp { 

  A => Temp | TempA  
  
  temp { 
    A => test/TempA | TempA 
  }  
} 


Temp => [ a:u32 b:u32 ]

TempA => [ a:u32 b:u32 ]

inferred_procedure ( test: &T? dest: &TempA ) => &T? { 
  test.a = 1 + 2
  test.b = test.b << 4
  test
}

main_procedure ( 
  t:&TempA? 
  d:&TempA? 
) =| {
  t.a = 2+2
  inferred_procedure(t, d) 
}
  "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );

  dbg!(scope);
}

#[test]
fn test_primitive_resolution() {
  let mut scope = crate::types::TypeScope::new();
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

  if let Some(ComplexType::Routine(r)) = scope.get(0, "test_primitive_variables".to_token()).and_then(|t| t.as_cplx_ref()) {
    let r = r.lock().unwrap();
    assert_eq!(*r.body.vars.entries[0].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u8);
    assert_eq!(*r.body.vars.entries[1].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u16);
    assert_eq!(*r.body.vars.entries[2].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u32);
    assert_eq!(*r.body.vars.entries[3].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::u64);
    assert_eq!(*r.body.vars.entries[4].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i8);
    assert_eq!(*r.body.vars.entries[5].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i16);
    assert_eq!(*r.body.vars.entries[6].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i32);
    assert_eq!(*r.body.vars.entries[7].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::i64);
    assert_eq!(*r.body.vars.entries[8].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32);
    assert_eq!(*r.body.vars.entries[9].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f64);
    assert_eq!(*r.body.vars.entries[10].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32v2);
    assert_eq!(*r.body.vars.entries[11].ty.as_prim().expect("Should Be A Primitive"), PrimitiveType::f32v4);
  } else {
    panic!("Routine not correctly built");
  }
}
