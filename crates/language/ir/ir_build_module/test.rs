use super::build_module;

#[test]
fn test() {
  let mut scope = crate::types::TypeScopes::new();
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

Temp => [
  bf32 
  : #desc8 
  | name  : u8 
  | id    : u8 
  | mem   : u8

  val:u32
]

Bent => [
  bf32: #desc6 | name: u16 
  val: u32
]

Union => Bent | Temp

main => () {
  PI = 3.14159
  data = 2.0 + PI * 22.22
  data + PI
}
    
  
  
  "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );
}
