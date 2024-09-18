use crate::{
  compiler::compile_binary_from_entry,
  ir::ir_build_module::build_module,
  istring::CachedString,
  x86::{print_instructions, x86_eval::x86Function},
};

const target_test_string: &'static str = r##"

BaseArray => [i32; 6]

heap_allocate ( size: u64, alignment: u64 ) => *u8 _malloc( size )


dandy () => u32 0

loop_iter(array: T?) => &i32 {
  i: u32 = 0
  iter if i is 
    < 2 {
      i = i + 1
      yield array[0]
    }
}

main (nest: u32) => D? {
  test: BaseArray = :[ 1 ]

  loop a in loop_iter(test) {
    a = 300 + nest
  }

  test
}

  "##;

const build_up_test_string: &'static str = r##"

add_two_numbers (l: u32, r: u32) => u32 {
  l + r
}


"##;

#[test]
fn compile_structures() {
  let mut db = build_module(&crate::parser::script_parser::parse_raw_module(&build_up_test_string).unwrap());

  let (entry_offset, binary) = compile_binary_from_entry("add_two_numbers".intern(), vec![], db.as_mut());

  print_instructions(&binary, 0);

  let fn_ = x86Function::new(&binary, entry_offset);

  #[derive(Debug)]
  #[repr(C)]
  struct OutStruct {
    data:   *const u8,
    params: u16,
    name:   f32,
  }

  let funct = fn_.access_as_call::<fn(i32, i32) -> i32>();

  let out = funct(150, 2);

  dbg!(out);
  //let out = unsafe { &*out };
  //
  //assert_eq!(out.name, 1234.934);
  assert_eq!(out, 152);

  dbg!(out);
}
