use crate::{
  compiler::compile_binary_from_entry,
  ir::{
    ir_build_module::build_module,
    ir_lowering::lower_iops,
    ir_register_allocator::{generate_register_assignments, CallRegisters, RegisterVariables},
    ir_type_analysis::{resolve_routine, resolve_struct_offset},
  },
  istring::CachedString,
  types::PrimitiveType,
  x86::{compile_from_ssa_fn, print_instructions, x86_eval::x86Function},
};

#[test]
fn compile_structures() {
  let mut db = build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

main () =| {
  test: [i32; 2] = [1, 2]
}

  "##,
    )
    .unwrap(),
  );

  let (entry_offset, binary) = compile_binary_from_entry("start".intern(), vec![], db.as_mut());

  print_instructions(&binary, 0);

  let fn_ = x86Function::new(&binary, entry_offset);

  #[derive(Debug)]
  #[repr(C)]
  struct OutStruct {
    data:   *const u8,
    params: u16,
    name:   f32,
  }

  let funct = fn_.access_as_call::<fn() -> *const OutStruct>();

  let out = funct();
  let out = unsafe { &*out };

  assert_eq!(out.name, 1234.934);
  assert_eq!(out.params, 22);

  dbg!(out);
}
