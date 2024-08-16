// # x86 Registers
//
//
// ## Caller / Callee saved registers
//
// - Linux:
//
// |                | RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 |
// |                | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 |
// | :---------     | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
// | Callee_Saved   |     |     |     |  X  |  X  |  X  |     |     |     |     |     |     |  X  |  X  |  X  |  X  |
// | Caller_Saved   |  X  |  X  |  X  |     |     |     |  X  |  X  |  X  |  X  |  X  |  X  |     |     |     |     |
// | C Calling Arg  |     |  4  |  3  |     |     |     |  2  |  1  |  5  |  6  |     |     |     |     |     |     |
// | C Return Arg   |  1  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
// | Syscall Args   |     |     |  3  |     |     |     |  2  |  1  |  5  |  6  |  4  |     |     |     |     |     |
// | Syscall Return |  1  |     |  2  |     |     |     |     |     |     |     |     |     |     |     |     |     |
//
//
// - Window:

use crate::{
  ir::{
    ir_build_module::build_module,
    ir_lowering::lower_iops,
    ir_register_allocator::{generate_register_assignments, CallRegisters, RegisterVariables},
    ir_type_analysis::{resolve_routine, resolve_struct_offset},
  },
  istring::*,
  x86::{compile_from_ssa_fn, x86_eval::x86Function},
};

#[test]
fn register_allocator() {
  #[repr(C)]
  struct Temp02 {
    a: u32,
    b: u32,
    c: u32,
  }

  let mut db = build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"
  02Temp => [
    a: u32,
    b: u32,
    c: u32,
  ]

  main (a: *02Temp) =| {
    a.a = 1
    a.b = 2
    a.c = 3
  }"##,
    )
    .unwrap(),
  );

  resolve_struct_offset("02Temp".intern(), &mut db);

  resolve_routine("main".intern(), &mut db);

  lower_iops("main".intern(), &mut db);

  use crate::x86::x86_types::*;
  let reg_pack = RegisterVariables {
    call_register_list: vec![CallRegisters {
      policy_name:         "default".intern(),
      arg_int_registers:   vec![7, 6, 3, 1, 8, 9],
      arg_float_registers: vec![],
      arg_ptr_registers:   vec![],
      ret_int_registers:   vec![1, 2, 3, 4, 5, 6],
      ret_float_registers: vec![],
      ret_ptr_registers:   vec![],
    }],
    ptr_registers:      vec![],
    int_registers:      vec![8, 9, 10, 11, 12, 14, 15, 7, 6, 3, 1, 8, 9, 0, 13],
    float_registers:    vec![],
    registers:          vec![
      RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
      XMM15,
    ],
  };

  let (spilled_variables, assignments) = generate_register_assignments("main".intern(), &mut db, &reg_pack);

  dbg!(&spilled_variables, &assignments);

  let linkable_x86 = compile_from_ssa_fn("main".intern(), &mut db, &assignments, &spilled_variables).expect("Could not create linkable");

  let x86_fn = x86Function::from(linkable_x86);

  let funct = x86_fn.access_as_call::<fn(&mut Temp02)>();

  let mut temp = Temp02 { a: 0, b: 0, c: 0 };

  funct(&mut temp);

  assert_eq!(temp.a, 1);
  assert_eq!(temp.b, 2);
  assert_eq!(temp.c, 3);
}
