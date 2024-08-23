use crate::{
  ir::{
    ir_lowering::lower_iops,
    ir_register_allocator::{generate_register_assignments, CallRegisters, RegisterVariables},
    ir_type_analysis::{resolve_routine, resolve_struct_offset},
  },
  istring::CachedString,
  types::PrimitiveType,
  x86::{compile_from_ssa_fn, x86_eval::x86Function},
};

use super::build_module;

#[test]
fn test_bitfield_structs() {
  build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

Bent => 
[
  bf32: #desc6 | name: u16,
  val: u32,
]

main () =|  { 
  d = Bent [
    name = 2
  ]
  r = d.name
}
  "##,
    )
    .unwrap(),
  );
}

#[test]
fn struct_instantiation() {
  let mut db = build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

// * =: nullable + movable


EMPTY_NAME => []

// ARRAY_TYPE => [u32; 2, 3]

LNX_MATH_CONSTS => f64 >
  : PI => 3.14567890
  : E  => 2.3344556677


LNX_MATH_CONSTS => u32 >
  : CONTINUE
  : NEW_GAME
  : OPTIONS
  : QUIT
  : SFX_VOL
  : MUSIC_VOL
  : FULLSCREEN
  : BACK


LNX_Clone_Flags => flag32 >
  : CLONE_CHILD_CLEARTID 
  : CLONE_CHILD_SETTID 
  : CLONE_CLEAR_SIGHAND
  : CLONE_DETACHED
  : CLONE_FILES
  : CLONE_FS
  : CLONE_INTO_CGROUP
  : CLONE_IO
  : CLONE_NEWCGROUP
  : CLONE_NEWIPC
  : CLONE_NEWNET
  : CLONE_NEWNS
  : CLONE_NEWPID
  : CLONE_NEWUSER
  : CLONE_NEWUTS
  : CLONE_PARENT
  : CLONE_PARENT_SETTID
  : CLONE_PID
  : CLONE_PIDFD
  : CLONE_PTRACE
  : CLONE_SETTLS
  : CLONE_SIGHAND
  : CLONE_THREAD
  : CLONE_SYSVSEM
  : CLONE_UNTRACED
  : CLONE_VFORK
  : CLONE_VM


StackFrame => [ 
  data:   *u8, // 8
  params: u16, // 2
  name:   f32, // 4
] 

#internal-str
str => [
  data: flag32 > :TRUE :FALSE
]

#internal-Type
Type => [
  name: str,
]

douglas => Type | str

heap_allocate ( size: u64, alignment: u64 ) => *u8 _malloc( size )

start () => g*StackFrame [ name = 1234.934, params = 22 ]

  "##,
    )
    .unwrap(),
  );

  resolve_struct_offset("StackFrame".intern(), &mut db);

  resolve_routine("start".intern(), &mut db);

  resolve_routine("heap_allocate".intern(), &mut db);

  lower_iops("start".intern(), &mut db);

  lower_iops("heap_allocate".intern(), &mut db);

  use crate::x86::x86_types::*;
  let reg_pack = RegisterVariables {
    call_register_list: vec![CallRegisters {
      policy_name:         "default".intern(),
      arg_int_registers:   vec![7, 6, 3, 1, 8, 9],
      arg_float_registers: vec![],
      arg_ptr_registers:   vec![],
      ret_int_registers:   vec![0],
      ret_float_registers: vec![],
      ret_ptr_registers:   vec![],
    }],
    ptr_registers:      vec![],
    int_registers:      vec![8, 9, 10, 11, 12, 14, 15, 7, 6, 3, 1, 8, 9, 0, 13],
    float_registers:    vec![16, 17],
    registers:          vec![
      RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
      XMM15,
    ],
  };

  let heap_allocate_linkable = {
    let (spilled_variables, assignments) = generate_register_assignments("heap_allocate".intern(), &mut db, &reg_pack);

    dbg!(&spilled_variables, &assignments);

    compile_from_ssa_fn("heap_allocate".intern(), &mut db, &assignments, &spilled_variables).expect("Could not create linkable")
  };

  let start_linkable = {
    let (spilled_variables, assignments) = generate_register_assignments("start".intern(), &mut db, &reg_pack);

    dbg!(&spilled_variables, &assignments);

    compile_from_ssa_fn("start".intern(), &mut db, &assignments, &spilled_variables).expect("Could not create linkable")
  };

  // Now we link

  let mut parts = vec![(0, start_linkable), (0, heap_allocate_linkable)];

  let mut out_binary = vec![];

  const MALLOC: unsafe extern "C" fn(usize) -> *mut libc::c_void = libc::malloc;
  const FREE: unsafe extern "C" fn(*mut libc::c_void) = libc::free;

  use crate::x86::*;

  push_bytes(&mut out_binary, MALLOC);
  push_bytes(&mut out_binary, FREE);

  for (offset, link) in &mut parts {
    *offset = out_binary.len();
    out_binary.extend(link.binary.clone());
  }

  for (offset, link) in &parts {
    for rt in &link.link_map {
      match rt.link_type {
        crate::linker::LinkType::DBGRoutine(name) => match name.to_str().as_str() {
          "_malloc" => {
            let diff = (0 as i32) - ((rt.binary_offset + *offset + 4) as i32);
            rt.replace(unsafe { out_binary.as_mut_ptr().offset(*offset as isize) }, diff);
          }
          name => panic!("could not recognize binary debug function: {name}"),
        },
        crate::linker::LinkType::Routine(name) => {
          if let Some((target_offset, _)) = parts.iter().find(|(_, l)| l.name == name) {
            let diff = (*target_offset as i32) - ((rt.binary_offset + *offset + 4) as i32);
            rt.replace(unsafe { out_binary.as_mut_ptr().offset(*offset as isize) }, diff);
          } else {
            panic!("Could not find target {name}");
          }
        }
        _ => {}
      }
    }
  }
  println!("\n\n\n");
  print_instructions(&out_binary, 0);

  let fn_ = x86Function::new(&out_binary, 16);

  #[derive(Debug)]
  #[repr(C)]
  struct OutStruct {
    data:   *const u8,
    params: u16,
    name:   u32,
  }

  let funct = fn_.access_as_call::<fn() -> *const OutStruct>();

  let out = funct();
  let out = unsafe { &*out };
  dbg!(out);
}

#[test]
fn test_type_inference() {
  let mut db = build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

#test
Temp => [ a:u32, b:u32 ]

#test
TempA   => [ a: harkness* u64, b: Ptr  ]
Ptr     => [ d: u32 ]
Message => [ u32; 1 ]

c_str => [ data: static* u8 ]


HPARAM => [ val: u64 ]          WPARAM => [ val: u64 ]

WNDPROC => [
  unnamedParam1: u64,           unnamedParam2: u32,
  unnamedParam3: WPARAM,        unnamedParam4: HPARAM
]

WNDCLASSA => [ 
  style: u32,                   lpfnWndProc: WNDPROC,
]

best ( test: gc* U?, dest: gc* U? ) => *U? {}

inferred_procedure ( test: gc* T?, dest: gc* TempA ) => *T? { 
  a = 200000
  test.a = 1 + 2 * test.a
  test.b = test.b.d << 4
  test
}

main_procedure ( 
  t: gen* TempA?,
  d: gen* TempA? 
) =| {
  d.a = 2 // Invalid assignment of 2 to d. Should use := syntax to declare a new type for d
  inferred_procedure(t, d) 
}
  "##,
    )
    .unwrap(),
  );

  resolve_routine("main_procedure".intern(), &mut db);

  lower_iops("main_procedure".intern(), &mut db);
}

/* #[test]
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
 */
