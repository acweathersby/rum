use crate::{
  compiler::compile_module,
  interpreter::interpret_node,
  types::{Database, Value},
};
use rum_lang::{ir::ir_rvsdg::SolveState, istring::CachedString};

#[test]
fn test_fn_call() {
  let mut db = Database::default();

  let global_constraints = compile_module(
    &mut db,
    "
    vec2 ( x: u32, y: u32) => ? :[x = x, y = y] 

    vec3 ( x: u32, y: u32, z: u32 ) => ? { 
      vec2 = vec2(x,y)
      
      
      if x > 2 { :[ x = vec2.x, y = vec2.y, z = z ] }
        otherwise { :[ x = vec2.x, y = vec2.y, z = z ] }
    }


  ",
  );

  if let Some(test) = db.get_routine_with_adhoc_polyfills("vec3".intern(), global_constraints) {
    let test = unsafe { &*test };

    dbg!(test);
    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(1), Value::u32(2), Value::u32(3)], &mut Vec::new(), 0);

      match val {
        Value::Ptr(ptr, _) => {
          let data = unsafe { std::slice::from_raw_parts(ptr as *const u32, 3) };
          assert_eq!(data, &[1, 2, 3])
        }
        val => unreachable!("Unexpected value: {val:?}"),
      }
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_interpreter_adhoc_struct() {
  let mut db = Database::default();

  let global_constraints = compile_module(
    &mut db,
    "vec ( x: u32, y: u32) => ? :[x = x + y, y = y, local = x + 4] 
  ",
  );

  if let Some(test) = db.get_routine_with_adhoc_polyfills("vec".intern(), global_constraints) {
    let test = unsafe { &*test };

    dbg!(test);
    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(30), Value::u32(33)], &mut Vec::new(), 0);

      match val {
        Value::Ptr(ptr, _) => {
          let data = unsafe { std::slice::from_raw_parts(ptr as *const u32, 3) };
          assert_eq!(data, &[30, 33, 34])
        }
        val => unreachable!("Unexpected value: {val:?}"),
      }
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_interpreter_fibonacci() {
  let mut db = Database::default();

  let global_constraints = compile_module(
    &mut db,
    "
  fib (a:?) => ? 
    { x1 = 0 x2 = 1 loop if a > 0 { r = x1 x1 = x1 + x2 x2 = r  a = a - 1 } x1 }
    
",
  );

  if let Some(test) = db.get_routine_with_adhoc_polyfills("fib".intern(), global_constraints) {
    let test = unsafe { &*test };
    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(30), Value::u32(2)], &mut Vec::new(), 0);

      assert_eq!(val, Value::f64(832040.0))
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}
