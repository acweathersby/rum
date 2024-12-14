use rum_lang::istring::CachedString;

use crate::{
  compiler::add_module,
  interpreter::interpret_node,
  types::{Database, SolveDatabase, SolveState, Value},
};

#[test]
fn test_missing_call_name() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    best (d: u32) => u32 202222
    
    guest (d: ?) => ? guest(best)
    
    test (d: ?) => ? best(d)

    scope () => ? { 
      test(32)
    }
  ",
  );

  if let Some((test, _)) = SolveDatabase::solve_for("scope".intern(), &db, true, global_constraints) {
    let test = test.get().unwrap();

    dbg!(test);
    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(1), Value::u32(2), Value::u32(3)], &mut Vec::new(), 0);

      dbg!(val);
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_fn_call_with_adhoc_structs() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    vec2 ( x: u32, y: u32) => ? :[x = x, y = y] 

    vec3 ( x: u32, y: u32, z: u32 ) => ? { 
      vec2 = vec2(x,y)
      
      if x > 2 { :[ x = vec2.x, y = vec2.y, z = z ] }
        otherwise { :[ x = vec2.y, y = vec2.x, z = z ] }
    }
  ",
  );

  if let Some((test, _)) = SolveDatabase::solve_for("vec3".intern(), &db, true, global_constraints) {
    let test = test.get().unwrap();

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

  let global_constraints = add_module(
    &mut db,
    "vec ( x: u32, y: u32) => ? :[x = x, y = y, local = x + 4] 
  ",
  );

  dbg!(&db);

  if let Some((test, _)) = SolveDatabase::solve_for("vec".intern(), &db, true, global_constraints) {
    let test = test.get().unwrap();

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

  let global_constraints = add_module(
    &mut db,
    "
  fib (a:?) => ? 
    { x1 = 0 x2 = 1 loop if a > 0 { r = x1 x1 = x1 + x2 x2 = r  a = a - 1 } x1 }
    
",
  );

  if let Some((test, _)) = SolveDatabase::solve_for("fib".intern(), &db, true, global_constraints) {
    let test = test.get().unwrap();

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
