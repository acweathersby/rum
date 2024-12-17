use rum_lang::istring::CachedString;

use crate::{
  compiler::add_module,
  interpreter::interpret_node,
  solver::solve,
  types::{Database, GetResult, SolveDatabase, SolveState, Value},
};

#[test]
fn allocator_binding() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    g => [ x: f32 ]

    slot => [ size: address ]

    named_heap => [ addr: u32 ]

    scope (i:?) => ? {
      hilbert* => named_heap(local*)

      b: g = hilbert*:[ x = i ]           
      
      d: g = hilbert*:[ x = i + b.x + b.x ]
      
      d.x
    }

  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "scope".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("scope".intern()) {
    let test = test.get().unwrap();

    dbg!(test);

    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::f32(11.0), Value::u32(2), Value::u32(3)], &mut Vec::new(), 0);

      dbg!(val);
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_missing_call_name() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    best (a: u32) => u32 if a > 10 { a + 2  } otherwise { a + 1 }
    
    guest (b: ?) => ? best(b)
    
    scope (input: ?) => ? { 
      test(input)
    }

    test (c: u32) => u32 if a > 10 { c+ 2  } otherwise { c + 1 }

  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "scope".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("scope".intern()) {
    let test = test.get().unwrap();

    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(11), Value::u32(2), Value::u32(3)], &mut Vec::new(), 0);

      dbg!(val);
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_missing_var() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "

    c(r: u32) => ? r + 2

    a(b: u32) => ? { d = b  + c(2) + 2 d }

    scope(i: ?) => ? {
      i + a(100) + 2
    }

  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "scope".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("scope".intern()) {
    let test = test.get().unwrap();

    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_node(test, &[Value::u32(7), Value::u32(2), Value::u32(3)], &mut Vec::new(), 0);

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

  let mut sdb: SolveDatabase<'_> = solve(&db, "vec3".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("vec3".intern()) {
    let test = test.get().unwrap();

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

  let mut sdb = solve(&db, "vec".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("vec".intern()) {
    let test = test.get().unwrap();

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
  let mut sdb = solve(&db, "fib".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name("fib".intern()) {
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
