use crate::{
  compiler::add_module,
  interpreter::{interpret, interpret_node},
  solver::solve,
  types::{Database, GetResult, SolveDatabase, SolveState, Type, Value},
};
use rum_lang::istring::CachedString;

#[test]
fn allocator_binding() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    g => [ x: f32 ]

    AllocatorI => [
      allocate: (ctx: AllocatorI, size: u64, par: AllocatorI) => addr,
      free: (ctx: AllocatorI, ptr: addr, par: AllocatorI) =|
    ]

    __root_allocator__ => [ x: u32 ]

    allocate (ctx: __root_allocator__, size: u64, par: AllocatorI) => addr { 
      __allocate__(size)
    } 

    free (ctx: __root_allocator__, ptr: addr) =| {}

    AppendOnlyAllocator => [ x: addr, capacity: u64 ]

    allocate (ctx: AppendOnlyAllocator, size: u64, par: AllocatorI) => addr {

      if ctx.capacity == 0 { 
        ctx.x = par.allocate(4096)
      }

      if ctx.capacity == 0 { 7 } otherwise { 2 }

      // returns address 10. This is normally not valid in 
      // most modern computing environments, but is allowed in 
      // testing. 
    }

    free (ctx: AppendOnlyAllocator, ptr: addr) =| {}


    scope () => ? {
      b = {       

        test* => AppendOnlyAllocator(test*) 
        
        b:g = test* :[x = 0]

        b.x = 2

        b
      }

      b.x
    }
  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "scope".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("scope".intern()) {
    let val = interpret(test, &[Value::f32(11.0), Value::u32(2), Value::u32(3)], &sdb);

    dbg!(val);
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn method_lookup() {
  let mut db: Database = Database::default();

  let global_constraints = add_module(
    &mut db,
    "


    #interface
    AllocatorI => [
      // par: *AllocatorI, requires VTable like system.
      allocate: (ctx: AllocatorI, size: u64, par: AllocatorI) > addr,
      free: (ctx: AllocatorI, ptr: addr, par: AllocatorI)
    ]


    __root_allocator__ => [ d: u32 ]

    allocate => (ctx: __root_allocator__, size: u64, par: AllocatorI) > addr 2020202

    allocate => (b: __root_allocator__, size: u64) > addr __malloc__(size)

    free => (ctx: __root_allocator__, ptr: addr, par: AllocatorI) { }

    allocate => (b: Dase, size: u64, par: AllocatorI) > addr {
      if par as par is __root_allocator__ {
        par.allocate(0) 
      } otherwise {
        0
      }
    }

    free => (base: Dase, ptr: addr, par: AllocatorI) {}

    Dase
      => [ val: u32 ]

    base 
      => [ val: u32 ]
      
    base_getter_method 
      => (b: base, test: f32) > ? b.val + test

    calls_method 
      => () > u32 {
        global* => Dase(global*) 

        // heap = :[]
        // 
        // 

        b: base = global*:[ val = 200 ]

        b.base_getter_method(2)
      }
  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "calls_method".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("calls_method".intern()) {
    let val = interpret(test, &[Value::f32(11.0), Value::u32(2), Value::u32(3)], &sdb);

    dbg!(val);
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn interface_structure() {
  let mut db = Database::default();

  let global_constraints = add_module(
    &mut db,
    "
    AllocatorI => [
      par: *AllocatorI,
      allocate: (ctx: AllocatorI, size: u64, par: AllocatorI) > addr,
      free: (ctx: AllocatorI, ptr: addr, par: AllocatorI)
    ]

   
  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "AllocatorI".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("AllocatorI".intern()) {
    let val = interpret(test, &[Value::f32(11.0), Value::u32(2), Value::u32(3)], &sdb);

    dbg!(val);
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

    test (c: u32) => u32 if a > 10 { c + 2  } otherwise { c + 1 }
  ",
  );

  let mut sdb: SolveDatabase<'_> = solve(&db, "scope".intern(), true);
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("scope".intern()) {
    let val = interpret(test, &[Value::u32(11), Value::u32(2), Value::u32(3)], &sdb);

    dbg!(val);
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
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("scope".intern()) {
    let val = interpret(test, &[Value::u32(7), Value::u32(2), Value::u32(3)], &sdb);

    // Create temporary types based on the type definitions
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
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("vec3".intern()) {
    // Create temporary types based on the type definitions

    let val = interpret(test, &[Value::u32(7), Value::u32(2), Value::u32(3)], &sdb);

    match val {
      Value::Ptr(ptr, _) => {
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u32, 3) };
        assert_eq!(data, &[1, 2, 3])
      }
      val => unreachable!("Unexpected value: {val:?}"),
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
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("vec".intern()) {
    // Create temporary types based on the type definitions

    let val = interpret(test, &[Value::u32(30), Value::u32(33)], &sdb);

    match val {
      Value::Ptr(ptr, _) => {
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u32, 3) };
        assert_eq!(data, &[30, 33, 34])
      }
      val => unreachable!("Unexpected value: {val:?}"),
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
  if let GetResult::Existing(test) = sdb.get_type_by_name_mut("fib".intern()) {
    let val = interpret(test, &[Value::u32(30), Value::u32(2)], &sdb);

    assert_eq!(val, Value::f64(832040.0))
  } else {
    panic!("routine test not found")
  }
}
