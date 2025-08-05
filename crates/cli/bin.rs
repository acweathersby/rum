#![allow(unused_variables, dead_code)]

use rum_compiler::{ir_compiler::add_module, linker, targets::{self, x86::{print_instructions, x86_eval}}, types::*};



// Stage 0 - Compiles single file and runs all "#test" annotated functions, which should have this signature `#test name => () `

fn main() {
  let args = std::env::args().collect::<Vec<_>>();

  if args.len() >= 2 {
    let path = std::path::PathBuf::from(args[1].clone());

    let path = if path.is_relative() { std::env::current_dir().expect("Could not read current directory").join(path) } else { path }.canonicalize().expect("Could not find path");

    if let Ok(input) = std::fs::read_to_string(path) {
      let input = input.as_str();

      let mut db = Database::default();

      add_module(&mut db, input);

      let sdb: SolveDatabase<'_> = SolveDatabase::solve_for("#test", &db);

      let sdb_fin = sdb.finalize();

      let sdb_opt = sdb_fin.optimize(rum_compiler::types::OptimizeLevel::MemoryOperations_01);

      let bin_functs = targets::x86::compile(&sdb_opt);

      // LINKER ====================================================

      let (entry_offset, binary) = linker::link(bin_functs, &sdb_opt);

      println!("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa\n\n\n");
      print_instructions(&binary, 0);

      let func = x86_eval::x86Function::new(&binary, entry_offset);

      let out = func.access_as_call::<fn(u32) -> &'static (f32, u32)>()(1);

      dbg!( out as *const _ as *const usize);

      assert_eq!(out, &(2f32, 3u32), "Failed to parse correctly");

      // TEMP: Run the binary.

      panic!("Finished: Have binary. Need to wrap in some kind of portable unit to allow progress of compilation and linking.");


    /*      let func = x86_eval::x86Function::new(&binary);

    assert_eq!(func.access_as_call::<fn(f32, f32) -> &'static (f32, u32)>()(10f32, 3f32), &(2f32, 3u32), "Failed to parse correctly");

    // TEMP: Run the binary.

    panic!("Finished: Have binary. Need to wrap in some kind of portable unit to allow progress of compilation and linking.");

    for item in sdb_opt.get("#test") {
      let val = interpret(item, &[], &sdb_opt);

      println!("{item:?} = {val:?}");
    } */
    } else {
      panic!("Could not read {}", args[0])
    }
  }
}
