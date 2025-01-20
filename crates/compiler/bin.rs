#![allow(unused_variables, dead_code)]
use compiler::add_module;
use interpreter::interpret;
use rum_compiler::*;
use types::{Database, SolveDatabase};
// Stage 0 - Compiles single file and runs all "#test" annotated functions, which should have this signature `#test name => () `

fn main() {
  let args = std::env::args().collect::<Vec<_>>();

  if args.len() >= 2 {
    let path = std::path::PathBuf::from(args[1].clone());

    let path = if path.is_relative() { std::env::current_dir().expect("Could not read current directory").join(path) } else { path }
      .canonicalize()
      .expect("Could not find path");

    if let Ok(input) = std::fs::read_to_string(path) {
      let input = input.as_str();

      let mut db = Database::default();

      add_module(&mut db, input);

      let sdb: SolveDatabase<'_> = SolveDatabase::solve_for("#test", &db);

      let sdb_opt = sdb.optimize(types::OptimizeLevel::MemoryOperations_01);

      for item in sdb_opt.get("#test") {
        let val = interpret(item, &[], &sdb_opt);

        println!("{item:?} = {val:?}");
      }
    } else {
      panic!("Could not read {}", args[0])
    }
  }
}
                                                                                                       