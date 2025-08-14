#![allow(unused_variables, dead_code)]

use std::{collections::HashMap, path::PathBuf};

use rum_common::IString;
use rum_compiler::{
  elf_link::elf_link, ir_compiler::add_module, linker::comptime_link, targets::{self, x86::x86_eval}, types::*
};

struct SymbolTable {
  symbol_lookup: HashMap<IString, usize>,
  symbols:       Vec<(usize, *const u8)>,
}

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

      // Process an executable file
      if true {
        let sdb: SolveDatabase<'_> = SolveDatabase::solve_for("#main", &db);
        
        let sdb_fin = sdb.finalize();
        
        let mut sdb_opt = sdb_fin.optimize(rum_compiler::types::OptimizeLevel::MemoryOperations_01);

        let bin_functs = targets::x86::compile(&sdb_opt);

        let main = sdb.roots[0].1;

        elf_link(&mut sdb_opt, main, bin_functs, &PathBuf::from("/home/work/test/"), "test_main");
      } else
      // Process a test file
      {
        let sdb: SolveDatabase<'_> = SolveDatabase::solve_for("#test", &db);
        
        let sdb_fin = sdb.finalize();

        let mut sdb_opt = sdb_fin.optimize(rum_compiler::types::OptimizeLevel::MemoryOperations_01);

        let bin_functs = targets::x86::compile(&sdb_opt);

        // LINKER ====================================================

        let (entries, binary) = comptime_link(&mut sdb_opt, bin_functs);

        if let Some(offset) = sdb_opt.roots.first().and_then(|f| entries.get(&f.1)) {
          let func = x86_eval::x86Function::new(&binary, *offset);

          let out = func.access_as_call::<fn() -> u8>()();

          dbg!(out, out);

          assert_eq!(out, 8, "Failed to parse correctly");
        }
      }
    } else {
      panic!("Could not read {}", args[0])
    }
  }
}
