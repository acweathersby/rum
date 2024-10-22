use rum_lang::{
  ir::ir_rvsdg::{
    lower::{self, lower_ast_to_rvsdg},
    solve_pipeline::{collect_constraints, get_type_from_db, IRModuleDatabase, SolveUnit},
    type_solve,
    RSDVGBinding,
    RVSDGInternalNode,
    RVSDGNode,
    RVSDGNodeType,
    TypeDatabase,
  },
  ir_interpreter::interpret,
  istring::CachedString,
  parser::{self, script_parser::parse_raw_module},
};
use std::{
  collections::{HashMap, VecDeque},
  sync::Arc,
};

fn main() -> Result<(), u8> {
  let mut args: VecDeque<String> = std::env::args().collect();
  let _ = args.pop_front();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "int" => {
        run_interpreter(args);
        return Ok(());
      }
      "ty" => {
        let mut module = RVSDGNode::new_module();

        let script = args.pop_front().expect("Expected an expression argument");
        let module_ast = parse_raw_module(&format!("{script}")).expect("Could not parse call expression");
        let mut ty_db = TypeDatabase::new();

        module = lower_ast_to_rvsdg(&module_ast, module, &mut ty_db);

        let type_target = args.pop_front().expect("Expected a type argument");

        let mut db = IRModuleDatabase { modules: HashMap::from_iter([("main".intern(), Box::into_raw(module))]) };
        dbg!(&db);

        match get_type_from_db(&mut db, type_target.intern()) {
          Some(node) => {
            let mut unit = SolveUnit { constraints: Default::default(), node, resolved: false };
            collect_constraints(&mut unit, &mut db);
          }
          _ => panic!("Could not find {type_target}"),
        }

        todo!("Return the type of the given expression")
      }
      cmd => panic!("Unrecognized command \"{cmd}\""),
    }
  }

  Ok(())
}

fn run_interpreter(mut args: VecDeque<String>) {
  let mut module = RVSDGNode::new_module();
  let mut ty_db = TypeDatabase::new();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "-c" => {
        let call_expr_str = args.pop_front().expect("Expected a call expression following -c");
        let module_ast = parse_raw_module(&format!("main () => Y? {call_expr_str}")).expect("Could not parse call expression");
        module = lower_ast_to_rvsdg(&module_ast, module, &mut ty_db);

        for RSDVGBinding { name, in_id, out_id, ty, input_index } in module.outputs.iter() {
          if *name == "main".to_token() {
            if let RVSDGInternalNode::Complex(cplx) = &module.nodes[in_id.usize()] {
              if cplx.ty == RVSDGNodeType::Function {
                match type_solve::solve(cplx, &module, &mut ty_db) {
                  Ok(type_info) => interpret(cplx, &type_info, &module, &mut ty_db),
                  Err(err) => panic!("{err:?}"),
                }
              }
            }
          }
        }
      }
      "-p" => {
        let script = args.pop_front().unwrap();

        let module_ast = parser::script_parser::parse_raw_module(&script).expect("Parsing Failed");

        module = lower_ast_to_rvsdg(&module_ast, module, &mut ty_db);

        /*    for (_, funct) in &mut module.functs {
          let constraints = type_solve::solve(funct);

          dbg!(funct, &constraints);
        } */
      }
      "-f" => {
        if let Some(path_str) = args.pop_front() {
          if let Ok(path) = std::fs::canonicalize(&std::path::PathBuf::from(path_str)) {
            if let Ok(string) = std::fs::read_to_string(path) {
              let module_ast = parser::script_parser::parse_raw_module(&string).expect("Parsing Failed");
              module = lower_ast_to_rvsdg(&module_ast, module, &mut ty_db);

              /*       for (_, funct) in &mut module.functs {
                let constraints = type_solve::solve(funct);
                dbg!(funct, &constraints);
              } */
            }
          }
        }
      }
      arg => panic!("Unrecognized arg \"{arg}\""),
    }
  }
}
