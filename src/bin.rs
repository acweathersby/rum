use rum_lang::{
  ir::{
    ir_rvsdg::{
      lower::lower_ast_to_rvsdg,
      solve_pipeline::{collect_op_constraints, get_type_from_db, solve_constraints},
      type_solve::{self, NodeTypeInfo},
      RSDVGBinding,
      RVSDGInternalNode,
      RVSDGNode,
      RVSDGNodeType,
    },
    types::TypeDatabase,
  },
  ir_interpreter::interpret,
  istring::CachedString,
  parser::{self, script_parser::parse_raw_module},
};
use std::collections::VecDeque;

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
        let script = args.pop_front().expect("Expected an expression argument");
        let module_ast = parse_raw_module(&format!("{script}")).expect("Could not parse call expression");
        let mut ty_db = TypeDatabase::new();

        lower_ast_to_rvsdg(&module_ast, RVSDGNode::new_module(), &mut ty_db);

        let type_target = args.pop_front().expect("Expected a type argument");

        match get_type_from_db(&mut ty_db, type_target.intern()) {
          Some(node) => {
            let node = unsafe { &*node };

            let op_constraints = collect_op_constraints(node);

            match solve_constraints(node, op_constraints, &mut ty_db) {
              Ok(types) => interpret(
                node,
                &NodeTypeInfo {
                  constraints: Default::default(),
                  inputs:      Default::default(),
                  node_types:  types,
                  outputs:     Default::default(),
                },
                &mut ty_db,
              ),
              Err(errors) => {
                for error in errors.iter() {
                  println!("{error}")
                }
              }
            }
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

        for RSDVGBinding { name, in_id, .. } in module.outputs.iter() {
          if *name == "main".to_token() {
            if let RVSDGInternalNode::Complex(cplx) = &module.nodes[in_id.usize()] {
              if cplx.ty == RVSDGNodeType::Function {
                match type_solve::solve(cplx, &module, &mut ty_db) {
                  Ok(type_info) => interpret(cplx, &type_info, &mut ty_db),
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
