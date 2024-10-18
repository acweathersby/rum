use rum_lang::{
  ir::ir_rvsdg::{lower::lower_ast_to_rvsdg, type_solve, RSDVGBinding, RVSDGInternalNode, RVSDGNodeType},
  ir_interpreter::interpret,
  istring::CachedString,
  parser::{self, script_parser::parse_raw_module},
  types::TypeDatabase,
};
use std::collections::VecDeque;

fn main() -> Result<(), u8> {
  let mut args: VecDeque<String> = std::env::args().collect();
  let _ = args.pop_front();
  let mut ty_db = TypeDatabase::new();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "-c" => {
        let call_expr_str = args.pop_front().expect("Expected a call expression following -c");
        let module = parse_raw_module(&format!("main () =| {call_expr_str}")).expect("Could not parse call expression");
        let mut module = lower_ast_to_rvsdg(&module, &mut ty_db);

        for RSDVGBinding { name, in_id, out_id, ty, input_index } in module.outputs.to_vec() {
          if name == "main".to_token() {
            if let RVSDGInternalNode::Complex(cplx) = &mut module.nodes[in_id] {
              if cplx.ty == RVSDGNodeType::Function {
                interpret(cplx);
              }
            }
          }
        }
      }
      "-p" => {
        let script = args.pop_front().unwrap();

        let parsed = parser::script_parser::parse_raw_module(&script).expect("Parsing Failed");

        let mut module = lower_ast_to_rvsdg(&parsed, &mut ty_db);

        /*    for (_, funct) in &mut module.functs {
          let constraints = type_solve::solve(funct);

          dbg!(funct, &constraints);
        } */
      }
      "-f" => {
        if let Some(path_str) = args.pop_front() {
          if let Ok(path) = std::fs::canonicalize(&std::path::PathBuf::from(path_str)) {
            if let Ok(string) = std::fs::read_to_string(path) {
              let parsed = parser::script_parser::parse_raw_module(&string).expect("Parsing Failed");
              let mut module = lower_ast_to_rvsdg(&parsed, &mut ty_db);

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

  Ok(())
}
