use rum_lang::{
  ir::ir_rvsdg::{lower::lower_ast_to_rvsdg, type_solve, RSDVGBinding, RVSDGInternalNode, RVSDGNode, RVSDGNodeType},
  ir_interpreter::interpret,
  istring::CachedString,
  parser::{self, script_parser::parse_raw_module},
  types::TypeDatabase,
};
use std::collections::VecDeque;

fn main() -> Result<(), u8> {
  let mut args: VecDeque<String> = std::env::args().collect();
  let _ = args.pop_front();
  let mut module = RVSDGNode::new_module();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "-c" => {
        let call_expr_str = args.pop_front().expect("Expected a call expression following -c");
        let module_ast = parse_raw_module(&format!("main () => Y? {call_expr_str}")).expect("Could not parse call expression");
        module = lower_ast_to_rvsdg(&module_ast, module);

        for RSDVGBinding { name, in_id, out_id, ty, input_index } in module.outputs.iter() {
          if *name == "main".to_token() {
            if let RVSDGInternalNode::Complex(cplx) = &module.nodes[in_id.usize()] {
              if cplx.ty == RVSDGNodeType::Function {
                match type_solve::solve(cplx, &module) {
                  Ok(type_info) => interpret(cplx, &type_info, &module),
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

        module = lower_ast_to_rvsdg(&module_ast, module);

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
              module = lower_ast_to_rvsdg(&module_ast, module);

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
