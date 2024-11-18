/**
 * Stores queryable information about types in the system.
 *
 * Should support
 * - Ability to match function based on input/output types.
 * - Overload functions based on signature
 * - Map functions with self like parameter to member type calls.
 * - Handle partially solved types
 */
// Draft - A template, or blueprint, of a type that may or may not be solved.
use std::collections::VecDeque;

use crate::{
  ir_interpreter::{interpreter::process_node, value::Value},
  parser::script_parser::parse_raw_module,
};

use super::{
  ir_rvsdg::{
    lower::lower_ast_to_rvsdg,
    solve_pipeline::{solve_node, OPConstraint},
  },
  types::{Type, TypeDatabase},
};

pub struct Database {
  db:     TypeDatabase,
  solver: Option<Solver>,
}

impl Database {
  // Returns one function that matches the required
  // get_matching_function_type
  // declare_type
  // update_type
  // cleanup_types

  pub fn new() -> Self {
    Self { db: TypeDatabase::new(), solver: Default::default() }
  }

  pub fn install_module(&mut self, module_source: &str) -> Result<(), String> {
    let module_ast = parse_raw_module(module_source)?;

    let s = lower_ast_to_rvsdg(&module_ast, &mut self.db);

    if let Some(solver) = &mut self.solver {
      solver.merge(s);
    } else {
      self.solver = Some(s);
      // -- dbg!(self.solver);
    }

    Ok(())
  }

  pub fn solve(&mut self) {
    if let Some(mut solver) = self.solver.take() {
      solver.solve(&mut self.db);
    }
  }

  pub fn interpret(&mut self, function_call: &str) -> Result<Value, String> {
    self.install_module(&format!("main () => ? {{ {function_call} }}"))?;

    self.solve();

    if let Some(entry) = self.db.get_ty_entry("main") {
      if let Some(node) = entry.get_node() {
        Ok(process_node(node, &[Value::f32(2.0), Value::f32(2.0)], &mut self.db))
      } else {
        Err("Main is not a complex type".to_string())
      }
    } else {
      Err("Failed to find main function".to_string())
    }
  }
}

#[derive(Debug, Clone)]
struct PartialSolution {
  ty:          Type,
  constraints: Vec<OPConstraint>,
}

pub struct Solver {
  ///
  in_flight: VecDeque<PartialSolution>,
}

impl Solver {
  pub fn merge(&mut self, other: Solver) {
    self.in_flight.extend(other.in_flight);
  }

  pub fn new() -> Self {
    Solver { in_flight: Default::default() }
  }

  pub fn add_type(&mut self, ty: Type, constraints: Vec<OPConstraint>) {
    self.in_flight.push_back(PartialSolution { ty, constraints });
  }

  pub fn solve(&mut self, ty_db: &mut TypeDatabase) {
    while let Some(PartialSolution { ty, constraints }) = self.in_flight.pop_front() {
      if let Some(mut ty) = ty_db.get_ty_entry_from_ty(ty) {
        if let Some(node) = ty.get_node_mut() {
          solve_node(node, &constraints, ty_db);
        }
      }
    }
  }
}
