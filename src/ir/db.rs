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
  container::get_aligned_value,
  ir::types::EntryOffsetData,
  ir_interpreter::{
    interpreter::{process_node, process_op},
    value::Value,
  },
  parser::script_parser::parse_raw_module,
};

use super::{
  ir_rvsdg::{
    lower::lower_ast_to_rvsdg,
    solve_pipeline::{solve_node, OPConstraint},
    SolveState,
    VarId,
  },
  types::{AggOffsetData, Type, TypeDatabase, TypeEntry},
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

  pub fn get_type_entry(&self, ty: Type) -> Option<&TypeEntry> {
    self.db.get_ty_entry_from_ty(ty).map(|t| &*t)
  }

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
        Ok(process_node(node, &mut self.db))
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
  hash:        u64,
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
    self.in_flight.push_back(PartialSolution { hash: 0, ty, constraints });
  }

  pub fn solve(&mut self, ty_db: &mut TypeDatabase) {
    while let Some(PartialSolution { hash, ty, constraints }) = self.in_flight.pop_front() {
      if let Some(mut entry) = ty_db.get_ty_entry_from_ty(ty) {
        if let Some(node) = entry.get_node_mut() {
          solve_node(node, &constraints, ty_db);

          if node.solved == SolveState::Solved {
            match node.ty {
              super::ir_rvsdg::RVSDGNodeType::Struct => {
                let mut agg_data = AggOffsetData { alignment: 0, byte_size: 0, ele_count: 0, member_offsets: vec![] };

                let types = &node.types;
                for output in node.outputs.iter() {
                  let ty = types[output.in_op.usize()];

                  let VarId::VarName(name) = output.id else {
                    continue;
                  };

                  match ty {
                    Type::Primitive(prim) => {
                      let offset = get_aligned_value(agg_data.byte_size as u64, prim.byte_size as u64) as usize;

                      agg_data.member_offsets.push(EntryOffsetData { ty, name, offset, size: prim.byte_size as usize });
                      agg_data.byte_size = offset + prim.byte_size as usize
                    }
                    ty => todo!("handle: {name}:{ty}"),
                  }
                }
                entry.offset_data = Some(agg_data);
              }
              super::ir_rvsdg::RVSDGNodeType::Array => {
                let mut agg_data = AggOffsetData { alignment: 0, byte_size: 0, ele_count: 0, member_offsets: vec![] };

                let mut ele_count = 0;

                for output in node.outputs.iter() {
                  match output.id {
                    VarId::ArraySize => {
                      let mut vec = vec![Value::Uninitialized; node.nodes.len()];
                      process_op(output.in_op, node, &mut vec, ty_db);

                      match vec[output.in_op] {
                        Value::u32(val) => agg_data.ele_count = val as usize,
                        _ => unreachable!(),
                      }
                    }
                    VarId::ArrayType => {
                      let ty = node.types[output.in_op.usize()];
                      match ty {
                        Type::Primitive(prim) => {
                          agg_data.alignment = prim.byte_size as usize;
                          agg_data.byte_size = prim.byte_size as usize;
                          agg_data.member_offsets.push(super::types::EntryOffsetData {
                            ty,
                            name: Default::default(),
                            offset: 0,
                            size: prim.byte_size as usize,
                          });
                        }
                        _ => unreachable!(),
                      }
                    }
                    _ => unreachable!(),
                  }
                }

                agg_data.byte_size = agg_data.byte_size * agg_data.ele_count;

                entry.offset_data = Some(agg_data);
              }

              ty => {
                println!("TODO: calculate {ty:?} agg data");
              }
            }
          }
        }
      }
    }
  }
}
