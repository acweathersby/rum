use std::sync::Arc;

use rum_lang::{
  ir::{ir_rvsdg::SolveState, types::TypeDatabase},
  istring::IString,
};

use super::RootNode;

pub(crate) struct Database {
  pub ops:      Vec<Arc<core_lang::parser::ast::Op>>,
  pub routines: Vec<Box<RootNode>>,
  pub ty_db:    TypeDatabase,
}

impl Default for Database {
  fn default() -> Self {
    Self { ops: Default::default(), routines: Default::default(), ty_db: TypeDatabase::new() }
  }
}

impl Database {
  pub fn get_routine(&self, fn_name: IString) -> Option<&RootNode> {
    for node in self.routines.iter() {
      if node.binding_name == fn_name {
        return Some(node);
      }
    }
    None
  }

  pub fn get_routine_with_adhoc_polyfills(&self, fn_name: IString) -> Option<&RootNode> {
    if let Some(routine) = self.get_routine(fn_name) {
      if routine.solve_state() == SolveState::Template {
        for ty in &routine.type_vars {
          if ty.ty.is_generic() {
            println!("Need to polyfill {ty}");
          }
        }

        todo!("Polyfill");
      } else {
        Some(routine)
      }
    } else {
      None
    }
  }
}

pub fn add_ops_to_db(db: &mut Database, ops: &str) {
  for op in core_lang::parser::parse_ops(ops).expect("Failed to parse ops").ops.iter() {
    db.ops.push(op.clone());
  }
}

pub fn get_op_from_db(db: &Database, name: &str) -> Option<Arc<core_lang::parser::ast::Op>> {
  for op in &db.ops {
    if op.name == name {
      return Some(op.clone());
    }
  }

  None
}
pub(crate) const ROUTINE_ID: &'static str = "---ROUTINE---";
pub(crate) const LOOP_ID: &'static str = "---LOOP---";
pub(crate) const MATCH_ID: &'static str = "---MATCH---";
pub(crate) const CLAUSE_SELECTOR_ID: &'static str = "---SELECT---";
pub(crate) const CLAUSE_ID: &'static str = "---CLAUSE---";
pub(crate) const CALL_ID: &'static str = "---CALL---";
