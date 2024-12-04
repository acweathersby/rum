use std::sync::Arc;

use rum_lang::{ir::types::TypeDatabase, istring::IString};

use crate::compiler::SuperNode;

pub(crate) struct Database {
  pub ops:      Vec<Arc<core_lang::parser::ast::Op>>,
  pub routines: Vec<Box<SuperNode>>,
  pub ty_db:    TypeDatabase,
}

impl Default for Database {
  fn default() -> Self {
    Self { ops: Default::default(), routines: Default::default(), ty_db: TypeDatabase::new() }
  }
}

impl Database {
  pub fn get_routine(&self, fn_name: IString) -> Option<&SuperNode> {
    for node in self.routines.iter() {
      if node.binding_name == fn_name {
        return Some(node);
      }
    }
    None
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
