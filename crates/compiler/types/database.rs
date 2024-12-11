use std::{
  collections::VecDeque,
  sync::{Arc, Mutex, MutexGuard},
};

use rum_lang::{
  ir::ir_rvsdg::SolveState,
  istring::{CachedString, IString},
};

use crate::{
  compiler::{compile_struct, OPS},
  solver::{polyfill, solve, GlobalConstraint},
  types::*,
};

use super::{CallLookup, RootNode};

#[derive(Debug)]
struct DatabaseCore {
  pub parent:  *const Database,
  pub ops:     Vec<Arc<core_lang::parser::ast::Op>>,
  pub objects: Vec<Box<RootNode>>,
}

#[derive(Clone, Debug)]
pub struct Database(std::sync::Arc<std::sync::Mutex<DatabaseCore>>);

impl Default for Database {
  fn default() -> Self {
    let mut core = DatabaseCore { ops: Default::default(), objects: Default::default(), parent: std::ptr::null() };
    add_ops_to_db(&mut core, &OPS);
    Self(Arc::new(Mutex::new(core)))
  }
}

impl Database {
  pub fn inherited(parent: &Database) -> Database {
    let mut db = Database::default();
    db.get_mut_ref().parent = parent;
    db
  }

  fn get_mut_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  fn get_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  pub fn add_object(&self, mut node: Box<RootNode>) -> Option<*const RootNode> {
    node.host_db = Some(self.clone());

    let name = node.binding_name;

    let mut db = self.get_mut_ref();

    for outgoing in &db.objects {
      if outgoing.binding_name == name {
        //panic!("Incoming {node:?} would replace outgoing node {outgoing:?}");
      }
    }

    let len = db.objects.len();
    db.objects.push(node);

    Some(Box::as_ptr(&db.objects[len]))
  }

  pub fn get_object_mut(&self, fn_name: IString) -> Option<*mut RootNode> {
    for node in self.get_ref().objects.iter_mut() {
      if node.binding_name == fn_name {
        return Some(Box::as_mut_ptr(node));
      }
    }

    if self.get_ref().parent != std::ptr::null() {
      unsafe { &*self.get_ref().parent }.get_object_mut(fn_name)
    } else {
      None
    }
  }

  pub fn get_routine_with_adhoc_polyfills(&mut self, fn_name: IString, global_constraints: Vec<GlobalConstraint>) -> Option<(*const RootNode)> {
    let mut proxy_db = solve(self, global_constraints, true);

    let db = proxy_db.as_mut().unwrap_or(self);

    if let Some(node_ref) = db.get_object_mut(fn_name).map(|n| unsafe { n as *mut _ }) {
      let node: &mut RootNode = unsafe { &mut *node_ref };

      if node.solve_state() == SolveState::Template {
        let global_constraints = polyfill(node, db);
        solve(db, global_constraints, true);
      }
      dbg!(node);

      return db.get_object_mut(fn_name).map(|f| f as *const _);
    } else {
      None
    }
  }
}

/// TODO: Ops should be a constant static dataset
pub fn add_ops_to_db(db: &mut DatabaseCore, ops: &str) {
  for op in core_lang::parser::parse_ops(ops).expect("Failed to parse ops").ops.iter() {
    db.ops.push(op.clone());
  }
}

pub fn get_op_from_db(db: &Database, name: &str) -> Option<Arc<core_lang::parser::ast::Op>> {
  for op in &db.get_ref().ops {
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
