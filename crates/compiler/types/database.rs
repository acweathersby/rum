use super::{CallLookup, RootNode};
use crate::{
  compiler::{compile_struct, OPS},
  solver::{solve, GlobalConstraint},
  types::*,
};
use rum_lang::istring::{CachedString, IString};
use std::{
  collections::{HashMap, VecDeque},
  sync::{Arc, Mutex, MutexGuard},
};

#[derive(Debug)]
pub struct DatabaseCore {
  pub parent:  *const Database,
  pub ops:     Vec<Arc<core_lang::parser::ast::Op>>,
  pub objects: Vec<(IString, NodeHandle)>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RootType {
  ExecutableEntry(IString),
  LibRoutine(IString),
  Test(IString),
}

#[derive(Debug)]
pub struct SolveDatabase<'a> {
  // Used to lookup
  pub db:          &'a Database,
  pub roots:       Vec<(RootType, NodeHandle)>,
  pub nodes:       Vec<NodeHandle>,
  pub sig_lookup:  Vec<(u64, NodeHandle)>,
  pub name_lookup: Vec<(IString, NodeHandle)>,
}

pub enum GetResult {
  Existing(NodeHandle),
  Introduced(NodeHandle),
  NotFound,
}

impl<'a> SolveDatabase<'a> {
  pub fn new<'b>(db: &'b Database) -> SolveDatabase<'b> {
    SolveDatabase {
      db,
      roots: Default::default(),
      sig_lookup: Default::default(),
      name_lookup: Default::default(),
      nodes: Default::default(),
    }
  }

  pub fn add_object(&mut self, name: IString, node: NodeHandle) {}

  pub fn get_root(&self, root_ty: RootType) -> Option<NodeHandle> {
    for ((c_root_ty, root)) in &self.roots {
      if *c_root_ty == root_ty {
        return Some(root.clone());
      }
    }

    None
  }

  pub fn add_root(&mut self, root_ty: RootType, root: NodeHandle) {
    self.roots.push((root_ty, root.clone()));
    self.nodes.push(root.clone());
  }

  // Returns a Type bound to a name in the user's binding namespace.
  pub fn get_type_by_name(&mut self, name: IString) -> GetResult {
    use GetResult::*;

    for ((root_ty, root)) in &self.roots {
      match root_ty {
        RootType::ExecutableEntry(root_name) => {
          if *root_name == name {
            return Existing(root.clone());
          }
        }
        _ => unreachable!(),
      }
    }

    if let Some(node) = self.name_lookup.iter().find(|(n, _)| *n == name) {
      return Existing(node.1.clone());
    }

    if let Some(node) = self.db.get_object_mut(name) {
      let node = node.duplicate();
      self.name_lookup.push((name, node.clone()));
      self.nodes.push(node.clone());
      return Introduced(node);
    }

    NotFound
  }

  // Returns a type generated from an inline definition. May return a virtual type.
  pub fn generate_type() -> Type {
    Type::Undefined
  }

  // Returns any type that matches the given signature.
  pub fn get_type_by_signature() -> Option<NodeHandle> {
    None
  }
}

#[derive(Clone, Debug)]
pub struct Database(pub std::sync::Arc<std::sync::Mutex<DatabaseCore>>);

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

  pub fn get_mut_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  pub fn get_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  pub fn add_object(&self, name: IString, node: NodeHandle) {
    let mut db = self.get_mut_ref();

    for (binding_name, outgoing) in &db.objects {
      if *binding_name == name {
        //panic!("Incoming {node:?} would replace outgoing node {outgoing:?}");
      }
    }

    db.objects.push((name, node));
  }

  pub fn get_object_mut_with_sig(&self, obj_name: IString, obj_sig: u64) -> Option<NodeHandle> {
    None
  }

  pub fn get_object_mut(&self, fn_name: IString) -> Option<NodeHandle> {
    for (binding_name, node) in self.get_ref().objects.iter_mut() {
      if *binding_name == fn_name {
        return Some(node.clone());
      }
    }

    if self.get_ref().parent != std::ptr::null() {
      unsafe { &*self.get_ref().parent }.get_object_mut(fn_name)
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
