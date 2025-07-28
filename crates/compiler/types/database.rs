use super::RootNode;
use crate::{
  finalizer::finalize,
  ir_compiler::STRUCT_ID,
  optimizer::optimize,
  solver::{solve, GlobalConstraint},
  types::*,
};
use rum_common::{CachedString, IString};
use rum_lang::parser::script_parser::entry_Value;
use std::{
  collections::{BTreeMap, HashMap},
  sync::{Arc, Mutex, MutexGuard},
};

#[derive(Debug)]
pub(crate) struct DatabaseCore {
  pub parent:   *const Database,
  pub ops:      Vec<Arc<core_lang::parser::ast::Op>>,
  pub nodes:    Vec<(IString, NodeHandle, Vec<NodeConstraint>)>,
  pub name_map: HashMap<IString, usize>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RootType {
  ExecutableEntry(IString),
  LibRoutine(IString),
  Test(IString),
  Any,
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Debug, Hash)]
pub struct CMPLXId(pub u32);

impl<'a, 'b> From<&'a SolveDatabase<'b>> for CMPLXId {
  fn from(value: &'a SolveDatabase<'b>) -> Self {
    Self(value.nodes.len() as u32)
  }
}

impl<'a> From<&'a mut SolveDatabase<'a>> for CMPLXId {
  fn from(value: &'a mut SolveDatabase<'a>) -> Self {
    Self(value.nodes.len() as u32)
  }
}

impl<'a> From<(CMPLXId, &'a SolveDatabase<'a>)> for NodeHandle {
  fn from((id, db): (CMPLXId, &'a SolveDatabase)) -> Self {
    let index = id.0 as usize;
    db.nodes[index].clone()
  }
}

#[derive(Debug, Clone)]
pub struct SolveDatabase<'a> {
  // Used to lookup
  pub db:           &'a Database,
  pub(crate) nodes: Vec<NodeHandle>,
  pub roots:        Vec<(RootType, CMPLXId)>,
  pub sig_lookup:   Vec<(u64, CMPLXId)>,
  pub name_map:     Vec<(IString, CMPLXId)>,

  /// Comptime type table.
  ///
  /// This is a parallel to the runtime time type table.
  /// ---
  /// Solver owns all type vars. Should destroy the objects of these pointers when this object
  /// goes out of scope.
  pub(crate) comptime_type_table:      Vec<*const RumTypeObject>,
  /// Maps type name to an entry in the type_table.
  pub comptime_type_name_lookup_table: HashMap<CMPLXId, usize>,

  pub(crate) interface_instances: BTreeMap<RumType, BTreeMap<RumType, BTreeMap<u64, CMPLXId>>>,
  pub heap_map:                   HashMap<IString, u32>,
  pub heap_count:                 usize,
}

pub enum OptimizeLevel {
  MemoryOperations_01,
  ExpressionOptimization_02,
  LoopOptimization_03,
  FunctionInlining_04,
}

pub(crate) enum GetResult {
  Existing(CMPLXId),
  Introduced((CMPLXId, Vec<NodeConstraint>)),
  NotFound,
}

impl<'a> SolveDatabase<'a> {
  pub fn new<'b>(db: &'b Database) -> SolveDatabase<'b> {
    let mut solve_db = SolveDatabase {
      db,
      roots: Default::default(),
      sig_lookup: Default::default(),
      name_map: Default::default(),
      nodes: Default::default(),
      interface_instances: Default::default(),
      heap_map: Default::default(),
      comptime_type_table: Default::default(),
      comptime_type_name_lookup_table: Default::default(),
      heap_count: 0,
    };

    for (index, (name, prim_ty, type_struct)) in [
      ("type".intern(), prim_ty_struct, &RUM_TYPE),
      ("type_prop".intern(), prim_ty_struct, &RUM_TYPE_PROP),
      ("type_prim".intern(), prim_ty_struct, &RUM_PRIM_TYPE),
      ("type_ref".intern(), prim_ty_struct, &RUM_TYPE_REF),
      ("core$$type_table".intern(), prim_ty_struct, &RUM_TYPE_TABLE),
      ("u32".intern(), prim_ty_u32, &RUM_TEMP_U32_TYPE),
      ("f32".intern(), prim_ty_f32, &RUM_TEMP_F32_TYPE),
      ("u64".intern(), prim_ty_u64, &RUM_TEMP_U64_TYPE),
      ("str".intern(), prim_ty_struct, &RUM_TEMP_STRING_TYPE),
    ]
    .into_iter()
    .enumerate()
    {
      let id = solve_db.add_object(
        name,
        NodeHandle::new(RootNode {
          nodes: vec![Node { children: vec![], index: 0, loop_type: LoopType::None, parent: -1, ports: vec![], type_str: STRUCT_ID }],
          ty: RumType { raw_type: prim_ty, type_id: index as _ },
          ..Default::default()
        }),
      );
      solve_db.comptime_type_table.push(unsafe { std::mem::transmute(type_struct) });
      solve_db.comptime_type_name_lookup_table.insert(id, index);
    }

    solve_db
  }

  pub fn solve_for<'b>(solve_ty: &str, db: &'b Database) -> SolveDatabase<'b> {
    let mut solver_db = Self::new(db);

    let node = rum_lang::parser::script_parser::parse_entry(solve_ty).expect("Input is not an entry");

    let mut global_constraints = Vec::new();

    match &node.entry {
      entry_Value::Annotation(annotation) => {
        let annotation_str = annotation.val.intern();

        for (name, node, node_constraints) in db.get_ref().nodes.iter().filter(|node| node.1.get().unwrap().annotations.iter().any(|s| *s == annotation_str)) {
          let node = node.duplicate();

          let node_id = solver_db.add_object(*name, node.clone());

          if node_constraints.len() > 0 {
            global_constraints.push(GlobalConstraint::ResolveObjectConstraints { node_id, constraints: node_constraints.clone() });
          }

          solver_db.add_root(RootType::Any, node_id);
          //global_constraints.push(GlobalConstraint::ExtractGlobals { node_id });
        }
      }
      entry_Value::NamedEntries(names) => {
        for name in &names.names {
          let name = name.id.intern();
          if let Some((node, node_constraints)) = db.get_object_mut(name) {
            let node = node.duplicate();

            let node_id = solver_db.add_object(name, node.clone());

            if node_constraints.len() > 0 {
              global_constraints.push(GlobalConstraint::ResolveObjectConstraints { node_id, constraints: node_constraints.clone() });
            }

            solver_db.add_root(RootType::Any, node_id);
          }
        }
      }
      entry_Value::RawRoutineDefinition(routine_sig) => {
        todo!("Routine sig");
      }
      _ => unreachable!(),
    }

    solve(&mut solver_db, global_constraints, false);

    solver_db
  }

  pub fn finalize(&self) -> SolveDatabase {
    finalize(self)
  }

  pub fn optimize(&self, optimize_level: OptimizeLevel) -> SolveDatabase {
    optimize(self, optimize_level)
  }

  pub fn get<'b>(&'b self, solve_ty: &str) -> impl Iterator<Item = CMPLXId> + 'b {
    let node = rum_lang::parser::script_parser::parse_entry(solve_ty).expect("Input is not an entry");
    match &node.entry {
      entry_Value::Annotation(annotation) => {
        let annotation_str = annotation.val.intern();

        self
          .nodes
          .iter()
          .enumerate()
          .filter(move |(_, n)| {
            let n = n.get().unwrap();
            n.annotations.contains(&annotation_str)
          })
          .map(|(index, _)| CMPLXId(index as u32))
      }
      entry_Value::NamedEntries(names) => {
        todo!("{names:?}");
      }
      entry_Value::RawRoutineDefinition(routine_sig) => {
        todo!("Routine sig");
      }
      _ => unreachable!(),
    }
  }

  pub fn get_root(&self, root_ty: RootType) -> Option<CMPLXId> {
    for (c_root_ty, root) in &self.roots {
      if *c_root_ty == root_ty {
        return Some(*root);
      }
    }

    None
  }

  pub fn add_root(&mut self, root_ty: RootType, root: CMPLXId) {
    self.roots.push((root_ty, root));
  }

  pub(crate) fn add_generated_node(&mut self, handle: NodeHandle) -> GetResult {
    let root_id = handle.get().unwrap().root_id;
    if let Some((i, _)) = self.nodes.iter().enumerate().find(|(_, n)| n.get().unwrap().root_id == root_id) {
      GetResult::Existing(CMPLXId(i as u32))
    } else {
      let id = CMPLXId(self.nodes.len() as u32);
      self.nodes.push(handle.duplicate());
      GetResult::Introduced((id, vec![]))
    }
  }

  pub(crate) fn get_type_by_name(&self, name: IString) -> GetResult {
    use GetResult::*;

    for ((root_ty, root)) in &self.roots {
      match root_ty {
        RootType::ExecutableEntry(root_name) => {
          if *root_name == name {
            return Existing(*root);
          }
        }
        _ => {}
      }
    }

    if let Some(node) = self.name_map.iter().find(|(n, _)| *n == name) {
      return Existing(node.1.clone());
    }

    NotFound
  }

  /// Returns a Type bound to a name in the user's binding namespace.
  pub(crate) fn get_type_by_name_mut(&mut self, name: IString) -> GetResult {
    use GetResult::*;

    for (root_ty, root) in &self.roots {
      match root_ty {
        RootType::ExecutableEntry(root_name) => {
          if *root_name == name {
            return Existing(root.clone());
          }
        }
        _ => {}
      }
    }

    if let Some(node) = self.name_map.iter().find(|(n, _)| *n == name) {
      return Existing(node.1.clone());
    }

    if let Some((node, constraints)) = self.db.get_object_mut(name) {
      return Introduced((self.add_object(name, node), constraints));
    }

    NotFound
  }

  fn add_object(&mut self, name: IString, node: NodeHandle) -> CMPLXId {
    let node = node.duplicate();
    let id = CMPLXId(self.nodes.len() as u32);
    self.name_map.push((name, id));
    self.nodes.push(node.clone());
    id
  }

  // Returns any type that matches the given signature.
  pub(crate) fn get_type_by_signature() -> Option<NodeHandle> {
    None
  }
}

#[derive(Clone, Debug)]
pub struct Database(pub(crate) std::sync::Arc<std::sync::Mutex<DatabaseCore>>);

impl Default for Database {
  fn default() -> Self {
    let mut core = DatabaseCore { ops: Default::default(), nodes: Default::default(), parent: std::ptr::null(), name_map: Default::default() };

    add_ops_to_db(&mut core, &OP_DEFINITIONS);
    let this = Self(Arc::new(Mutex::new(core)));

    this
  }
}

impl Database {
  pub fn inherited(parent: &Database) -> Database {
    let mut db = Database::default();
    db.get_mut_ref().parent = parent;
    db
  }

  pub(crate) fn get_mut_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<'a, DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  pub(crate) fn get_ref<'a: 'b, 'b>(&'a self) -> MutexGuard<'a, DatabaseCore> {
    self.0.lock().expect("Failed to lock database")
  }

  pub(crate) fn add_object(&self, name: IString, node: NodeHandle, constraints: Vec<NodeConstraint>) {
    let mut db = self.get_mut_ref();
    let obj = node.get().unwrap();

    for (binding_name, outgoing, _) in &db.nodes {
      if *binding_name == name {
        //panic!("Incoming {node:?} would replace outgoing node {outgoing:?}");
      }
    }

    node.get_mut().unwrap().root_id = db.nodes.len() as isize;

    db.nodes.push((name, node, constraints));
  }

  pub(crate) fn get_object_mut_with_sig(&self, obj_name: IString, obj_sig: u64) -> Option<NodeHandle> {
    None
  }

  pub(crate) fn get_object_mut<'a: 'b, 'b>(&'a self, fn_name: IString) -> Option<(NodeHandle, Vec<NodeConstraint>)> {
    for (binding_name, node, constraints) in self.get_ref().nodes.iter_mut() {
      if *binding_name == fn_name {
        return Some((node.clone(), constraints.clone()));
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
pub(crate) fn add_ops_to_db(db: &mut DatabaseCore, ops: &str) {
  for op in core_lang::parser::parse_ops(ops).expect("Failed to parse ops").ops.iter() {
    db.ops.push(op.clone());
  }
}

pub fn get_op_from_db(db: &Database, op: Op) -> Option<Arc<core_lang::parser::ast::Op>> {
  let op_name = op.get_name();
  for op in &db.get_ref().ops {
    if op.name == op_name {
      return Some(op.clone());
    }
  }

  None
}
