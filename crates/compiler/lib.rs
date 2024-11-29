use container::ArrayVec;
use core_lang::parser::ast::annotation_Value;
use ir::{
  ir_rvsdg::{lower::get_type, VarId},
  types::{Type, TypeDatabase, *},
};
use istring::{CachedString, IString};
use parser::script_parser::{assignment_var_Value, member_group_Value, statement_Value, RawBlock};
use rum_lang::{
  parser::script_parser::{expression_Value, routine_type_Value, RawRoutine},
  *,
};
use std::{
  collections::{HashMap, VecDeque},
  fmt::{Debug, Display},
  sync::Arc,
};
use types::ConstVal;

struct Database {
  pub ops:      Vec<Arc<core_lang::parser::ast::Op>>,
  pub routines: Vec<()>,
  pub ty_db:    TypeDatabase,
}

impl Default for Database {
  fn default() -> Self {
    Self { ops: Default::default(), routines: Default::default(), ty_db: TypeDatabase::new() }
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

pub fn compile_module(db: &mut Database, module: &str) {
  use rum_lang::parser::script_parser::*;
  let module_ast = rum_lang::parser::script_parser::parse_raw_module(module).expect("Failed to parse module");

  for module_mem in module_ast.members.members.iter() {
    match &module_mem {
      module_members_group_Value::AnnotatedModMember(mem) => match &mem.member {
        module_member_Value::RawRoutine(routine) => {
          compile_routine(db, routine.as_ref());
        }
        ty => todo!("handle {ty:#?}"),
      },

      ty => todo!("handle {ty:#?}"),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum OPConstraint {
  OpToTy(OpId, Type),
  // The type of op at src must match te type of the op at dst.
  // If both src and dst are resolved, a conversion must be made.
  OpToOp { src: OpId, dst: OpId },
  BindOpToOp { src: OpId, dst: OpId },
  MemOp { ptr_op: OpId, val_op: OpId },
  Deref { ptr_ty: Type, val_ty: Type, mutable: bool },
  Num(Type),
  Member { name: IString, ref_dst: OpId, par: OpId },
  Mutable(u32, u32),
  Agg(OpId),
  GenTyToTy(Type, Type),
  GenTyToGenTy(Type, Type),
}
struct OpAddress(*mut Operation, usize);

#[derive(Debug)]
struct Var {
  id: VarId,
  op: OpId,
  ty: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MemberEntry {
  pub name:      IString,
  pub origin_op: u32,
  pub ty:        Type,
}

#[derive(Clone)]
pub struct TypeVar {
  pub id:          u32,
  pub ref_id:      i32,
  pub ty:          Type,
  pub ref_count:   u32,
  pub constraints: ArrayVec<2, VarConstraint>,
  pub members:     ArrayVec<2, MemberEntry>,
}

impl Default for TypeVar {
  fn default() -> Self {
    Self {
      id:          Default::default(),
      ref_id:      -1,
      ref_count:   0,
      ty:          Default::default(),
      constraints: Default::default(),
      members:     Default::default(),
    }
  }
}

impl TypeVar {
  pub fn new(id: u32) -> Self {
    Self { id: id, ..Default::default() }
  }

  #[track_caller]
  pub fn has(&self, constraint: VarConstraint) -> bool {
    self.constraints.find_ordered(&constraint).is_some()
  }

  #[track_caller]
  pub fn add(&mut self, constraint: VarConstraint) {
    let _ = self.constraints.push_unique(constraint);
  }

  pub fn add_mem(&mut self, name: IString, ty: Type, origin_node: u32) {
    self.constraints.push_unique(VarConstraint::Agg).unwrap();

    for (index, MemberEntry { name: n, origin_op: origin_node, ty }) in self.members.iter().enumerate() {
      if *n == name {
        self.members.remove(index);
        break;
      }
    }

    let _ = self.members.insert_ordered(MemberEntry { name, origin_op: origin_node, ty });
  }

  pub fn get_mem(&self, name: IString) -> Option<(u32, Type)> {
    for MemberEntry { name: n, origin_op: origin_node, ty } in self.members.iter() {
      if *n == name {
        return Some((*origin_node, *ty));
      }
    }
    None
  }
}

impl Debug for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let Self { id, ty, constraints, members, ref_id, ref_count } = self;

    if ty.is_generic() {
      f.write_fmt(format_args!("[{id}] refs:{ref_count:03} {}{ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    } else {
      f.write_fmt(format_args!("[{id}] refs:{ref_count:03} {}v{id}: {ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    }
    if !constraints.is_empty() {
      f.write_str(" <")?;
      for constraint in constraints.iter() {
        f.write_fmt(format_args!("{constraint:?},"))?;
      }
      f.write_str(">")?;
    }

    if !members.is_empty() {
      f.write_str(" [\n")?;
      for MemberEntry { name, origin_op: origin_node, ty } in members.iter() {
        f.write_fmt(format_args!("  {name}: {ty} @ `{origin_node},\n"))?;
      }
      f.write_str("]")?;
    }

    Ok(())
  }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum VarConstraint {
  Agg,
  Indexable,
  Method,
  Member,
  Index(u32),
  Numeric,
  Float,
  Unsigned,
  Ptr,
  Load(u32, u32),
  MemOp {
    ptr_ty: Type,
    val_ty: Type,
  },
  Convert {
    dst: OpId,
    src: OpId,
  },
  Callable,
  Mutable,
  Default(Type),
  /// Node index, node port index, is_output
  Binding(u32, u32, bool),
}

impl Debug for VarConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarConstraint::*;
    match self {
      Indexable => f.write_fmt(format_args!("[*]",)),
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      MemOp { ptr_ty: ptr, val_ty: val } => f.write_fmt(format_args!("memop  *{ptr} = {val}",)),
      Load(a, b) => f.write_fmt(format_args!("load (@ `{a}, src: `{b})",)),
      Convert { dst, src } => f.write_fmt(format_args!("{src} => {dst}",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Agg => f.write_fmt(format_args!("agg",)),
      Mutable => f.write_fmt(format_args!("mut",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      Numeric => f.write_fmt(format_args!("numeric",)),
      Float => f.write_fmt(format_args!("floating-point",)),
      Unsigned => f.write_fmt(format_args!("unsigned",)),
      Ptr => f.write_fmt(format_args!("* = *ptr",)),
      &Default(ty) => f.write_fmt(format_args!("could be {ty}",)),
      Binding(node_index, binding_index, output) => {
        if *output {
          f.write_fmt(format_args!("`{node_index} => output[{binding_index}]"))
        } else {
          f.write_fmt(format_args!("`{node_index} => input[{binding_index}]"))
        }
      }
    }
  }
}

enum Operation {
  Param(VarId, u32),
  OutputPort(u32, Vec<(u32, OpId)>),
  Op { op_name: &'static str, operands: [OpId; 3] },
  Const(ConstVal),
  Name(IString),
}

impl Debug for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Operation::Name(name) => f.write_fmt(format_args!("\"{name}\"",)),
      Operation::OutputPort(root, ops) => f.write_fmt(format_args!(
        "{:12}  {}",
        format!("PORT@{root}"),
        ops.iter().map(|(a, b)| { format!("{:5}", format!("{b}@{a}")) }).collect::<Vec<String>>().join("  ")
      )),
      Operation::Param(name, index) => f.write_fmt(format_args!("{:12}  {name}[{index}]", "PARAM")),
      Operation::Op { op_name, operands } => {
        f.write_fmt(format_args!("{op_name:12}  {:}", operands.iter().map(|o| format!("{:5}", format!("{o}"))).collect::<Vec<_>>().join("  ")))
      }
      Operation::Const(const_val) => f.write_fmt(format_args!("{const_val}",)),
    }
  }
}

struct SuperNode {
  nodes:       Vec<Node>,
  operands:    Vec<Operation>,
  types:       Vec<Type>,
  type_vars:   Vec<TypeVar>,
  errors:      Vec<String>,
  constraints: Vec<OPConstraint>,
}

fn write_agg(var: &TypeVar, vars: &[TypeVar]) -> String {
  let mut string = Default::default();

  string += format!("{} => {{", var.ty).as_str();

  for (index, mem) in var.members.iter().enumerate() {
    let mem_var = &vars[mem.ty.generic_id().unwrap()];

    string += mem.name.to_str().as_str();
    string += ": ";

    if mem_var.has(VarConstraint::Agg) {
      string += write_agg(mem_var, vars).as_str();
    } else {
      string += format!("{}", mem_var.ty).as_str();
    }

    if index < var.members.len() - 1 {
      string += ", ";
    }
  }

  string += format!("}}").as_str();

  string
}

impl Debug for SuperNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    {
      f.write_str("\n###################### \n")?;
      let sig_node = &self.nodes[0];

      let vars = self
        .type_vars
        .iter()
        .filter_map(|f| {
          if f.ty.is_open() {
            if f.has(VarConstraint::Agg) {
              Some(write_agg(f, &self.type_vars))
            } else {
              Some(format!("∀{}={:?}", f.id, &f.constraints))
            }
          } else {
            None
          }
        })
        .collect::<Vec<_>>();

      if vars.len() > 0 {
        f.write_str("<")?;
        f.write_str(&vars.join(", "))?;
        f.write_str(">")?;
      }

      f.write_str("(")?;
      f.write_str(
        &sig_node
          .inputs
          .iter()
          .map(|input| {
            let ty = self.get_base_ty(self.types[input.0 .0 as usize]);
            format!("{}: {ty}", input.1)
          })
          .collect::<Vec<_>>()
          .join(", "),
      )?;
      f.write_str(")")?;
      f.write_str(" => ")?;
      for input in sig_node.outputs.iter() {
        let ty = self.get_base_ty(self.types[input.0 .0 as usize]);

        if input.1 == VarId::Return {
          f.write_fmt(format_args!(" {ty}"))?;
        } else {
          f.write_fmt(format_args!(" [{}: {ty}]", input.1))?;
        }
      }

      f.write_str("\n###################### \n")?;
    }

    for ((index, op), ty) in self.operands.iter().enumerate().zip(self.types.iter()) {
      let ty = self.get_base_ty(*ty);

      let err = &self.errors[index];

      f.write_fmt(format_args!("\n  {index:3} <= {:36} :{ty} {err}", format!("{op}")))?
    }
    if !self.type_vars.is_empty() {
      f.write_str("\nty_vars:\n")?;
      for (index, var) in self.type_vars.iter().enumerate() {
        f.write_fmt(format_args!("{index:3}: {var:?}\n"))?;
      }
    }
    if !self.constraints.is_empty() {
      f.write_str("\nconstriants:")?;
      for node in self.constraints.iter() {
        f.write_str("\n")?;
        Debug::fmt(node, f)?;
      }
    }

    f.write_str("\nnodes:")?;

    for node in self.nodes.iter() {
      Display::fmt(node, f)?;
      f.write_str("\n")?;
    }

    Ok(())
  }
}

impl SuperNode {
  fn get_base_ty(&self, ty: Type) -> Type {
    if let Some(index) = ty.generic_id() {
      let r_ty = get_root_var(index, &self.type_vars).ty;
      if r_ty.is_open() {
        ty
      } else {
        r_ty
      }
    } else {
      ty
    }
  }
}

struct Node {
  index:    usize,
  type_str: &'static str,
  inputs:   Vec<(OpId, VarId)>,
  outputs:  Vec<(OpId, VarId)>,
}

impl Debug for Node {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Node {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("[{}] {}\n", self.index, self.type_str))?;

    f.write_str("inputs:\n")?;
    for (op, id) in self.inputs.iter() {
      f.write_fmt(format_args!("  {op}[{}]\n", id.to_string()))?;
    }

    f.write_str("outputs:\n")?;

    for (op, id) in self.outputs.iter() {
      f.write_fmt(format_args!("  {op}[{}]\n", id.to_string()))?;
    }

    Ok(())
  }
}

#[derive(Debug)]
struct NodeScope {
  node_index: usize,
  vars:       Vec<Var>,
  var_lu:     HashMap<VarId, usize>,
}

#[derive(Debug)]
struct BuildPack {
  super_node: Box<SuperNode>,
  node_stack: Vec<NodeScope>,
  db:         *mut Database,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpId(u32);

impl Debug for OpId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for OpId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_valid() {
      f.write_fmt(format_args!("{:>3}", format!("`{}", self.0)))
    } else {
      f.write_fmt(format_args!("`xxx"))
    }
  }
}

impl Default for OpId {
  fn default() -> Self {
    Self(u32::MAX)
  }
}

impl OpId {
  fn is_invalid(&self) -> bool {
    self.0 == u32::MAX
  }
  fn is_valid(&self) -> bool {
    !self.is_invalid()
  }
}

pub fn push_node(bp: &mut BuildPack, node_name: &'static str) {
  let node_index = bp.super_node.nodes.len();
  bp.super_node.nodes.push(Node {
    index:    node_index,
    type_str: node_name,
    inputs:   Default::default(),
    outputs:  Default::default(),
  });
  bp.node_stack.push(NodeScope { node_index, vars: Default::default(), var_lu: Default::default() });
}

pub fn pop_node(bp: &mut BuildPack) -> NodeScope {
  bp.node_stack.pop().unwrap()
}

pub fn add_var(bp: &mut BuildPack, var_id: VarId, op: OpId, ty: Type) {
  let node_data = bp.node_stack.last_mut().unwrap();

  let var_index = node_data.vars.len();
  node_data.vars.push(Var { id: var_id, op, ty });
  node_data.var_lu.insert(var_id.clone(), var_index);
}

/// Update the op id of a variable. The new op should have the same type as the existing op.
pub fn update_var(bp: &mut BuildPack, var_id: VarId, op: OpId, ty: Type) {
  let node_data = bp.node_stack.last_mut().unwrap();

  if let Some(var) = node_data.var_lu.get(&var_id) {
    let var = &mut node_data.vars[*var];
    //debug_assert_eq!(var.ty, ty);
    var.op = op;
  } else {
    add_var(bp, var_id, op, ty);
  }
}

pub fn remove_var(bp: &mut BuildPack, var_id: VarId) {
  let node_data = bp.node_stack.last_mut().unwrap();
  node_data.var_lu.remove(&var_id);
}

pub fn get_var(bp: &mut BuildPack, var_id: VarId) -> Option<(OpId, Type)> {
  for node_stack_index in (0..bp.node_stack.len()).rev() {
    if let Some(var) = bp.node_stack[node_stack_index].var_lu.get(&var_id) {
      let var: &Var = &bp.node_stack[node_stack_index].vars[*var];
      if node_stack_index == bp.node_stack.len() - 1 {
        return Some((var.op, var.ty));
      } else {
        return Some((var.op, var.ty));
        todo!("Import variable {var_id}");
      }
    }
  }

  return None;
}

pub fn get_context(bp: &mut BuildPack, var_id: VarId) -> (OpId, Type) {
  debug_assert_eq!(var_id, VarId::HeapContext);

  if let Some(var) = get_var(bp, var_id) {
    var
  } else {
    // Contexts need to be added as a params in the root
    let ty = add_ty_var(bp).ty;

    let root = &mut bp.super_node.nodes[0];
    let index = root.inputs.len();

    let op = add_op(bp, Operation::Param(var_id, index as u32), ty, Default::default());

    let root = &mut bp.super_node.nodes[0];

    root.inputs.push((op, var_id));

    add_var(bp, var_id, op, ty);

    return get_context(bp, var_id);
  }
}

pub fn update_context(bp: &mut BuildPack, var_id: VarId, op: OpId) {
  println!("_____________________________________________________ {}", op);
  debug_assert_eq!(var_id, VarId::HeapContext);
  update_var(bp, var_id, op, Default::default())
}

fn add_op(bp: &mut BuildPack, operation: Operation, ty: Type, error: String) -> OpId {
  let op_id = OpId(bp.super_node.operands.len() as u32);
  bp.super_node.operands.push(operation);
  bp.super_node.types.push(ty);
  bp.super_node.errors.push(error);
  op_id
}

pub fn add_ty_var(bp: &mut BuildPack) -> &mut TypeVar {
  let ty_index = bp.super_node.type_vars.len();
  let ty = Type::generic(ty_index);
  let mut ty_var = TypeVar::new(ty_index as u32);
  ty_var.ty = ty;
  bp.super_node.type_vars.push(ty_var);
  let last_index = bp.super_node.type_vars.len() - 1;
  &mut bp.super_node.type_vars[last_index]
}

pub fn add_input(bp: &mut BuildPack, op_id: OpId, var_id: VarId) {
  let top = bp.node_stack.last().unwrap().node_index;
  bp.super_node.nodes[top].inputs.push((op_id, var_id));
}

pub fn add_output(bp: &mut BuildPack, op_id: OpId, var_id: VarId) {
  let top = bp.node_stack.last().unwrap().node_index;
  bp.super_node.nodes[top].outputs.push((op_id, var_id));
}

fn add_constraint(bp: &mut BuildPack, constraint: OPConstraint) {
  bp.super_node.constraints.push(constraint)
}

fn compile_routine(db: &mut Database, routine: &RawRoutine<Token>) -> Box<SuperNode> {
  let mut bp = BuildPack {
    db,
    super_node: Box::new(SuperNode {
      nodes:       Vec::with_capacity(8),
      operands:    Vec::with_capacity(8),
      types:       Vec::with_capacity(8),
      errors:      Vec::with_capacity(8),
      constraints: Vec::with_capacity(8),
      type_vars:   Vec::with_capacity(8),
    }),
    node_stack: Default::default(),
  };

  push_node(&mut bp, "fn");

  let mut out_ty = Default::default();

  match &routine.def.ty {
    routine_type_Value::RawFunctionType(fn_ty) => {
      for (index, param) in fn_ty.params.params.iter().enumerate() {
        let name = &param.var.id;

        let ty = add_ty_var(&mut bp).ty;
        let var_id = VarId::Name(name.intern());
        let op_id = add_op(&mut bp, Operation::Param(var_id, index as u32), ty, Default::default());
        add_var(&mut bp, var_id, op_id, ty);
        add_input(&mut bp, op_id, var_id);

        if let Some(defined_ty) = get_type(&param.ty.ty, false, &mut unsafe { &mut *bp.db }.ty_db) {
          if !defined_ty.is_open() {
            add_constraint(&mut bp, OPConstraint::GenTyToTy(ty, defined_ty));
          }
        }
      }

      out_ty = get_type(&fn_ty.return_type.ty, false, &mut db.ty_db).unwrap_or_default();
    }
    routine_type_Value::RawProcedureType(proc_ty) => {
      todo!()
    }
    routine_type_Value::None => {
      unreachable!()
    }
  }

  let (out_op, out_gen_ty) = compile_expression(&routine.def.expression.expr, &mut bp);

  if out_op.is_valid() {
    bp.super_node.nodes[0].outputs.push((out_op, VarId::Return));

    if !out_ty.is_open() {
      add_constraint(&mut bp, OPConstraint::GenTyToTy(out_gen_ty, out_ty));
    }
  }

  if let Some((op, ty)) = get_var(&mut bp, VarId::HeapContext) {
    if bp.super_node.nodes[0].inputs.iter().any(|c| c.1 == VarId::HeapContext && c.0 != op) {
      bp.super_node.nodes[0].outputs.push((op, VarId::HeapContext));
      add_constraint(&mut bp, OPConstraint::GenTyToTy(ty, Type::MemContext));
    } else {
      add_constraint(&mut bp, OPConstraint::GenTyToTy(ty, Type::NoUse));
    }
  }

  let mut routine = bp.super_node;

  solve(&mut routine, &mut db.ty_db);

  routine
}

pub fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<OPConstraint>, ty_db: &TypeDatabase) {
  if !var.ty.is_open() {
    for (index, constraint) in var.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarConstraint::Convert { src, dst } => {
          queue.push_back(OPConstraint::BindOpToOp { dst, src });
          var.constraints.remove(index);
        }
        VarConstraint::MemOp { ptr_ty, val_ty } => {
          queue.push_back(OPConstraint::Deref { ptr_ty, val_ty, mutable: false });
          var.constraints.remove(index);
        }
        _ => {}
      }
    }

    var.constraints.sort();

    /*     if var.has(VarConstraint::Agg) {
      let mut ty = var.ty;
      let members = var.members.as_slice();

      while let Some(new_ty) = ty_db.from_ptr(ty) {
        ty = new_ty;
      }

      if let Type::Complex { ty_index, .. } = ty {
        let agg_ty = &ty_db.types[ty_index as usize];

        if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty: node_ty, types, .. }) = agg_ty.get_node() {
          let mut have_name = false;

          for MemberEntry { name: member_name, origin_op, ty } in members.iter() {
            if member_name.is_empty() {
              if *node_ty == RVSDGNodeType::Array {
                for output in outputs.iter() {
                  if output.id == VarId::BaseType {
                    let ty = types[output.in_op.usize()];
                    if !ty.is_open() && *origin_op > 0 {
                      queue.push_back(OPConstraint::OpToTy(IRGraphId(*origin_op), ty_db.to_ptr(ty).unwrap()));
                    }
                  }
                }
              } else {
                todo!("Handle index lookup of node type");
              }
            } else {
              let var_id = VarId::Name(*member_name);

              if let Some(output) = outputs.iter().find(|o| o.id == var_id) {
                let ty = types[output.in_op.usize()];
                if !ty.is_open() && *origin_op > 0 {
                  queue.push_back(OPConstraint::OpToTy(IRGraphId(*origin_op), ty_db.to_ptr(ty).unwrap()));
                }
              } else {
                //let node = &src_node[mem_op as usize];
                //errors.push(blame(node, &format!("Member [{ref_name}] not found in type {:}", agg_ty.get_node().unwrap().id)));
              }
            }
          }
        }
      }
    } */
  }
}

fn solve(node: &mut SuperNode, ty_db: &mut TypeDatabase) {
  dbg!(&node);
  let SuperNode { nodes, operands, types, type_vars, constraints, .. } = node;
  let mut constraint_queue = VecDeque::from_iter(constraints.drain(..));

  while let Some(constraint) = constraint_queue.pop_front() {
    match constraint {
      OPConstraint::Deref { ptr_ty, val_ty, mutable } => {
        let ptr_index = ptr_ty.generic_id().expect("ptr_ty should be generic");
        let val_index = val_ty.generic_id().expect("val_ty should be generic");

        let var_ptr = get_root_var_mut(ptr_index, type_vars);
        let var_val = get_root_var_mut(val_index, type_vars);

        if mutable {
          var_ptr.add(VarConstraint::Mutable);
        }

        if !var_ptr.ty.is_open() {
          constraint_queue.push_back(OPConstraint::GenTyToTy(val_ty, ty_db.from_ptr(var_ptr.ty).unwrap()));
        } else if !var_val.ty.is_open() {
          constraint_queue.push_back(OPConstraint::GenTyToTy(ptr_ty, ty_db.to_ptr(var_val.ty).unwrap()));
        } else {
          var_ptr.add(VarConstraint::Ptr);
          var_ptr.add(VarConstraint::MemOp { ptr_ty, val_ty });
          var_val.add(VarConstraint::MemOp { ptr_ty, val_ty });
        }
      }
      OPConstraint::GenTyToGenTy(a, b) => {
        let a_index = a.generic_id().expect("ty should be generic");
        let b_index = b.generic_id().expect("ty should be generic");
        let var_a = get_root_var_mut(a_index, type_vars);
        let var_b = get_root_var_mut(b_index, type_vars);

        //dbg!(((a_index, &var_a), (b_index, &var_b)));

        if var_a.ty.is_poison() || var_b.ty.is_poison() {
          var_a.ty = ty_poison;
          var_b.ty = ty_poison;
        }

        if var_a.id == var_b.id {
          continue;
        } else if var_a.id < var_b.id {
          var_b.id = var_a.id;

          var_a.constraints.extend_unique(var_b.constraints.iter().cloned());

          if !var_b.ty.is_open() && var_a.ty.is_open() {
            var_a.ty = var_b.ty;
            process_variable(var_b, &mut constraint_queue, ty_db);
          }
        } else {
          var_a.id = var_b.id;

          var_b.constraints.extend_unique(var_a.constraints.iter().cloned());

          if !var_a.ty.is_open() && var_b.ty.is_open() {
            var_b.ty = var_a.ty;
            process_variable(var_a, &mut constraint_queue, ty_db);
          }
        }
      }
      OPConstraint::Num(ty) => {
        let index = ty.generic_id().expect("Left ty should be generic");
        let var = get_root_var_mut(index, type_vars);
        var.add(VarConstraint::Numeric);
      }
      OPConstraint::GenTyToTy(ty_a, ty_b) => {
        debug_assert!(ty_a.is_generic() && !ty_b.is_open());

        let index = ty_a.generic_id().expect("Left ty should be generic");

        let var = get_root_var_mut(index, type_vars);

        if var.ty.is_open() {
          var.ty = ty_b;
          process_variable(var, &mut constraint_queue, ty_db);
        } else if var.ty != ty_b {
          panic!("{}, {}  {var}", ty_a, ty_b);
        }
      }
      cs => todo!("Handle {cs:?}"),
    }
  }

  let mut out_map = vec![Default::default(); node.type_vars.len()];
  let mut output_types = vec![];

  for (index) in 0..node.type_vars.len() {
    let var = &mut node.type_vars[index];
    if var.id as usize == index {
      let mut clone = var.clone();
      clone.id = output_types.len() as u32;
      var.ref_id = output_types.len() as i32;
      output_types.push(clone);
    }
    out_map[index] = Type::generic(get_root_var(index, &node.type_vars).ref_id as usize);
  }

  for var_ty in node.types.iter_mut() {
    match var_ty {
      Type::Generic { .. } => {
        let index = var_ty.generic_id().expect("Type is not generic");
        *var_ty = out_map[index];
      }
      _ => {}
    }
  }

  for var in output_types.iter_mut() {
    for mem in var.members.iter_mut() {
      mem.ty = out_map[mem.ty.generic_id().expect("index") as usize];
    }

    for cstr in var.constraints.iter_mut() {
      match cstr {
        VarConstraint::MemOp { ptr_ty, val_ty } => {
          *val_ty = out_map[val_ty.generic_id().expect("index") as usize];
          *ptr_ty = out_map[ptr_ty.generic_id().expect("index") as usize];
        }
        _ => {}
      }
    }

    var.constraints.sort();
  }

  node.type_vars = output_types;

  dbg!(node);
}

fn get_root_var<'a>(mut index: usize, type_vars: &'a [TypeVar]) -> &'a TypeVar {
  unsafe {
    let mut var = type_vars.as_ptr().offset(index as isize);

    while (&*var).id != index as u32 {
      index = (&*var).id as usize;
      var = type_vars.as_ptr().offset(index as isize);
    }

    &*var
  }
}

fn get_root_var_mut<'a>(mut index: usize, type_vars: &mut [TypeVar]) -> &'a mut TypeVar {
  unsafe {
    let mut var = type_vars.as_mut_ptr().offset(index as isize);

    while (&*var).id != index as u32 {
      index = (&*var).id as usize;
      var = type_vars.as_mut_ptr().offset(index as isize);
    }

    &mut *var
  }
}

enum VarLookup {
  Var(OpId, Type, IString),
  Ptr(OpId, Type),
}

/// Returns either the underlying value assigned to a variable name, or the caclulated pointer to the value.
fn get_mem_op(bp: &mut BuildPack, mem: &Arc<parser::script_parser::MemberCompositeAccess<Token>>) -> VarLookup {
  let ty_vars = bp.super_node.type_vars.as_mut_ptr();
  let var_name = mem.root.name.id.intern();
  if let Some((mut op, mut ty)) = get_var(bp, VarId::Name(var_name)) {
    if mem.sub_members.is_empty() {
      VarLookup::Var(op, ty, var_name)
    } else {
      // Ensure context is added to this node.

      let mut ty_var_index = ty.generic_id().expect("All vars should have generic ids");

      let mut ptr_op = op;
      let mut ptr_ty = ty;

      for (index, mem_val) in mem.sub_members.iter().enumerate() {
        let var = unsafe { &mut *ty_vars.offset(ty_var_index as isize) };

        match mem_val {
          member_group_Value::IndexedMember(index) => {
            todo!("handle indexed lookup")
          }
          member_group_Value::NamedMember(name) => {
            let name = name.name.id.intern();
            let name_op = add_op(bp, Operation::Name(name), Type::NoUse, Default::default());
            let (ref_op, ref_ty) = process_op("NAMED_PTR", &[ptr_op, name_op], bp);
            ptr_ty = ref_ty;
            ptr_op = ref_op;

            if let Some(mem) = var.members.iter().find(|m| m.name == name) {
              add_constraint(bp, OPConstraint::Deref { ptr_ty: ref_ty, val_ty: mem.ty, mutable: false });
            } else {
              let mem_ty = add_ty_var(bp);
              mem_ty.add(VarConstraint::Member);
              let mem_ty = mem_ty.ty;

              var.add_mem(name, mem_ty, Default::default());

              add_constraint(bp, OPConstraint::Deref { ptr_ty: ref_ty, val_ty: mem_ty, mutable: false });
            }
          }
          _ => unreachable!(),
        }

        ty_var_index = ty.generic_id().unwrap();

        if index != mem.sub_members.len() - 1 {
          // load the value of the pointer
          let (loaded_val_op, loaded_val_ty) = process_op("LOAD", &[ptr_op], bp);
          ptr_op = loaded_val_op;
          ptr_ty = loaded_val_ty;
        }
      }

      VarLookup::Ptr(ptr_op, ptr_ty)
    }
  } else {
    let ty = add_ty_var(bp).ty;
    add_var(bp, VarId::Name(mem.root.name.id.intern()), Default::default(), ty);
    return get_mem_op(bp, mem);
  }
}

fn compile_scope(block: &RawBlock<Token>, bp: &mut BuildPack) -> (OpId, Type) {
  let mut output = Default::default();

  for stmt in block.statements.iter() {
    match stmt {
      statement_Value::Expression(expr) => {
        output = compile_expression(&expr.expr, bp);
      }
      statement_Value::RawAssignment(assign) => {
        let (expr_op, expr_ty) = compile_expression(&assign.expression.expr, bp);

        match &assign.var {
          assignment_var_Value::MemberCompositeAccess(mem) => match get_mem_op(bp, mem) {
            VarLookup::Ptr(ptr_op, ..) => {
              process_op("STORE", &[ptr_op, expr_op], bp);
            }
            VarLookup::Var(.., var_name) => {
              update_var(bp, VarId::Name(var_name), expr_op, expr_ty);
            }
          },
          assign => todo!("{assign:#?}"),
        }

        output = Default::default()
      }
      ty => {
        output = Default::default();
        todo!("{ty:?}")
      }
    }
  }

  output
}

fn compile_expression(expr: &expression_Value<Token>, bp: &mut BuildPack) -> (OpId, Type) {
  use rum_lang::parser::script_parser::*;
  match expr {
    expression_Value::RawBlock(block_scope) => compile_scope(&block_scope, bp),
    expression_Value::MemberCompositeAccess(mem) => match get_mem_op(bp, mem) {
      VarLookup::Ptr(ptr_op, ..) => process_op("LOAD", &[ptr_op], bp),
      VarLookup::Var(op, ty, ..) => (op, ty),
    },
    expression_Value::RawNum(num) => {
      let ty_db = unsafe { &mut (&mut *bp.db).ty_db };
      let string_val = num.tok.to_string();

      let const_val = if string_val.contains(".") {
        ConstVal::new(ty_db.get_ty("f64").expect("f64 should exist").to_primitive().unwrap(), num.val)
      } else {
        ConstVal::new(ty_db.get_ty("i64").expect("i64 should exist").to_primitive().unwrap(), string_val.parse::<i64>().unwrap())
      };
      let ty = add_ty_var(bp).ty;
      let op = add_op(bp, Operation::Const(const_val), ty, Default::default());

      (op, ty)
    }
    expression_Value::Add(add) => {
      let left = compile_expression(&add.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&add.right.clone().to_ast().into_expression_Value().expect("super_node.operands  be convertible"), bp).0;
      process_op("ADD", &[left, right], bp)
    }
    expression_Value::Sub(sub) => {
      let left = compile_expression(&sub.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&sub.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("SUB", &[left, right], bp)
    }
    expression_Value::Div(div) => {
      let left = compile_expression(&div.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&div.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("DIV", &[left, right], bp)
    }
    expression_Value::Mul(mul) => {
      let left = compile_expression(&mul.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&mul.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("MUL", &[left, right], bp)
    }
    expression_Value::Pow(pow) => {
      let left = compile_expression(&pow.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&pow.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("POW", &[left, right], bp)
    }
    expression_Value::RawMatch(match_) => {
      // Build input test
      let input_op = compile_expression(&expression_Value::MemberCompositeAccess(match_.expression.clone()), bp);

      push_node(bp, "---MATCH---");
      let output_ty = add_ty_var(bp).ty;
      bp.super_node.constraints.push(OPConstraint::GenTyToTy(output_ty, ty_u32));

      let mut clauses = Vec::new();

      let clause_ast = match_.clauses.iter().chain(match_.default_clause.iter()).enumerate();

      for (index, clause) in clause_ast.clone() {
        push_node(bp, "clause_selector");

        let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.to_primitive().unwrap(), index as u32)), output_ty, Default::default());

        if let Some(expr) = &clause.expr {
          let (expr_op, _) = compile_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), bp);

          let cmp_op_name = match expr.op.as_str() {
            ">" => "GR",
            "<" => "LS",
            ">=" => "GE",
            "<=" => "LE",
            "==" => "EQ",
            "!=" => "NE",
            _ => todo!(),
          };

          add_input(bp, input_op.0, VarId::MatchInputExpr);

          let (bool_op, _) = process_op(cmp_op_name, &[input_op.0, expr_op], bp);

          let (out_op, output_ty_new) = process_op("SEL", &[bool_op, sel_op], bp);

          if output_ty_new != output_ty {
            add_constraint(bp, OPConstraint::GenTyToGenTy(output_ty_new, output_ty));
          }

          add_var(bp, VarId::MatchActivation, out_op, output_ty_new);
        } else {
          add_var(bp, VarId::MatchActivation, sel_op, output_ty);
        }

        clauses.push(pop_node(bp));
      }

      if match_.default_clause.is_none() {
        push_node(bp, "clause_selector");
        let sel_op: OpId =
          add_op(bp, Operation::Const(ConstVal::new(ty_u32.to_primitive().unwrap(), match_.clauses.len() as u32)), output_ty, Default::default());
        add_var(bp, VarId::MatchActivation, sel_op, output_ty);
        clauses.push(pop_node(bp));
      }

      join_nodes(clauses, bp);

      let selector_op = get_var(bp, VarId::MatchActivation).unwrap().0;

      remove_var(bp, VarId::MatchActivation);
      add_output(bp, selector_op, VarId::MatchActivation);

      let mut clauses = Vec::new();

      let match_ty = add_ty_var(bp).ty;

      for (_, clause) in clause_ast {
        push_node(bp, "clause");

        let (op, output_ty) = compile_scope(&clause.scope, bp);

        add_var(bp, VarId::MatchOutputVal, op, output_ty);
        add_constraint(bp, OPConstraint::GenTyToGenTy(match_ty, output_ty));

        clauses.push(pop_node(bp));
      }

      if match_.default_clause.is_none() {
        push_node(bp, "clause");
        let (poison_op, output_ty) = process_op("POISON", &[], bp);
        add_var(bp, VarId::MatchOutputVal, poison_op, output_ty);
        add_constraint(bp, OPConstraint::GenTyToGenTy(match_ty, output_ty));
        clauses.push(pop_node(bp));
      }

      remove_var(bp, VarId::MatchActivation);
      join_nodes(clauses, bp);

      remove_var(bp, VarId::MatchInputExpr);

      join_nodes(vec![pop_node(bp)], bp);

      let out = get_var(bp, VarId::MatchOutputVal).unwrap();

      remove_var(bp, VarId::MatchOutputVal);

      out
    }
    ty => todo!("{ty:#?}"),
  }
}

fn join_nodes(clauses: Vec<NodeScope>, bp: &mut BuildPack) {
  let current_node = bp.node_stack.last().unwrap();
  let current_node_index = current_node.node_index;
  for clause in clauses {
    for var in clause.var_lu {
      let var = &clause.vars[var.1];
      bp.super_node.nodes[clause.node_index].outputs.push((var.op, var.id));

      if let Some((op, ty)) = get_var(bp, var.id.clone()) {
        match &mut bp.super_node.operands[op.0 as usize] {
          Operation::OutputPort(root, vars) => {
            vars.push((clause.node_index as u32, var.op));
          }
          _ => {
            let op = add_op(
              bp,
              Operation::OutputPort(current_node_index as u32, vec![(Default::default(), op), (clause.node_index as u32, var.op)]),
              ty,
              Default::default(),
            );
            add_var(bp, var.id, op, var.ty);
          }
        }
      } else {
        let op = add_op(bp, Operation::OutputPort(current_node_index as u32, vec![(clause.node_index as u32, var.op)]), var.ty, Default::default());
        add_var(bp, var.id, op, var.ty);
      }
    }
  }
}

fn process_op(op_name: &'static str, inputs: &[OpId], bp: &mut BuildPack) -> (OpId, Type) {
  let op_def = get_op_from_db(unsafe { &mut *bp.db }, op_name).expect(&format!("{op_name} op not loaded"));

  let mut operands = [OpId::default(); 3];

  let mut ty_lu = HashMap::new();

  let mut op_index: isize = -1;
  for (port_index, port) in op_def.inputs.iter().enumerate() {
    let type_ref_name = port.var.name.as_str();

    match type_ref_name {
      "read_ctx" => {
        let (op, ty) = get_context(bp, VarId::HeapContext);
        operands[port_index] = op;
      }
      type_ref_name => {
        op_index += 1;

        let op_index = op_index as usize;

        operands[port_index] = inputs[op_index];

        let op_id = operands[port_index];

        let ty = if op_id.is_valid() { bp.super_node.types[op_id.0 as usize] } else { add_ty_var(bp).ty };

        match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => {
            // Create a link
            if *d.get() != ty {
              let other_ty = *d.get();
              bp.super_node.constraints.push(OPConstraint::GenTyToGenTy(other_ty, ty));
            }
          }
          std::collections::hash_map::Entry::Vacant(entry) => {
            for annotation in port.var.annotations.iter() {
              match annotation {
                _ => {}
              }
            }

            entry.insert(ty);
          }
        }

        add_annotations(port.var.annotations.iter(), &ty_lu, bp, ty);
      }
    }
  }

  let op_id = add_op(bp, Operation::Op { op_name, operands }, Default::default(), Default::default());
  let mut ty = Default::default();

  let mut have_output = false;
  for output in op_def.outputs.iter() {
    match output.var.name.as_str() {
      "write_ctx" => {
        update_context(bp, VarId::HeapContext, op_id);
      }
      _ => {
        debug_assert_eq!(have_output, false);
        have_output = true;

        let type_ref_name = output.var.name.as_str();

        ty = match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => *d.get(),
          std::collections::hash_map::Entry::Vacant(..) => {
            let ty = add_ty_var(bp).ty;
            add_annotations(output.var.annotations.iter(), &ty_lu, bp, ty);
            ty
          }
        };
      }
    }
  }

  bp.super_node.types[op_id.0 as usize] = ty;

  (op_id, ty)
  // Add constraints
}

fn add_annotations(annotations: std::slice::Iter<'_, annotation_Value>, ty_lu: &HashMap<&str, Type>, bp: &mut BuildPack, ty: Type) {
  for annotation in annotations {
    match annotation {
      annotation_Value::Deref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, OPConstraint::Deref { ptr_ty: *target, val_ty: ty, mutable: false });
        }
      }

      annotation_Value::MutDeref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, OPConstraint::Deref { ptr_ty: *target, val_ty: ty, mutable: true });
        }
      }

      annotation_Value::Annotation(val) => match val.name.as_str() {
        "Numeric" => add_constraint(bp, OPConstraint::Num(ty)),
        "poison" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_poison)),
        "bool" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_bool)),
        "u8" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_u8)),
        "u16" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_u16)),
        "u32" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_u32)),
        "u64" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_u64)),
        "i8" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_s8)),
        "i16" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_s16)),
        "i32" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_s32)),
        "i64" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_s64)),
        "f32" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_f32)),
        "f64" => add_constraint(bp, OPConstraint::GenTyToTy(ty, ty_f64)),
        _ => {}
      },
      _ => {}
    }
  }
}

// Side effect modifiers

const OPS: &'static str = r###"

op: PARAM
x [A: Input] => out [A]

op: SEL  l [A: bool]  r [B: Numeric]  => out [B]

op: POISON  => out [B: poison]

op: OFFSET_PTR  b [Base: agg]  n [Offset: Numeric]  => out [MemPtr]
op: NAMED_PTR  b [Base: agg]  n [MemName: label]  => out [MemPtr]

op: LOAD  ptr [ptr] ctx [read_ctx] => out [val: deref(ptr)]
op: STORE  ptr [ptr] val [val: mut_deref(ptr)] ctx [read_ctx] => ctx [write_ctx]

op: GE  l [A: Numeric]  r [A]  => out [B: bool]
op: LE  l [A: Numeric]  r [A]  => out [B: bool]
op: EQ  l [A: Numeric]  r [A]  => out [B: bool]
op: GR  l [A: Numeric]  r [A]  => out [B: bool]
op: LS  l [A: Numeric]  r [A]  => out [B: bool]
op: NE  l [A: Numeric]  r [A]  => out [B: bool]

op: MOD  l [A: Numeric]  r [A]  => out [A]
op: POW  l [A: Numeric]  r [A]  => out [A]

op: MUL  l [A: Numeric]  r [A]  => out [A]
op: DIV  l [A: Numeric]  r [A]  => out [A]

op: SUB  l [A: Numeric]  r [A]  => out [A]
op: ADD  l [A: Numeric]  r [A]  => out [A] impl { 
x86: 
  - A = u32: "add l r"
  - A = u8:  "add l r"
int:
  - A = u32: "+ l r"
}

"###;
#[test]
pub fn test() {
  let mut db: Database = Database::default();

  add_ops_to_db(&mut db, OPS);

  compile_module(
    &mut db,
    "

  test (a:?, b:?) => u32 { 
    a.g = 2 
    a.g + b 
  }
  
  ",
  );
}
