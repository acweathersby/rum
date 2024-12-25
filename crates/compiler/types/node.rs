use super::*;
use rum_lang::{
  istring::{CachedString, IString},
  Token,
};
use std::{
  alloc::Layout,
  default,
  fmt::{Debug, Display},
  hash::{DefaultHasher, Hash, Hasher},
  ptr::drop_in_place,
  thread::sleep,
  time::Duration,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum SolveState {
  #[default]
  Unsolved,
  Solved,
  Template,
}

struct NodeWrapper {
  owners:  std::sync::atomic::AtomicU16,
  writers: std::sync::atomic::AtomicU16,
  readers: std::sync::atomic::AtomicU16,
  node:    RootNode,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeHandle(*mut NodeWrapper);

impl Default for NodeHandle {
  fn default() -> Self {
    NodeHandle(std::ptr::null_mut())
  }
}
impl Debug for NodeHandle {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let r = unsafe { self.0.as_ref().expect("Invalid internal pointer for NodeHandle") };
    r.node.fmt(f)
  }
}

impl Clone for NodeHandle {
  fn clone(&self) -> Self {
    let other = Self(self.0);

    let r = unsafe { other.0.as_ref().expect("Invalid internal pointer for NodeHandle") };

    r.owners.fetch_add(1, std::sync::atomic::Ordering::Release);

    other
  }
}

impl Drop for NodeHandle {
  fn drop(&mut self) {
    let r = unsafe { self.0.as_ref().expect("Invalid internal pointer for NodeHandle") };

    let val = r.owners.fetch_sub(1, std::sync::atomic::Ordering::Acquire);

    if val == 1 {
      unsafe { drop_in_place(self.0) };
      let layout = Layout::new::<NodeWrapper>();
      unsafe { std::alloc::dealloc(self.0 as *mut _, layout) };
    }
  }
}

impl NodeHandle {
  pub fn new(node: RootNode) -> Self {
    unsafe {
      let layout = Layout::new::<NodeWrapper>();
      let ptr = std::alloc::alloc(layout) as *mut NodeWrapper;

      if ptr.is_null() {
        panic!("Could not allocate space for Node");
      }

      let mut wrapper = NodeWrapper {
        node,
        owners: std::sync::atomic::AtomicU16::new(1),
        readers: Default::default(),
        writers: Default::default(),
      };
      std::mem::swap(ptr.as_mut().unwrap(), &mut wrapper);
      std::mem::forget(wrapper);

      Self(ptr)
    }
  }

  /// Creates a clone of the original Node, instead of cloning the handle.
  pub fn duplicate(&self) -> Self {
    unsafe {
      let layout = Layout::new::<NodeWrapper>();
      let ptr = std::alloc::alloc(layout) as *mut NodeWrapper;

      if ptr.is_null() {
        panic!("Could not allocate space for Node");
      }

      let mut wrapper = NodeWrapper {
        node:    self.get().unwrap().clone(),
        owners:  std::sync::atomic::AtomicU16::new(1),
        readers: Default::default(),
        writers: Default::default(),
      };
      std::mem::swap(ptr.as_mut().unwrap(), &mut wrapper);
      std::mem::forget(wrapper);

      Self(ptr)
    }
  }

  pub fn get_mut<'a>(&'a self) -> Option<&'a mut RootNode> {
    let wrapper = unsafe { self.0.as_mut().expect("Invalid internal pointer for NodeHandle") };

    Some(&mut wrapper.node)
  }

  pub fn get<'a>(&'a self) -> Option<&'a RootNode> {
    let wrapper = unsafe { self.0.as_mut().expect("Invalid internal pointer for NodeHandle") };

    wrapper.readers.fetch_add(1, std::sync::atomic::Ordering::AcqRel);

    while wrapper.writers.load(std::sync::atomic::Ordering::Acquire) > 0 {
      sleep(Duration::from_nanos(300));
    }

    Some(&mut wrapper.node)
  }
}

#[derive(Default, Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub enum VarId {
  #[default]
  Undefined,
  Name(IString),
  MemName(usize, IString),
  SideEffect(usize),
  MemRef(usize),
  MatchInputExpr,
  OutputVal,
  MatchActivation,
  LoopActivation,
  Return,
  GlobalContext,
  Generic,
  MemCTX,
  DeletedHeap,
  Param(usize),
  CallRef,
  BaseType,
  ElementCount,
  AggSize,
}

impl Display for VarId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Name(id) => f.write_str(id.to_str().as_str()),
      Self::SideEffect(id) => f.write_fmt(format_args!("--{id}--")),
      Self::MemRef(id) => f.write_fmt(format_args!("--*{id}--")),
      Self::MatchInputExpr => f.write_str("MATCH_INPUT_VALUE"),
      Self::OutputVal => f.write_str("OUTPUT_VALUE"),
      Self::MatchActivation => f.write_str("MATCH_ACTIVATION"),
      Self::Return => f.write_str("RETURN"),
      Self::LoopActivation => f.write_str("LOOP_ACTIVATION"),
      Self::MemCTX => f.write_fmt(format_args!("MemCtx")),
      _ => f.write_fmt(format_args!("{self:?}")),
    }
  }
}

impl VarId {
  pub fn to_string(&self) -> IString {
    format!("{self}").intern()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OpId(pub(crate) u32);

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
  pub(crate) fn is_invalid(&self) -> bool {
    self.0 == u32::MAX
  }
  pub(crate) fn is_valid(&self) -> bool {
    !self.is_invalid()
  }
  pub fn usize(&self) -> usize {
    self.0 as usize
  }
}

#[derive(Clone)]
pub(crate) enum Operation {
  Param(VarId, u32),
  Heap(VarId),
  MemCheck(OpId),
  OutputPort(u32, Vec<(u32, OpId)>),
  Op { op_name: &'static str, operands: [OpId; 3] },
  Const(ConstVal),
  Name(IString),
  CallTarget(NodeHandle),
  Allocate,
  Free,
}

impl Debug for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Operation::CallTarget(target) => f.write_fmt(format_args!("Target [cmplx {}]", target.0 as usize)),
      Operation::Name(name) => f.write_fmt(format_args!("\"{name}\"",)),
      Operation::MemCheck(op) => f.write_fmt(format_args!("MemCheck({op})",)),
      Operation::OutputPort(root, ops) => f.write_fmt(format_args!(
        "{:12}  {}",
        format!("PORT@{root}"),
        ops.iter().map(|(a, b)| { format!("{:5}", format!("{b}@{a}")) }).collect::<Vec<String>>().join("  ")
      )),
      Operation::Param(name, index) => f.write_fmt(format_args!("{:12}  {name}[{index}]", "PARAM")),
      Operation::Heap(name) => f.write_fmt(format_args!("{:12}  {name}", "HEAP")),
      Operation::Op { op_name, operands } => {
        f.write_fmt(format_args!("{op_name:12}  {:}", operands.iter().map(|o| format!("{:5}", format!("{o}"))).collect::<Vec<_>>().join("  ")))
      }
      Operation::Const(const_val) => f.write_fmt(format_args!("{const_val}",)),
      Operation::Allocate => f.write_fmt(format_args!("allocate",)),
      Operation::Free => f.write_fmt(format_args!("free",)),
    }
  }
}

#[derive(Debug, Clone)]
pub struct CallLookup {
  pub name:        IString,
  pub args:        Vec<Type>,
  pub ret:         Type,
  pub origin_node: usize,
}

#[derive(Clone, Copy, Default)]
pub enum HeapData {
  #[default]
  Undefined,
  Named(IString),
  Local, // The stack in other languages
  Primitive,
}

impl Debug for HeapData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for HeapData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      HeapData::Local => f.write_str("~local"),
      HeapData::Undefined => f.write_str("~"),
      HeapData::Primitive => f.write_str("~reg"),
      HeapData::Named(d) => f.write_fmt(format_args!("*{d}")),
    }
  }
}

#[derive(Clone)]
pub(crate) struct RootNode {
  pub(crate) nodes:         Vec<Node>,
  pub(crate) operands:      Vec<Operation>,
  pub(crate) types:         Vec<Type>,
  pub(crate) type_vars:     Vec<TypeVar>,
  pub(crate) heap_id:       Vec<usize>,
  pub(crate) source_tokens: Vec<rum_lang::parser::script_parser::ast::ASTNode<Token>>,
}

impl Default for RootNode {
  fn default() -> Self {
    RootNode {
      nodes:         Vec::with_capacity(8),
      operands:      Vec::with_capacity(8),
      types:         Vec::with_capacity(8),
      type_vars:     Vec::with_capacity(8),
      source_tokens: Vec::with_capacity(8),
      heap_id:       Vec::with_capacity(8),
    }
  }
}

pub(crate) fn write_agg(var: &TypeVar, vars: &[TypeVar]) -> String {
  let mut string = Default::default();

  string += format!("{} => {{", var.ty).as_str();

  for (index, mem) in var.members.iter().enumerate() {
    let mem_var = &vars[mem.ty.generic_id().unwrap()];

    string += mem.name.to_str().as_str();
    string += ": ";

    if mem_var.has(VarAttribute::Agg) {
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

pub fn get_signature(node: &RootNode) -> u64 {
  Signature::new(
    &node.nodes[0].inputs.iter().map(|i| node.get_base_ty(node.types[i.0.usize()].clone())).collect::<Vec<_>>(),
    &node.nodes[0].outputs.iter().map(|i| node.get_base_ty(node.types[i.0.usize()].clone())).collect::<Vec<_>>(),
  )
  .hash()
}

impl Debug for RootNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.nodes.len() > 0 {
      f.write_str("\n###################### \n")?;

      let sig_node = &self.nodes[0];

      let vars = self
        .type_vars
        .iter()
        .filter_map(|f| {
          if f.ty.is_open() {
            if f.has(VarAttribute::Agg) {
              Some(write_agg(f, &self.type_vars))
            } else {
              Some(format!("∀{}={:?}", f.id, &f.attributes))
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
            if input.0.is_valid() {
              let ty = self.get_base_ty(self.types[input.0 .0 as usize].clone());
              format!("{}: {ty}", input.1)
            } else {
              format!("{}: --", input.1)
            }
          })
          .collect::<Vec<_>>()
          .join(", "),
      )?;
      f.write_str(")")?;
      f.write_str(" => ")?;
      for input in sig_node.outputs.iter() {
        if input.0.is_valid() {
          let ty = self.get_base_ty(self.types[input.0 .0 as usize].clone());

          if input.1 == VarId::Return {
            f.write_fmt(format_args!(" {ty}"))?;
          } else {
            f.write_fmt(format_args!(" [{}: {ty}]", input.1))?;
          }
        } else {
          f.write_fmt(format_args!("[{}: --]", input.1))?;
        }
      }

      f.write_str("\n###################### \n")?;
    }

    for ((index, op), ty) in self.operands.iter().enumerate().zip(self.types.iter()) {
      let ty = self.get_base_ty(ty.clone());
      let mut tok = self.source_tokens[index].token();
      let source = tok.to_string();
      let heap = self.heap_id.get(index).cloned().unwrap_or_default();

      //let heap = if heap < self.type_vars.len() { self.type_vars[heap].ty.to_string() } else { "???".to_string() };

      let heap = if heap == usize::MAX { "".to_string() } else { heap.to_string() };

      f.write_fmt(format_args!("\n  {index:3} <= {:36} @{heap:10} :{:32} {}: {:}", format!("{op}"), format!("{ty}"), tok.get_line(), source))?
    }
    f.write_str("\nnodes:")?;

    for node in self.nodes.iter() {
      Display::fmt(node, f)?;
      f.write_str("\n")?;
    }
    if !self.type_vars.is_empty() {
      f.write_str("\nty_vars:\n")?;
      for (index, var) in self.type_vars.iter().enumerate() {
        f.write_fmt(format_args!("{index:3}: {var:?}\n"))?;
      }
    }

    Ok(())
  }
}

impl RootNode {
  pub(crate) fn get_base_ty(&self, ty: Type) -> Type {
    if let Some(index) = ty.generic_id() {
      let r_ty = get_root_var(index, &self.type_vars).ty.clone();
      if r_ty.is_open() {
        ty
      } else {
        r_ty
      }
    } else {
      ty
    }
  }

  pub fn solve_state(&self) -> SolveState {
    if self.type_vars.iter().any(|v| v.ty.is_open()) {
      SolveState::Template
    } else {
      SolveState::Solved
    }
  }
}
#[derive(Clone)]
pub(crate) struct Node {
  pub(crate) index:    usize,
  pub(crate) type_str: &'static str,
  pub(crate) inputs:   Vec<(OpId, VarId)>,
  pub(crate) outputs:  Vec<(OpId, VarId)>,
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

pub struct Signature {
  inputs:  Vec<Type>,
  outputs: Vec<Type>,
}

impl Signature {
  pub fn new(inputs: &[Type], outputs: &[Type]) -> Self {
    Self { inputs: inputs.to_vec(), outputs: outputs.to_vec() }
  }

  pub fn hash(&self) -> u64 {
    let mut h = DefaultHasher::new();
    for ty in &self.inputs {
      if ty.is_generic() {
        0u64.hash(&mut h);
      } else {
        ty.hash(&mut h);
      }
    }

    for ty in &self.outputs {
      if ty.is_generic() {
        0u64.hash(&mut h);
      } else {
        ty.hash(&mut h);
      }
    }

    h.finish()
  }
}
