use super::*;
use crate::_interpreter::get_op_type;
use radlr_rust_runtime::types::Token;
use rum_common::{CachedString, IString};
use std::{
  alloc::Layout,
  fmt::{Debug, Display},
  hash::{DefaultHasher, Hash, Hasher},
  ptr::drop_in_place,
  thread::sleep,
  time::Duration,
  usize,
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
pub(crate) struct NodeHandle(*mut NodeWrapper, pub usize);

impl Default for NodeHandle {
  fn default() -> Self {
    NodeHandle(std::ptr::null_mut(), usize::MAX)
  }
}
impl Debug for NodeHandle {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.1 != usize::MAX {
      f.write_fmt(format_args!("\nCMPLXid({}) => \n", self.1))?;
    } else {
      f.write_fmt(format_args!("\n{:16X} => \n", self.0 as *const _ as usize,))?;
    }
    let r = unsafe { self.0.as_ref().expect("Invalid internal pointer for NodeHandle") };
    r.node.fmt(f)
  }
}

impl Clone for NodeHandle {
  fn clone(&self) -> Self {
    let other = Self(self.0, self.1);

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

      let mut wrapper = NodeWrapper { node, owners: std::sync::atomic::AtomicU16::new(1), readers: Default::default(), writers: Default::default() };
      std::mem::swap(ptr.as_mut().unwrap(), &mut wrapper);
      std::mem::forget(wrapper);

      Self(ptr, usize::MAX)
    }
  }

  pub fn get_rum_ty(&self) -> RumTypeRef {
    self.get().unwrap().ty
  }

  pub fn get_type(&self) -> &'static str {
    self.get().unwrap().nodes[0].type_str
  }

  /// Creates a clone of the original Node, instead of cloning the handle.
  pub fn duplicate(&self) -> Self {
    unsafe {
      let layout = Layout::new::<NodeWrapper>();
      let ptr = std::alloc::alloc(layout) as *mut NodeWrapper;

      if ptr.is_null() {
        panic!("Could not allocate space for Node");
      }

      let mut wrapper = NodeWrapper { node: self.get().unwrap().clone(), owners: std::sync::atomic::AtomicU16::new(1), readers: Default::default(), writers: Default::default() };
      std::mem::swap(ptr.as_mut().unwrap(), &mut wrapper);
      std::mem::forget(wrapper);

      Self(ptr, self.1)
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

#[derive(Default, Clone, Copy, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarId {
  #[default]
  Undefined,
  Name(IString),
  ASM(IString),
  MemName(usize, IString),
  ArrayMem(usize),
  SideEffect(usize),
  MemRef(usize),
  MatchInputExpr,
  OutputVal,
  /// Match activation are bound to comparison operators and represent a branch pending the result of the comparison
  MatchBooleanSelector,
  LoopActivation,
  Return,
  MemReturn,
  VoidReturn,
  GlobalContext,
  Generic,
  MemCTX,
  Heap,
  Freed,
  Param(usize, IString),
  Arg(usize, IString),
  CallId(IString),
  BaseType,
  ElementCount,
  AggSize,
  CallTarget(CMPLXId),
  IntrinsicCallTarget(IString),
}

impl Display for VarId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Name(id) => f.write_str(id.to_str().as_str()),
      Self::SideEffect(id) => f.write_fmt(format_args!("--{id}--")),
      Self::MemRef(id) => f.write_fmt(format_args!("--*{id}--")),
      Self::MatchInputExpr => f.write_str("MATCH_INPUT_VALUE"),
      Self::OutputVal => f.write_str("OUTPUT_VALUE"),
      Self::MatchBooleanSelector => f.write_str("MATCH_ACTIVATION"),
      Self::Return => f.write_str("RETURN"),
      Self::LoopActivation => f.write_str("LOOP_ACTIVATION"),
      Self::MemCTX => f.write_fmt(format_args!("MemCtx")),
      Self::Freed => f.write_fmt(format_args!("Freed")),
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
pub struct OpId(pub(crate) u32);

impl Debug for OpId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for OpId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_valid() {
      f.write_fmt(format_args!("{:>6}", format!("`{}", self.0)))
    } else {
      if self.0 == u32::MAX {
        f.write_fmt(format_args!("    xxx"))
      } else {
        f.write_fmt(format_args!("[*{:>3}]", self.meta()))
      }
    }
  }
}
impl Default for OpId {
  fn default() -> Self {
    Self(u32::MAX)
  }
}

impl OpId {
  pub(crate) fn meta(&self) -> usize {
    (self.0 & !0x1000_0000) as usize
  }
  pub(crate) fn is_invalid(&self) -> bool {
    self.0 & 0x1000_0000 > 0
  }
  pub(crate) fn is_valid(&self) -> bool {
    !self.is_invalid()
  }
  pub fn usize(&self) -> usize {
    self.0 as usize
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PortType {
  Phi,
  In,
  Out,
  #[default]
  Merge,
  Passthrough,
  CallTarget,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct NodePort {
  pub ty:   PortType,
  pub slot: OpId,
  pub id:   VarId,
}

#[derive(Clone, Copy)]
pub(crate) enum Reference {
  UnresolvedName(IString),
  Object(CMPLXId),
  Type(RumTypeRef),
  Intrinsic(IString),
  Integer(usize),
  Pointer(usize),
  SmallStruct(usize),
  Unknown,
}

impl Debug for Reference {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::UnresolvedName(name) => f.write_fmt(format_args!("{name}?")),
      Self::Intrinsic(name) => f.write_fmt(format_args!("`{name}")),
      Self::Object(id) => f.write_fmt(format_args!("{id:?}")),
      Self::Integer(id) => f.write_fmt(format_args!("{id}")),
      Self::Pointer(id) => f.write_fmt(format_args!("[{id}]")),
      Self::SmallStruct(id) => f.write_fmt(format_args!("Obj{id}")),
      Self::Type(id) => f.write_fmt(format_args!("{id}")),
      Self::Unknown => f.write_fmt(format_args!("??")),
    }
  }
}

#[derive(Clone)]
pub(crate) enum Operation {
  /// - VarId - Variable representing the param value.
  /// - u32   - List index position of the parameter
  Param(VarId, u32),
  Φ(u32, Vec<OpId>),
  _Gamma(u32, OpId),
  Call {
    routine: OpId,
    args:    Vec<OpId>,
    seq_op:  OpId,
  },
  AggDecl {
    /// Used to define number or repeating elements in this structure
    reps:      OpId,
    ty_op: OpId,
    seq_op:    OpId,
  },
  NamedOffsetPtr {
    reference: Reference,
    base:      OpId,
    seq_op:    OpId,
  },
  CalcOffsetPtr {
    index:  OpId,
    base:      OpId,
    seq_op:    OpId,
  },
  Op {
    op_name:  Op,
    operands: [OpId; 3],
    seq_op:   OpId,
  },
  Const(ConstVal),
  //Data,
  Str(IString),
  /// Provides a pointer to the static type entry for the ref type,
  MetaType(RumTypeRef),
  /// Provides a TypeReference object for a given type reference
  MetaTypeReference(RumTypeRef),
  // Extracts type information from operation
  InlineTypeRef(OpId),
  /// Reference to non local object
  StaticObj(Reference),
  Dead,
}

impl Debug for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Operation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Operation::Str(name) => f.write_fmt(format_args!("\"{name}\"",)),
      Operation::MetaType(ty) => f.write_fmt(format_args!("type_of::{ty}",)),
      Operation::MetaTypeReference(ty) => f.write_fmt(format_args!("type_ref_of::{ty}",)),
      Operation::InlineTypeRef(op) => f.write_fmt(format_args!("type_ref_of::{op}",)),
      Operation::StaticObj(name) => f.write_fmt(format_args!("obj::{name:?}",)),
      Operation::Call { routine: routine_op, args, seq_op } => f.write_fmt(format_args!("{routine_op:?} ( {args:?} ) @ {seq_op}",)),
      Operation::NamedOffsetPtr { reference, base, seq_op } => f.write_fmt(format_args!("MEM [{base} + {reference:?}] @ ({seq_op})",)),
      Operation::CalcOffsetPtr { index, base, seq_op } => f.write_fmt(format_args!("MEM [{base} + {index:?}] @ ({seq_op})",)),
      Operation::AggDecl { reps,  seq_op, ty_op: ty_ref_op, .. } => f.write_fmt(format_args!("###### AGG_ALLOC (reps:{reps:?} type:{ty_ref_op}) @ {seq_op:?}",)),
      // Operation::MemCheck(op) => f.write_fmt(format_args!("MemCheck({op})",)),
      Operation::Param(name, index) => f.write_fmt(format_args!("{:12}  {name}[{index}]", "PARAM")),
      //  Operation::Heap(name) => f.write_fmt(format_args!("{:12}  {name}", "HEAP")),
      Operation::Op { op_name, operands, seq_op } => f.write_fmt(format_args!("{op_name:12} [{seq_op}]  {:}", operands.iter().map(|o| format!("{:5}", format!("{o}"))).collect::<Vec<_>>().join("  "))),
      Operation::_Gamma(node, op) => f.write_fmt(format_args!("Gamma  {op:?} @ {node}",)),
      Operation::Φ(node, ops) => f.write_fmt(format_args!("PHI  {ops:?} @ {node}",)),
      Operation::Const(const_val) => f.write_fmt(format_args!("{const_val}",)),
      // Operation::Data => f.write_fmt(format_args!("DATA",)),
      Operation::Dead => f.write_fmt(format_args!("XXXX",)),
    }
  }
}

#[derive(Clone)]
pub(crate) struct RootNode {
  pub(crate) nodes:             Vec<Node>,
  pub(crate) annotations:       Vec<IString>,
  pub(crate) operands:          Vec<Operation>,
  /// Maps operands to the node they belong to.
  pub(crate) operand_node:      Vec<usize>,
  pub(crate) op_types:          Vec<RumTypeRef>,
  pub(crate) type_vars:         Vec<TypeVar>,
  pub(crate) heap_id:           Vec<usize>,
  pub(crate) source_tokens:     Vec<Token>,
  pub(crate) global_references: Vec<ExternalReference>,
  pub(crate) root_id:           isize,
  pub(crate) ty:                RumTypeRef,
}

impl Default for RootNode {
  fn default() -> Self {
    RootNode {
      nodes:             Vec::with_capacity(8),
      annotations:       Vec::with_capacity(8),
      operands:          Vec::with_capacity(8),
      operand_node:      Vec::with_capacity(8),
      op_types:          Vec::with_capacity(8),
      type_vars:         Vec::with_capacity(8),
      source_tokens:     Vec::with_capacity(8),
      global_references: Vec::with_capacity(8),
      heap_id:           Vec::with_capacity(8),
      root_id:           -1,
      ty:                Default::default(),
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

pub(crate) fn get_signature(node: &RootNode) -> Signature {
  get_internal_node_signature(node, 0)
}

pub(crate) fn get_internal_node_signature(node: &RootNode, internal_node_index: usize) -> Signature {
  let RootNode { nodes, op_types: types, type_vars, .. } = node;
  let call_node = &nodes[internal_node_index];

  Signature::new(
    &call_node
      .ports
      .iter()
      .filter(|p| p.ty == PortType::In)
      .filter_map(|p| match p.id {
        VarId::Param(..) => Some((p.slot, get_op_type(node, p.slot))),
        VarId::Name(_) => Some((p.slot, get_op_type(node, p.slot))),
        _ => None,
      })
      .collect::<Vec<_>>(),
    &call_node
      .ports
      .iter()
      .filter(|p| p.ty == PortType::Out)
      .filter_map(|p| match p.id {
        VarId::VoidReturn => Some((p.slot, RumTypeRef::NoUse)),
        VarId::Return => Some((p.slot, get_op_type(node, p.slot))),
        _ => None,
      })
      .collect::<Vec<_>>(),
  )
}

pub(crate) fn get_call_op_signature(node: &RootNode, call_op: OpId) -> Signature {
  if let Operation::Call { args, .. } = &node.operands[call_op.usize()] {
    Signature::new(
      &args
        .iter()
        .filter_map(|op| {
          let ty = get_op_type(node, *op);
          if ty.is_mem_ctx() {
            None
          } else {
            Some((*op, ty))
          }
        })
        .collect::<Vec<_>>(),
      &{ vec![(call_op, get_op_type(node, call_op))] },
    )
  } else {
    unreachable!("Should be call operation")
  }
}

impl Debug for RootNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.nodes.len() > 0 {
      f.write_str("\n###################### \n")?;

      let vars = self
        .type_vars
        .iter()
        .filter_map(|f| {
          if f.ori_id != f.id {
            None
          } else if f.ty.is_open() {
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

      let args = self.nodes[0].ports.iter().filter(|p| p.ty == PortType::In).map(|s| format!("{}", get_op_type(&self, s.slot))).collect::<Vec<_>>().join(", ");

      f.write_str(&args);

      f.write_str(")")?;
      f.write_str(" => ")?;

      let returns = self.nodes[0].ports.iter().filter(|p| p.ty == PortType::Out).map(|s| format!("{}", get_op_type(&self, s.slot))).collect::<Vec<_>>().join("- ");

      f.write_str(&returns);

      f.write_str("\n###################### \n")?;
    }

    for ((index, op), ty) in self.operands.iter().enumerate().zip(self.op_types.iter()) {
      if matches!(op, Operation::Dead) {
        f.write_str("\n  DEAD xxxx")?;
      } else {
        let ty = self.get_base_ty(*ty);
        let mut tok = self.source_tokens[index].clone();
        let source = tok.to_string();
        let heap = self.heap_id.get(index).cloned().unwrap_or_default();
        let parent_node = self.operand_node[index];

        //let heap = if heap < self.type_vars.len() { self.type_vars[heap].ty.to_string() } else { "???".to_string() };

        let heap = if heap == usize::MAX { "".to_string() } else { heap.to_string() };

        f.write_fmt(format_args!("\n  {index:3} @ [{parent_node:3}] <= {:36} @{heap:10} :{:32} {}: {:}", format!("{op}"), format!("{ty}"), tok.get_line(), source))?
      }
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

    Debug::fmt(&self.global_references, f);

    Ok(())
  }
}

impl RootNode {
  pub(crate) fn get_base_ty_from_op(&self, op: OpId) -> RumTypeRef {
    self.get_base_ty(self.op_types[op.usize()])
  }

  pub(crate) fn get_base_ty(&self, ty: RumTypeRef) -> RumTypeRef {
    if let Some(index) = ty.generic_id() {
      let r_ty = get_root_var(index, &self.type_vars).ty;
      let w_ty = get_root_var(index, &self.type_vars).weak_ty;
      if r_ty.is_open() {
        if w_ty.is_open() {
          ty
        } else {
          w_ty
        }
      } else {
        r_ty
      }
    } else {
      ty
    }
  }

  pub fn solve_state(&self) -> SolveState {
    if self.type_vars.iter().any(|v| v.ty.is_open() && v.ori_id == v.id) {
      SolveState::Unsolved
    } else {
      SolveState::Solved
    }
  }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoopType {
  #[default]
  None,
  Head(u16),
  Break(u16),
}

#[derive(Clone)]
pub(crate) struct Node {
  pub(crate) index:     usize,
  pub(crate) parent:    isize,
  pub(crate) type_str:  &'static str,
  pub(crate) children:  Vec<usize>,
  pub(crate) loop_type: LoopType,
  pub(crate) ports:     Vec<NodePort>,
}

impl Node {
  pub fn get_inputs(&self) -> Vec<(OpId, VarId)> {
    let mut out = Vec::new();
    for port in self.ports.iter().filter(|p| p.ty == PortType::In) {
      out.push((port.slot, port.id));
    }
    out
  }

  pub fn get_outputs(&self) -> Vec<(OpId, VarId)> {
    let mut out = Vec::new();
    for port in self.ports.iter().filter(|p| p.ty == PortType::Out) {
      out.push((port.slot, port.id));
    }
    out
  }
}

impl Debug for Node {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Node {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("[{}] {} {:?} <= {}\n", self.index, self.type_str, self.loop_type, self.parent))?;
    f.write_str("ports:\n")?;

    for NodePort { slot: slots, id: name, ty } in self.ports.iter() {
      f.write_fmt(format_args!("  {ty:?} of {name} {:?}\n", slots))?;
    }

    if !self.children.is_empty() {
      f.write_fmt(format_args!("c: {:?} \n", self.children))?;
    }

    /*  f.write_str("inputs:\n")?;
    for (op, id) in self.inputs.iter() {
      f.write_fmt(format_args!("  {op}[{}]\n", id.to_string()))?;
    }

    f.write_str("outputs:\n")?;

    for (op, id) in self.outputs.iter() {
      f.write_fmt(format_args!("  {op}[{}]\n", id.to_string()))?;
    } */

    Ok(())
  }
}

pub(crate) struct Signature {
  pub inputs:  Vec<(OpId, RumTypeRef)>,
  pub outputs: Vec<(OpId, RumTypeRef)>,
}

impl Debug for Signature {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("(")?;
    f.write_str(&self.inputs.iter().map(|(op, ty)| if op.is_valid() { format!("{op}:{ty}") } else { format!("XXXX") }).collect::<Vec<_>>().join(", "))?;
    f.write_str(") => ")?;

    if self.outputs.is_empty() {
      f.write_str("XXXX")?;
    } else {
      f.write_str(&self.outputs.iter().map(|(op, ty)| if op.is_valid() { format!("{ty}") } else { format!("XXXX") }).collect::<Vec<_>>().join(" - "))?;
    }

    Ok(())
  }
}

impl Signature {
  pub fn new(inputs: &[(OpId, RumTypeRef)], outputs: &[(OpId, RumTypeRef)]) -> Self {
    Self { inputs: inputs.to_vec(), outputs: outputs.to_vec() }
  }

  /// Changes the type of the givin paramater to `ty`, where index is
  /// a value in the range 0...(inputs.len + outputs.len)
  pub fn set_param_ty(&mut self, index: usize, ty: RumTypeRef) {
    if index >= self.inputs.len() {
      let index = index - self.inputs.len();
      self.outputs[index].1 = ty;
    } else {
      self.inputs[index].1 = ty;
    }
  }

  pub fn hash(&self) -> u64 {
    let mut h = DefaultHasher::new();
    for (_, ty) in &self.inputs {
      if ty.is_generic() {
        0u64.hash(&mut h);
      } else {
        ty.hash(&mut h);
      }
    }

    for (_, ty) in &self.outputs {
      if ty.is_generic() {
        0u64.hash(&mut h);
      } else {
        ty.hash(&mut h);
      }
    }

    h.finish()
  }
}
