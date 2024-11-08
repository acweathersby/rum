use super::types::Type;
use crate::{
  container::ArrayVec,
  ir_interpreter::blame::blame,
  istring::{CachedString, IString},
  parser::script_parser::ASTNode,
  types::ConstVal,
};
use radlr_rust_runtime::types::Token;
use std::{
  collections::BTreeMap,
  fmt::{Debug, Display, Pointer, Write},
  sync::Arc,
};
use type_solve::TypeVar;

pub mod lower;
pub mod solve_pipeline;
pub mod type_check;
pub mod type_solve;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RVSDGNodeType {
  #[default]
  Undefined,
  Routine,
  MatchHead,
  MatchClause,
  MatchActivation,
  MatchBody,
  Call,
  Struct,
  Array,
  Module,
  Loop,
  GenericBlock,
}

#[derive(Default, Clone)]
pub struct RVSDGNode {
  pub id:           u32,
  pub ty:           RVSDGNodeType,
  pub inputs:       ArrayVec<4, RSDVGBinding>,
  pub outputs:      ArrayVec<4, RSDVGBinding>,
  pub nodes:        Vec<RVSDGInternalNode>,
  pub source_nodes: Vec<ASTNode>,
  pub ty_vars:      Vec<TypeVar>,
  pub types:        Vec<Type>,
  pub solved:       SolveState,
}

impl RVSDGNode {
  pub fn set_type_if_undefined(&mut self, op: IRGraphId, ty: Type) -> Type {
    let existing_type = &mut self.types[op.usize()];

    if existing_type.is_undefined() {
      *existing_type = ty;
    }

    *existing_type
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum SolveState {
  #[default]
  Unsolved,
  Solved,
  PartiallySolved,
}

impl RVSDGNode {
  pub fn new_module() -> Box<Self> {
    Box::new(RVSDGNode {
      id:           Default::default(),
      ty:           RVSDGNodeType::Module,
      inputs:       Default::default(),
      outputs:      Default::default(),
      nodes:        Default::default(),
      source_nodes: Default::default(),
      ty_vars:      Default::default(),
      types:        Default::default(),
      solved:       Default::default(),
    })
  }
}

impl Debug for RVSDGNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone, Copy, Default)]
pub struct RSDVGBinding {
  // Temporary identifier of the binding
  pub name:   IString,
  /// The input node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a node in the parent scope
  ///
  /// if the binding is an output then this value corresponds to a local node
  pub in_id:  IRGraphId,
  /// The output node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a local node
  ///
  /// if the binding is an output then this value corresponds to a node in the parent scope
  pub out_id: IRGraphId,
}

impl Debug for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:<4}  => {:<3} [{}]", self.in_id, self.out_id, self.name.to_string(),))
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
  ParamBinding,
  IntraBinding,
}

#[derive(Clone)]
pub enum RVSDGInternalNode {
  PlaceHolder,
  Label(IString),
  Const(ConstVal),
  Complex(Box<RVSDGNode>),
  Simple { op: IROp, operands: [IRGraphId; 3] },
  Binding { ty: BindingType },
  Sink { src: IRGraphId, ty: BindingType },
}

impl Debug for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RVSDGInternalNode::PlaceHolder => f.write_str("--- place holder --- "),
      RVSDGInternalNode::Label(name) => f.write_fmt(format_args!("\"{:#}\"", name)),
      RVSDGInternalNode::Complex(complex) => f.write_fmt(format_args!("{:#}", complex)),
      RVSDGInternalNode::Const(r#const) => f.write_fmt(format_args!("{}", r#const)),
      RVSDGInternalNode::Simple { op, operands } => f.write_fmt(format_args!(
        "{:10} {:10}",
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
      RVSDGInternalNode::Binding { ty } => f.write_fmt(format_args!("{ty:?}")),
      RVSDGInternalNode::Sink { ty, src: op } => f.write_fmt(format_args!("{ty:?}({op:?}) ")),
    }
  }
}

#[cfg(test)]
mod test;

fn get_node_by_name(name: IString, node: &mut RVSDGNode) -> Option<&mut RVSDGNode> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_nodes: source_tokens, .. } = node;

  let node_ptr = nodes.as_mut_ptr();

  for RSDVGBinding { name: n_name, in_id, out_id } in outputs.iter().cloned() {
    if name == n_name {
      match unsafe { &mut *node_ptr.offset(in_id.usize() as isize) } {
        RVSDGInternalNode::Complex(node) => return Some(node),
        _ => {}
      }
    }
  }
  None
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IROp {
  // Encoding Oriented operators
  /// Calculates a ptr to a member variable based on a base aggregate pointer
  /// and a const offset. This is also used to get the address of a stack
  /// variable, by taking address of the difference between the sp and stack
  /// offset.
  MEMB_PTR_CALC,
  /// Declares a stack or heap variable and its type
  MATCH_LOC,
  /// Declares a stack or heap variable and its type
  VAR_DECL,
  /// Declares a location to store a local value
  AGG_DECL,
  ///
  PARAM_VAL,
  /// Declares a location to store a parameter value
  PARAM_DECL,
  /// Declares a location to store a return value
  RET_VAL,
  /// Declares a constant and its type
  CONST_DECL,

  REF,

  // Arithmetic & Logic functions - MATH
  ADD,
  SUB,
  MUL,
  DIV,
  LOG,
  POW,
  GR,
  LE,
  GE,
  LS,
  NE,
  EQ,
  OR,
  XOR,
  AND,
  NOT,
  NEG,
  SHL,
  SHR,
  // End MATH --------------------------
  /// Stores a value into a memory location denoted by a pointer.
  STORE,
  /// Loads must proceed from a STORE, a PARAM_DECL, or a RET_VAL
  LOAD,
  /// Zeroes all bytes of a type pointer or an array pointer.
  ZERO,
  /// Copies data from one type pointer to another type pointer.
  COPY,

  /// Declares a variable output value for an iteration step
  ITER_OUT_VAL,
  ITER_IN_VAL,
  ITER_ARG,

  DBG_CALL,
  CALL,
  CALL_ARG,
  CALL_RET,
  // Deliberate movement of data from one location to another
  MOVE,
  // Clone one memory structure to another memory structure. Operands MUST be pointer values.
  // Depending on type, may require deep cloning, which will probably be handled through a dynamically generated function.
  CLONE,
  /// Returns the address of op1 as a pointer
  LOAD_ADDR,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct IRGraphId(pub u32);

impl<T> std::ops::Index<IRGraphId> for Vec<T> {
  type Output = T;
  fn index(&self, index: IRGraphId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T> std::ops::IndexMut<IRGraphId> for Vec<T> {
  fn index_mut(&mut self, index: IRGraphId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum GraphIdType {
  SSA,
  CALL,
  STORED_REGISTER,
  REGISTER,
  VAR_LOAD,
  VAR_STORE,
  INVALID = 0xF,
}

impl Default for IRGraphId {
  fn default() -> Self {
    Self::INVALID
  }
}

impl IRGraphId {
  pub const INVALID: IRGraphId = IRGraphId(u32::MAX);
  pub const INDEX_MASK: u64 = 0x0000_0000_00FF_FFFF;
  pub const VAR_MASK: u64 = 0x0000_FFFF_FF00_0000;
  pub const REG_MASK: u64 = 0x0FFF_0000_0000_0000;
  pub const NEEDS_LOAD_VAL: u64 = 0x7000_0000_0000_0000;
  pub const LOAD_MASK_OUT: u64 = 0x0FFF_FFFF_FFFF_FFFF;

  pub const fn usize(&self) -> usize {
    self.0 as usize
  }

  pub const fn new(index: usize) -> IRGraphId {
    IRGraphId(index as u32)
  }

  pub const fn is_invalid(&self) -> bool {
    self.0 == Self::INVALID.0
  }

  pub const fn is_valid(&self) -> bool {
    !self.is_invalid()
  }
}

impl Display for IRGraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_invalid() {
      f.write_fmt(format_args!("xxx"))
    } else {
      f.write_fmt(format_args!("{:>3}", format!("`{}", self.0)))
    }
  }
}

impl Debug for IRGraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl From<IRGraphId> for usize {
  fn from(value: IRGraphId) -> Self {
    value.0 as usize
  }
}

impl RVSDGNode {
  fn fmt_internal(&self, f: &mut std::fmt::Formatter<'_>, ty_vars: &[TypeVar]) -> std::fmt::Result {
    let index = 0;
    let types = &self.types;

    f.write_fmt(format_args!("--- [{}]  {:?}  ---\n", self.id, self.ty))?;
    f.write_fmt(format_args!("!# {:20} #!\n", format!("{:?}", self.solved)))?;

    if self.inputs.len() > 0 {
      f.write_str("\ninputs:\n")?;

      for input in self.inputs.iter() {
        f.write_fmt(format_args!("{:<49} | {:}\n", format!("{input}"), get_type_string(input.out_id.usize(), types, ty_vars)))?;
      }
    }

    f.write_str("nodes:\n")?;
    for (index, node) in self.nodes.iter().enumerate() {
      match node {
        RVSDGInternalNode::Complex(node) => {
          f.write_fmt(format_args!(
            "`{index:<4} ---------------- {}\n  {} \n---------------------- | \n",
            get_type_string(index, types, ty_vars),
            format!("{}", node).split("\n").collect::<Vec<_>>().join("\n  "),
          ));
        }
        _ => {
          f.write_fmt(format_args!("`{index:<4} <= {:<40} | {:}\n", format!("{node}"), format!("{}", get_type_string(index, types, ty_vars))));
        }
      }
    }

    if self.outputs.len() > 0 {
      f.write_str("outputs:\n")?;

      for output in self.outputs.iter() {
        f.write_fmt(format_args!("{:<49} | {:}\n", format!("{output}"), get_type_string(output.in_id.usize(), types, ty_vars)))?;
      }
    }

    if !self.ty_vars.is_empty() {
      f.write_str("type vars:\n")?;
      for var in &self.ty_vars {
        f.write_fmt(format_args!("  {var}\n"))?;
      }
    }

    Ok(())
  }
}

impl Display for RVSDGNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let ty_vars = &self.ty_vars;

    self.fmt_internal(f, ty_vars);

    Ok(())
  }
}

pub fn __debug_node_types__(node: &RVSDGNode) {
  println!("{node}");
}

fn get_type_string(index: usize, types: &Vec<Type>, ty_vars: &[TypeVar]) -> String {
  if index > types.len() {
    Default::default()
  } else {
    let ty: Type = types[index];
    if let Some(gen_index) = ty.generic_id() {
      if gen_index < ty_vars.len() {
        format!("{}", ty_vars[gen_index])
      } else {
        format!("A{}", gen_index)
      }
    } else if ty.is_undefined() {
      Default::default()
    } else {
      format!("{ty}")
    }
  }
}
