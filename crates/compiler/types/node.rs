use super::*;
use rum_lang::{
  ir::ir_rvsdg::SolveState,
  istring::{CachedString, IString},
};
use std::fmt::{Debug, Display};

#[derive(Default, Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub enum VarId {
  #[default]
  Undefined,
  Name(IString),
  SideEffect(usize),
  MemRef(usize),
  MatchInputExpr,
  MatchOutputVal,
  MatchActivation,
  LoopActivation,
  Return,
  GlobalContext,
  Generic,
  HeapContext,
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
      Self::MatchOutputVal => f.write_str("MATCH_OUTPUT_VALUE"),
      Self::MatchActivation => f.write_str("MATCH_ACTIVATION"),
      Self::Return => f.write_str("RETURN"),
      Self::LoopActivation => f.write_str("LOOP_ACTIVATION"),
      Self::HeapContext => f.write_str("HeapCTX"),
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

#[derive(Debug, Clone)]
pub struct CallLookup {
  pub name:        IString,
  pub args:        Vec<Type>,
  pub ret:         Type,
  pub origin_node: usize,
}

#[derive(Clone)]
pub(crate) struct RootNode {
  pub(crate) host_db:      Option<Database>,
  pub(crate) binding_name: IString,
  pub(crate) nodes:        Vec<Node>,
  pub(crate) operands:     Vec<Operation>,
  pub(crate) types:        Vec<Type>,
  pub(crate) type_vars:    Vec<TypeVar>,
  pub(crate) errors:       Vec<String>,
}

pub(crate) fn write_agg(var: &TypeVar, vars: &[TypeVar]) -> String {
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

impl Debug for RootNode {
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
