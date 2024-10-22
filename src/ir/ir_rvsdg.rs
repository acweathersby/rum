use radlr_rust_runtime::types::Token;

use super::ir_graph::{IRGraphId, IROp};
use crate::{
  container::ArrayVec,
  istring::{CachedString, IString},
  types::ConstVal,
};
use std::{
  collections::BTreeMap,
  fmt::{Debug, Display, Pointer, Write},
};

pub mod lower;
pub mod solve_pipeline;
pub mod type_check;
pub mod type_solve;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RVSDGNodeType {
  #[default]
  Undefined,
  Function,
  MatchHead,
  Switch,
  SwitchBody,
  Call,
  Struct,
  Array,
  Module,
}

#[derive(Default, Clone, Debug)]
pub struct RVSDGNode {
  pub id:            IString,
  pub ty:            RVSDGNodeType,
  pub inputs:        ArrayVec<4, RSDVGBinding>,
  pub outputs:       ArrayVec<4, RSDVGBinding>,
  pub nodes:         Vec<RVSDGInternalNode>,
  pub source_tokens: Vec<Token>,
}

impl RVSDGNode {
  pub fn new_module() -> Box<Self> {
    Box::new(RVSDGNode {
      id:            Default::default(),
      ty:            RVSDGNodeType::Module,
      inputs:        Default::default(),
      outputs:       Default::default(),
      nodes:         Default::default(),
      source_tokens: Default::default(),
    })
  }
}

impl Display for RVSDGNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("Node");
    s.field("ty", &self.ty);
    s.field("id", &self.id);

    s.field_with("in", |f| {
      for i in self.inputs.iter() {
        f.write_str("\n    ")?;
        Display::fmt(&i, f)?;
      }
      Ok(())
    });

    if self.nodes.len() > 0 {
      s.field_with("nodes", |f| {
        for i in self.nodes.iter() {
          f.write_fmt(format_args!("\n"))?;
          Display::fmt(&i, f)?;
        }
        Ok(())
      });
    }

    s.field_with("out", |f| {
      for i in self.outputs.iter() {
        f.write_str("\n     ")?;
        Display::fmt(&i, f)?;
      }
      Ok(())
    });
    s.finish()
  }
}

#[derive(Clone, Copy, Default)]
pub struct RSDVGBinding {
  // Temporary identifier of the binding
  pub name:        IString,
  /// The input node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a node in the parent scope
  ///
  /// if the binding is an output then this value corresponds to a local node
  pub in_id:       IRGraphId,
  /// The output node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a local node
  ///
  /// if the binding is an output then this value corresponds to a node in the parent scope
  pub out_id:      IRGraphId,
  /// The type of the binding. This must match the types of the in_id and out_id nodes
  pub ty:          Type,
  pub input_index: u32,
}

impl Debug for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:>3} => {:<3} {:>3} [{}]", self.in_id, self.out_id, self.ty, self.name.to_string(),))
  }
}

#[derive(Clone)]
pub enum RVSDGInternalNode {
  Label(IRGraphId, IString),
  Const(u32, ConstVal),
  Complex(Box<RVSDGNode>),
  Simple { id: IRGraphId, op: IROp, operands: [IRGraphId; 2], ty: Type },
  Input { id: IRGraphId, ty: Type, input_index: usize },
  Output { id: IRGraphId, ty: Type, output_index: usize },
}

impl Debug for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RVSDGInternalNode::Label(id, name) => f.write_fmt(format_args!("{}: \"{:#}\"", id, name)),
      RVSDGInternalNode::Complex(complex) => f.write_fmt(format_args!("{:#}", complex)),
      RVSDGInternalNode::Const(id, r#const) => f.write_fmt(format_args!("{id:3} : {}", r#const)),
      RVSDGInternalNode::Simple { id, op, operands, ty } => f.write_fmt(format_args!(
        "{id:03}: {:6} = {:6} {:3}",
        format!("{:?}", ty),
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
      RVSDGInternalNode::Input { id, ty, .. } => f.write_fmt(format_args!("{}:=: {} ", id, ty)),
      RVSDGInternalNode::Output { id, ty, output_index } => f.write_fmt(format_args!("{}:{:5} => [@{:03}]", id, ty, output_index)),
    }
  }
}

#[cfg(test)]
mod test;

fn get_node_by_name(name: IString, node: &mut RVSDGNode) -> Option<&mut RVSDGNode> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_tokens } = node;

  let node_ptr = nodes.as_mut_ptr();

  for RSDVGBinding { name: n_name, in_id, out_id, ty, input_index } in outputs.iter().cloned() {
    if name == n_name {
      match unsafe { &mut *node_ptr.offset(in_id.usize() as isize) } {
        RVSDGInternalNode::Complex(node) => return Some(node),
        _ => {}
      }
    }
  }
  None
}

#[derive(Debug, Clone, Copy)]
pub struct TypeEntry {
  pub ty: Type,
  node:   Option<*mut RVSDGNode>,
}

impl TypeEntry {
  pub fn get_node(&self) -> Option<&RVSDGNode> {
    self.node.map(|n| unsafe { &*n })
  }

  pub fn get_node_mut(&mut self) -> Option<&mut RVSDGNode> {
    self.node.map(|n| unsafe { &mut *n })
  }
}

#[derive(Debug)]
pub struct TypeDatabase {
  types:         Vec<TypeEntry>,
  name_to_entry: BTreeMap<IString, usize>,
}

macro_rules! create_primitive {
  ($db:ident $index:ident) => { };
  ($db:ident $index:ident $name:literal $($rest:literal)*) => {
    let n = $name.intern();
    $db.name_to_entry.insert(n, $index);
    create_primitive!($db $index $($rest)*)
  };
  ($db:ident $primitive_name:tt,  $size:literal  $ele_count:literal  $($name:literal)+) => {
    let index = $db.types.len();
    let ty = Type::Primitive(PrimitiveType{  base_ty: PrimitiveBaseType::$primitive_name,  base_index: index as u8,  byte_size: $size, ele_count: $ele_count   });
    $db.types.push(TypeEntry { ty, node: None });
    create_primitive!($db index $($name)+)
  };
}

impl TypeDatabase {
  pub fn new() -> Self {
    let mut db = Self { types: Default::default(), name_to_entry: Default::default() };
    "".intern();

    create_primitive!(db Unsigned, 1 1 "u8");
    create_primitive!(db Unsigned, 2 1 "u16");
    create_primitive!(db Unsigned, 4 1 "u32");
    create_primitive!(db Unsigned, 8 1 "u64");
    create_primitive!(db Unsigned, 8 1 "u128");

    create_primitive!(db Signed, 1 1 "i8"   "s8" );
    create_primitive!(db Signed, 2 1 "i16"  "s16");
    create_primitive!(db Signed, 4 1 "i32"  "s32");
    create_primitive!(db Signed, 8 1 "i64"  "s64");
    create_primitive!(db Signed, 8 1 "i128" "s128");

    create_primitive!(db Float, 2 1 "f16");
    create_primitive!(db Float, 4 1 "f32");
    create_primitive!(db Float, 8 1 "f64");

    db
  }

  pub fn get_ty_entry(&self, name: &str) -> Option<TypeEntry> {
    self.name_to_entry.get(&name.to_token()).map(|i| self.types[*i])
  }

  pub fn get_ty(&self, name: &str) -> Option<Type> {
    self.name_to_entry.get(&name.to_token()).map(|i| self.types[*i].ty)
  }

  pub fn get_ptr(&mut self, ty: Type) -> Option<Type> {
    match ty {
      Type::Undefined => None,
      Type::Generic { ptr_count, gen_index } => Some(Type::Generic { ptr_count: ptr_count + 1, gen_index }),
      Type::Complex { ty_index } => Some(Type::Pointer { count: 1, ty_index }),
      Type::Pointer { count, ty_index } => Some(Type::Pointer { count: count + 1, ty_index }),
      Type::Primitive(PrimitiveType { base_index, .. }) => Some(Type::Pointer { count: 1, ty_index: base_index as u32 }),
    }
  }

  pub fn add_ty(&mut self, name: IString, node: Box<RVSDGNode>) -> Option<Type> {
    let index = self.types.len();
    let ty = Type::Complex { ty_index: index as u32 };

    if self.get_ty_entry(&name.to_str().as_str()).is_some() {
      None
    } else {
      self.name_to_entry.insert(name, index);
      self.types.push(TypeEntry { ty, node: Some(Box::into_raw(node)) });
      Some(ty)
    }
  }
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default)]
pub enum Type {
  #[default]
  Undefined,
  Generic {
    ptr_count: u8,
    gen_index: u32,
  },
  Primitive(PrimitiveType),
  Complex {
    ty_index: u32,
  },
  Pointer {
    count:    u8,
    ty_index: u32,
  },
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub enum PrimitiveBaseType {
  Unsigned,
  Signed,
  Float,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct PrimitiveType {
  pub base_ty:    PrimitiveBaseType,
  pub base_index: u8,
  pub byte_size:  u8,
  pub ele_count:  u8,
}

impl Debug for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let PrimitiveType { base_ty, base_index, byte_size, ele_count } = self;
    use PrimitiveBaseType::*;
    match base_ty {
      Signed => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("s{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("s{}", byte_size * 8))
        }
      }
      Unsigned => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("u{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("u{}", byte_size * 8))
        }
      }
      Float => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("f{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("f{}", byte_size * 8))
        }
      }
    }
  }
}

impl Type {
  pub fn generic(ty_index: usize) -> Type {
    Type::Pointer { count: 1, ty_index: ty_index as u32 }
  }

  pub fn generic_id(&self) -> Option<usize> {
    match *self {
      Type::Generic { ptr_count, gen_index } => Some(gen_index as usize),
      _ => None,
    }
  }

  pub fn to_primitive(&self) -> Option<PrimitiveType> {
    match self {
      Type::Primitive(prim) => Some(*prim),
      _ => None,
    }
  }

  pub fn is_primitive(&self) -> bool {
    matches!(self, Type::Primitive(..))
  }

  pub fn is_generic(&self) -> bool {
    matches!(self, Type::Generic { .. })
  }

  pub fn is_undefined(&self) -> bool {
    matches!(self, Type::Undefined)
  }
}

impl Debug for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use Type::*;
    match self {
      Undefined => f.write_str("und"),
      Generic { ptr_count, gen_index } => f.write_fmt(format_args!("∀{}", gen_index)),
      Primitive(prim) => f.write_fmt(format_args!("{prim}")),
      Complex { ty_index } => f.write_fmt(format_args!("cplx@[{}]", ty_index)),
      Pointer { ty_index, .. } => f.write_fmt(format_args!("* -> [{}]", ty_index)),
    }
  }
}

#[test]
fn test_primitive_type() {
  let ty = TypeDatabase::new();

  let f64 = ty.get_ty_entry("f64").expect("Should have f64").ty;
  assert_eq!(format!("{f64}"), "f64");

  let f32 = ty.get_ty_entry("f32").expect("Should have f32").ty;
  assert_eq!(format!("{f32}"), "f32");

  let u64 = ty.get_ty_entry("u64").expect("Should have u64").ty;
  assert_eq!(format!("{u64}"), "u64");

  let u32 = ty.get_ty_entry("u32").expect("Should have u32").ty;
  assert_eq!(format!("{u32}"), "u32");

  let u16 = ty.get_ty_entry("u16").expect("Should have u16").ty;
  assert_eq!(format!("{u16}"), "u16");

  let u8 = ty.get_ty_entry("u8").expect("Should have u8").ty;
  assert_eq!(format!("{u8}"), "u8");

  let s64 = ty.get_ty_entry("s64").expect("Should have s64").ty;
  assert_eq!(format!("{s64}"), "s64");

  let s32 = ty.get_ty_entry("s32").expect("Should have s32").ty;
  assert_eq!(format!("{s32}"), "s32");

  let s16 = ty.get_ty_entry("s16").expect("Should have s16").ty;
  assert_eq!(format!("{s16}"), "s16");

  let s8 = ty.get_ty_entry("s8").expect("Should have s8").ty;
  assert_eq!(format!("{s8}"), "s8");
}
