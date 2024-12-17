use std::fmt::{Debug, Display};

use rum_lang::{container::ArrayVec, istring::IString};

use super::{OpId, Type, VarId};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MemberEntry {
  pub name:      IString,
  pub origin_op: u32,
  pub ty:        Type,
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NodeConstraint {
  /// Used to bind a variable to a type that is not defined in the current
  /// routine scope.
  GlobalNameReference(Type, IString),
  OpToTy(OpId, Type),
  // The type of op at src must match te type of the op at dst.
  // If both src and dst are resolved, a conversion must be made.
  OpToOp {
    src: OpId,
    dst: OpId,
  },
  BindOpToOp {
    src: OpId,
    dst: OpId,
  },
  MemOp {
    ptr_op: OpId,
    val_op: OpId,
  },
  Deref {
    ptr_ty:  Type,
    val_ty:  Type,
    mutable: bool,
  },
  Num(Type),
  Member {
    name:    IString,
    ref_dst: OpId,
    par:     OpId,
  },
  Mutable(u32, u32),
  Agg(OpId),
  GenTyToTy(Type, Type),
  GenTyToGenTy(Type, Type),
  OpConvertTo {
    target_op: OpId,
    arg_index: usize,
    target_ty: Type,
  },
}

#[derive(Clone)]
pub struct TypeVar {
  pub id:         u32,
  pub ref_id:     i32,
  pub ty:         Type,
  pub ref_count:  u32,
  pub attributes: ArrayVec<2, VarAttribute>,
  pub members:    ArrayVec<2, MemberEntry>,
}

impl Default for TypeVar {
  fn default() -> Self {
    Self {
      id:         Default::default(),
      ref_id:     -1,
      ref_count:  0,
      ty:         Default::default(),
      attributes: Default::default(),
      members:    Default::default(),
    }
  }
}

impl TypeVar {
  pub fn new(id: u32) -> Self {
    Self { id: id, ..Default::default() }
  }

  #[track_caller]
  pub fn has(&self, constraint: VarAttribute) -> bool {
    self.attributes.find_ordered(&constraint).is_some()
  }

  #[track_caller]
  pub fn add(&mut self, constraint: VarAttribute) {
    let _ = self.attributes.push_unique(constraint);
  }

  pub fn add_mem(&mut self, name: IString, ty: Type, origin_node: u32) {
    self.attributes.push_unique(VarAttribute::Agg).unwrap();

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
        return Some((*origin_node, ty.clone()));
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
    let Self { id, ty, attributes: constraints, members, ref_id, ref_count } = self;

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

#[derive(Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum VarAttribute {
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
  /// Node index, node port index, is_output
  Binding(u32, u32, bool),
  ForeignType,
  Global(IString),
}

impl Debug for VarAttribute {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarAttribute::*;
    match self {
      ForeignType => f.write_str("FOREIGN"),
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
      &Global(ty) => f.write_fmt(format_args!("typeof({ty})",)),
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

pub(crate) fn get_root_var<'a>(mut index: usize, type_vars: &'a [TypeVar]) -> &'a TypeVar {
  unsafe {
    let mut var = type_vars.as_ptr().offset(index as isize);

    while (&*var).id != index as u32 {
      index = (&*var).id as usize;
      var = type_vars.as_ptr().offset(index as isize);
    }

    &*var
  }
}
