use std::fmt::Display;

use crate::istring::{CachedString, IString};

use std::collections::BTreeMap;

use std::fmt::Debug;

use super::ir_rvsdg::RVSDGNode;

#[derive(Clone, Copy)]
pub struct TypeEntry {
  pub ty:                 Type,
  pub(crate) node:        Option<*mut RVSDGNode>,
  pub(crate) offset_data: Option<(usize, usize, *mut usize)>,
  pub(crate) size:        usize,
}

impl Debug for TypeEntry {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if let Some(node) = self.node {
      f.write_fmt(format_args!("{} => \n{:#?}", self.ty, unsafe { &*node }))
    } else {
      Debug::fmt(&self.ty, f)
    }
  }
}

impl TypeEntry {
  pub fn get_node(&self) -> Option<&RVSDGNode> {
    self.node.map(|n| unsafe { &*n })
  }

  pub fn get_node_mut(&mut self) -> Option<&mut RVSDGNode> {
    self.node.map(|n| unsafe { &mut *n })
  }

  pub fn get_offset_data(&self) -> Option<&[usize]> {
    self.offset_data.map(|(len, capacity, data)| unsafe { std::slice::from_raw_parts(data, len as usize) })
  }
}

#[derive(Debug)]
pub struct TypeDatabase {
  pub(crate) types:         Vec<TypeEntry>,
  pub(crate) name_to_entry: BTreeMap<IString, usize>,
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
    $db.types.push(TypeEntry { ty, node: None, offset_data:None, size: 0 });
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
  pub fn get_or_insert_complex_type(&mut self, name: &str) -> Type {
    match self.name_to_entry.entry(name.intern()) {
      std::collections::btree_map::Entry::Occupied(ty) => self.types[*ty.get()].ty,
      std::collections::btree_map::Entry::Vacant(mut entry) => {
        let index = self.types.len();
        entry.insert(index);

        let entry = TypeEntry {
          node:        None,
          ty:          Type::Complex { ty_index: index as u32, hash: 0 },
          offset_data: None,
          size:        0,
        };
        self.types.push(entry);
        entry.ty
      }
    }
  }

  pub fn get_ty_entry_from_ty(&self, ty: Type) -> Option<TypeEntry> {
    let index = match ty {
      Type::Complex { ty_index, .. } => ty_index as usize,
      Type::Pointer { count, ty_index } => ty_index as usize,
      Type::Primitive(prim) => usize::MAX,
      _ => usize::MAX,
    };

    if index < self.types.len() {
      Some(self.types[index])
    } else {
      None
    }
  }

  pub fn get_ty_entry(&self, name: &str) -> Option<TypeEntry> {
    self.name_to_entry.get(&name.to_token()).map(|i| self.types[*i])
  }

  pub fn get_ty(&self, name: &str) -> Option<Type> {
    self.name_to_entry.get(&name.to_token()).map(|i| self.types[*i].ty)
  }

  pub fn get_ptr(&self, ty: Type) -> Option<Type> {
    match ty {
      Type::ComplexHash(_) => Some(Type::Undefined),
      Type::Undefined => Some(Type::Undefined),
      Type::Generic { ptr_count, gen_index } => Some(Type::Generic { ptr_count: ptr_count + 1, gen_index }),
      Type::Complex { ty_index, .. } => Some(Type::Pointer { count: 1, ty_index }),
      Type::Pointer { count, ty_index } => Some(Type::Pointer { count: count + 1, ty_index }),
      Type::Primitive(PrimitiveType { base_index, .. }) => Some(Type::Pointer { count: 1, ty_index: base_index as u32 }),
    }
  }

  pub fn add_ty(&mut self, name: IString, node: Box<RVSDGNode>) -> Option<Type> {
    let index = self.types.len();
    let ty = Type::Complex { ty_index: index as u32, hash: 0 };

    if let Some(mut entry) = self.get_ty_entry(&name.to_str().as_str()) {
      if entry.node.is_none() {
        if let Type::Complex { ty_index, .. } = entry.ty {
          let index = ty_index as usize;
          entry.node = Some(Box::into_raw(node));
          self.types[index] = entry;
          Some(entry.ty)
        } else {
          None
        }
      } else {
        None
      }
    } else {
      self.name_to_entry.insert(name, index);
      self.types.push(TypeEntry { ty, node: Some(Box::into_raw(node)), offset_data: None, size: 0 });
      Some(ty)
    }
  }
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
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
    hash:     u32,
  },
  ComplexHash(u64),
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
    Type::Generic { ptr_count: 0, gen_index: ty_index as u32 }
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

  pub fn is_open(&self) -> bool {
    self.is_generic() || self.is_undefined()
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
      ComplexHash(ty_hash) => f.write_fmt(format_args!("cplx#{:016X}", ty_hash)),
      Undefined => f.write_str("und"),
      Generic { ptr_count, gen_index } => f.write_fmt(format_args!("∀{}", gen_index)),
      Primitive(prim) => f.write_fmt(format_args!("{prim}")),
      Complex { ty_index, .. } => f.write_fmt(format_args!("cplx@[{}]", ty_index)),
      Pointer { ty_index, .. } => f.write_fmt(format_args!("* -> [{}]", ty_index)),
    }
  }
}

#[test]
pub(crate) fn test_primitive_type() {
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

pub fn dbg_ty(ty: Type, ty_db: &TypeDatabase) {
  match ty {
    Type::Complex { ty_index, .. } => {
      debug_assert!((ty_index as usize) < ty_db.types.len(), "type index is out side the range of TypeDatabase");

      let ty = ty_db.types[ty_index as usize];

      eprintln!("{ty:?}")
    }
    ty => {
      dbg!(ty);
    }
  }
}
