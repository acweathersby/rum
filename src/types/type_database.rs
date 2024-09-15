use super::{ArrayType, BitFieldType, EnumType, Lifetime, RoutineType, RumType, ScopeType, StructType, UnionType};
use crate::{
  ir::ir_graph::{IRGraphId, VarId},
  istring::{CachedString, IString},
  parser::script_parser::Var,
};
use std::{
  borrow::BorrowMut,
  cell::{Ref, RefCell, RefMut},
  collections::{BTreeMap, HashMap},
  fmt::{Debug, Display, Pointer},
  ops::Deref,
  rc::Rc,
};

pub use ctx::*;
pub use scope::*;
pub use ty::*;
pub use type_slot::*;
pub use variable::*;

pub struct TypeDatabase {
  pub types:     Vec<Box<Type>>,
  pub lifetimes: Vec<Box<Lifetime>>,
  /// Maps a typename to a to a databese type type.
  lookup:        BTreeMap<IString, usize>,
}

impl TypeDatabase {
  pub fn new() -> Self {
    Self { types: Default::default(), lifetimes: Default::default(), lookup: Default::default() }
  }

  pub fn get_type_index(&self, name: IString) -> Option<usize> {
    if let Some(index) = self.lookup.get(&name) {
      Some(*index)
    } else {
      None
    }
  }

  pub fn get_or_add_type_index(&mut self, name: IString, ty: Type) -> usize {
    if let Some(index) = self.lookup.get(&name) {
      *index
    } else {
      self.insert_type(name, ty)
    }
  }

  pub fn get_type_mut<'a>(&mut self, name: IString) -> Option<(&'a mut Type, RumType)> {
    if let Some(index) = self.get_type_index(name) {
      let ptr = self.types.as_mut_ptr();
      let val = unsafe { ptr.offset(index as isize) };
      Some((unsafe { &mut *val }, RumType::Undefined.to_aggregate_id(index)))
    } else {
      None
    }
  }

  pub fn get_type<'a>(&self, name: IString) -> Option<(&'a Type, RumType)> {
    if let Some(index) = self.get_type_index(name) {
      let ptr = self.types.as_ptr();
      let val = unsafe { ptr.offset(index as isize) };
      Some((unsafe { &*val }, RumType::Undefined.to_aggregate_id(index)))
    } else {
      None
    }
  }

  pub fn insert_type(&mut self, name: IString, ty: Type) -> usize {
    let index = self.types.len();
    self.lookup.insert(name, index);
    self.types.push(Box::new(ty));
    index
  }

  pub fn insert_anonymous_type(&mut self, ty: Type) -> usize {
    let index = self.types.len();
    self.types.push(Box::new(ty));
    index
  }
}

mod ctx {
  use std::{
    cell::RefMut,
    fmt::{Debug, Display},
  };

  use super::{ty::Type, variable::Variable, RumType, TypeDatabase};
  use crate::{
    ir::ir_graph::VarId,
    istring::{CachedString, IString},
    types::{type_context::MemberName, type_database::ty::TypeRef, RumSubType, ScopeLookup},
  };

  #[derive(Clone)]
  pub struct TypeVarContext {
    pub db:            *mut TypeDatabase,
    pub vars:          Vec<Variable>,
    pub type_slots:    Vec<(RumType, VarId)>,
    pub generics:      Vec<Vec<VarId>>,
    pub var_scopes:    ScopeLookup,
    pub ty_var_scopes: ScopeLookup,
  }

  impl Debug for TypeVarContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      Display::fmt(&self, f)
    }
  }

  impl Display for TypeVarContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      let db = unsafe { &mut *self.db };
      f.write_str("\nTVContext\n  vars:\n")?;

      for (index, var) in self.vars.iter().enumerate() {
        if var.par.is_valid() {
          if var.mem_name.is_empty() {
            f.write_fmt(format_args!("  {index:5}: {}[{}]({})", var.par, var.mem_index, var.ty))?;
          } else {
            f.write_fmt(format_args!("  {index:5}: {}.{}({})", var.par, var.mem_name, var.ty))?;
          }
        } else {
          f.write_fmt(format_args!("  {index:5}: {} {} ", var.mem_name, var.ty))?;
        }
        f.write_str("\n")?;
      }

      f.write_str("  types:\n")?;

      for (index, scope) in self.var_scopes.scopes.iter().enumerate() {
        f.write_fmt(format_args!("  scope {index}:\n"))?;
        for (name, index, entry_index) in &scope.1 {
          let ty = &self.type_slots[*entry_index].0;
          if !name.is_empty() {
            f.write_fmt(format_args!(" {entry_index:5}: {name} => {}", ty))?;
          } else if *index < usize::MAX {
            f.write_fmt(format_args!(" {entry_index:5}: [{index}] => {}", ty))?;
          } else {
            f.write_fmt(format_args!(" {entry_index:5}: ____ => {}", ty))?;
          }
          f.write_str("\n")?;
        }
      }

      Ok(())
    }
  }

  impl TypeVarContext {
    pub fn new(db: &mut TypeDatabase) -> Self {
      Self {
        db:            db,
        vars:          Default::default(),
        type_slots:    Default::default(),
        generics:      Default::default(),
        ty_var_scopes: ScopeLookup::new(),
        var_scopes:    ScopeLookup::new(),
      }
    }

    pub fn rename_var(&mut self, old: IString, new: IString) -> bool {
      for scope_id in self.var_scopes.scope_stack.iter().rev() {
        let scope_id = *scope_id;
        for (scope_var_index, (name, var_index, slot_index)) in self.var_scopes.scopes[scope_id].1.iter().enumerate() {
          if *name == old {
            let slot_index = *slot_index;
            self.var_scopes.scopes[scope_id].1[scope_var_index].0 = new;
            self.vars[slot_index].mem_name = new;
            return true;
          }
        }
      }

      return false;
    }

    pub fn db_mut<'a>(&self) -> &'a mut TypeDatabase {
      unsafe { &mut *self.db }
    }

    pub fn db<'a>(&self) -> &'a TypeDatabase {
      unsafe { &*self.db }
    }

    pub fn get_type(&mut self, ty_name: IString) -> Option<(RumType, VarId)> {
      if let Some(index) = self.ty_var_scopes.get_index_from_id(ty_name) {
        Some(self.type_slots[index])
      } else if let Some(g_index) = self.db_mut().get_type_index(ty_name) {
        let index = self.type_slots.len();
        self.ty_var_scopes.add_index(ty_name, usize::MAX, index);
        self.type_slots.push((RumType::Undefined.to_aggregate_id(g_index), VarId::default()));
        Some(self.type_slots[index])
      } else {
        None
      }
    }

    pub fn get_type_local(&self, name: IString) -> Option<(RumType, VarId)> {
      if let Some(index) = self.ty_var_scopes.get_index_from_id(name) {
        Some(self.type_slots[index])
      } else {
        None
      }
    }

    pub fn get_member_var(&mut self, parent_var: VarId, member_name: MemberName) -> Option<&mut Variable> {
      let host = self as *mut _;

      let par_var = self.vars[parent_var];

      debug_assert!(par_var.ctx as usize == host as usize);

      let ts_index = self.type_slots.len();
      let par_index = parent_var.usize();

      let member_var_index = self.get_member_var_index(par_var, par_index, member_name);

      if member_var_index == self.type_slots.len() {
        let var = self.get_member_type_var(parent_var, member_name);
        var.temporary = false;
        var.mem_name = match member_name {
          MemberName::String(name) => name,
          _ => Default::default(),
        };
        var.mem_index = match member_name {
          MemberName::Index(id) => id,
          _ => usize::MAX,
        };
        var.par = VarId::new(par_index as u32);

        Some(var)
      } else {
        let var = self.type_slots[member_var_index].1;
        Some(&mut self.vars[var])
      }
    }

    fn get_member_type_var(&mut self, par_var_id: VarId, member_name: MemberName) -> &mut Variable {
      let par_var = &self.vars[par_var_id];

      let ptr1: RumType = RumType::Undefined.increment_pointer();

      if par_var.ty.is_generic() {
        let child_ty = self.create_generic_type(Default::default());
        return self.insert_new_var(Default::default(), child_ty.increment_pointer());
      }

      let entry = match par_var.ty.sub_type() {
        RumSubType::Generic => {
          let child_ty = self.create_generic_type(Default::default());
          return self.insert_new_var(Default::default(), child_ty.increment_pointer());
        }
        RumSubType::Aggregate => match par_var.ty.aggregate(&self.db()).unwrap() {
          Type::Array(array) => return self.insert_new_var_internal(Default::default(), array.element_type.increment_pointer(), true),
          Type::Structure(strct) => match member_name {
            MemberName::Index(index) => {
              if let Some(mem) = strct.members.iter().find(|m| m.original_index == index) {
                return self.insert_new_var_internal(Default::default(), mem.ty.increment_pointer(), true);
              } else {
                let child_ty = self.create_generic_type(Default::default());
                return self.insert_new_var(Default::default(), child_ty.increment_pointer());
              }
            }
            MemberName::String(member_name) => {
              if let Some(mem) = strct.members.iter().find(|m| m.name == member_name) {
                return self.insert_new_var_internal(Default::default(), mem.ty.increment_pointer(), true);
              } else {
                let child_ty = self.create_generic_type(Default::default());
                return self.insert_new_var(Default::default(), child_ty.increment_pointer());
              }
            }
          },
          tr => {
            panic!("invalid operation on {}", par_var.ty);
          }
        },

        tr => {
          panic!("invalid operation on {}", par_var.ty);
        }
      };
    }

    fn get_member_var_index(&mut self, par_var: Variable, par_index: usize, member_name: MemberName) -> usize {
      let par_index = par_var.id.usize();

      let member_var_index = self.vars.len();
      if par_var.mem_scope > 0 {
        let (_, scope) = &self.var_scopes.scopes[par_var.mem_scope];

        match member_name {
          MemberName::String(member_name) => {
            if let Some((.., index)) = scope.iter().find(|(.., id)| self.vars[*id].mem_name == member_name) {
              return *index;
            } else {
              self.var_scopes.scopes[par_var.mem_scope].1.push((member_name, usize::MAX, member_var_index));
              return member_var_index;
            }
          }
          MemberName::Index(index) => {
            if let Some((.., index)) = scope.iter().find(|(.., id)| self.vars[*id].mem_index == index) {
              return *index;
            } else {
              self.var_scopes.scopes[par_var.mem_scope].1.push((Default::default(), index, member_var_index));
              return member_var_index;
            }
          }
        }
      } else {
        self.var_scopes.push_scope();
        dbg!(self.var_scopes.current_scope, member_name);
        let par_scope = self.var_scopes.current_scope;
        self.vars[par_index].mem_scope = par_scope;

        match member_name {
          MemberName::Index(index) => {
            self.var_scopes.scopes[par_scope].1.push((Default::default(), index, member_var_index));
          }
          MemberName::String(name) => {
            self.var_scopes.scopes[par_scope].1.push((name, usize::MAX, member_var_index));
          }
        }
        self.var_scopes.pop_scope();

        return member_var_index;
      }
    }

    pub fn alias_variable(&mut self, existing_var: &Variable, new_name: IString) {
      self.var_scopes.add_index(new_name.into(), usize::MAX, existing_var.id.usize());
    }

    pub fn insert_new_var(&mut self, var_name: IString, ty: RumType) -> &mut Variable {
      self.insert_new_var_internal(var_name, ty, false)
    }

    pub fn insert_new_var_internal(&mut self, var_name: IString, ty: RumType, use_parent_scope: bool) -> &mut Variable {
      let var_index: usize = self.vars.len();
      let ts_index: usize = self.type_slots.len();

      let db = self.db_mut();

      if (!use_parent_scope) {
        self.var_scopes.add_index(var_name.into(), usize::MAX, var_index);
      }

      let var = Variable {
        ctx:         self as *const _ as *mut _,
        temporary:   false,
        id:          VarId::new(var_index as u32),
        par:         Default::default(),
        declaration: Default::default(),
        ty:          ty,
        mem_name:    var_name,
        mem_index:   usize::MAX,
        mem_scope:   0,
      };

      if let Some(generic_index) = ty.generic_id() {
        self.generics[generic_index].push(var.id);
      }

      self.vars.push(var);

      self.type_slots.push((ty, VarId::new(var_index as u32)));

      self.vars.last_mut().unwrap()
    }

    pub fn create_generic_type(self: &mut TypeVarContext, generic_type_name: IString) -> RumType {
      let generic_index = match (!generic_type_name.is_empty()).then_some(self.ty_var_scopes.get_index_from_id(generic_type_name)).flatten() {
        Some(index) => index,
        None => {
          let index = self.generics.len();
          self.generics.push(Vec::new());
          index
        }
      };
      RumType::Undefined.to_generic_id(generic_index)
    }

    pub fn get_var(&mut self, var_name: IString) -> Option<&mut Variable> {
      if let Some(index) = self.var_scopes.get_index_from_id(var_name) {
        Some(&mut self.vars[index])
      } else {
        None
      }
    }

    pub fn push_scope(&mut self) {
      self.ty_var_scopes.push_scope();
      self.var_scopes.push_scope();
    }

    pub fn pop_scope(&mut self) {
      self.ty_var_scopes.pop_scope();
      self.var_scopes.pop_scope();
    }
  }
}

mod type_slot {
  use std::fmt::{Debug, Display};

  use crate::{
    istring::IString,
    types::{ArrayType, BitFieldType, EnumType, RoutineType, ScopeType, StructType, UnionType},
  };

  use super::{ty::TypeRef, Type, TypeDatabase, TypeVarContext};
}

pub enum MemName {
  String(IString),
  Index(usize),
}

mod variable {
  use super::{TypeRef, TypeVarContext};
  use crate::{
    ir::ir_graph::{IRGraphId, VarId},
    istring::IString,
    types::RumType,
  };

  #[derive(Clone, Copy, Debug)]
  pub struct Variable {
    pub ctx:         *mut TypeVarContext,
    pub temporary:   bool,
    /// The index of the type slot for this variable.
    pub ty:          RumType,
    pub id:          VarId,
    pub par:         VarId,
    pub declaration: IRGraphId,
    pub mem_name:    IString,
    pub mem_index:   usize,
    pub mem_scope:   usize,
  }
}

mod scope {
  use crate::istring::IString;

  #[derive(Clone)]
  pub struct ScopeLookup {
    pub scopes:        Vec<(usize, Vec<(IString, usize, usize)>)>,
    pub free_scopes:   Vec<usize>,
    pub scope_stack:   Vec<usize>,
    pub current_scope: usize,
  }

  impl ScopeLookup {
    pub fn new() -> Self {
      Self {
        scopes:        vec![(usize::MAX, Default::default())],
        free_scopes:   Default::default(),
        scope_stack:   vec![0],
        current_scope: 0,
      }
    }

    pub fn add_index(&mut self, var_name: IString, var_index: usize, index: usize) {
      self.scopes[self.current_scope].1.push((var_name, var_index, index));
    }

    pub fn get_index_from_id(&self, name: IString) -> Option<usize> {
      for scope_index in self.scope_stack.iter().rev() {
        let (_, scope) = &self.scopes[*scope_index];

        for (candidate_name, candidate_index, slot_index) in scope.iter() {
          if *candidate_name == name {
            return Some(*slot_index);
          }
        }
      }

      return None;
    }

    pub fn get_index_from_index(&self, var_index: usize) -> Option<usize> {
      for scope_index in self.scope_stack.iter().rev() {
        let (_, scope) = &self.scopes[*scope_index];

        for (candidate_name, candidate_index, slot_index) in scope.iter() {
          if *candidate_index == var_index {
            return Some(*slot_index);
          }
        }
      }

      return None;
    }

    pub fn push_scope(&mut self) {
      if let Some(free_scope) = self.free_scopes.pop() {
        self.scopes[free_scope].0 = self.current_scope;
        self.scope_stack.push(free_scope);
        self.current_scope = free_scope;
      } else {
        self.scopes.push((self.current_scope, Default::default()));
        self.scope_stack.push(self.scopes.len() - 1);
        self.current_scope = self.scopes.len() - 1;
      }
    }

    pub fn pop_scope(&mut self) {
      if self.scope_stack.len() > 1 {
        let old_scope = self.scope_stack.pop().unwrap();
        if self.scopes[old_scope].1.is_empty() {
          self.free_scopes.push(old_scope);
        }

        self.current_scope = self.scope_stack[self.scope_stack.len() - 1];
      }
    }
  }
}

mod ty {
  use std::fmt::Display;

  use crate::{
    istring::IString,
    types::{ArrayType, BitFieldType, EnumType, FlagEnumType, RoutineType, ScopeType, StructType, UnionType},
  };

  use super::{RumType, TypeDatabase};

  pub enum Type {
    Scope(ScopeType),
    Structure(StructType),
    Union(UnionType),
    Routine(RoutineType),
    Syscall(IString),
    DebugCall(IString),
    Enum(EnumType),
    BitField(BitFieldType),
    Array(ArrayType),
    Flag(FlagEnumType),
  }

  #[derive(Clone, Copy)]
  pub enum TypeRef<'a> {
    UNRESOLVED { var_name: &'a IString, var_index: &'a u32, slot_index: &'a u32 },
    Scope(&'a ScopeType),
    Struct(&'a StructType),
    Union(&'a UnionType),
    Routine(&'a RoutineType),
    Enum(&'a EnumType),
    BitField(&'a BitFieldType),
    Array(&'a ArrayType),
    Syscall(IString),
    DebugCall(IString),
    Flag(&'a FlagEnumType),
    Undefined,
  }

  impl<'a> From<&'a Type> for TypeRef<'a> {
    fn from(value: &'a Type) -> Self {
      match value {
        Type::Scope(ty) => TypeRef::Scope(ty),
        Type::Structure(ty) => TypeRef::Struct(ty),
        Type::Union(ty) => TypeRef::Union(ty),
        Type::Routine(ty) => TypeRef::Routine(ty),
        Type::Enum(ty) => TypeRef::Enum(ty),
        Type::BitField(ty) => TypeRef::BitField(ty),
        Type::Array(ty) => TypeRef::Array(ty),
        Type::Syscall(name) => TypeRef::Syscall(*name),
        Type::DebugCall(name) => TypeRef::DebugCall(*name),
        Type::Flag(ty) => TypeRef::Flag(ty),
      }
    }
  }

  impl From<ScopeType> for Type {
    fn from(value: ScopeType) -> Self {
      Self::Scope(value)
    }
  }

  impl From<StructType> for Type {
    fn from(value: StructType) -> Self {
      Self::Structure(value)
    }
  }

  impl From<UnionType> for Type {
    fn from(value: UnionType) -> Self {
      Self::Union(value)
    }
  }

  impl From<RoutineType> for Type {
    fn from(value: RoutineType) -> Self {
      Self::Routine(value)
    }
  }

  impl From<EnumType> for Type {
    fn from(value: EnumType) -> Self {
      Self::Enum(value)
    }
  }

  impl From<BitFieldType> for Type {
    fn from(value: BitFieldType) -> Self {
      Self::BitField(value)
    }
  }

  impl From<ArrayType> for Type {
    fn from(value: ArrayType) -> Self {
      Self::Array(value)
    }
  }

  impl<'a> TypeRef<'a> {
    pub fn is_unresolved(&self) -> bool {
      matches!(self, TypeRef::UNRESOLVED { .. })
    }

    /*     pub fn byte_alignment(&self, ctx: &TypeDatabase) -> u64 {
      match self {
        TypeRef::Struct(struc) => struc.alignment,
        TypeRef::Primitive(prim) => prim.alignment(),
        TypeRef::Array(array) => array.element_type.ty_gb(ctx).byte_alignment(ctx),
        val => todo!("Byte alignment of {val}"),
      }
    }

    pub fn byte_size(&self, ctx: &TypeDatabase) -> u64 {
      match self {
        TypeRef::Struct(struc) => struc.size,
        TypeRef::Primitive(prim) => prim.byte_size(),
        TypeRef::Array(array) => array.element_type.ty_gb(ctx).byte_size(ctx) * array.size as u64,
        val => todo!("Byte size of {val}"),
      }
    } */

    /*     pub fn bit_size(&self, ctx: &TypeDatabase) -> u64 {
      self.byte_size(ctx) * 8
    } */

    pub fn base_type<'b>(&'b self, ctx: &'b TypeDatabase) -> TypeRef<'b> {
      match self {
        _ => *self,
      }
    }

    pub fn ptr_depth(&self) -> usize {
      match self {
        _ => 0,
      }
    }

    pub fn pointer_name(&self) -> IString {
      match self {
        _ => Default::default(),
      }
    }
  }

  /* impl From<ReferenceType> for Type {
    fn from(value: PrimitiveType) -> Self {
      Self::Reference(value)
    }
  } */

  impl std::fmt::Display for TypeRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      match self {
        Self::Scope(s) => f.write_fmt(format_args!("scope {}", s.name.to_str().as_str())),
        Self::Struct(s) => f.write_fmt(format_args!("struct {}", s.name.to_str().as_str())),
        Self::Flag(s) => f.write_fmt(format_args!("flag {}", s.name.to_str().as_str())),
        Self::Routine(s) => {
          if s.returns.is_empty() {
            f.write_str("proc ")?;
          } else {
            f.write_str("func ")?;
          }

          f.write_str(&s.name.to_string())?;

          f.write_str("(")?;

          for (name, .., param) in &s.parameters {
            f.write_fmt(format_args!("{param}"));
            f.write_str(", ")?;
          }

          f.write_str(")")?;

          if s.returns.len() > 0 {
            f.write_str(" => ");
            for ((ty, _)) in &s.returns {
              f.write_fmt(format_args!("{ty} "));
            }
          }

          Ok(())
        }
        Self::Union(s) => f.write_fmt(format_args!("union {}", s.name.to_str().as_str())),
        Self::Enum(s) => f.write_fmt(format_args!("enum {}", s.name.to_str().as_str())),
        Self::BitField(s) => f.write_fmt(format_args!("bf {}", s.name.to_str().as_str())),
        Self::Array(s) => f.write_fmt(format_args!("{}[{}]", s.name.to_str().as_str(), s.element_type)),
        Self::UNRESOLVED { var_name, var_index, slot_index } => f.write_fmt(format_args!("({var_name}[{var_index}]?)",)),
        Self::Syscall(name) => f.write_fmt(format_args!("sys::{}()", name)),
        Self::DebugCall(name) => f.write_fmt(format_args!("dbg::{}()", name)),
        Self::Undefined => f.write_str("undef"),
      }
    }
  }
}
