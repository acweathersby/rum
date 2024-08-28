use super::{ArrayType, BitFieldType, EnumType, PrimitiveType, RoutineType, ScopeType, StructType, UnionType};
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
  pub types: Vec<Box<Type>>,
  /// Maps a typename to a to a databese type type.
  lookup:    BTreeMap<IString, usize>,
}

impl TypeDatabase {
  pub fn new() -> Self {
    Self { types: Default::default(), lookup: Default::default() }
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

  pub fn get_type_mut<'a>(&mut self, name: IString) -> Option<(&'a mut Type, TypeSlot)> {
    if let Some(index) = self.get_type_index(name) {
      let ptr = self.types.as_mut_ptr();
      let val = unsafe { ptr.offset(index as isize) };
      Some((unsafe { &mut *val }, TypeSlot::GlobalIndex(0, index as u32)))
    } else {
      None
    }
  }

  pub fn get_type<'a>(&self, name: IString) -> Option<(&'a Type, TypeSlot)> {
    if let Some(index) = self.get_type_index(name) {
      let ptr = self.types.as_ptr();
      let val = unsafe { ptr.offset(index as isize) };
      Some((unsafe { &*val }, TypeSlot::GlobalIndex(0, index as u32)))
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

  use crate::{
    ir::ir_graph::VarId,
    istring::{CachedString, IString},
    types::{type_context::MemberName, type_database::ty::TypeRef, ScopeLookup},
  };

  use super::{ty::Type, variable::Variable, TypeDatabase, TypeSlot};

  #[derive(Clone)]
  pub struct TypeVarContext {
    pub db:            *mut TypeDatabase,
    pub vars:          Vec<Variable>,
    pub type_slots:    Vec<TypeSlot>,
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
            f.write_fmt(format_args!("  {index:5}: {}[{}]({})", var.par, var.mem_index, var.ty_slot.ty(self)))?;
          } else {
            f.write_fmt(format_args!("  {index:5}: {}.{}({})", var.par, var.mem_name, var.ty_slot.ty(self)))?;
          }
        } else {
          f.write_fmt(format_args!("  {index:5}: {} {} ", var.mem_name, var.ty_slot.ty(self)))?;
        }
        f.write_str("\n")?;
      }

      f.write_str("  types:\n")?;

      for (index, scope) in self.ty_var_scopes.scopes.iter().enumerate() {
        f.write_fmt(format_args!("  scope {index}:\n"))?;
        for (.., entry_index) in &scope.1 {
          let slot = &self.type_slots[*entry_index];
          let ty = slot.ty(self);
          f.write_fmt(format_args!(" {entry_index:5}: {}", ty))?;
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

    pub fn get_type(&mut self, ty_name: IString) -> Option<TypeSlot> {
      if let Some(index) = self.ty_var_scopes.get_index_from_id(ty_name) {
        Some(self.convert_into_out_ctx_slot(index))
      } else if let Some(g_index) = self.db_mut().get_type_index(ty_name) {
        let index = self.type_slots.len();
        self.ty_var_scopes.add_index(ty_name, usize::MAX, index);
        self.type_slots.push(TypeSlot::GlobalIndex(0, g_index as u32));
        Some(self.convert_into_out_ctx_slot(index))
      } else {
        None
      }
    }

    pub fn get_type_local(&self, name: IString) -> Option<TypeSlot> {
      if let Some(index) = self.ty_var_scopes.get_index_from_id(name) {
        Some(self.convert_into_out_ctx_slot(index))
      } else {
        None
      }
    }

    fn insert_ty_internal(self: &mut TypeVarContext, ty_name: IString, ty: Type) -> TypeSlot {
      let index = self.type_slots.len();
      self.ty_var_scopes.add_index(ty_name, usize::MAX, index);
      let g_index = self.db_mut().insert_anonymous_type(ty);
      self.type_slots.push(TypeSlot::GlobalIndex(0, g_index as u32));
      self.convert_into_out_ctx_slot(index)
    }

    pub fn insert_generic(self: &mut TypeVarContext, name: MemberName) -> TypeSlot {
      if let Some(index) = match name {
        MemberName::Index(index) => self.ty_var_scopes.get_index_from_index(index),
        MemberName::String(name) => self.ty_var_scopes.get_index_from_id(name),
      } {
        self.convert_into_out_ctx_slot(index)
      } else {
        let index = self.type_slots.len();

        let slot = match name {
          MemberName::Index(mem_index) => {
            self.ty_var_scopes.add_index(Default::default(), mem_index, index);
            TypeSlot::UNRESOLVED {
              ptr_depth:  0,
              var_name:   Default::default(),
              var_index:  mem_index as u32,
              slot_index: index as u32,
            }
          }
          MemberName::String(name) => {
            self.ty_var_scopes.add_index(name, usize::MAX, index);
            TypeSlot::UNRESOLVED { ptr_depth: 0, var_name: name, var_index: u32::MAX, slot_index: index as u32 }
          }
        };

        self.type_slots.push(slot);
        self.convert_into_out_ctx_slot(index)
      }
    }

    fn convert_into_out_ctx_slot(self: &TypeVarContext, index: usize) -> TypeSlot {
      match self.type_slots[index] {
        TypeSlot::UNRESOLVED { .. } => TypeSlot::CtxIndex(0, index as u32),
        slot => slot,
      }
    }

    pub fn get_or_add_type(&mut self, ty_name: IString, ty_string: Type) -> TypeSlot {
      if let Some(ty_index) = self.get_type(ty_name) {
        ty_index
      } else {
        self.insert_ty_internal(ty_name.into(), ty_string)
      }
    }

    pub fn get_or_add_type_local(&mut self, ty_name: IString, ty_string: Type) -> TypeSlot {
      if let Some(ty_index) = self.get_type_local(ty_name) {
        ty_index
      } else {
        self.insert_ty_internal(ty_name.into(), ty_string)
      }
    }

    pub fn insert_ty(&mut self, ty_name: IString, ty_string: Type) -> TypeSlot {
      if let Some(ty_index) = self.ty_var_scopes.get_index_from_id(ty_name) {
        panic!("Type {} => {} has already been added into this context", ty_name.to_string(), TypeRef::from(&ty_string));
      } else {
        self.insert_ty_internal(ty_name.into(), ty_string)
      }
    }

    pub fn insert_var(&mut self, var_name: IString, slot_index: TypeSlot) -> &mut Variable {
      debug_assert!(!matches!(slot_index, TypeSlot::UNRESOLVED { .. }));
      let index: usize = self.vars.len();
      let db = self.db_mut();

      self.var_scopes.add_index(var_name.into(), usize::MAX, index);

      let var = Variable {
        ctx:       self as *const _ as *mut _,
        temporary: false,
        id:        VarId::new(index as u32),
        par:       Default::default(),
        reference: Default::default(),
        ty_slot:   slot_index,
        mem_name:  var_name,
        mem_index: usize::MAX,
        mem_scope: 0,
      };

      self.vars.push(var);

      self.vars.last_mut().unwrap()
    }

    pub fn get_var(&mut self, var_name: IString) -> Option<&mut Variable> {
      if let Some(index) = self.var_scopes.get_index_from_id(var_name) {
        Some(&mut self.vars[index])
      } else {
        None
      }
    }

    pub fn get_var_member(&mut self, par_var: VarId, member_name: MemberName) -> Option<&mut Variable> {
      let host = self as *mut _;

      let par_var = self.vars[par_var];

      debug_assert!(par_var.ctx as usize == host as usize);

      let member_index = self.vars.len();

      let par_index = par_var.id.usize();

      if par_var.mem_scope > 0 {
        let (_, scope) = &self.var_scopes.scopes[par_var.mem_scope];
        match member_name {
          MemberName::String(member_name) => {
            if let Some((.., index)) = scope.iter().find(|(.., id)| self.vars[*id].mem_name == member_name) {
              return Some(&mut self.vars[*index]);
            } else {
              self.var_scopes.scopes[par_var.mem_scope].1.push((member_name, usize::MAX, member_index));
            }
          }
          MemberName::Index(index) => {
            if let Some((.., index)) = scope.iter().find(|(.., id)| self.vars[*id].mem_index == index) {
              return Some(&mut self.vars[*index]);
            } else {
              self.var_scopes.scopes[par_var.mem_scope].1.push((Default::default(), index, member_index));
            }
          }
        }
      } else {
        self.var_scopes.push_scope();
        self.vars[par_index].mem_scope = self.var_scopes.current_scope;

        match member_name {
          MemberName::Index(index) => {
            self.var_scopes.scopes[self.vars[par_index].mem_scope].1.push((Default::default(), index, member_index));
          }
          MemberName::String(name) => {
            self.var_scopes.scopes[self.vars[par_index].mem_scope].1.push((name, usize::MAX, member_index));
          }
        }

        self.var_scopes.pop_scope();
      }

      let entry = self.get_member_type(par_var.id, member_name).unwrap_or_default();

      let var = Variable {
        ctx:       self as *const _ as *mut _,
        temporary: false,
        id:        VarId::new(member_index as u32),
        par:       VarId::new(par_index as u32),
        reference: Default::default(),
        ty_slot:   entry,
        mem_name:  match member_name {
          MemberName::String(name) => name,
          _ => Default::default(),
        },
        mem_index: match member_name {
          MemberName::Index(id) => id,
          _ => usize::MAX,
        },
        mem_scope: 0,
      };

      self.vars.push(var);

      Some(&mut self.vars[member_index])
    }

    pub fn get_member_type(&mut self, par_var_id: VarId, member_name: MemberName) -> Option<TypeSlot> {
      let par_var = &self.vars[par_var_id];
      let entry = match par_var.ty_slot.ty_base(self) {
        TypeRef::Undefined | TypeRef::UNRESOLVED { .. } => {
          self.ty_var_scopes.push_scope();
          let entry = self.insert_generic(member_name);
          self.ty_var_scopes.pop_scope();
          entry
        }
        TypeRef::Array(array) => array.element_type.increment_ptr(),
        TypeRef::Struct(strct) => match member_name {
          MemberName::Index(index) => {
            if let Some(mem) = strct.members.iter().find(|m| m.original_index == index) {
              mem.ty.increment_ptr()
            } else {
              return None;
            }
          }
          MemberName::String(member_name) => {
            if let Some(mem) = strct.members.iter().find(|m| m.name == member_name) {
              mem.ty.increment_ptr()
            } else {
              return None;
            }
          }
        },
        tr => {
          println!("invalid operation on {tr}");
          return None;
        }
      };
      Some(entry)
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
    types::{ArrayType, BitFieldType, EnumType, PrimitiveType, RoutineType, ScopeType, StructType, UnionType},
  };

  use super::{ty::TypeRef, Type, TypeDatabase, TypeVarContext};

  #[derive(Clone, Copy, Default)]
  pub enum TypeSlot {
    // Mapping to a global type. Stores index to a user type.
    GlobalIndex(u32, u32),
    // Mapping to a local context's type table.
    CtxIndex(u32, u32),
    // Mapping to a primitive type. Stores the primitive type.
    Primitive(u32, PrimitiveType),
    // Slot is unresolved. Stores the index to the local ctx's type_slot,
    UNRESOLVED {
      var_name:   IString,
      var_index:  u32,
      slot_index: u32,
      ptr_depth:  u32,
    },

    #[default]
    None,
  }

  impl TypeSlot {
    pub fn increment_ptr(&self) -> Self {
      match *self {
        TypeSlot::Primitive(ptr_depth, prim) => Self::Primitive(ptr_depth + 1, prim),
        TypeSlot::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => TypeSlot::UNRESOLVED { ptr_depth: ptr_depth + 1, slot_index, var_index, var_name },
        TypeSlot::GlobalIndex(ptr_depth, index) => TypeSlot::GlobalIndex(ptr_depth + 1, index),
        TypeSlot::CtxIndex(ptr_depth, index) => TypeSlot::CtxIndex(ptr_depth + 1, index),
        TypeSlot::None => TypeSlot::None,
      }
    }

    pub fn decrement_ptr(&self) -> Self {
      match *self {
        TypeSlot::Primitive(ptr_depth, prim) => Self::Primitive(ptr_depth.checked_sub(1).unwrap_or_default(), prim),
        TypeSlot::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => {
          TypeSlot::UNRESOLVED { ptr_depth: ptr_depth.checked_sub(1).unwrap_or_default(), slot_index, var_index, var_name }
        }
        TypeSlot::GlobalIndex(ptr_depth, index) => TypeSlot::GlobalIndex(ptr_depth.checked_sub(1).unwrap_or_default(), index),
        TypeSlot::CtxIndex(ptr_depth, index) => TypeSlot::CtxIndex(ptr_depth.checked_sub(1).unwrap_or_default(), index),
        TypeSlot::None => TypeSlot::None,
      }
    }

    /// The number of pointer dereferences required to get to the base value of this type.
    pub fn ptr_depth(&self, ctx: &TypeVarContext) -> usize {
      let ref_depth = match *self {
        TypeSlot::Primitive(ptr_depth, prim) => ptr_depth as usize,
        TypeSlot::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => ptr_depth as usize,
        TypeSlot::GlobalIndex(ptr_depth, index) => ptr_depth as usize,
        TypeSlot::CtxIndex(ptr_depth, index) => ptr_depth as usize,
        TypeSlot::None => 0,
      };

      let base_depth = self.ty(ctx).ptr_depth();

      ref_depth + base_depth
    }

    pub fn is_unresolved(&self) -> bool {
      matches!(self, TypeSlot::UNRESOLVED { .. })
    }

    pub fn is_primitive(&self) -> bool {
      matches!(self, TypeSlot::Primitive(..))
    }

    pub fn resolve_to_outer_slot(&self, ctx: &TypeVarContext) -> TypeSlot {
      match *self {
        Self::CtxIndex(0, index) => ctx.type_slots[index as usize],
        slot => slot,
      }
    }

    pub fn ty_base<'a>(&'a self, ctx: &'a TypeVarContext) -> TypeRef<'a> {
      match self.ty(ctx) {
        TypeRef::Pointer(.., slot) => slot.ty_base(ctx),
        ty => ty,
      }
    }

    pub fn ty_pointer_name<'a>(&'a self, ctx: &'a TypeVarContext) -> Option<IString> {
      match self.ty(ctx) {
        TypeRef::Pointer(name, slot) => Some(name),
        ty => None,
      }
    }

    pub fn ty<'a>(&'a self, ctx: &'a TypeVarContext) -> TypeRef<'a> {
      match self {
        TypeSlot::Primitive(ptr_depth, prim) => TypeRef::Primitive(prim),
        TypeSlot::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => TypeRef::UNRESOLVED { slot_index, var_index, var_name },
        TypeSlot::GlobalIndex(ptr_depth, index) => ctx.db().types[*index as usize].as_ref().into(),
        TypeSlot::CtxIndex(ptr_depth, index) => ctx.type_slots[*index as usize].ty(ctx),
        TypeSlot::None => TypeRef::Undefined,
      }
    }

    pub fn ty_gb<'a>(&'a self, db: &'a TypeDatabase) -> TypeRef<'a> {
      match self {
        TypeSlot::Primitive(ptr_depth, prim) => TypeRef::Primitive(prim),
        TypeSlot::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => TypeRef::UNRESOLVED { slot_index, var_index, var_name },
        TypeSlot::GlobalIndex(ptr_depth, index) => db.types[*index as usize].as_ref().into(),
        TypeSlot::CtxIndex(ptr_depth, index) => TypeRef::Undefined,
        TypeSlot::None => TypeRef::Undefined,
      }
    }
  }

  impl Display for TypeSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      match self {
        Self::Primitive(ptr_depth, prim) => Display::fmt(prim, f),
        Self::GlobalIndex(ptr_depth, index) => f.write_fmt(format_args!("ty @ {index}")),
        Self::CtxIndex(ptr_depth, index) => f.write_fmt(format_args!("ty @ local {index}")),
        Self::UNRESOLVED { ptr_depth, slot_index, var_index, var_name } => f.write_fmt(format_args!("{var_name}[{var_index}]? @ local {slot_index}")),
        Self::None => Ok(()),
      }
    }
  }

  impl Debug for TypeSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      Display::fmt(&self, f)
    }
  }
}

pub enum MemName {
  String(IString),
  Index(usize),
}

mod variable {
  use super::{TypeRef, TypeSlot, TypeVarContext};
  use crate::{
    ir::ir_graph::{IRGraphId, VarId},
    istring::IString,
  };

  #[derive(Clone, Copy, Debug)]
  pub struct Variable {
    pub ctx:       *mut TypeVarContext,
    pub temporary: bool,
    pub ty_slot:   TypeSlot,
    pub id:        VarId,
    pub par:       VarId,
    pub reference: IRGraphId,
    pub mem_name:  IString,
    pub mem_index: usize,
    pub mem_scope: usize,
  }

  impl Variable {
    pub fn ty<'a>(&'a self) -> TypeRef<'a> {
      let par = unsafe { &*self.ctx };
      self.ty_slot.ty(par)
    }
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
    types::{ArrayType, BitFieldType, EnumType, FlagEnumType, PrimitiveType, RoutineType, ScopeType, StructType, UnionType},
  };

  use super::{TypeDatabase, TypeSlot};

  pub enum Type {
    Pointer(IString, u32, TypeSlot),
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
    Primitive(&'a PrimitiveType),
    UNRESOLVED { var_name: &'a IString, var_index: &'a u32, slot_index: &'a u32 },
    Pointer(IString, &'a TypeSlot),
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
        Type::Pointer(name, depth, ty) => TypeRef::Pointer(*name, ty),
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
    pub fn is_primitive(&self) -> bool {
      matches!(self, TypeRef::Primitive(..))
    }

    pub fn is_pointer(&self) -> bool {
      matches!(self, TypeRef::Pointer(..))
    }

    pub fn is_unresolved(&self) -> bool {
      matches!(self, TypeRef::UNRESOLVED { .. })
    }

    pub fn byte_alignment(&self, ctx: &TypeDatabase) -> u64 {
      match self {
        TypeRef::Struct(struc) => struc.alignment,
        TypeRef::Primitive(prim) => prim.alignment(),
        TypeRef::Pointer(..) => 8,
        TypeRef::Array(array) => array.element_type.ty_gb(ctx).byte_alignment(ctx),
        val => todo!("Byte alignment of {val}"),
      }
    }

    pub fn byte_size(&self, ctx: &TypeDatabase) -> u64 {
      match self {
        TypeRef::Struct(struc) => struc.size,
        TypeRef::Primitive(prim) => prim.byte_size(),
        TypeRef::Pointer(..) => 8,
        TypeRef::Array(array) => array.element_type.ty_gb(ctx).byte_size(ctx) * array.size as u64,
        val => todo!("Byte size of {val}"),
      }
    }

    pub fn bit_size(&self, ctx: &TypeDatabase) -> u64 {
      self.byte_size(ctx) * 8
    }

    pub fn base_type<'b>(&'b self, ctx: &'b TypeDatabase) -> TypeRef<'b> {
      match self {
        TypeRef::Pointer(.., ty) => ty.ty_gb(ctx),
        _ => *self,
      }
    }

    pub fn ptr_depth(&self) -> usize {
      match self {
        TypeRef::Pointer(..) => 1,
        _ => 0,
      }
    }

    pub fn pointer_name(&self) -> IString {
      match self {
        TypeRef::Pointer(name, ..) => *name,
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
        // Self::Reference(ptr) => f.write_fmt(format_args!("&{}", ptr.base_type)),
        Self::Pointer(ty, base) => {
          if ty.is_empty() {
            f.write_fmt(format_args!("*{}", base))
          } else {
            f.write_fmt(format_args!("{}* {}", ty.to_string(), base))
          }
        }
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
        Self::Primitive(prim) => Display::fmt(prim, f),
        Self::Syscall(name) => f.write_fmt(format_args!("sys::{}()", name)),
        Self::DebugCall(name) => f.write_fmt(format_args!("dbg::{}()", name)),
        Self::Undefined => f.write_str("undef"),
      }
    }
  }
}
