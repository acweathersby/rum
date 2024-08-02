#![allow(unused)]

use rum_istring::CachedString;
mod base;
mod bitsize;
mod complex;
mod primitive;

mod const_val;

use crate::IString;
pub use base::*;
pub use bitsize::*;
pub use complex::*;
pub use const_val::*;
pub use primitive::*;
use std::collections::HashMap;

#[derive(Debug)]
pub enum VariableEntry {
  WeakCandidates { types: Vec<Type>, generics: Vec<Type> },
  StrongCandidate(Type),
}

#[derive(Debug)]
pub struct TypeScopes {
  scopes:        Vec<TypeScopeEntry>,
  current_index: usize,
}

impl TypeScopes {
  pub fn new() -> Self {
    Self {
      scopes:        vec![TypeScopeEntry { entries: Default::default(), parent_index: None }],
      current_index: 0,
    }
  }

  pub fn add_scope(&mut self, parent_scope_index: usize) -> usize {
    let out_index = self.scopes.len();

    self.scopes.push(TypeScopeEntry { parent_index: Some(parent_scope_index), entries: Default::default() });

    out_index
  }

  pub fn set(&mut self, mut scope_index: usize, name: IString, ty: ComplexType) {
    if let Some(scope) = self.scopes.get_mut(scope_index) {
      if scope.entries.contains_key(&name) {
        panic!("The type {name:?} has already been declared")
      } else {
        scope.entries.insert(name, Box::new(ty));
      }
    } else {
      panic!("Scope {scope_index} does not exist");
    }
  }

  pub fn get(&self, mut scope_index: usize, name: IString) -> Option<&ComplexType> {
    loop {
      let scope = &self.scopes[scope_index];

      if let Some(var) = scope.entries.get(&name) {
        return Some(var);
      } else if let Some(par_index) = scope.parent_index {
        scope_index = par_index;
      } else {
        return None;
      }
    }
  }
}

#[derive(Debug)]
struct TypeScopeEntry {
  parent_index: Option<usize>,
  entries:      HashMap<IString, Box<ComplexType>>,
}

struct VarScope<'v_scope> {
  t_scope_index: usize,
  type_scope:    *mut TypeScopes,
  parent:        Option<&'v_scope mut VarScope<'v_scope>>,
  variables:     HashMap<IString, VariableEntry>,
}

impl<'v_scope> VarScope<'v_scope> {
  pub fn new(type_scope: &mut TypeScopes, scope_index: usize) -> Self {
    Self {
      t_scope_index: scope_index,
      type_scope:    type_scope,
      parent:        None,
      variables:     Default::default(),
    }
  }

  pub fn get(&self, name: IString) -> Option<&VariableEntry> {
    if let Some(var) = self.variables.get(&name) {
      Some(var)
    } else if let Some(par) = &self.parent {
      par.get(name)
    } else {
      None
    }
  }

  pub fn set(&mut self, name: IString, var: VariableEntry) {
    self.variables.insert(name, var);
  }

  pub fn create_child_scope<'new_v_scope: 'v_scope>(&'new_v_scope mut self) -> VarScope<'new_v_scope> {
    Self {
      t_scope_index: self.ts().add_scope(self.t_scope_index),
      type_scope:    self.type_scope,
      parent:        Some(self),
      variables:     Default::default(),
    }
  }

  pub fn add_type(&self, name: IString, ty: ComplexType) {
    let t_scope = self.ts();
    t_scope.scopes[self.t_scope_index].entries.insert(name, Box::new(ty));
  }

  fn ts(&self) -> &mut TypeScopes {
    unsafe { &mut *self.type_scope }
  }
}

#[test]
fn test_type_scope() {
  let mut t_scope = TypeScopes::new();

  t_scope.set(t_scope.current_index, "test".to_token(), ComplexType::Struct(StructType { name: "test".intern(), members: Default::default(), size: 0, alignment: 0 }));

  dbg!(t_scope);
}

#[test]
fn test_variable_scope() {
  let mut t_scope = TypeScopes::new();
  let mut v_scope = VarScope::new(&mut t_scope, 0);

  let mut c_v_scope = v_scope.create_child_scope();

  c_v_scope.set("test".intern(), VariableEntry::WeakCandidates { types: vec![PrimitiveType::i64.into()], generics: vec![] });

  c_v_scope.add_type("main".intern(), ComplexType::Struct(StructType { name: "main".to_token(), size: 0, alignment: 0, members: Default::default() }));

  if let Some(var) = c_v_scope.get("test".to_token()) {
    dbg!(var);
    dbg!(t_scope);
  } else {
    panic!("Variable test not found");
  }
}
