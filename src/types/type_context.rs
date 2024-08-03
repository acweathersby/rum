use std::collections::HashMap;

  use crate::istring::IString;

  use super::ComplexType;

  pub struct TypeContext {
pub(crate) scopes:        Vec<std::cell::Cell<Box<TypeScopeEntry>>>,
pub(crate) current_index: usize,
  }

  impl TypeContext {
pub fn new() -> Self {
  Self {
    scopes:        vec![std::cell::Cell::new(Box::new(TypeScopeEntry { entries: Default::default(), parent_index: None }))],
    current_index: 0,
  }
}

pub fn add_scope(&mut self, parent_scope_index: usize) -> usize {
  let out_index = self.scopes.len();

  self.scopes.push(std::cell::Cell::new(Box::new(TypeScopeEntry { parent_index: Some(parent_scope_index), entries: Default::default() })));

  out_index
}

pub fn set(&self, mut scope_index: usize, name: IString, ty: ComplexType) {
  if let Some(scope) = self.scopes.get(scope_index) {
    let scope = unsafe { &mut *scope.as_ptr() };
    if scope.entries.contains_key(&name) {
      panic!("The type {name:?} has already been declared")
    } else {
      scope.entries.insert(name, std::cell::Cell::new(Box::new(ty)));
    }
  } else {
    panic!("Scope {scope_index} does not exist");
  }
}

pub fn get(&self, mut scope_index: usize, name: IString) -> Option<&ComplexType> {
  loop {
    let scope = &self.scopes[scope_index];
    let scope = unsafe { &mut *scope.as_ptr() };
    if let Some(var) = scope.entries.get(&name) {
      return Some(unsafe { &*var.as_ptr() });
    } else if let Some(par_index) = scope.parent_index {
      scope_index = par_index;
    } else {
      return None;
    }
  }
}

pub fn get_mut<'ty>(&self, mut scope_index: usize, name: IString) -> Option<&'ty mut ComplexType> {
  loop {
    let scope = &self.scopes[scope_index];
    let scope = unsafe { &mut *scope.as_ptr() };
    if let Some(var) = scope.entries.get(&name) {
      return Some(unsafe { &mut *var.as_ptr() });
    } else if let Some(par_index) = scope.parent_index {
      scope_index = par_index;
    } else {
      return None;
    }
  }
}
  }

  pub(crate) struct TypeScopeEntry {
pub(crate) parent_index: Option<usize>,
pub(crate) entries:      HashMap<IString, std::cell::Cell<Box<ComplexType>>>,
  }
