use std::{
  collections::HashMap,
  fmt::{Debug, Display},
  rc::Rc,
};

use crate::istring::IString;

use super::{ComplexType, Type};

pub struct TypeScope {
  pub(crate) types:         std::cell::Cell<(Vec<(IString, Type)>)>,
  pub(crate) scopes:        std::cell::Cell<Vec<(usize, Vec<usize>)>>,
  pub(crate) current_index: usize,
}

impl Clone for TypeScope {
  fn clone(&self) -> Self {
    Self {
      types:         std::cell::Cell::new((unsafe { &*self.types.as_ptr() }).clone()),
      scopes:        std::cell::Cell::new((unsafe { &*self.scopes.as_ptr() }).clone()),
      current_index: self.current_index,
    }
  }
}

impl Display for TypeScope {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("TypeContext");
    let scopes = unsafe { &*self.scopes.as_ptr() };
    let types = unsafe { &*self.types.as_ptr() };

    for (i, (par_i, ty_ids)) in scopes.iter().enumerate() {
      st.field(&format!("scope {i} <-> {par_i}: \n"), &ty_ids.iter().map(|id| types[*id].0.to_string() + " mapped to " + &types[*id].1.to_string()).collect::<Vec<_>>());
    }

    st.finish()
  }
}

impl Debug for TypeScope {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl TypeScope {
  pub fn new() -> Self {
    Self {
      types:         Default::default(),
      scopes:        std::cell::Cell::new(vec![(usize::MAX, Default::default())]),
      current_index: 0,
    }
  }

  pub fn is_empty(&self) -> bool {
    unsafe { &mut *self.types.as_ptr() }.is_empty()
  }

  pub fn add_scope(&self, parent_scope_index: usize) -> usize {
    let scopes = unsafe { &mut *self.scopes.as_ptr() };

    let new_scope_id = scopes.len();

    scopes.push((parent_scope_index, Default::default()));

    return new_scope_id;
  }

  pub fn set<'ty>(&'ty self, mut scope_index: usize, name: IString, ty: Type) -> Result<&'ty Type, &'ty Type> {
    if let Some((_, existing)) = self.get_internal(scope_index, name, false) {
      Err(existing)
    } else {
      let scopes = unsafe { &mut *self.scopes.as_ptr() };
      let types = unsafe { &mut *self.types.as_ptr() };
      let type_index = types.len();
      types.push((name, ty));
      scopes[scope_index].1.push(type_index);
      Ok(&types.last().unwrap().1)
    }
  }

  pub fn replace<'ty>(&'ty self, mut scope_index: usize, name: IString, ty: Type) -> &'ty Type {
    if let Some((existing_index, _)) = self.get_internal(scope_index, name, false) {
      let types = unsafe { &mut *self.types.as_ptr() };
      types[existing_index] = (name, ty);
      &types[existing_index].1
    } else {
      let scopes = unsafe { &mut *self.scopes.as_ptr() };
      let types = unsafe { &mut *self.types.as_ptr() };
      let type_index = types.len();
      types.push((name, ty));
      scopes[scope_index].1.push(type_index);
      &types.last().unwrap().1
    }
  }

  pub fn get(&self, mut scope_index: usize, name: IString) -> Option<(&Type)> {
    self.get_internal(scope_index, name, true).map(|t| &*t.1)
  }

  /*   pub fn get_mut<'ty>(&'ty self, mut scope_index: usize, name: IString) -> Option<&'ty mut Type> {
    self.get_internal(scope_index, name, true)
  } */

  fn get_internal<'ty>(&'ty self, mut scope_index: usize, name: IString, recurse: bool) -> Option<(usize, &'ty Type)> {
    let scopes = unsafe { &*self.scopes.as_ptr() };

    let mut index = scope_index;

    while index != usize::MAX {
      let (parent_index, types_ids) = &scopes[index];
      index = *parent_index;

      for ty_id in types_ids.iter() {
        let types = unsafe { &mut *self.types.as_ptr() };
        let ty = &mut types[*ty_id];
        if ty.0 == name {
          return Some((*ty_id, &ty.1));
        }
      }

      if !recurse {
        break;
      }
    }

    return None;
  }
}
