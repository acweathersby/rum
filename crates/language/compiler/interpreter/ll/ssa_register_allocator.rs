use rum_container::ArrayVec;

use super::types::GraphId;

#[derive(Debug, Clone, Copy)]
pub struct RegisterEntry {
  pub var: GraphId,
  pub reg: GraphId,
}

impl PartialEq for RegisterEntry {
  fn eq(&self, other: &Self) -> bool {
    self.var.eq(&other.var)
  }
}
impl Eq for RegisterEntry {}
impl PartialOrd for RegisterEntry {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.var.partial_cmp(&other.var)
  }
}
impl Ord for RegisterEntry {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.var.cmp(&other.var)
  }
}

#[derive(Debug, Clone)]
pub(super) struct RegisterAllocator {
  pub allocations:    ArrayVec<16, RegisterEntry>,
  pub register_index: usize,
}

impl RegisterAllocator {
  pub fn new() -> Self {
    Self { allocations: Default::default(), register_index: Default::default() }
  }
  pub fn push_allocation(&mut self, new: RegisterEntry) {
    debug_assert!(self.allocations.data_is_ordered());

    for (i, existing) in self.allocations.iter().enumerate() {
      if existing.reg == new.reg {
        if existing.var == new.var {
          // Already inserted.
          return;
        } else {
          todo!(
            "
  Handle deconfliction of register. 
    Most likely by returning a value indicating that a store and load will need to occur; 
          
    new {new:?}
          
    existing: {existing:?}"
          )
        }
      }
    }
    self.register_index |= 1 << new.reg.raw_val();
    self.allocations.insert_ordered(new).unwrap();
  }

  pub fn get_register(&self, var_name: GraphId) -> Option<RegisterEntry> {
    if let Some(i) =
      self.allocations.find_ordered(&RegisterEntry { reg: GraphId::default(), var: var_name })
    {
      Some(self.allocations[i])
    } else {
      None
    }
  }

  pub fn allocate_register(&mut self, var_name: GraphId) -> Option<RegisterEntry> {
    let mut free_register = None;

    for i in 0..16 {
      if self.register_index & 1 << i == 0 {
        free_register = Some(i);
        break;
      }
    }

    if let Some(i) = free_register {
      if let Ok(i) = self
        .allocations
        .insert_ordered(RegisterEntry { var: var_name, reg: GraphId(i as u32).as_register() })
      {
        self.register_index |= 1 << i;
        Some(self.allocations[i])
      } else {
        None
      }
    } else {
      None
    }
  }
  pub fn get_assignements(&self) -> ArrayVec<16, RegisterEntry> {
    self.allocations.clone()
  }
}
