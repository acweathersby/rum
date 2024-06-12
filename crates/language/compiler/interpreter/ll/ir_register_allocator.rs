use rum_container::ArrayVec;
use rum_logger::todo_note;

use super::ir_types::{BlockId, GraphId, TypeInfo};

#[derive(Debug, Clone, Copy)]
pub struct RegisterEntry {
  pub var:   GraphId,
  pub ty:    TypeInfo,
  pub reg:   GraphId,
  pub block: BlockId,
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

#[derive(Debug, Clone, Copy)]
pub struct PreferredRegister {
  pub var: GraphId,
  pub reg: GraphId,
}

pub enum SetPreferredResult {
  Ok,
  Conflict,
}

#[derive(Debug, Clone)]
pub(super) struct RegisterAllocator {
  /// Map variable to the preferred register name
  pub mappings:        ArrayVec<16, RegisterEntry>,
  pub preferred:       ArrayVec<16, PreferredRegister>,
  pub preferred_index: u64,
  pub register_index:  u64,
}

impl RegisterAllocator {
  pub fn new() -> Self {
    Self {
      mappings:        Default::default(),
      register_index:  Default::default(),
      preferred:       Default::default(),
      preferred_index: Default::default(),
    }
  }

  pub fn set_preferred_register(&mut self, entry: &RegisterEntry) -> SetPreferredResult {
    if let Some(reg) = self.get_preferred_register(entry.var) {
      if reg.reg != entry.reg {
        return SetPreferredResult::Conflict;
      } else {
        return SetPreferredResult::Ok;
      }
    } else {
      self.preferred_index |= (1u64 << entry.reg.as_index());
      self.preferred.push(PreferredRegister { var: entry.var, reg: entry.reg });
      return SetPreferredResult::Ok;
    }
  }

  pub fn get_preferred_register(&self, var_name: GraphId) -> Option<PreferredRegister> {
    for var in self.preferred.iter() {
      if var.var == var_name {
        return Some(*var);
      }
    }
    None
  }

  pub fn push_allocation(&mut self, new: RegisterEntry) {
    debug_assert!(self.mappings.data_is_ordered());

    for (i, existing) in self.mappings.iter().enumerate() {
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
    self.mappings.insert_ordered(new).unwrap();
  }

  pub fn get_register_assigned_to_var(
    &self,
    var_name: GraphId,
    ty: TypeInfo,
  ) -> Option<RegisterEntry> {
    if let Some(i) = self.mappings.find_ordered(&RegisterEntry {
      reg: GraphId::default(),
      var: var_name,
      ty,
      block: Default::default(),
    }) {
      Some(self.mappings[i])
    } else {
      None
    }
  }

  pub fn get_register_for_var(&mut self, var_name: GraphId) -> AllocationProxy {
    // Check for existing allocation
    if let Some(i) = self.mappings.find_ordered(&RegisterEntry {
      reg:   GraphId::default(),
      var:   var_name,
      ty:    Default::default(),
      block: Default::default(),
    }) {
      let register = self.mappings[i].reg.as_index() as u64;
      self.register_index &= !(1 << register);
      AllocationProxy { register, prev: self.mappings.remove(i), allocator: self }
    } else if let Some(reg) = self.get_preferred_register(var_name) {
      self.allocate_register(reg.reg.as_index() as u64)
    } else {
      let mut free_register = None;
      for i in 0..16 {
        if ((self.register_index | self.preferred_index) & (1 << i)) == 0 {
          free_register = Some(i);
          break;
        }
      }

      if let Some(id) = free_register {
        self.allocate_register(id as u64)
      } else {
        // Evict a register.
        todo_note!("Create a register eviction policy");
        self.allocate_register(0)
      }
    }
  }

  fn allocate_register_for_var(
    &mut self,
    var_name: GraphId,
    ty: TypeInfo,
    block: BlockId,
  ) -> Option<RegisterEntry> {
    let mut free_register = None;

    for i in 0..16 {
      if ((self.register_index | self.preferred_index) & (1 << i)) == 0 {
        free_register = Some(i);
        break;
      }
    }

    if let Some(i) = free_register {
      if let Ok(reg_index) = self.mappings.insert_ordered(RegisterEntry {
        var: var_name,
        reg: GraphId(i as u32).as_register(),
        ty,
        block,
      }) {
        self.register_index |= 1 << i;
        Some(self.mappings[reg_index])
      } else {
        None
      }
    } else {
      None
    }
  }

  pub fn allocate_register(&mut self, register: u64) -> AllocationProxy {
    let reg_bit = 1 << register;
    let target_reg = GraphId(register as u32).as_register();
    if (reg_bit & self.register_index) > 0 {
      for (i, reg) in self.mappings.as_slice().iter().enumerate() {
        if reg.reg == target_reg {
          self.register_index = self.register_index & !reg_bit;
          return AllocationProxy { prev: self.mappings.remove(i), allocator: self, register };
        }
      }
      unreachable!("{register} {:032b}", self.register_index);
    } else {
      AllocationProxy { prev: None, allocator: self, register }
    }
  }

  fn set(
    &mut self,
    register: u64,
    var_name: GraphId,
    ty: TypeInfo,
    block: BlockId,
  ) -> Option<RegisterEntry> {
    let reg_bit = 1 << register;
    debug_assert!(var_name.is_var());
    debug_assert!(reg_bit & self.register_index == 0);
    if let Ok(reg_index) = self.mappings.insert_ordered(RegisterEntry {
      var: var_name,
      reg: GraphId(register as u32).as_register(),
      ty,
      block,
    }) {
      self.register_index |= 1 << register;
      Some(self.mappings[reg_index])
    } else {
      None
    }
  }

  pub fn get_assignements(&self) -> ArrayVec<16, RegisterEntry> {
    self.mappings.clone()
  }
}

trait RegisterAllocatorSpecialization {}

pub struct AllocationProxy<'a> {
  allocator: &'a mut RegisterAllocator,
  prev:      Option<RegisterEntry>,
  register:  u64,
}

impl<'a> AllocationProxy<'a> {
  pub fn get_previous(&self) -> Option<RegisterEntry> {
    self.prev
  }

  pub fn reg(&self) -> GraphId {
    GraphId(self.register as u32).as_register()
  }

  pub fn set_current(self, var: GraphId, ty: TypeInfo, block: BlockId) -> Option<RegisterEntry> {
    self.allocator.set(self.register, var, ty, block)
  }

  pub fn revert(self) {
    if let Some(RegisterEntry { var, ty, reg, block }) = self.prev {
      self.allocator.set(reg.as_index() as u64, var, ty, block);
    }
  }
}
