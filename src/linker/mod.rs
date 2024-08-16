use crate::istring::IString;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Endianess {
  Big,
  Little,
}

#[derive(Clone, Copy)]
pub enum LinkType {
  Block(IString),
  Routine(IString),
  DBGRoutine(IString),
}

#[derive(Clone, Copy)]
pub struct RetargetingLink {
  pub binary_offset: usize,
  pub byte_size:     u64,
  pub endianess:     Endianess,
  pub link_type:     LinkType,
}

impl RetargetingLink {
  pub fn replace<T: Sized>(&self, ptr: *mut u8, val: T) {
    let size_of_t = std::mem::size_of::<T>();

    debug_assert_eq!(size_of_t, self.byte_size as usize);

    let src_ptr = unsafe { std::mem::transmute::<&T, *const u8>(&val) };

    unsafe { std::ptr::copy(src_ptr, ptr.offset(self.binary_offset as isize), size_of_t) };
  }
}

pub struct LinkableBinary {
  pub name:     IString,
  pub binary:   Vec<u8>,
  pub link_map: Vec<RetargetingLink>,
}
