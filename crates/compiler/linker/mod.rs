use std::collections::HashMap;

use crate::targets::x86::{
  print_instructions,
  x86_binary_writer::{BinaryFunction, PatchType},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Endianess {
  Big,
  Little,
}

pub fn link(mut bin_functs: Vec<BinaryFunction>) -> (usize, Vec<u8>) {
  let total_size = bin_functs.iter().fold(0, |size, f| f.byte_size() + size);

  let mut binary = vec![0u8; total_size];
  let mut offset = 0;

  let mut id_offset = HashMap::new();

  let mut entry_offset = -1isize;

  for bin_funct in &mut bin_functs {
    id_offset.insert(bin_funct.id, offset + bin_funct.entry_offset);

    if entry_offset < 0 {
      entry_offset = bin_funct.entry_offset as _;
    }

    for (byte, byte_entry) in bin_funct.binary.iter().cloned().zip(&mut binary[offset..offset + bin_funct.byte_size()]) {
      *byte_entry = byte;
    }

    for (pp_offset, _) in &mut bin_funct.patch_points {
      *pp_offset += offset;
    }

    offset += bin_funct.byte_size();
  }

  for bin_funct in bin_functs {
    for (instr_offset, pp) in bin_funct.patch_points {
      match pp {
        PatchType::Function(id) => {
          if let Some(offset) = id_offset.get(&id) {
            let relative_offset = *offset as i32 - instr_offset as i32;
            let ptr = binary[instr_offset - 4..].as_mut_ptr();
            unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
          } else {
            panic!("Could not find offset for {id:?}")
          }
        }
      }
    }
  }

  (entry_offset as usize, binary)
}
