use crate::{
  targets::x86::{self, print_instructions, x86_binary_writer::PatchType},
  types::{CMPLXId, RumString, RumTypeRef, SolveDatabase},
};
use rum_common::{align_buffer_to, get_aligned_value, IString};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaticResolution {
  // Stores a copy of the object in the GLOBAL_SYMBOL_TABLE and resolves to program counter relative
  // address
  PCRelative,
  // Store a copy of the object in instruction address space.
  RoutineStatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Endianess {
  Big,
  Little,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Symbol {
  StaticString(IString),
  // Resolves to the address of the matching type.
  Type(RumTypeRef),
  RoutineAddress(CMPLXId),
  Object(IString),
  StaticData64([u8; 8]),
  StaticData32(u32),
  StaticData16(u16),
  // Resolves to a type reference object
  StaticTypeReference(RumTypeRef),
  EntryPoint(CMPLXId),
}

#[derive(Debug, Clone, Copy)]
pub struct Relocation {
  pub(crate) resolution: StaticResolution,
  /// The name of the relocated symbol
  pub(crate) symbol:     Symbol,
  /// Size in bytes of the integer that stores the relocation value
  pub(crate) byte_size:  u8,
  /// Location of the first byte of the relocatable value
  pub(crate) offset:     usize,
  /// The endianess of the of relocation value
  pub(crate) endianess:  Endianess,
}

pub(crate) fn apply_relocation(data: &mut [u8], reloc: &Relocation, val: i64) {
  let mut val_bytes = val.to_le_bytes();
  let mut bytes_to_write = &mut val_bytes[..reloc.byte_size as usize];

  const host_is_big: bool = cfg!(target_endian = "big");
  let target_is_big = reloc.endianess == Endianess::Big;

  if host_is_big != target_is_big {
    bytes_to_write.reverse();
  }

  data[reloc.offset..reloc.offset + reloc.byte_size as usize].copy_from_slice(bytes_to_write);
}

#[derive(Debug, Default)]
pub struct BinaryObject {
  pub text:        Vec<u8>,
  pub relocations: Vec<Relocation>,
}

impl BinaryObject {
  pub(crate) fn merge(left: BinaryObject, mut right: BinaryObject, align_to: u64) -> BinaryObject {
    if left.text.len() == 0 {
      right
    } else if right.text.len() == 0 {
      left
    } else {
      let mut text = left.text;

      let mut relocations = Vec::from_iter(left.relocations.into_iter().filter_map(|r| {
        // Add RoutineLocal relocations into data structure
        match r.symbol {
          Symbol::StaticData64(bytes) => {
            align_buffer_to(&mut text, 8, 0x90);

            let address = text.len();

            text.extend(bytes);

            dbg!(address, address as i64 - (r.offset as i64 + r.byte_size as i64));

            apply_relocation(&mut text, &r, address as i64 - (r.offset as i64 + r.byte_size as i64));

            None
          }
          _ => Some(r),
        }
      }));

      if align_to > 0 {
        align_buffer_to(&mut text, align_to, 0x90);
      }

      let mut base_offset = text.len();

      relocations.extend(right.relocations.into_iter().map(|mut i| {
        i.offset += base_offset;
        i
      }));

      text.append(&mut right.text);

      BinaryObject { relocations, text }
    }
  }
}

/*

- Static Linking Addresses x86
mov  r## 32      Move of static pointer into register       12 byte instruction
mov  r## 64      -
call r## 32      Call of static pointer                     9 byte instruction
call r## 64      -

- Relative Linking Address
  - PC
    lea r## rip((+|-) rel_offset) - Load of pointer to data location relative to program counter


*/

pub fn link(db: &SolveDatabase<'_>, mut bin_functs: Vec<(CMPLXId, BinaryObject)>) -> (HashMap<CMPLXId, usize>, Vec<u8>) {
  let mut data = bin_functs.pop().unwrap().1;

  for (_, func) in bin_functs {
    data = BinaryObject::merge(data, func, 4);
  }

  // Collect entry points
  let mut entry_points = data
    .relocations
    .iter()
    .filter_map(|e| match e.symbol {
      Symbol::EntryPoint(id) => Some((id, e.offset)),
      _ => None,
    })
    .collect::<HashMap<_, _>>();

  // Create preamble data
  let mut comptime_executable = vec![];
  let mut pending_relocations = vec![];
  //let mut string_data = HashMap::new();

  for relocation in data.relocations.drain(0..) {
    match relocation.symbol {
      Symbol::Type(type_ref) => {
        let ty = db.comptime_type_table[type_ref.type_id as usize] as i64;
        apply_relocation(&mut data.text, &relocation, ty);
      }
      Symbol::StaticString(str) => {
        todo!("Handle String");
      }
      Symbol::RoutineAddress(node) => pending_relocations.push(relocation),
      Symbol::Object(name) => match name.to_str().as_str() {
        "core$$type_table" => {
          let address = db.comptime_type_table.as_ptr();
          apply_relocation(&mut data.text, &relocation, unsafe { std::mem::transmute(address) });
        }
        "core$$alloc" => {
          let address: usize = x86::allocate as _;
          apply_relocation(&mut data.text, &relocation, unsafe { std::mem::transmute(address) });
        }
        _ => {
          panic!("Unknown object: {name}")
        }
      },
      Symbol::EntryPoint(..) => {}
      sym => unreachable!("{sym:?}"),
    }
  }

  // Build out string list
  /*   for (str_value, offset) in &mut string_data {
    align_buffer_to(&mut comptime_executable, 8, 0x90 /* x86 noop */);

    *offset = comptime_executable.len();
    let str = str_value.to_str();
    let str = str.as_str();
    let len = str.len() as u32;

    comptime_executable.extend_from_slice(&len.to_le_bytes());
    comptime_executable.extend_from_slice(str.as_bytes());
    comptime_executable.push(0);
  } */

  for mut pending_relocation in pending_relocations {
    match pending_relocation.symbol {
      Symbol::RoutineAddress(node) => {
        let val = *entry_points.get(&node).unwrap() as i64 - (pending_relocation.offset as i64 + 4);
        apply_relocation(&mut data.text, &pending_relocation, val);
      }
      _ => unreachable!(),
    }
  }

  let static_data_offset = comptime_executable.len();

  comptime_executable.extend(data.text.into_iter());

  for (_, entry_point) in entry_points.iter_mut() {
    *entry_point += static_data_offset
  }

  (entry_points, comptime_executable)
}

pub fn comptime_link(db: &mut SolveDatabase<'_>, mut bin_functs: Vec<(CMPLXId, BinaryObject)>) -> (HashMap<CMPLXId, usize>, Vec<u8>) {
  let mut data = bin_functs.pop().unwrap().1;

  for (_, func) in bin_functs {
    data = BinaryObject::merge(data, func, 4);
  }

  // Collect entry points
  let mut entry_points = data
    .relocations
    .iter()
    .filter_map(|e| match e.symbol {
      Symbol::EntryPoint(id) => Some((id, e.offset)),
      _ => None,
    })
    .collect::<HashMap<_, _>>();

  // Create preamble data
  let mut comptime_executable = vec![];
  let mut pending_relocations = vec![];
  //let mut string_data = HashMap::new();

  for relocation in data.relocations.drain(0..) {
    match relocation.symbol {
      Symbol::Type(type_ref) => {
        let ty = db.comptime_type_table[type_ref.type_id as usize] as i64;
        apply_relocation(&mut data.text, &relocation, ty);
      }
      Symbol::StaticString(str) => {
        let str_ref = str.to_str();
        let str = *db.comptime_strings.entry(str).or_insert_with(|| RumString::new(str_ref.as_str()));
        apply_relocation(&mut data.text, &relocation, unsafe { std::mem::transmute(str) });

        //string_data.insert(str, 0usize);
        //pending_relocations.push(relocation);
      }
      Symbol::RoutineAddress(node) => pending_relocations.push(relocation),
      Symbol::Object(name) => match name.to_str().as_str() {
        "core$$type_table" => {
          let address = db.comptime_type_table.as_ptr();
          apply_relocation(&mut data.text, &relocation, unsafe { std::mem::transmute(address) });
        }
        "core$$alloc" => {
          let address: usize = x86::allocate as _;
          apply_relocation(&mut data.text, &relocation, unsafe { std::mem::transmute(address) });
        }
        _ => {
          panic!("Unknown object: {name}")
        }
      },
      Symbol::EntryPoint(..) => {}
      sym => unreachable!("{sym:?}"),
    }
  }

  // Build out string list
  /*   for (str_value, offset) in &mut string_data {
    align_buffer_to(&mut comptime_executable, 8, 0x90 /* x86 noop */);

    *offset = comptime_executable.len();
    let str = str_value.to_str();
    let str = str.as_str();
    let len = str.len() as u32;

    comptime_executable.extend_from_slice(&len.to_le_bytes());
    comptime_executable.extend_from_slice(str.as_bytes());
    comptime_executable.push(0);
  } */

  for mut pending_relocation in pending_relocations {
    match pending_relocation.symbol {
      /*  Symbol::StaticString(str) => {
        if let Some(offset) = string_data.get(&str) {
          let val = *offset as i64 - (pending_relocation.offset + 4 + comptime_executable.len()) as i64;
          apply_relocation(&mut data.text, &pending_relocation, val);
        }
      } */
      Symbol::RoutineAddress(node) => {
        let val = *entry_points.get(&node).unwrap() as i64 - (pending_relocation.offset as i64 + 4);
        apply_relocation(&mut data.text, &pending_relocation, val);
      }
      _ => unreachable!(),
    }
  }

  let static_data_offset = comptime_executable.len();

  comptime_executable.extend(data.text.into_iter());

  for (_, entry_point) in entry_points.iter_mut() {
    *entry_point += static_data_offset
  }

  (entry_points, comptime_executable)
}
