use std::{
  any::Any,
  collections::{HashSet, VecDeque},
};

use x86_binary_writer::{ PatchType};

use crate::{
  basic_block_compiler::{self},
  ir_compiler::{ROUTINE_ID, STRUCT_ID},
  linker::BinaryObject,
  types::{CMPLXId, NodeHandle, Operation, Reference, RumTypeObject, RumTypeProp, RumTypeRef, SolveDatabase, TypeVar},
};

//pub(crate) mod x86_compiler;
//pub use x86_compiler::compile_from_ssa_fn;
pub mod x86_binary_writer;
pub mod x86_encoder;
pub mod x86_eval;
pub mod x86_instructions;
pub mod x86_types;

pub(crate) extern "C" fn allocate(reps: u64, ty: &RumTypeObject) -> *mut u8 {

  println!("AAAA");
  println!("reps: {reps}, ty: {}", ty as *const _ as usize);

  dbg!(reps, ty);
  let ptr = if reps > 0 {
    if ty.name.as_str() == "type" {
      let prop_size = std::mem::size_of::<RumTypeProp>();
      let prop_size = prop_size * reps as usize;
      let size = ty.base_byte_size as usize + prop_size;

      dbg!(size);
      let alignment = ty.alignment as usize;
      let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(alignment).expect("");
      unsafe { std::alloc::alloc(layout) }
    } else {
      todo!("Implement variable length structs to allocate objects with reps larger than 1")
    }
  } else {
    let size = ty.base_byte_size as usize;
    let alignment = ty.alignment as usize;
    dbg!(size, alignment);
    let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(alignment).expect("");
    unsafe { std::alloc::alloc(layout) }
  };

  dbg!(ptr);

  ptr
}

pub(crate) extern "C" fn free(ptr: *mut u8, size: u64, allocator_slot: u64) {
  dbg!(size, ptr, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(8 as _).expect("");
  unsafe { std::alloc::dealloc(ptr, layout) };
}

pub fn compile(db: &SolveDatabase) -> Vec<(CMPLXId, BinaryObject)> {
  let mut functions = vec![];
  let mut queue = VecDeque::from_iter(db.roots.iter().map(|(_, id)| *id));
  let mut seen = HashSet::new();

  while let Some(id) = queue.pop_front() {
    if seen.insert(id) {
      let handle: NodeHandle = (id, db).into();
      if handle.get_type() == ROUTINE_ID || handle.get_type() == STRUCT_ID {
        let super_node = handle.get_mut().unwrap();


        for op in &super_node.operands {
          match op {
            Operation::StaticObj(reference) => match reference {
              Reference::Object(obj) => {
                queue.push_back(*obj);
              }

              _ => {}
            },
            _ => {}
          }
        }

        let register_assigned_basic_blocks = basic_block_compiler::encode_function(id, super_node, db);

        let binary = x86_binary_writer::encode_routine(id, super_node, &register_assigned_basic_blocks, db);

        functions.push((id, binary));
      }
    }
  }

  functions
}



pub(crate) fn compile_function(db: &SolveDatabase<'_>, handle: NodeHandle, id: crate::types::CMPLXId) -> Option<BinaryObject> {
  if handle.get_type() == ROUTINE_ID || handle.get_type() == STRUCT_ID {
    let super_node = handle.get_mut().unwrap();
    let register_assigned_basic_blocks = basic_block_compiler::encode_function(id, super_node, db);
    let binary = x86_binary_writer::encode_routine(id, super_node, &register_assigned_basic_blocks, db);

    Some(binary)
  } else {
    None
  }
}

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
pub fn push_bytes<T: Sized>(binary: &mut Vec<u8>, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;
  binary.extend(unsafe { std::slice::from_raw_parts(data_as_bytes, byte_size) });
}

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
pub fn set_bytes<T: Sized>(binary: &mut Vec<u8>, offset: usize, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;

  debug_assert!(offset + byte_size <= binary.len());

  unsafe { binary.as_mut_ptr().offset(offset as isize).copy_from(data_as_bytes, byte_size) }
}

mod test {
  #![cfg(test)]
}

pub fn print_instructions(binary: &[u8], mut offset: u64) -> u64 {
  use iced_x86::{Decoder, DecoderOptions, Formatter, MasmFormatter};

  let decoder = Decoder::with_ip(64, &binary, offset, DecoderOptions::NONE);
  let mut formatter = MasmFormatter::new();

  formatter.options_mut().set_digit_separator("_");
  formatter.options_mut().set_number_base(iced_x86::NumberBase::Decimal);
  formatter.options_mut().set_add_leading_zero_to_hex_numbers(true);
  formatter.options_mut().set_first_operand_char_index(2);
  formatter.options_mut().set_always_show_scale(true);
  formatter.options_mut().set_rip_relative_addresses(true);

  for instruction in decoder {
    let mut output = String::default();
    formatter.format(&instruction, &mut output);
    print!("{:0>4} ", instruction.ip());
    println!(" {}", output);

    offset = instruction.ip() + instruction.len() as u64
  }
  println!("\n\n");

  offset
}

fn print_instruction(binary: &[u8]) -> String {
  use iced_x86::{Decoder, DecoderOptions, Formatter, MasmFormatter};

  let decoder = Decoder::with_ip(64, &binary, 0, DecoderOptions::NONE);
  let mut formatter = MasmFormatter::new();

  formatter.options_mut().set_digit_separator("_");
  formatter.options_mut().set_number_base(iced_x86::NumberBase::Decimal);
  formatter.options_mut().set_add_leading_zero_to_hex_numbers(true);
  formatter.options_mut().set_leading_zeros(true);
  formatter.options_mut().set_first_operand_char_index(2);
  formatter.options_mut().set_always_show_scale(true);
  formatter.options_mut().set_rip_relative_addresses(true);

  for instruction in decoder {
    let mut output = String::default();
    formatter.format(&instruction, &mut output);
    return output;
  }
  Default::default()
}
