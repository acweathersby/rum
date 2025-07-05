use std::{
  any::Any,
  collections::{HashSet, VecDeque},
};

use x86_binary_writer::{BinaryFunction, PatchType};

use crate::{
  basic_block_compiler::{self},
  ir_compiler::{ROUTINE_ID, STRUCT_ID},
  types::{NodeHandle, SolveDatabase, TypeVar, VarId},
};

//pub(crate) mod x86_compiler;
//pub use x86_compiler::compile_from_ssa_fn;
pub mod x86_binary_writer;
pub mod x86_encoder;
pub mod x86_eval;
pub mod x86_instructions;
pub mod x86_types;

extern "C" fn allocate(size: u64, alignment: u64, allocator_slot: u64) -> *mut u8 {
  dbg!(size, alignment, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(alignment as _).expect("");
  let ptr = unsafe { std::alloc::alloc(layout) };

  dbg!(ptr);
  ptr
}

extern "C" fn free(ptr: *mut u8, size: u64, allocator_slot: u64) {
  dbg!(size, ptr, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(8 as _).expect("");
  unsafe { std::alloc::dealloc(ptr, layout) };
}

pub fn compile(db: &SolveDatabase) -> Vec<BinaryFunction> {
  let mut functions = vec![];
  let mut queue = VecDeque::from_iter(db.roots.iter().map(|(_, id)| *id));
  let mut seen = HashSet::new();

  while let Some(id) = queue.pop_front() {
    if seen.insert(id) {
      let handle: NodeHandle = (id, db).into();

      if handle.get_type() == ROUTINE_ID || handle.get_type() == STRUCT_ID {
        let super_node = handle.get_mut().unwrap();

        // Collect complex node requirements
        for TypeVar { ty, .. } in super_node.type_vars.iter() {
          if ty.is_cmplx() {
            queue.push_back(ty.cmplx_data().unwrap());
          }
        }

        let register_assigned_basic_blocks = basic_block_compiler::encode_function(id, super_node, db);

        let binary = x86_binary_writer::encode_routine(super_node, &register_assigned_basic_blocks, db, allocate as _, free as _);

        for (_, patch) in &binary.patch_points {
          match patch {
            PatchType::Function(cmplx_id) => {
              queue.push_back(*cmplx_id);
            }
            _ => unreachable!(),
          }
        }

        functions.push(binary);
      }
    }
  }

  functions
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
