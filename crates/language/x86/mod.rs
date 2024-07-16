//! # x86 Registers
//!
//!
//! ## Caller / Callee saved registers
//!
//! - Linux:
//!
//! |                | RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 |
//! |                | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 |
//! | :---------     | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
//! | Callee_Saved   |     |     |     |  X  |  X  |  X  |     |     |     |     |     |     |  X  |  X  |  X  |  X  |    
//! | Caller_Saved   |  X  |  X  |  X  |     |     |     |  X  |  X  |  X  |  X  |  X  |  X  |     |     |     |     |
//! | C Calling Arg  |     |  4  |  3  |     |     |     |  2  |  1  |  5  |  6  |     |     |     |     |     |     |
//! | C Return Arg   |  1  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
//! | Syscall Args   |     |     |  3  |     |     |     |  2  |  1  |  5  |  6  |  4  |     |     |     |     |     |
//! | Syscall Return |  1  |     |  2  |     |     |     |     |     |     |     |     |     |     |     |     |     |
//!
//!
//! - Window:

pub(crate) mod x86_compiler;
pub use x86_compiler::compile_from_ssa_fn;

pub(crate) mod x86_encoder;
pub(crate) mod x86_instructions;
pub(crate) mod x86_types;

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
fn push_bytes<T: Sized>(binary: &mut Vec<u8>, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;
  binary.extend(unsafe { std::slice::from_raw_parts(data_as_bytes, byte_size) });
}

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
fn set_bytes<T: Sized>(binary: &mut Vec<u8>, offset: usize, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;

  debug_assert!(offset + byte_size <= binary.len());

  unsafe { binary.as_mut_ptr().offset(offset as isize).copy_from(data_as_bytes, byte_size) }
}

mod test {
  #![cfg(test)]
}

fn print_instructions(binary: &[u8], mut offset: u64) -> u64 {
  use iced_x86::{Decoder, DecoderOptions, Formatter, MasmFormatter};

  let mut decoder = Decoder::with_ip(64, &binary, offset, DecoderOptions::NONE);
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
    print!("{:016} ", instruction.ip());
    println!(" {}", output);

    offset = instruction.ip() + instruction.len() as u64
  }

  offset
}

fn print_instruction(binary: &[u8]) -> String {
  use iced_x86::{Decoder, DecoderOptions, Formatter, MasmFormatter};

  let mut decoder = Decoder::with_ip(64, &binary, 0, DecoderOptions::NONE);
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
    return output;
  }
  Default::default()
}
