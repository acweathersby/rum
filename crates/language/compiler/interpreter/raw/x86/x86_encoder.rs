#![allow(unused, non_upper_case_globals)]

use super::{push_bytes, set_bytes, x86_types::*};
use crate::compiler::interpreter::raw::ir::ir_types::BitSize;
use std::collections::HashMap;
use OperandType as OT;

type OpSignature = (u16, OperandType, OperandType, OperandType);
type OpEncoder = fn(
  binary: &mut InstructionProps,
  op_code: u32,
  bit_size: BitSize,
  enc: OpEncoding,
  op1: Arg,
  op2: Arg,
  op3: Arg,
  ext: u8,
);

pub(super) struct InstructionProps<'bin> {
  instruction_name:      &'static str,
  bin:                   &'bin mut Vec<u8>,
  displacement_index:    usize,
  displacement_bit_size: usize,
}

impl<'bin> InstructionProps<'bin> {
  #[track_caller]
  pub fn displace_too(&mut self, offset: usize) {
    if self.displacement_bit_size > 0 {
      let ip_offset = self.bin.len() as i64;
      let dis = offset as i64 - ip_offset;

      match self.displacement_bit_size {
        8 => set_bytes(self.bin, self.displacement_index, dis as i8),
        32 => set_bytes(self.bin, self.displacement_index, dis as i32),
        64 => set_bytes(self.bin, self.displacement_index, dis as i64),
        size => panic!("Invalid displacement size {size}. {}", self.instruction_name),
      }
    } else {
      panic!(
        "Attempt to adjust displacement of instruction that has no such value. {}",
        self.instruction_name
      )
    }
  }
}

macro_rules! op_table {
  ($name: ident $value: expr) => {
    pub(super) const $name: (
      &'static str,
      [(OpSignature, (u32, u8, OpEncoding, OpEncoder)); $value.len()],
    ) = (stringify!($name), $value);
  };
}

/// https://www.felixcloutier.com/x86/call
op_table!(call [
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x00E8, 0x00, OpEncoding::D, gen_unary_op)),
//
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x02, OpEncoding::M, gen_unary_op)),
//
((64, OT::REG, OT::NONE, OT::NONE), (0x00FF, 0x02, OpEncoding::M, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/ret
op_table!(ret [
((8, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op)),
((16, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op)),
((32, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op)),
((64, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op)),
]);

/// https://www.felixcloutier.com/x86/ret
op_table!(ret_pop [
((16, OT::IMM_INT, OT::NONE, OT::NONE), (0x00C2, 0x02, OpEncoding::I, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/push
op_table!(push [
((16, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x06, OpEncoding::M, gen_unary_op)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x06, OpEncoding::M, gen_unary_op)),
//
((16, OT::REG, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::O, gen_unary_op)),
((64, OT::REG, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::O, gen_unary_op)),
//
((16, OT::IMM_INT, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::I, gen_unary_op)),
((64, OT::IMM_INT, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::I, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/pop
op_table!(pop [
((16, OT::MEM, OT::NONE, OT::NONE), (0x008F, 0x06, OpEncoding::M, gen_unary_op)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x008F, 0x06, OpEncoding::M, gen_unary_op)),
//
((16, OT::REG, OT::NONE, OT::NONE), (0x0058, 0x00, OpEncoding::O, gen_unary_op)),
((64, OT::REG, OT::NONE, OT::NONE), (0x0058, 0x00, OpEncoding::O, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jmp
op_table!(jmp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x00EB, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x00E9, 0x00, OpEncoding::D, gen_unary_op)),
//
((64, OT::REG, OT::NONE, OT::NONE), (0x00FF, 0x04, OpEncoding::M, gen_unary_op)),
//
((16, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op)),
((32, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jb [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0072, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F82, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jae [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0073, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F83, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(je [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0074, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F84, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jne [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0075, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F85, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jbe [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0076, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F86, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(ja [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0077, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F87, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(js [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0078, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F88, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jns [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0079, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F89, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jpe [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007A, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8A, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jpo [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007B, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8B, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jl [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007C, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8C, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jge [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007D, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8D, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jle [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007E, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8E, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jg [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007F, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8F, 0x00, OpEncoding::D, gen_unary_op)),
]);

/// https://www.felixcloutier.com/x86/mov
op_table!(mov [
  ((08, OT::REG, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x008A, 0x00, OpEncoding::RM, gen_bin_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_bin_op)),
  //
  ((8, OT::REG, OT::IMM_INT, OT::NONE), (0x00B0, 0x00, OpEncoding::OI, gen_bin_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_bin_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_bin_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_bin_op)),
  //
  ((8, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C6, 0x00, OpEncoding::MI, gen_bin_op)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_bin_op)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_bin_op)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_bin_op)),

]);

/// https://www.felixcloutier.com/x86/cmp
op_table!(cmp [
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_bin_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ///
  ((08, OT::MEM, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_bin_op)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_bin_op)),
  ///
  ((08, OT::REG, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_bin_op)),
  ///
  ((08, OT::REG, OT::MEM, OT::NONE), (0x003A, 0x00, OpEncoding::RM, gen_bin_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_bin_op)),
]);

/// https://www.felixcloutier.com/x86/add
op_table!(add [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x0002, 0x00, OpEncoding::RM, gen_bin_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_bin_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_bin_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_bin_op)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x00, OpEncoding::MI, gen_bin_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_bin_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_bin_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_bin_op)),
]);

/// https://www.felixcloutier.com/x86/sub
op_table!(sub [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0028, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x002A, 0x00, OpEncoding::MR, gen_bin_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_bin_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_bin_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_bin_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x002A, 0x00, OpEncoding::RM, gen_bin_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_bin_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_bin_op)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x05, OpEncoding::MI, gen_bin_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_bin_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_bin_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_bin_op)),
]);

/// https://www.felixcloutier.com/x86/mul
op_table!(mul [  //
  ((08, OT::REG, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_bin_op)),
  ((16, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
  ((32, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
  ((64, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
  ((08, OT::MEM, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_bin_op)),
  ((16, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
  ((32, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
  ((64, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_bin_op)),
]);

/// https://www.felixcloutier.com/x86/imul
op_table!(imul [  //
  ((16, OT::REG, OT::REG, OT::NONE), (0x00AF, 0x00, OpEncoding::RM, gen_tri_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x00AF, 0x00, OpEncoding::RM, gen_tri_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x00AF, 0x00, OpEncoding::RM, gen_tri_op)),

  ((08, OT::REG, OT::REG, OT::IMM_INT), (0x0068, 0x00, OpEncoding::RMI, gen_tri_op)),
  ((16, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op)),
  ((32, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op)),
  ((64, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op)),
]);

pub(super) fn encode_zero<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
) -> InstructionProps<'bin> {
  encode(binary, table, bit_size, Arg::None, Arg::None, Arg::None)
}

pub(super) fn encode_unary<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
) -> InstructionProps<'bin> {
  encode(binary, table, bit_size, op1, Arg::None, Arg::None)
}

pub(super) fn encode_binary<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
  op2: Arg,
) -> InstructionProps<'bin> {
  encode(binary, table, bit_size, op1, op2, Arg::None)
}

pub(super) fn encode<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> InstructionProps<'bin> {
  debug_assert!(bit_size.as_u64() <= 512);
  let signature = (bit_size.as_u64() as u16, op1.ty(), op2.ty(), op3.ty());

  for (sig, (op_code, ext, encoding, encoder)) in &table.1 {
    if *sig == signature {
      let mut props = InstructionProps {
        instruction_name:      table.0,
        bin:                   binary,
        displacement_index:    0,
        displacement_bit_size: 0,
      };
      encoder(&mut props, *op_code, bit_size, *encoding, op1, op2, op3, *ext);
      return props;
    }
  }

  panic!(
    "Could not find operation for {signature:?} in encoding table \n\n{}",
    format!(
      "{}:\n{}",
      table.0,
      table.1.iter().map(|v| format!("{:?} {:?}", v.0, v.1)).collect::<Vec<_>>().join("\n")
    )
  );
}

pub(super) fn encoded_vec(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> usize {
  let mut bin = vec![];
  encode(&mut bin, table, bit_size, op1, op2, op3);
  bin.len()
}

fn insert_rex_if_required(props: &mut InstructionProps, bit_size: BitSize, r_m: Arg, reg: Arg) {
  const REX_W_64B: u8 = 0b0100_1000;
  const REX_R_REG_EX: u8 = 0b0100_0100;
  const REX_X_SIP: u8 = 0b0100_0010;
  const REX_B_MEM_REG_EX: u8 = 0b0100_0001;

  let mut rex = 0;
  rex |= (bit_size == BitSize::b64).then_some(REX_W_64B).unwrap_or(0);
  rex |= (r_m.is_64_extended()).then_some(REX_B_MEM_REG_EX).unwrap_or(0);
  rex |= (reg.is_64_extended()).then_some(REX_R_REG_EX).unwrap_or(0);
  if rex > 0 {
    props.bin.push(rex);
  }
}

fn push_mod_rm_reg_op(props: &mut InstructionProps, r_m: Arg, reg: Arg) {
  const SIB_SCALE_OFFSET: u8 = 6;
  const SIB_INDEX_OFFSET: u8 = 3;
  const SIB_INDEX_NOT_USED: u8 = 0b100 << SIB_INDEX_OFFSET;
  const SIB_NO_INDEX_SCALE: u8 = 0b00 << SIB_SCALE_OFFSET;
  const DISPLACEMENT_INDEX: u8 = 0b101;
  const SIB_RIP_BASE: u8 = 0b101;

  let mut mem_encoding = 0b00;
  let mut displace_val = 0 as u64;
  let mut rm_index = r_m.reg_index();

  let sib = match rm_index {
    4 => match r_m {
      Arg::Mem(RSP) | Arg::Mem(R12) => {
        // use sib index to access the RSP register
        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | RSP.index()) as u8
      }

      Arg::RSP_REL(val) => {
        if (val & !0xFF) > 0 {
          mem_encoding = 0b10
        } else {
          mem_encoding = 0b01;
        }

        displace_val = val;

        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | RSP.index()) as u8
      }
      Arg::Reg(RSP) | Arg::Reg(R12) => {
        ///
        0
      }
      arg => unreachable!("{arg:?}"),
    },
    5 => match r_m {
      Arg::RIP_REL(val) => {
        displace_val = val;
        0
      }
      Arg::Mem(RBP) | Arg::Mem(R13) => {
        // use sib index to access the RSP register
        mem_encoding = 0b01;
        (SIB_NO_INDEX_SCALE | (0b000 << 3) | 0b000) as u8
      }
      _ => unreachable!(),
    },
    _ => 0,
  };

  let mod_bits = match r_m {
    Arg::RSP_REL(_) | Arg::RIP_REL(_) | Arg::Mem(_) => mem_encoding,
    Arg::Reg(_) => 0b11,
    op => panic!("Invalid r_m operand {op:?}"),
  };

  props.bin.push(((mod_bits & 0b11) << 6) | ((reg.reg_index() & 0x7) << 3) | (rm_index & 0x7));

  if sib != 0 {
    props.bin.push(sib)
  }

  match mod_bits {
    0b01 => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 8;
      push_bytes(props.bin, displace_val as u8);
    }
    0b00 if rm_index == DISPLACEMENT_INDEX => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 32;
      push_bytes(props.bin, displace_val as u32);
    }
    0b10 => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 32;
      push_bytes(props.bin, displace_val as u32);
    }
    _ => {}
  }
}

pub(super) fn gen_zero_op(
  props: &mut InstructionProps,
  op_code: u32,
  bit_size: BitSize,
  enc: OpEncoding,
  _: Arg,
  _: Arg,
  _: Arg,
  ext: u8,
) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    Zero => {
      insert_op_code_bytes(props.bin, op_code as u32);
    }
    enc => panic!("{enc:?} not valid for unary operations"),
  }
}

pub(super) fn gen_unary_op(
  props: &mut InstructionProps,
  op_code: u32,
  bit_size: BitSize,
  enc: OpEncoding,
  op1: Arg,
  _: Arg,
  _: Arg,
  ext: u8,
) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    O => match op1 {
      Reg(_) => {
        insert_rex_if_required(props, bit_size, op1, OpExt(ext));
        insert_op_code_bytes(props.bin, op_code | op1.reg_index() as u32);
      }
      imm => panic!("Invalid immediate arg op1 of {imm:?} for MI encoding"),
    },
    I => match op1 {
      Imm_Int(imm) => {
        insert_op_code_bytes(props.bin, op_code as u32);
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          BitSize::b32 => push_bytes(props.bin, imm as u32),
          BitSize::b64 => push_bytes(props.bin, imm as u64),
          size => panic!("Invalid immediate size {size:?} for OI encoding"),
        }
      }
      imm => panic!("Invalid immediate arg op1 of {imm:?} for MI encoding"),
    },
    M => {
      insert_rex_if_required(props, bit_size, op1, OpExt(ext));
      insert_op_code_bytes(props.bin, op_code);
      push_mod_rm_reg_op(props, op1, OpExt(ext))
    }
    D => {
      insert_op_code_bytes(props.bin, op_code);
      match op1 {
        Arg::Imm_Int(imm) => match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          _ => push_bytes(props.bin, imm as u32),
        },
        _ => unreachable!(),
      }
    }
    enc => panic!("{enc:?} not valid for unary operations"),
  }
}

pub(super) fn gen_bin_op(
  props: &mut InstructionProps,
  op_code: u32,
  bit_size: BitSize,
  enc: OpEncoding,
  op1: Arg,
  op2: Arg,
  _: Arg,
  ext: u8,
) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    MR => {
      insert_rex_if_required(props, bit_size, op1, op2);
      insert_op_code_bytes(props.bin, op_code);
      push_mod_rm_reg_op(props, op1, op2);
    }
    RM => {
      insert_rex_if_required(props, bit_size, op2, op1);
      insert_op_code_bytes(props.bin, op_code);
      push_mod_rm_reg_op(props, op2, op1);
    }
    MI => match op2 {
      Imm_Int(imm) => {
        insert_rex_if_required(props, bit_size, op1, OpExt(ext));
        insert_op_code_bytes(props.bin, op_code);

        push_mod_rm_reg_op(props, op1, OpExt(ext));
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          _ => push_bytes(props.bin, imm as u32),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    OI => match op2 {
      Imm_Int(imm) => {
        insert_rex_if_required(props, bit_size, op1, OpExt(ext));
        insert_op_code_bytes(props.bin, op_code | op1.reg_index() as u32);
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          BitSize::b32 => push_bytes(props.bin, imm as u32),
          BitSize::b64 => push_bytes(props.bin, imm as u64),
          size => panic!("Invalid immediate size of {size:?} for OI encoding"),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    I => match op2 {
      Imm_Int(imm) => {
        insert_rex_if_required(props, bit_size, OpExt(0), OpExt(ext));
        insert_op_code_bytes(props.bin, op_code);
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          BitSize::b64 | BitSize::b32 => push_bytes(props.bin, 3 as u32),
          size => panic!("Invalid immediate size of {size:?} for OI encoding"),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    enc => panic!("{enc:?} not valid for binary operations on {op1:?} on {op2:?}"),
  }
}

pub(super) fn gen_tri_op(
  props: &mut InstructionProps,
  op_code: u32,
  bit_size: BitSize,
  enc: OpEncoding,
  op1: Arg,
  op2: Arg,
  op3: Arg,
  ext: u8,
) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    RMI => match op3 {
      Imm_Int(imm) => {
        insert_rex_if_required(props, bit_size, op1, op2);
        insert_op_code_bytes(props.bin, op_code);

        push_mod_rm_reg_op(props, op1, op2);
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          _ => push_bytes(props.bin, imm as u32),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    enc => panic!("{enc:?} not valid for binary operations on {op1:?} on {op2:?}"),
  }
}

fn insert_op_code_bytes(binary: &mut Vec<u8>, op_code: u32) {
  for byte in op_code.to_be_bytes() {
    if byte != 0 {
      push_bytes(binary, byte);
    }
  }
}
