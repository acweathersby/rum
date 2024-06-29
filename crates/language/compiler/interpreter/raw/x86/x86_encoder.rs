#![allow(unused, non_upper_case_globals)]

use super::{push_bytes, set_bytes, x86_types::*};
use crate::compiler::interpreter::raw::{
  ir::{ir_types::BitSize, SSAFunction},
  x86::print_instruction,
};
use std::collections::HashMap;
use BitSize::*;
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

#[test]
fn test_call() {
  assert_eq!("call 37", test_enc_uno(&call, b32, Arg::Imm_Int(32)));
  assert_eq!("call r8", test_enc_uno(&call, b64, R8.as_reg_op()));
  assert_eq!("call qword ptr [r8]", test_enc_uno(&call, b64, R8.as_mem_op()));
  assert_eq!("call qword ptr [rip+23]", test_enc_uno(&call, b64, Arg::RIP_REL(23)));
  assert_eq!("call qword ptr [rsp+23]", test_enc_uno(&call, b64, Arg::RSP_REL(23)));
  assert_eq!("call qword ptr [rbx]", test_enc_uno(&call, b64, RBX.as_mem_op()));
  assert_eq!("call qword ptr [r13]", test_enc_uno(&call, b64, R13.as_mem_op()));
}

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

#[test]
fn test_jb() {
  assert_eq!("jb short 18_446_744_073_709_551_586", test_enc_uno(&jb, b8, Arg::Imm_Int(-32)));
  assert_eq!("jb near ptr 38", test_enc_uno(&jb, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jae [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0073, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F83, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jae() {
  assert_eq!("jae short 18_446_744_073_709_551_586", test_enc_uno(&jae, b8, Arg::Imm_Int(-32)));
  assert_eq!("jae near ptr 38", test_enc_uno(&jae, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(je [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0074, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F84, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_je() {
  assert_eq!("je short 18_446_744_073_709_551_586", test_enc_uno(&je, b8, Arg::Imm_Int(-32)));
  assert_eq!("je near ptr 38", test_enc_uno(&je, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jne [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0075, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F85, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jne() {
  assert_eq!("jne short 18_446_744_073_709_551_586", test_enc_uno(&jne, b8, Arg::Imm_Int(-32)));
  assert_eq!("jne near ptr 38", test_enc_uno(&jne, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jbe [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0076, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F86, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jbe() {
  assert_eq!("jbe short 18_446_744_073_709_551_586", test_enc_uno(&jbe, b8, Arg::Imm_Int(-32)));
  assert_eq!("jbe near ptr 38", test_enc_uno(&jbe, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(ja [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0077, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F87, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_ja() {
  assert_eq!("ja short 18_446_744_073_709_551_586", test_enc_uno(&ja, b8, Arg::Imm_Int(-32)));
  assert_eq!("ja near ptr 38", test_enc_uno(&ja, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(js [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0078, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F88, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_js() {
  assert_eq!("js short 18_446_744_073_709_551_586", test_enc_uno(&js, b8, Arg::Imm_Int(-32)));
  assert_eq!("js near ptr 38", test_enc_uno(&js, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jns [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0079, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F89, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jns() {
  assert_eq!("jns short 18_446_744_073_709_551_586", test_enc_uno(&jns, b8, Arg::Imm_Int(-32)));
  assert_eq!("jns near ptr 38", test_enc_uno(&jns, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007A, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8A, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jp() {
  assert_eq!("jp short 18_446_744_073_709_551_586", test_enc_uno(&jp, b8, Arg::Imm_Int(-32)));
  assert_eq!("jp near ptr 38", test_enc_uno(&jp, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jnp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007B, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8B, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jnp() {
  assert_eq!("jnp short 18_446_744_073_709_551_586", test_enc_uno(&jnp, b8, Arg::Imm_Int(-32)));
  assert_eq!("jnp near ptr 38", test_enc_uno(&jnp, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jl [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007C, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8C, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jl() {
  assert_eq!("jl short 18_446_744_073_709_551_586", test_enc_uno(&jl, b8, Arg::Imm_Int(-32)));
  assert_eq!("jl near ptr 38", test_enc_uno(&jl, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jge [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007D, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8D, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jge() {
  assert_eq!("jge short 18_446_744_073_709_551_586", test_enc_uno(&jge, b8, Arg::Imm_Int(-32)));
  assert_eq!("jge near ptr 38", test_enc_uno(&jge, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jle [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007E, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8E, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jle() {
  assert_eq!("jle short 18_446_744_073_709_551_586", test_enc_uno(&jle, b8, Arg::Imm_Int(-32)));
  assert_eq!("jle near ptr 38", test_enc_uno(&jle, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jg [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007F, 0x00, OpEncoding::D, gen_unary_op)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8F, 0x00, OpEncoding::D, gen_unary_op)),
]);

#[test]
fn test_jg() {
  assert_eq!("jg short 18_446_744_073_709_551_586", test_enc_uno(&jg, b8, Arg::Imm_Int(-32)));
  assert_eq!("jg near ptr 38", test_enc_uno(&jg, b32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_o [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_o() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovo r8d,r11d", test_enc_dos(&mov_o, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovo r8,r11", test_enc_dos(&mov_o, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovo r8d,[r11]", test_enc_dos(&mov_o, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovo r8,[r11]", test_enc_dos(&mov_o, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_no [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_no() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovno r8d,r11d", test_enc_dos(&mov_no, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovno r8,r11", test_enc_dos(&mov_no, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovno r8d,[r11]", test_enc_dos(&mov_no, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovno r8,[r11]", test_enc_dos(&mov_no, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_b [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op)),
]);

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ae [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_ae() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovae r8d,r11d", test_enc_dos(&mov_ae, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovae r8,r11", test_enc_dos(&mov_ae, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovae r8d,[r11]", test_enc_dos(&mov_ae, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovae r8,[r11]", test_enc_dos(&mov_ae, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_e [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_e() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmove r8d,r11d", test_enc_dos(&mov_e, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmove r8,r11", test_enc_dos(&mov_e, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmove r8d,[r11]", test_enc_dos(&mov_e, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmove r8,[r11]", test_enc_dos(&mov_e, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ne [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_ne() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovne r8d,r11d", test_enc_dos(&mov_ne, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovne r8,r11", test_enc_dos(&mov_ne, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovne r8d,[r11]", test_enc_dos(&mov_ne, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovne r8,[r11]", test_enc_dos(&mov_ne, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_be [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_be() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovbe r8d,r11d", test_enc_dos(&mov_be, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovbe r8,r11", test_enc_dos(&mov_be, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovbe r8d,[r11]", test_enc_dos(&mov_be, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovbe r8,[r11]", test_enc_dos(&mov_be, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_a [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_a() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmova r8d,r11d", test_enc_dos(&mov_a, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmova r8,r11", test_enc_dos(&mov_a, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmova r8d,[r11]", test_enc_dos(&mov_a, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmova r8,[r11]", test_enc_dos(&mov_a, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_s [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_s() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovs r10d,r11d", test_enc_dos(&mov_s, b32, R10.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovs r10,r11", test_enc_dos(&mov_s, b64, R10.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovs r10d,[r11]", test_enc_dos(&mov_s, b32, R10.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovs r10,[r11]", test_enc_dos(&mov_s, b64, R10.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ns [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_ns() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovns r8d,r11d", test_enc_dos(&mov_ns, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovns r8,r11", test_enc_dos(&mov_ns, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovns r8d,[r11]", test_enc_dos(&mov_ns, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovns r8,[r11]", test_enc_dos(&mov_ns, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_pe [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_pe() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovp r8d,r11d", test_enc_dos(&mov_pe, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovp r8,r11", test_enc_dos(&mov_pe, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovp r8d,[r11]", test_enc_dos(&mov_pe, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovp r8,[r11]", test_enc_dos(&mov_pe, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_po [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_po() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovnp r8d,r11d", test_enc_dos(&mov_po, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovnp r8,r11", test_enc_dos(&mov_po, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovnp r8d,[r11]", test_enc_dos(&mov_po, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovnp r8,[r11]", test_enc_dos(&mov_po, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_l [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_l() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovl r8d,r11d", test_enc_dos(&mov_l, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovl r8,r11", test_enc_dos(&mov_l, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovl r8d,[r11]", test_enc_dos(&mov_l, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovl r8,[r11]", test_enc_dos(&mov_l, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ge [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_ge() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovge r8d,r11d", test_enc_dos(&mov_ge, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovge r8,r11", test_enc_dos(&mov_ge, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovge r8d,[r11]", test_enc_dos(&mov_ge, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovge r8,[r11]", test_enc_dos(&mov_ge, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_le [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_le() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovle r8d,r11d", test_enc_dos(&mov_le, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovle r8,r11", test_enc_dos(&mov_le, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovle r8d,[r11]", test_enc_dos(&mov_le, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovle r8,[r11]", test_enc_dos(&mov_le, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_g [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op)),
]);

#[test]
fn test_mov_g() {
  let c = SSAFunction::default();
  let c = &c;
  let s = &[];
  assert_eq!("cmovg r8d,r11d", test_enc_dos(&mov_g, b32, R8.as_op(c, s), R11.as_op(c, s)));
  assert_eq!("cmovg r8,r11", test_enc_dos(&mov_g, b64, R8.as_op(c, s), R11.as_op(c, s)));

  assert_eq!("cmovg r8d,[r11]", test_enc_dos(&mov_g, b32, R8.as_op(c, s), R11.as_addr_op(c, s)));
  assert_eq!("cmovg r8,[r11]", test_enc_dos(&mov_g, b64, R8.as_op(c, s), R11.as_addr_op(c, s)));
}

/// https://www.felixcloutier.com/x86/mov
op_table!(mov [
  ((08, OT::REG, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x008A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op)),
  //
  ((8, OT::REG, OT::IMM_INT, OT::NONE), (0x00B0, 0x00, OpEncoding::OI, gen_multi_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op)),
  //
  ((8, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C6, 0x00, OpEncoding::MI, gen_multi_op)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op)),
]);

/// https://www.felixcloutier.com/x86/mov
op_table!(vmov32_align [
  ((128, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((128, OT::REG, OT::MEM, OT::NONE), (0x660F6F, 0x00, OpEncoding::RM, gen_multi_op)),
  ((128, OT::MEM, OT::REG, OT::NONE), (0x660F7F, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((256, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::VEX_RM, gen_multi_op)),
  ((256, OT::REG, OT::MEM, OT::NONE), (0x660F6F, 0x00, OpEncoding::VEX_RM, gen_multi_op)),
  ((256, OT::MEM, OT::REG, OT::NONE), (0x660F7F, 0x00, OpEncoding::VEX_MR, gen_multi_op)),

  ((512, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:0 }, gen_multi_op)),
]);

op_table!(vmov64_align [
  ((512, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:1 }, gen_multi_op)),
  ((512, OT::REG, OT::REG, OT::REG), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:1 }, gen_multi_op)),
]);

#[test]
fn test_vec_mov() {
  assert_eq!(
    "vmovdqa64 zmm31{k1},zmm31",
    test_enc_tres(&vmov64_align, b512, ZMM31.as_reg_op(), ZMM31.as_reg_op(), K1.as_reg_op())
  );

  assert_eq!(
    "vmovdqa32 zmm31,zmm31",
    test_enc_dos(&vmov32_align, b512, ZMM31.as_reg_op(), ZMM31.as_reg_op())
  );

  assert_eq!(
    "vmovdqa32 zmm0,zmm0",
    test_enc_dos(&vmov32_align, b512, ZMM0.as_reg_op(), ZMM0.as_reg_op())
  );

  assert_eq!(
    "movdqa xmm0,xmm1",
    test_enc_dos(&vmov32_align, b128, XMM0.as_reg_op(), XMM1.as_reg_op())
  );
  assert_eq!(
    "movdqa [rbx],xmm3",
    test_enc_dos(&vmov32_align, b128, RBX.as_mem_op(), XMM3.as_reg_op())
  );
  assert_eq!(
    "vmovdqa ymm15,ymmword ptr [r8]",
    test_enc_dos(&vmov32_align, b256, XMM15.as_reg_op(), R8.as_mem_op())
  );
  assert_eq!(
    "vmovdqa ymmword ptr [r15],ymm8",
    test_enc_dos(&vmov32_align, b256, XMM15.as_mem_op(), R8.as_reg_op())
  );
}

/// https://www.felixcloutier.com/x86/cmp
op_table!(cmp [
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_multi_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ///
  ((08, OT::MEM, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_multi_op)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op)),
  ///
  ((08, OT::REG, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op)),
  ///
  ((08, OT::REG, OT::MEM, OT::NONE), (0x003A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op)),
]);

/// https://www.felixcloutier.com/x86/add
op_table!(add [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x0002, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x00, OpEncoding::MI, gen_multi_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op)),
]);

/// https://www.felixcloutier.com/x86/sub
op_table!(sub [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0028, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x002A, 0x00, OpEncoding::MR, gen_multi_op)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x002A, 0x00, OpEncoding::RM, gen_multi_op)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x05, OpEncoding::MI, gen_multi_op)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op)),
]);

/// https://www.felixcloutier.com/x86/mul
op_table!(mul [  //
  ((08, OT::REG, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_multi_op)),
  ((16, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
  ((32, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
  ((64, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
  ((08, OT::MEM, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_multi_op)),
  ((16, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
  ((32, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
  ((64, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op)),
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

pub(super) fn test_enc_uno<'bin>(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
) -> String {
  let mut bin = vec![];
  encode(&mut bin, table, bit_size, op1, Arg::None, Arg::None);
  print_instruction(&bin)
}

pub(super) fn test_enc_dos<'bin>(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
  op2: Arg,
) -> String {
  let mut bin = vec![];
  encode(&mut bin, table, bit_size, op1, op2, Arg::None);
  //println!("{:02X?}", &bin);
  print_instruction(&bin)
}

pub(super) fn test_enc_tres<'bin>(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, OpEncoder))]),
  bit_size: BitSize,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> String {
  let mut bin = vec![];
  encode(&mut bin, table, bit_size, op1, op2, op3);
  //println!("{:02X?}", &bin);
  print_instruction(&bin)
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
        encode_rex(props, bit_size, op1, OpExt(ext));
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
      encode_rex(props, bit_size, op1, OpExt(ext));
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op1, OpExt(ext))
    }
    D => {
      insert_op_code_bytes(props.bin, op_code);
      match op1 {
        Arg::Imm_Int(imm) => match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as i8),
          _ => push_bytes(props.bin, imm as i32),
        },
        _ => unreachable!(),
      }
    }
    enc => panic!("{enc:?} not valid for unary operations"),
  }
}

pub(super) fn gen_multi_op(
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
    A => {
      insert_op_code_bytes(props.bin, 0x66);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op2, op1);
    }
    B => {
      insert_op_code_bytes(props.bin, 0x66);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op1, op2);
    }
    EVEX_RM { w } => {
      let op_code = encode_evex(op_code, op2, op1, op3, bit_size, props, w);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op2, op1);
    }
    EVEX_MR { w } => {
      let op_code = encode_evex(op_code, op1, op2, op3, bit_size, props, w);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op1, op2);
    }
    VEX_RM => {
      let op_code = encode_vex(op_code, op2, op1, bit_size, props);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op2, op1);
    }
    VEX_MR => {
      let op_code = encode_vex(op_code, op1, op2, bit_size, props);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op1, op2);
    }
    MR => {
      encode_rex(props, bit_size, op1, op2);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op1, op2);
    }
    RM => {
      encode_rex(props, bit_size, op2, op1);
      insert_op_code_bytes(props.bin, op_code);
      encode_mod_rm_reg(props, op2, op1);
    }
    MI => match op2 {
      Imm_Int(imm) => {
        encode_rex(props, bit_size, op1, OpExt(ext));
        insert_op_code_bytes(props.bin, op_code);

        encode_mod_rm_reg(props, op1, OpExt(ext));
        match bit_size {
          BitSize::b8 => push_bytes(props.bin, imm as u8),
          _ => push_bytes(props.bin, imm as u32),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    OI => match op2 {
      Imm_Int(imm) => {
        encode_rex(props, bit_size, op1, OpExt(ext));
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
        encode_rex(props, bit_size, OpExt(0), OpExt(ext));
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

fn encode_mod_rm_reg(props: &mut InstructionProps, r_m: Arg, reg: Arg) {
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
        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | (RSP.reg_id() & 7) as u8) as u8
      }

      Arg::RSP_REL(val) => {
        if (val & !0xFF) > 0 {
          mem_encoding = 0b10
        } else {
          mem_encoding = 0b01;
        }

        displace_val = val;

        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | (RSP.reg_id() & 7) as u8) as u8
      }
      _ => {
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
      Arg::Reg(RBP) | Arg::Reg(R13) => {
        // use sib index to access the RSP register
        0
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

fn encode_rex(props: &mut InstructionProps, bit_size: BitSize, r_m: Arg, reg: Arg) {
  const REX_W_64B: u8 = 0b0100_1000;
  const REX_R_REG_EX: u8 = 0b0100_0100;
  const REX_X_SIP: u8 = 0b0100_0010;
  const REX_B_MEM_REG_EX: u8 = 0b0100_0001;

  let mut rex = 0;
  rex |= (bit_size == BitSize::b64).then_some(REX_W_64B).unwrap_or(0);
  rex |= (r_m.is_upper_8_reg()).then_some(REX_B_MEM_REG_EX).unwrap_or(0);
  rex |= (reg.is_upper_8_reg()).then_some(REX_R_REG_EX).unwrap_or(0);
  if rex > 0 {
    props.bin.push(rex);
  }
}

fn encode_evex(
  op_code: u32,
  r_m: Arg,
  reg: Arg,
  op3: Arg,
  bit_size: BitSize,
  props: &mut InstructionProps<'_>,
  w: u8,
) -> u32 {
  insert_op_code_bytes(props.bin, 0x62);

  let rex_b = ((r_m.is_upper_8_reg() as u8) ^ 1) << 5;
  let rex_x = ((r_m.is_upper_16_reg() as u8) ^ 1) << 6;

  let rex_r = ((reg.is_upper_8_reg() as u8) ^ 1) << 7;
  let rex_r_prime = ((reg.is_upper_16_reg() as u8) ^ 1) << 4;

  let rex_w = w << 7;

  // two byte form
  let op_ext: [u8; 4] = unsafe { std::mem::transmute(op_code) };
  let mut pp = 0;
  let mut mmm = 0;

  for i in (0..4).rev() {
    match op_ext[i] {
      0x66 => pp = 1,
      0xF3 => pp = 2,
      0xF2 => pp = 3,
      0x0F => {
        match op_ext[i - 1] {
          0x3A => mmm = 3,
          0x38 => mmm = 2,
          _ => mmm = 1,
        }
        break;
      }
      _ => {}
    }
  }

  let (vvvv, V) = match op3 {
    Arg::Reg(reg) if !op3.is_mask_register() => {
      (!(reg.reg_id() as u8) & 0xF, ((reg.is_upper_16_reg() as u8) ^ 0x1))
    }
    _ => (0xF, 0x1),
  };
  let (vvvv, V) = (vvvv << 3, V << 3);

  let z = 0 << 7;
  let ll = match bit_size {
    b512 => 2,
    b256 => 1,
    b128 | _ => 0,
  } << 5;
  let _b = 0 << 4;

  let aaa = if op3.is_mask_register() { op3.reg_index() } else { 0 };

  let mut byte1 = rex_r | rex_x | rex_b | rex_r_prime | 0 | mmm;
  let mut byte2 = rex_w | vvvv | (1 << 2) | pp;
  let mut byte3 = z | ll | _b | V | aaa;

  insert_op_code_bytes(props.bin, byte1 as u32);
  insert_op_code_bytes(props.bin, byte2 as u32);
  insert_op_code_bytes(props.bin, byte3 as u32);

  op_ext[0] as u32
}

fn encode_vex(
  op_code: u32,
  op2: Arg,
  op1: Arg,
  bit_size: BitSize,
  props: &mut InstructionProps<'_>,
) -> u32 {
  // two byte form
  let op_ext: [u8; 4] = unsafe { std::mem::transmute(op_code) };
  let mut pp = 0;
  let mut m_mmmmm = 0;

  for i in (0..4).rev() {
    match op_ext[i] {
      0x66 => pp = 1,
      0xF3 => pp = 2,
      0xF2 => pp = 3,
      0x0F => {
        match op_ext[i - 1] {
          0x3A => m_mmmmm = 3,
          0x38 => m_mmmmm = 2,
          _ => m_mmmmm = 1,
        }
        break;
      }
      _ => {}
    }
  }

  let op_code = op_ext[0] as u32;

  let use_three = op2.is_upper_8_reg();
  let rex_r = ((op1.is_upper_8_reg() as u8) ^ 1) << 7;
  let rex_x = 0;
  let rex_b = ((op2.is_upper_8_reg() as u8) ^ 1) << 6;
  let rex_w = 0;

  let vvvv = 0xF << 3;

  let vec_len = match bit_size {
    b256 => 0b100u8,
    b128 => 0b000u8,
    _ => 0,
  };

  if use_three {
    insert_op_code_bytes(props.bin, 0xC4);
    let mut byte1 = rex_r | rex_x | rex_b | m_mmmmm;
    let mut byte2 = rex_w | vvvv | vec_len | pp;
    insert_op_code_bytes(props.bin, byte1 as u32);
    insert_op_code_bytes(props.bin, byte2 as u32);
  } else {
    insert_op_code_bytes(props.bin, 0xC5);
    let mut byte1 = rex_r | vvvv | vec_len | pp;
    insert_op_code_bytes(props.bin, byte1 as u32);
  }
  op_code
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
        encode_rex(props, bit_size, op1, op2);
        insert_op_code_bytes(props.bin, op_code);

        encode_mod_rm_reg(props, op1, op2);
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
