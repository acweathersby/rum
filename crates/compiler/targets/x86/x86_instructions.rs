#![allow(unused, non_upper_case_globals)]

use super::{set_bytes, x86_types::*};

use super::x86_encoder::*;

use std::collections::HashMap;
use OperandType as OT;

macro_rules! op_table {
  ($name: ident $value: expr) => {
    pub(crate) const $name: (&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder)); $value.len()]) = (stringify!($name), $value);
  };
}

/// https://www.felixcloutier.com/x86/call
op_table!(
  syscall [
    ((8, OT::NONE, OT::NONE, OT::NONE), (0x0F05, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder as *const OpEncoder)),
    //
    ((16, OT::NONE, OT::NONE, OT::NONE), (0x0F05, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder as *const OpEncoder)),
    //
    ((32, OT::NONE, OT::NONE, OT::NONE), (0x0F05, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder as *const OpEncoder)),
    //
    ((64, OT::NONE, OT::NONE, OT::NONE), (0x0F05, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder as *const OpEncoder)),
  ]
);

#[test]
fn test_syscall() {
  assert_eq!("syscall", test_enc_uno(&syscall, 8, Arg::None));
  assert_eq!("syscall", test_enc_uno(&syscall, 16, Arg::None));
  assert_eq!("syscall", test_enc_uno(&syscall, 32, Arg::None));
  assert_eq!("syscall", test_enc_uno(&syscall, 64, Arg::None));
}

/// https://www.felixcloutier.com/x86/call
op_table!(call [
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x00E8, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder as *const OpEncoder)),
//
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x02, OpEncoding::M, gen_unary_op as *const OpEncoder as *const OpEncoder)),
//
((64, OT::REG, OT::NONE, OT::NONE), (0x00FF, 0x02, OpEncoding::M, gen_unary_op as *const OpEncoder as *const OpEncoder)),
]);

#[test]
fn test_call() {
  assert_eq!("call 37", test_enc_uno(&call, 32, Arg::Imm_Int(32)));
  assert_eq!("call r8", test_enc_uno(&call, 64, R8.as_reg_op()));
  assert_eq!("call qword ptr [r8]", test_enc_uno(&call, 64, R8.as_mem_op()));
  assert_eq!("call qword ptr [rip+23]", test_enc_uno(&call, 64, Arg::RIP_REL(23)));
  assert_eq!("call qword ptr [rsp+23]", test_enc_uno(&call, 64, Arg::RSP_REL(23)));
  assert_eq!("call qword ptr [rbx]", test_enc_uno(&call, 64, RBX.as_mem_op()));
  assert_eq!("call qword ptr [r13]", test_enc_uno(&call, 64, R13.as_mem_op()));
}

/// https://www.felixcloutier.com/x86/ret
op_table!(ret [
((8, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder)),
((16, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder)),
((32, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder)),
((64, OT::NONE, OT::NONE, OT::NONE), (0x00C3, 0x00, OpEncoding::Zero, gen_zero_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/ret
op_table!(ret_pop [
((16, OT::IMM_INT, OT::NONE, OT::NONE), (0x00C2, 0x02, OpEncoding::I, gen_unary_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/push
op_table!(push [
((16, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x06, OpEncoding::M, gen_unary_op as *const OpEncoder)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x06, OpEncoding::M, gen_unary_op as *const OpEncoder)),
//
((16, OT::REG, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::O, gen_unary_op as *const OpEncoder)),
((64, OT::REG, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::O, gen_unary_op as *const OpEncoder)),
//
((16, OT::IMM_INT, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::I, gen_unary_op as *const OpEncoder)),
((64, OT::IMM_INT, OT::NONE, OT::NONE), (0x0050, 0x00, OpEncoding::I, gen_unary_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/pop
op_table!(pop [
((16, OT::MEM, OT::NONE, OT::NONE), (0x008F, 0x06, OpEncoding::M, gen_unary_op as *const OpEncoder)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x008F, 0x06, OpEncoding::M, gen_unary_op as *const OpEncoder)),
//
((16, OT::REG, OT::NONE, OT::NONE), (0x0058, 0x00, OpEncoding::O, gen_unary_op as *const OpEncoder)),
((64, OT::REG, OT::NONE, OT::NONE), (0x0058, 0x00, OpEncoding::O, gen_unary_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/jmp
op_table!(jmp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x00EB, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x00E9, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
//
((64, OT::REG, OT::NONE, OT::NONE), (0x00FF, 0x04, OpEncoding::M, gen_unary_op as *const OpEncoder)),
//
((16, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op as *const OpEncoder)),
((32, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op as *const OpEncoder)),
((64, OT::MEM, OT::NONE, OT::NONE), (0x00FF, 0x05, OpEncoding::M, gen_unary_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/jcc
op_table!(jb [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0072, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F82, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jb() {
  assert_eq!("jb short 18_446_744_073_709_551_586", test_enc_uno(&jb, 8, Arg::Imm_Int(-32)));
  assert_eq!("jb near ptr 38", test_enc_uno(&jb, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jae [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0073, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F83, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jae() {
  assert_eq!("jae short 18_446_744_073_709_551_586", test_enc_uno(&jae, 8, Arg::Imm_Int(-32)));
  assert_eq!("jae near ptr 38", test_enc_uno(&jae, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(je [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0074, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F84, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_je() {
  assert_eq!("je short 18_446_744_073_709_551_586", test_enc_uno(&je, 8, Arg::Imm_Int(-32)));
  assert_eq!("je near ptr 38", test_enc_uno(&je, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jne [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0075, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F85, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jne() {
  assert_eq!("jne short 18_446_744_073_709_551_586", test_enc_uno(&jne, 8, Arg::Imm_Int(-32)));
  assert_eq!("jne near ptr 38", test_enc_uno(&jne, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jbe [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0076, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F86, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jbe() {
  assert_eq!("jbe short 18_446_744_073_709_551_586", test_enc_uno(&jbe, 8, Arg::Imm_Int(-32)));
  assert_eq!("jbe near ptr 38", test_enc_uno(&jbe, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(ja [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0077, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F87, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_ja() {
  assert_eq!("ja short 18_446_744_073_709_551_586", test_enc_uno(&ja, 8, Arg::Imm_Int(-32)));
  assert_eq!("ja near ptr 38", test_enc_uno(&ja, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(js [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0078, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F88, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_js() {
  assert_eq!("js short 18_446_744_073_709_551_586", test_enc_uno(&js, 8, Arg::Imm_Int(-32)));
  assert_eq!("js near ptr 38", test_enc_uno(&js, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jns [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x0079, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F89, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jns() {
  assert_eq!("jns short 18_446_744_073_709_551_586", test_enc_uno(&jns, 8, Arg::Imm_Int(-32)));
  assert_eq!("jns near ptr 38", test_enc_uno(&jns, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007A, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8A, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jp() {
  assert_eq!("jp short 18_446_744_073_709_551_586", test_enc_uno(&jp, 8, Arg::Imm_Int(-32)));
  assert_eq!("jp near ptr 38", test_enc_uno(&jp, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jnp [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007B, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8B, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jnp() {
  assert_eq!("jnp short 18_446_744_073_709_551_586", test_enc_uno(&jnp, 8, Arg::Imm_Int(-32)));
  assert_eq!("jnp near ptr 38", test_enc_uno(&jnp, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jl [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007C, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8C, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jl() {
  assert_eq!("jl short 18_446_744_073_709_551_586", test_enc_uno(&jl, 8, Arg::Imm_Int(-32)));
  assert_eq!("jl near ptr 38", test_enc_uno(&jl, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jge [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007D, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8D, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jge() {
  assert_eq!("jge short 18_446_744_073_709_551_586", test_enc_uno(&jge, 8, Arg::Imm_Int(-32)));
  assert_eq!("jge near ptr 38", test_enc_uno(&jge, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jle [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007E, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8E, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jle() {
  assert_eq!("jle short 18_446_744_073_709_551_586", test_enc_uno(&jle, 8, Arg::Imm_Int(-32)));
  assert_eq!("jle near ptr 38", test_enc_uno(&jle, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/jcc
op_table!(jg [
((08, OT::IMM_INT, OT::NONE, OT::NONE), (0x007F, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
((32, OT::IMM_INT, OT::NONE, OT::NONE), (0x0F8F, 0x00, OpEncoding::D, gen_unary_op as *const OpEncoder)),
]);

#[test]
fn test_jg() {
  assert_eq!("jg short 18_446_744_073_709_551_586", test_enc_uno(&jg, 8, Arg::Imm_Int(-32)));
  assert_eq!("jg near ptr 38", test_enc_uno(&jg, 32, Arg::Imm_Int(32)));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_o [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F40, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_o() {
  assert_eq!("cmovo r8d,r11d", test_enc_dos(&mov_o, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovo r8,r11", test_enc_dos(&mov_o, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovo r8d,[r11]", test_enc_dos(&mov_o, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovo r8,[r11]", test_enc_dos(&mov_o, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_no [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F41, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_no() {
  assert_eq!("cmovno r8d,r11d", test_enc_dos(&mov_no, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovno r8,r11", test_enc_dos(&mov_no, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovno r8d,[r11]", test_enc_dos(&mov_no, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovno r8,[r11]", test_enc_dos(&mov_no, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_b [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F42, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ae [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F43, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_ae() {
  assert_eq!("cmovae r8d,r11d", test_enc_dos(&mov_ae, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovae r8,r11", test_enc_dos(&mov_ae, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovae r8d,[r11]", test_enc_dos(&mov_ae, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovae r8,[r11]", test_enc_dos(&mov_ae, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_e [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F44, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_e() {
  assert_eq!("cmove r8d,r11d", test_enc_dos(&mov_e, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmove r8,r11", test_enc_dos(&mov_e, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmove r8d,[r11]", test_enc_dos(&mov_e, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmove r8,[r11]", test_enc_dos(&mov_e, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ne [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F45, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_ne() {
  assert_eq!("cmovne r8d,r11d", test_enc_dos(&mov_ne, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovne r8,r11", test_enc_dos(&mov_ne, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovne r8d,[r11]", test_enc_dos(&mov_ne, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovne r8,[r11]", test_enc_dos(&mov_ne, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_be [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F46, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_be() {
  assert_eq!("cmovbe r8d,r11d", test_enc_dos(&mov_be, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovbe r8,r11", test_enc_dos(&mov_be, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovbe r8d,[r11]", test_enc_dos(&mov_be, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovbe r8,[r11]", test_enc_dos(&mov_be, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_a [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F47, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_a() {
  assert_eq!("cmova r8d,r11d", test_enc_dos(&mov_a, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmova r8,r11", test_enc_dos(&mov_a, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmova r8d,[r11]", test_enc_dos(&mov_a, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmova r8,[r11]", test_enc_dos(&mov_a, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_s [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F48, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_s() {
  assert_eq!("cmovs r10d,r11d", test_enc_dos(&mov_s, 32, R10.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovs r10,r11", test_enc_dos(&mov_s, 64, R10.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovs r10d,[r11]", test_enc_dos(&mov_s, 32, R10.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovs r10,[r11]", test_enc_dos(&mov_s, 64, R10.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ns [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F49, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_ns() {
  assert_eq!("cmovns r8d,r11d", test_enc_dos(&mov_ns, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovns r8,r11", test_enc_dos(&mov_ns, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovns r8d,[r11]", test_enc_dos(&mov_ns, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovns r8,[r11]", test_enc_dos(&mov_ns, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_pe [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_pe() {
  assert_eq!("cmovp r8d,r11d", test_enc_dos(&mov_pe, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovp r8,r11", test_enc_dos(&mov_pe, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovp r8d,[r11]", test_enc_dos(&mov_pe, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovp r8,[r11]", test_enc_dos(&mov_pe, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_po [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_po() {
  assert_eq!("cmovnp r8d,r11d", test_enc_dos(&mov_po, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovnp r8,r11", test_enc_dos(&mov_po, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovnp r8d,[r11]", test_enc_dos(&mov_po, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovnp r8,[r11]", test_enc_dos(&mov_po, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_l [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4C, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_l() {
  assert_eq!("cmovl r8d,r11d", test_enc_dos(&mov_l, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovl r8,r11", test_enc_dos(&mov_l, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovl r8d,[r11]", test_enc_dos(&mov_l, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovl r8,[r11]", test_enc_dos(&mov_l, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_ge [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_ge() {
  assert_eq!("cmovge r8d,r11d", test_enc_dos(&mov_ge, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovge r8,r11", test_enc_dos(&mov_ge, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovge r8d,[r11]", test_enc_dos(&mov_ge, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovge r8,[r11]", test_enc_dos(&mov_ge, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_le [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4E, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_le() {
  assert_eq!("cmovle r8d,r11d", test_enc_dos(&mov_le, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovle r8,r11", test_enc_dos(&mov_le, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovle r8d,[r11]", test_enc_dos(&mov_le, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovle r8,[r11]", test_enc_dos(&mov_le, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/cmovcc
op_table!(mov_g [
  ((16, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0F4F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_mov_g() {
  assert_eq!("cmovg r8d,r11d", test_enc_dos(&mov_g, 32, R8.as_reg_op(), R11.as_reg_op()));
  assert_eq!("cmovg r8,r11", test_enc_dos(&mov_g, 64, R8.as_reg_op(), R11.as_reg_op()));

  assert_eq!("cmovg r8d,[r11]", test_enc_dos(&mov_g, 32, R8.as_reg_op(), R11.as_addr_op()));
  assert_eq!("cmovg r8,[r11]", test_enc_dos(&mov_g, 64, R8.as_reg_op(), R11.as_addr_op()));
}

/// https://www.felixcloutier.com/x86/mov
op_table!(mov [
  ((08, OT::REG, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0088, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0089, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x008A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x008B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  //
  ((8, OT::REG, OT::IMM_INT, OT::NONE), (0x00B0, 0x00, OpEncoding::OI, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x00B8, 0x00, OpEncoding::OI, gen_multi_op as *const OpEncoder)),
  //
  ((8, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C6, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x00C7, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/mov
op_table!(lea [
  ((08, OT::REG, OT::MEM, OT::NONE), (0x008D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x008D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x008D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x008D, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),

]);

#[test]
fn test_lea() {
  assert_eq!("lea r8d,[r11]", test_enc_dos(&lea, 32, R8.as_reg_op(), R11.as_reg_op().to_mem()));
  assert_eq!("cmovg r8,r11", test_enc_dos(&lea, 64, R8.as_reg_op(), Arg::RSP_REL(128)));
}

/// https://www.felixcloutier.com/x86/mov
op_table!(vmov32_align [
  ((128, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((128, OT::REG, OT::MEM, OT::NONE), (0x660F6F, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((128, OT::MEM, OT::REG, OT::NONE), (0x660F7F, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((256, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::VEX_RM, gen_multi_op as *const OpEncoder)),
  ((256, OT::REG, OT::MEM, OT::NONE), (0x660F6F, 0x00, OpEncoding::VEX_RM, gen_multi_op as *const OpEncoder)),
  ((256, OT::MEM, OT::REG, OT::NONE), (0x660F7F, 0x00, OpEncoding::VEX_MR, gen_multi_op as *const OpEncoder)),

  ((512, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:0 }, gen_multi_op as *const OpEncoder)),
]);

op_table!(vmov64_align [
  ((512, OT::REG, OT::REG, OT::NONE), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:1 }, gen_multi_op as *const OpEncoder)),
  ((512, OT::REG, OT::REG, OT::REG), (0x660F6F, 0x00, OpEncoding::EVEX_RM{ w:1 }, gen_multi_op as *const OpEncoder)),
]);

#[test]
fn test_vec_mov() {
  assert_eq!("vmovdqa64 zmm31{k1},zmm31", test_enc_tres(&vmov64_align, 512, ZMM31.as_reg_op(), ZMM31.as_reg_op(), K1.as_reg_op()));

  assert_eq!("vmovdqa32 zmm31,zmm31", test_enc_dos(&vmov32_align, 512, ZMM31.as_reg_op(), ZMM31.as_reg_op()));

  assert_eq!("vmovdqa32 zmm0,zmm0", test_enc_dos(&vmov32_align, 512, ZMM0.as_reg_op(), ZMM0.as_reg_op()));

  assert_eq!("movdqa xmm0,xmm1", test_enc_dos(&vmov32_align, 128, XMM0.as_reg_op(), XMM1.as_reg_op()));
  assert_eq!("movdqa [rbx],xmm3", test_enc_dos(&vmov32_align, 128, RBX.as_mem_op(), XMM3.as_reg_op()));
  assert_eq!("vmovdqa ymm15,ymmword ptr [r8]", test_enc_dos(&vmov32_align, 256, XMM15.as_reg_op(), R8.as_mem_op()));
  assert_eq!("vmovdqa ymmword ptr [r15],ymm8", test_enc_dos(&vmov32_align, 256, XMM15.as_mem_op(), R8.as_reg_op()));
}

/// https://www.felixcloutier.com/x86/cmp
op_table!(cmp [
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ///
  ((08, OT::MEM, OT::IMM_INT, OT::NONE), (0x0080, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::IMM_INT, OT::NONE), (0x0081, 0x07, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ///
  ((08, OT::REG, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0038, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0039, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ///
  ((08, OT::REG, OT::MEM, OT::NONE), (0x003A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x003B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/add
op_table!(add [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x0000, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0001, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x0002, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x0003, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x00, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/sub
op_table!(sub [
  ((08, OT::MEM, OT::REG, OT::NONE), (0x0028, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::REG, OT::NONE), (0x0029, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::REG, OT::NONE), (0x002A, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x002B, 0x00, OpEncoding::MR, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::MEM, OT::NONE), (0x002A, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::MEM, OT::NONE), (0x002B, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  //
  ((08, OT::REG, OT::IMM_INT, OT::NONE), (0x0080, 0x05, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::IMM_INT, OT::NONE), (0x0081, 0x05, OpEncoding::MI, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/mul
op_table!(mul [  //
  ((08, OT::REG, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((16, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((08, OT::MEM, OT::NONE, OT::NONE), (0x00F6, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((16, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((32, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
  ((64, OT::MEM, OT::NONE, OT::NONE), (0x00F7, 0x04, OpEncoding::M, gen_multi_op as *const OpEncoder)),
]);

/// https://www.felixcloutier.com/x86/imul
op_table!(imul [  //
  ((16, OT::REG, OT::REG, OT::NONE), (0x0FAF, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::NONE), (0x0FAF, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::NONE), (0x0FAF, 0x00, OpEncoding::RM, gen_multi_op as *const OpEncoder)),

  ((08, OT::REG, OT::REG, OT::IMM_INT), (0x0068, 0x00, OpEncoding::RMI, gen_tri_op as *const OpEncoder)),
  ((16, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op as *const OpEncoder)),
  ((32, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op as *const OpEncoder)),
  ((64, OT::REG, OT::REG, OT::IMM_INT), (0x0069, 0x00, OpEncoding::RMI, gen_tri_op as *const OpEncoder)),
]);

#[test]
fn test_imul() {
  assert_eq!("imul r8,r8", test_enc_dos(&imul, 64, R8.as_reg_op(), R8.as_reg_op()));
  assert_eq!("imul rax,r15", test_enc_dos(&imul, 64, RAX.as_reg_op(), R15.as_reg_op()));
}
