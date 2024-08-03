use crate::{
  ir::{ir_graph::IRGraphId, ir_register_allocator::Reg},
  types::ConstVal,
};
use std::collections::BTreeMap;

#[derive(Debug, Hash, Clone, Copy)]
pub(super) enum OpEncoding {
  Zero,
  VEX_MR,
  VEX_RM,
  EVEX_RM {
    w: u8,
  },
  EVEX_MR {
    w: u8,
  },
  /// ### Register to Memory/Register (SSE/AVX)
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op(0x66)   | mod:rm(w)     | mod:reg(r)    |
  A,
  /// ### Memory/Register to Register (SSE/AVX)
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op(0x66)   | mod:reg(w)    | mod:rm(r)     |
  B,
  /// ### Register to Memory/Register
  ///
  /// | opcode      
  /// | ------    
  /// | opcode + rd (r)
  O,
  /// ### Register to Memory/Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | offset        |               |
  D,
  /// ### Memory/Register Value
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | mod:reg(w)    |               |
  M,
  /// ### Register to Memory/Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | mod:rm(w)     | mod:reg(r)    |
  MR,
  /// ### Memory/Register to Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | mod:reg(w)    | mod:rm(r)     |
  RM,
  /// ### SEG/OFFSET to Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | RAX/EAX/AX/AL | Moffs         |
  FD,
  /// ### Register to SEG/OFFSET
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | Moffs(w)      | RAX/EAX/AX/AL |
  TD,
  /// ### Immediate to Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op + rd(2) | imm           |
  OI,
  /// ### Immediate to Memory
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | mod:rm(w)     | imm           |
  MI,
  /// ### Immediate to Register
  ///
  /// | opcode     | operand1      | operand2      |
  /// | ------     | ------        | ------        |
  /// | op         | Reg           | imm           |
  I,

  RMI,
}

const GENERAL_REGISTER: u16 = 0 << 7;
const VECTOR_REGISTER: u16 = 1 << 8;
const MASK_REGISTER: u16 = 1 << 9;

pub const K0: Reg = Reg::new(00 | MASK_REGISTER);
pub const K1: Reg = Reg::new(01 | MASK_REGISTER);
pub const K2: Reg = Reg::new(02 | MASK_REGISTER);
pub const K3: Reg = Reg::new(03 | MASK_REGISTER);
pub const K4: Reg = Reg::new(04 | MASK_REGISTER);
pub const K5: Reg = Reg::new(05 | MASK_REGISTER);
pub const K6: Reg = Reg::new(06 | MASK_REGISTER);
pub const K7: Reg = Reg::new(07 | MASK_REGISTER);

pub const RAX: Reg = Reg::new(00 | GENERAL_REGISTER);
pub const RCX: Reg = Reg::new(01 | GENERAL_REGISTER);
pub const RDX: Reg = Reg::new(02 | GENERAL_REGISTER);
pub const RBX: Reg = Reg::new(03 | GENERAL_REGISTER);
pub const RSP: Reg = Reg::new(04 | GENERAL_REGISTER);
pub const RBP: Reg = Reg::new(05 | GENERAL_REGISTER);
pub const RSI: Reg = Reg::new(06 | GENERAL_REGISTER);
pub const RDI: Reg = Reg::new(07 | GENERAL_REGISTER);
pub const R8: Reg = Reg::new(08 | GENERAL_REGISTER);
pub const R9: Reg = Reg::new(09 | GENERAL_REGISTER);
pub const R10: Reg = Reg::new(10 | GENERAL_REGISTER);
pub const R11: Reg = Reg::new(11 | GENERAL_REGISTER);
pub const R12: Reg = Reg::new(12 | GENERAL_REGISTER);
pub const R13: Reg = Reg::new(13 | GENERAL_REGISTER);
pub const R14: Reg = Reg::new(14 | GENERAL_REGISTER);
pub const R15: Reg = Reg::new(15 | GENERAL_REGISTER);

pub const XMM0: Reg = Reg::new(00 | VECTOR_REGISTER);
pub const XMM1: Reg = Reg::new(01 | VECTOR_REGISTER);
pub const XMM2: Reg = Reg::new(02 | VECTOR_REGISTER);
pub const XMM3: Reg = Reg::new(03 | VECTOR_REGISTER);
pub const XMM4: Reg = Reg::new(04 | VECTOR_REGISTER);
pub const XMM5: Reg = Reg::new(05 | VECTOR_REGISTER);
pub const XMM6: Reg = Reg::new(06 | VECTOR_REGISTER);
pub const XMM7: Reg = Reg::new(07 | VECTOR_REGISTER);
pub const XMM8: Reg = Reg::new(08 | VECTOR_REGISTER);
pub const XMM9: Reg = Reg::new(09 | VECTOR_REGISTER);
pub const XMM10: Reg = Reg::new(10 | VECTOR_REGISTER);
pub const XMM11: Reg = Reg::new(11 | VECTOR_REGISTER);
pub const XMM12: Reg = Reg::new(12 | VECTOR_REGISTER);
pub const XMM13: Reg = Reg::new(13 | VECTOR_REGISTER);
pub const XMM14: Reg = Reg::new(14 | VECTOR_REGISTER);
pub const XMM15: Reg = Reg::new(15 | VECTOR_REGISTER);

pub const YMM0: Reg = Reg::new(00 | VECTOR_REGISTER);
pub const YMM1: Reg = Reg::new(01 | VECTOR_REGISTER);
pub const YMM2: Reg = Reg::new(02 | VECTOR_REGISTER);
pub const YMM3: Reg = Reg::new(03 | VECTOR_REGISTER);
pub const YMM4: Reg = Reg::new(04 | VECTOR_REGISTER);
pub const YMM5: Reg = Reg::new(05 | VECTOR_REGISTER);
pub const YMM6: Reg = Reg::new(06 | VECTOR_REGISTER);
pub const YMM7: Reg = Reg::new(07 | VECTOR_REGISTER);
pub const YMM8: Reg = Reg::new(08 | VECTOR_REGISTER);
pub const YMM9: Reg = Reg::new(09 | VECTOR_REGISTER);
pub const YMM10: Reg = Reg::new(10 | VECTOR_REGISTER);
pub const YMM11: Reg = Reg::new(11 | VECTOR_REGISTER);
pub const YMM12: Reg = Reg::new(12 | VECTOR_REGISTER);
pub const YMM13: Reg = Reg::new(13 | VECTOR_REGISTER);
pub const YMM14: Reg = Reg::new(14 | VECTOR_REGISTER);
pub const YMM15: Reg = Reg::new(15 | VECTOR_REGISTER);

pub const ZMM0: Reg = Reg::new(00 | VECTOR_REGISTER);
pub const ZMM1: Reg = Reg::new(01 | VECTOR_REGISTER);
pub const ZMM2: Reg = Reg::new(02 | VECTOR_REGISTER);
pub const ZMM3: Reg = Reg::new(03 | VECTOR_REGISTER);
pub const ZMM4: Reg = Reg::new(04 | VECTOR_REGISTER);
pub const ZMM5: Reg = Reg::new(05 | VECTOR_REGISTER);
pub const ZMM6: Reg = Reg::new(06 | VECTOR_REGISTER);
pub const ZMM7: Reg = Reg::new(07 | VECTOR_REGISTER);
pub const ZMM8: Reg = Reg::new(08 | VECTOR_REGISTER);
pub const ZMM9: Reg = Reg::new(09 | VECTOR_REGISTER);
pub const ZMM10: Reg = Reg::new(10 | VECTOR_REGISTER);
pub const ZMM11: Reg = Reg::new(11 | VECTOR_REGISTER);
pub const ZMM12: Reg = Reg::new(12 | VECTOR_REGISTER);
pub const ZMM13: Reg = Reg::new(13 | VECTOR_REGISTER);
pub const ZMM14: Reg = Reg::new(14 | VECTOR_REGISTER);
pub const ZMM15: Reg = Reg::new(15 | VECTOR_REGISTER);
pub const ZMM16: Reg = Reg::new(16 | VECTOR_REGISTER);
pub const ZMM17: Reg = Reg::new(17 | VECTOR_REGISTER);
pub const ZMM18: Reg = Reg::new(18 | VECTOR_REGISTER);
pub const ZMM19: Reg = Reg::new(19 | VECTOR_REGISTER);
pub const ZMM20: Reg = Reg::new(20 | VECTOR_REGISTER);
pub const ZMM21: Reg = Reg::new(21 | VECTOR_REGISTER);
pub const ZMM22: Reg = Reg::new(22 | VECTOR_REGISTER);
pub const ZMM23: Reg = Reg::new(23 | VECTOR_REGISTER);
pub const ZMM24: Reg = Reg::new(24 | VECTOR_REGISTER);
pub const ZMM25: Reg = Reg::new(25 | VECTOR_REGISTER);
pub const ZMM26: Reg = Reg::new(26 | VECTOR_REGISTER);
pub const ZMM27: Reg = Reg::new(27 | VECTOR_REGISTER);
pub const ZMM28: Reg = Reg::new(28 | VECTOR_REGISTER);
pub const ZMM29: Reg = Reg::new(29 | VECTOR_REGISTER);
pub const ZMM30: Reg = Reg::new(30 | VECTOR_REGISTER);
pub const ZMM31: Reg = Reg::new(31 | VECTOR_REGISTER);

impl Reg {
  const SIB_RM: u8 = 0b100;

  pub(super) fn index(&self) -> u8 {
    match *self {
      R8 | RAX | XMM0 => 0x00,
      R9 | RCX | XMM1 => 0x01,
      R10 | RDX | XMM2 => 0x02,
      R11 | RBX | XMM3 => 0x03,
      R12 | RSP | XMM4 => 0x04,
      R13 | RBP | XMM5 => 0x05,
      R14 | RSI | XMM6 => 0x06,
      R15 | RDI | XMM7 => 0x07,
      _ => Self::SIB_RM, // uses SIB byte
    }
  }

  pub(super) fn is_general_purpose(&self) -> bool {
    (self.0 & (VECTOR_REGISTER | MASK_REGISTER)) == 0
  }

  /// The register is one of R8-R15
  pub(super) fn is_ext_8_reg(&self) -> bool {
    (self.0 & 0x8) > 0
  }

  pub(super) fn is_upper_16_reg(&self) -> bool {
    (self.0 & 0x1F) >= 16
  }

  pub fn as_addr_op(&self) -> Arg {
    Arg::Mem(*self)
  }

  /// Returns an Arg::Reg op for the given register. Panics if the Graphid is
  /// not a register
  pub fn as_reg_op(&self) -> Arg {
    Arg::Reg(*self)
  }

  /// Returns an Arg::Mem op for the given register. Panics if the Graphid is
  /// not a register
  pub fn as_mem_op(&self) -> Arg {
    Arg::Mem(*self)
  }
}

#[derive(PartialEq, Debug, Hash)]
pub(super) enum OperandType {
  REG,
  MEM,
  IMM_INT,
  NONE,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arg {
  Reg(Reg),
  Mem(Reg),
  RSP_REL(u64),
  RIP_REL(u64),
  Imm_Int(i64),
  OpExt(u8),
  None,
}

impl Arg {
  pub(super) fn ty(&self) -> OperandType {
    match self {
      Arg::Imm_Int(..) => OperandType::IMM_INT,
      Arg::Reg(..) => OperandType::REG,
      Arg::Mem(..) => OperandType::MEM,
      Arg::RSP_REL(..) => OperandType::MEM,
      Arg::RIP_REL(..) => OperandType::MEM,
      _ => OperandType::NONE,
    }
  }

  pub(super) fn from_const(c: ConstVal) -> Self {
    let val: i128 = unsafe { std::mem::transmute(c.val) };
    Arg::Imm_Int(val as i64)
  }

  /// Converts the argument from an operation on a value stored in a register to
  /// an operation performed on the memory location resolved from the registers
  /// value
  pub(super) fn to_mem(&self) -> Arg {
    match self {
      Arg::Reg(reg) => Arg::Mem(*reg),
      arg => *arg,
    }
  }

  pub(super) fn is_reg(&self) -> bool {
    matches!(self, Arg::Reg(..))
  }

  pub(super) fn reg_index(&self) -> u8 {
    match self {
      Arg::RIP_REL(_) => 0x5,
      Arg::RSP_REL(_) => 0x4,
      Arg::Reg(reg) | Arg::Mem(reg) => (reg.0 & 7) as u8,
      Self::OpExt(index) => *index,

      arg => unreachable!("{arg:?}"),
    }
  }

  pub(super) fn is_mask_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.0 & MASK_REGISTER) > 0,
      _ => false,
    }
  }

  pub(super) fn is_vector_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.0 & VECTOR_REGISTER) > 0,
      _ => false,
    }
  }

  pub(super) fn is_general_purpose(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.0 & (VECTOR_REGISTER | MASK_REGISTER)) == 0,
      _ => false,
    }
  }

  pub(super) fn is_upper_8_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) => reg.is_ext_8_reg(),
      _ => false,
    }
  }

  pub(super) fn is_upper_16_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) => reg.is_upper_16_reg(),
      _ => false,
    }
  }
}
