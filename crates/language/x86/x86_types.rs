use crate::ir::{ir_context::IRCallable, ir_graph::IRGraphId};
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

const GENERAL_REGISTER: usize = 0 << 7;
const VECTOR_REGISTER: usize = 1 << 8;
const MASK_REGISTER: usize = 1 << 9;

pub const K0: IRGraphId = IRGraphId::register(00 | MASK_REGISTER);
pub const K1: IRGraphId = IRGraphId::register(01 | MASK_REGISTER);
pub const K2: IRGraphId = IRGraphId::register(02 | MASK_REGISTER);
pub const K3: IRGraphId = IRGraphId::register(03 | MASK_REGISTER);
pub const K4: IRGraphId = IRGraphId::register(04 | MASK_REGISTER);
pub const K5: IRGraphId = IRGraphId::register(05 | MASK_REGISTER);
pub const K6: IRGraphId = IRGraphId::register(06 | MASK_REGISTER);
pub const K7: IRGraphId = IRGraphId::register(07 | MASK_REGISTER);

pub const RAX: IRGraphId = IRGraphId::register(00 | GENERAL_REGISTER);
pub const RCX: IRGraphId = IRGraphId::register(01 | GENERAL_REGISTER);
pub const RDX: IRGraphId = IRGraphId::register(02 | GENERAL_REGISTER);
pub const RBX: IRGraphId = IRGraphId::register(03 | GENERAL_REGISTER);
pub const RSP: IRGraphId = IRGraphId::register(04 | GENERAL_REGISTER);
pub const RBP: IRGraphId = IRGraphId::register(05 | GENERAL_REGISTER);
pub const RSI: IRGraphId = IRGraphId::register(06 | GENERAL_REGISTER);
pub const RDI: IRGraphId = IRGraphId::register(07 | GENERAL_REGISTER);
pub const R8: IRGraphId = IRGraphId::register(08 | GENERAL_REGISTER);
pub const R9: IRGraphId = IRGraphId::register(09 | GENERAL_REGISTER);
pub const R10: IRGraphId = IRGraphId::register(10 | GENERAL_REGISTER);
pub const R11: IRGraphId = IRGraphId::register(11 | GENERAL_REGISTER);
pub const R12: IRGraphId = IRGraphId::register(12 | GENERAL_REGISTER);
pub const R13: IRGraphId = IRGraphId::register(13 | GENERAL_REGISTER);
pub const R14: IRGraphId = IRGraphId::register(14 | GENERAL_REGISTER);
pub const R15: IRGraphId = IRGraphId::register(15 | GENERAL_REGISTER);

pub const XMM0: IRGraphId = IRGraphId::register(00 | VECTOR_REGISTER);
pub const XMM1: IRGraphId = IRGraphId::register(01 | VECTOR_REGISTER);
pub const XMM2: IRGraphId = IRGraphId::register(02 | VECTOR_REGISTER);
pub const XMM3: IRGraphId = IRGraphId::register(03 | VECTOR_REGISTER);
pub const XMM4: IRGraphId = IRGraphId::register(04 | VECTOR_REGISTER);
pub const XMM5: IRGraphId = IRGraphId::register(05 | VECTOR_REGISTER);
pub const XMM6: IRGraphId = IRGraphId::register(06 | VECTOR_REGISTER);
pub const XMM7: IRGraphId = IRGraphId::register(07 | VECTOR_REGISTER);
pub const XMM8: IRGraphId = IRGraphId::register(08 | VECTOR_REGISTER);
pub const XMM9: IRGraphId = IRGraphId::register(09 | VECTOR_REGISTER);
pub const XMM10: IRGraphId = IRGraphId::register(10 | VECTOR_REGISTER);
pub const XMM11: IRGraphId = IRGraphId::register(11 | VECTOR_REGISTER);
pub const XMM12: IRGraphId = IRGraphId::register(12 | VECTOR_REGISTER);
pub const XMM13: IRGraphId = IRGraphId::register(13 | VECTOR_REGISTER);
pub const XMM14: IRGraphId = IRGraphId::register(14 | VECTOR_REGISTER);
pub const XMM15: IRGraphId = IRGraphId::register(15 | VECTOR_REGISTER);

pub const YMM0: IRGraphId = IRGraphId::register(00 | VECTOR_REGISTER);
pub const YMM1: IRGraphId = IRGraphId::register(01 | VECTOR_REGISTER);
pub const YMM2: IRGraphId = IRGraphId::register(02 | VECTOR_REGISTER);
pub const YMM3: IRGraphId = IRGraphId::register(03 | VECTOR_REGISTER);
pub const YMM4: IRGraphId = IRGraphId::register(04 | VECTOR_REGISTER);
pub const YMM5: IRGraphId = IRGraphId::register(05 | VECTOR_REGISTER);
pub const YMM6: IRGraphId = IRGraphId::register(06 | VECTOR_REGISTER);
pub const YMM7: IRGraphId = IRGraphId::register(07 | VECTOR_REGISTER);
pub const YMM8: IRGraphId = IRGraphId::register(08 | VECTOR_REGISTER);
pub const YMM9: IRGraphId = IRGraphId::register(09 | VECTOR_REGISTER);
pub const YMM10: IRGraphId = IRGraphId::register(10 | VECTOR_REGISTER);
pub const YMM11: IRGraphId = IRGraphId::register(11 | VECTOR_REGISTER);
pub const YMM12: IRGraphId = IRGraphId::register(12 | VECTOR_REGISTER);
pub const YMM13: IRGraphId = IRGraphId::register(13 | VECTOR_REGISTER);
pub const YMM14: IRGraphId = IRGraphId::register(14 | VECTOR_REGISTER);
pub const YMM15: IRGraphId = IRGraphId::register(15 | VECTOR_REGISTER);

pub const ZMM0: IRGraphId = IRGraphId::register(00 | VECTOR_REGISTER);
pub const ZMM1: IRGraphId = IRGraphId::register(01 | VECTOR_REGISTER);
pub const ZMM2: IRGraphId = IRGraphId::register(02 | VECTOR_REGISTER);
pub const ZMM3: IRGraphId = IRGraphId::register(03 | VECTOR_REGISTER);
pub const ZMM4: IRGraphId = IRGraphId::register(04 | VECTOR_REGISTER);
pub const ZMM5: IRGraphId = IRGraphId::register(05 | VECTOR_REGISTER);
pub const ZMM6: IRGraphId = IRGraphId::register(06 | VECTOR_REGISTER);
pub const ZMM7: IRGraphId = IRGraphId::register(07 | VECTOR_REGISTER);
pub const ZMM8: IRGraphId = IRGraphId::register(08 | VECTOR_REGISTER);
pub const ZMM9: IRGraphId = IRGraphId::register(09 | VECTOR_REGISTER);
pub const ZMM10: IRGraphId = IRGraphId::register(10 | VECTOR_REGISTER);
pub const ZMM11: IRGraphId = IRGraphId::register(11 | VECTOR_REGISTER);
pub const ZMM12: IRGraphId = IRGraphId::register(12 | VECTOR_REGISTER);
pub const ZMM13: IRGraphId = IRGraphId::register(13 | VECTOR_REGISTER);
pub const ZMM14: IRGraphId = IRGraphId::register(14 | VECTOR_REGISTER);
pub const ZMM15: IRGraphId = IRGraphId::register(15 | VECTOR_REGISTER);
pub const ZMM16: IRGraphId = IRGraphId::register(16 | VECTOR_REGISTER);
pub const ZMM17: IRGraphId = IRGraphId::register(17 | VECTOR_REGISTER);
pub const ZMM18: IRGraphId = IRGraphId::register(18 | VECTOR_REGISTER);
pub const ZMM19: IRGraphId = IRGraphId::register(19 | VECTOR_REGISTER);
pub const ZMM20: IRGraphId = IRGraphId::register(20 | VECTOR_REGISTER);
pub const ZMM21: IRGraphId = IRGraphId::register(21 | VECTOR_REGISTER);
pub const ZMM22: IRGraphId = IRGraphId::register(22 | VECTOR_REGISTER);
pub const ZMM23: IRGraphId = IRGraphId::register(23 | VECTOR_REGISTER);
pub const ZMM24: IRGraphId = IRGraphId::register(24 | VECTOR_REGISTER);
pub const ZMM25: IRGraphId = IRGraphId::register(25 | VECTOR_REGISTER);
pub const ZMM26: IRGraphId = IRGraphId::register(26 | VECTOR_REGISTER);
pub const ZMM27: IRGraphId = IRGraphId::register(27 | VECTOR_REGISTER);
pub const ZMM28: IRGraphId = IRGraphId::register(28 | VECTOR_REGISTER);
pub const ZMM29: IRGraphId = IRGraphId::register(29 | VECTOR_REGISTER);
pub const ZMM30: IRGraphId = IRGraphId::register(30 | VECTOR_REGISTER);
pub const ZMM31: IRGraphId = IRGraphId::register(31 | VECTOR_REGISTER);

impl IRGraphId {
  const SIB_RM: u8 = 0b100;

  pub(super) fn index(&self) -> u8 {
    debug_assert!(self.reg_id().is_some());
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
    (self.reg_id().unwrap() & (VECTOR_REGISTER | MASK_REGISTER)) == 0
  }

  /// The register is one of R8-R15
  pub(super) fn is_ext_8_reg(&self) -> bool {
    (self.reg_id().unwrap() & 0x8) > 0
  }

  pub(super) fn is_upper_16_reg(&self) -> bool {
    (self.reg_id().unwrap() & 0x1F) >= 16
  }

  pub fn as_addr_op(&self, ctx: &IRCallable, stack_offsets: &BTreeMap<usize, u64>) -> Arg {
    if let Some(reg) = self.reg_id() {
      Arg::Mem(Self::register(reg))
    } else {
      self.as_op(ctx, stack_offsets)
    }
  }

  /// Returns an Arg::Reg op for the given register. Panics if the Graphid is
  /// not a register
  pub fn as_reg_op(&self) -> Arg {
    if let Some(reg) = self.reg_id() {
      Arg::Reg(Self::register(reg))
    } else {
      panic!("GraphID node is not a register {self}");
    }
  }

  /// Returns an Arg::Mem op for the given register. Panics if the Graphid is
  /// not a register
  pub fn as_mem_op(&self) -> Arg {
    if let Some(reg) = self.reg_id() {
      Arg::Mem(Self::register(reg))
    } else {
      panic!("GraphID node is not a register {self}");
    }
  }

  pub fn as_op(&self, ctx: &IRCallable, stack_offsets: &BTreeMap<usize, u64>) -> Arg {
    if let Some(reg) = self.reg_id() {
      Arg::Reg(Self::register(reg))
    } else if ctx.graph[self.graph_id()].is_const() {
      let constant = ctx.graph[self.graph_id()].constant().unwrap();
      let val: i128 = unsafe { std::mem::transmute(constant.val) };
      Arg::Imm_Int(val as i64)
    } else if let Some(var_id) = self.var_id() {
      Arg::RSP_REL(*stack_offsets.get(&var_id).unwrap())
    } else {
      Arg::None
    }
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
  Reg(IRGraphId),
  Mem(IRGraphId),
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
      Arg::Reg(reg) | Arg::Mem(reg) => (reg.reg_id().unwrap() & 7) as u8,
      Self::OpExt(index) => *index,

      arg => unreachable!("{arg:?}"),
    }
  }

  pub(super) fn is_mask_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.reg_id().unwrap() & MASK_REGISTER) > 0,
      _ => false,
    }
  }

  pub(super) fn is_vector_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.reg_id().unwrap() & VECTOR_REGISTER) > 0,
      _ => false,
    }
  }

  pub(super) fn is_general_purpose(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.reg_id().unwrap() & (VECTOR_REGISTER | MASK_REGISTER)) == 0,
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
