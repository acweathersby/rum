use crate::compiler::interpreter::ll::types::OpArg;

impl OpArg<x86Reg> {
  pub fn arg(&self, so: &[usize]) -> Arg {
    match self {
      OpArg::REG(reg, _) => Arg::Reg(*reg),
      OpArg::STACK(_, l) => {
        let stack_id = l.info.stack_id().expect("Loads should have an associated stack id");
        let offset = (so[stack_id] as isize);
        Arg::Mem(x86Reg::RSP_REL(offset as u64))
      }
      OpArg::Lit(literal) => {
        let val = literal.load::<u64>().unwrap();
        Arg::Imm_Int(val)
      }
      _ => Arg::None,
    }
  }
}

#[derive(Debug, Hash, Clone, Copy)]
pub(super) enum OpEncoding {
  Zero,
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
}

#[allow(non_camel_case_types, unused)]
pub(super) enum RegisterName {
  GeneralRegisterName(x86Reg),
  AVX_64,
  AVX_128,
  AVX_512,
}

#[repr(u32)]
#[allow(non_camel_case_types, unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum x86Reg {
  FLAGS = 0,
  RAX   = 0x00 | 0x0_00 | 0xF0_00,
  RCX   = 0x01 | 0x0_00 | 0xF0_00,
  RDX   = 0x02 | 0x0_00 | 0xF0_00,
  RBX   = 0x03 | 0x0_00 | 0xF0_00,
  RSP   = 0x04 | 0x0_00 | 0xF0_00,
  RBP   = 0x05 | 0x0_00 | 0xF0_00,
  RSI   = 0x06 | 0x0_00 | 0xF0_00,
  RDI   = 0x07 | 0x0_00 | 0xF0_00,
  //
  R8    = 0x00 | 0x1_00 | 0xF0_00,
  R9    = 0x01 | 0x1_00 | 0xF0_00,
  R10   = 0x02 | 0x1_00 | 0xF0_00,
  R11   = 0x03 | 0x1_00 | 0xF0_00,
  R12   = 0x04 | 0x1_00 | 0xF0_00,
  R13   = 0x05 | 0x1_00 | 0xF0_00,
  R14   = 0x06 | 0x1_00 | 0xF0_00,
  R15   = 0x07 | 0x1_00 | 0xF0_00,
  XMM1,
  XMM2,
  XMM3,
  XMM4,
  XMM5,
  XMM6,
  XMM7,
  XMM8,
  // Allows displacement bytes to be inserted for relative addressing.
  RSP_REL(u64),
  RIP_REL(u64),
}

impl x86Reg {
  const SIB_RM: u8 = 0b100;

  pub(super) fn displacement(&self) -> Option<u64> {
    match self {
      Self::RSP_REL(val) => Some(*val),
      _ => None,
    }
  }

  pub(super) fn index(&self) -> u8 {
    use x86Reg::*;
    match self {
      R8 | RAX | XMM1 => 0x00,
      R9 | RCX | XMM2 => 0x01,
      R10 | RDX | XMM3 => 0x02,
      R11 | RBX | XMM4 => 0x03,
      R12 | RSP | XMM5 => 0x04,
      R13 | RBP | XMM6 => 0x05,
      R14 | RSI | XMM7 => 0x06,
      R15 | RDI | XMM8 => 0x07,
      Self::RSP_REL(..) => Self::SIB_RM, // uses SIB byte
      Self::RIP_REL(..) => 0x5,          // uses SIB byte
      _ => 0xFF,
    }
  }

  pub(super) fn is_general_purpose(&self) -> bool {
    use x86Reg::*;
    match self {
      R8 | RAX | R9 | RCX | R10 | RDX | R11 | RBX | R12 | RSP | R13 | RBP | R14 | RSI | R15
      | RDI => true,
      _ => false,
    }
  }
  /// The register is one of R8-R15
  pub(super) fn is_64_extended(&self) -> bool {
    use x86Reg::*;
    match self {
      R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 => true,
      _ => false,
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

#[derive(Clone, Copy, Debug)]
pub(super) enum Arg {
  Reg(x86Reg),
  Mem(x86Reg),
  Imm_Int(u64),
  OpExt(u8),
  None,
}

impl Arg {
  pub(super) fn ty(&self) -> OperandType {
    match self {
      Arg::Imm_Int(..) => OperandType::IMM_INT,
      Arg::Reg(..) => OperandType::REG,
      Arg::Mem(..) => OperandType::MEM,
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
      Arg::Reg(reg) | Arg::Mem(reg) => reg.index(),
      Self::OpExt(index) => *index,
      arg => unreachable!("{arg:?}"),
    }
  }

  pub(super) fn is_64_extended(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) => reg.is_64_extended(),
      _ => false,
    }
  }
}
