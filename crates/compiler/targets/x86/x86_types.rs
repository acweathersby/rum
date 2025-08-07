use crate::targets::reg::Reg;

#[derive(Debug, Hash, Clone, Copy)]
pub(crate) enum OpEncoding {
  Zero,
  VEX_MR_3,
  VEX_RM_3,
  VEX_RM_2 {
    w: bool,
  },
  VEX_MR_2 {
    w: bool,
  },
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

const GENERAL_REGISTER: u8 = 1;
const VECTOR_REGISTER: u8 = 2;
const MASK_REGISTER: u8 = 4;
const MMX_REGISTER: u8 = 8;
pub const STACK_PTR: u8 = 8;

pub const RAX: Reg = Reg::new(00, 00, 8, GENERAL_REGISTER);
pub const RCX: Reg = Reg::new(01, 01, 8, GENERAL_REGISTER);
pub const RDX: Reg = Reg::new(02, 02, 8, GENERAL_REGISTER);
pub const RBX: Reg = Reg::new(03, 03, 8, GENERAL_REGISTER);
pub const RSP: Reg = Reg::new(04, 04, 8, GENERAL_REGISTER);
pub const RBP: Reg = Reg::new(05, 05, 8, GENERAL_REGISTER);
pub const RSI: Reg = Reg::new(06, 06, 8, GENERAL_REGISTER);
pub const RDI: Reg = Reg::new(07, 07, 8, GENERAL_REGISTER);
pub const R8: Reg = Reg::new(08, 08, 8, GENERAL_REGISTER);
pub const R9: Reg = Reg::new(09, 09, 8, GENERAL_REGISTER);
pub const R10: Reg = Reg::new(10, 10, 8, GENERAL_REGISTER);
pub const R11: Reg = Reg::new(11, 11, 8, GENERAL_REGISTER);
pub const R12: Reg = Reg::new(12, 12, 8, GENERAL_REGISTER);
pub const R13: Reg = Reg::new(13, 13, 8, GENERAL_REGISTER);
pub const R14: Reg = Reg::new(14, 14, 8, GENERAL_REGISTER);
pub const R15: Reg = Reg::new(15, 15, 8, GENERAL_REGISTER);

//pub const MMX0: Reg = Reg::new(16, 00, 8, MMX_REGISTER);
//pub const MMX1: Reg = Reg::new(17, 01, 8, MMX_REGISTER);
//pub const MMX2: Reg = Reg::new(18, 02, 8, MMX_REGISTER);
//pub const MMX3: Reg = Reg::new(19, 03, 8, MMX_REGISTER);
//pub const MMX4: Reg = Reg::new(20, 04, 8, MMX_REGISTER);
//pub const MMX5: Reg = Reg::new(21, 05, 8, MMX_REGISTER);
//pub const MMX6: Reg = Reg::new(22, 06, 8, MMX_REGISTER);
//pub const MMX7: Reg = Reg::new(23, 07, 8, MMX_REGISTER);

pub const VEC0: Reg = Reg::new(16, 00, 64, VECTOR_REGISTER);
pub const VEC1: Reg = Reg::new(17, 01, 64, VECTOR_REGISTER);
pub const VEC2: Reg = Reg::new(18, 02, 64, VECTOR_REGISTER);
pub const VEC3: Reg = Reg::new(19, 03, 64, VECTOR_REGISTER);
pub const VEC4: Reg = Reg::new(20, 04, 64, VECTOR_REGISTER);
pub const VEC5: Reg = Reg::new(21, 05, 64, VECTOR_REGISTER);
pub const VEC6: Reg = Reg::new(22, 06, 64, VECTOR_REGISTER);
pub const VEC7: Reg = Reg::new(23, 07, 64, VECTOR_REGISTER);
pub const VEC8: Reg = Reg::new(24, 08, 64, VECTOR_REGISTER);
pub const VEC9: Reg = Reg::new(25, 09, 64, VECTOR_REGISTER);
pub const VEC10: Reg = Reg::new(26, 10, 64, VECTOR_REGISTER);
pub const VEC11: Reg = Reg::new(27, 11, 64, VECTOR_REGISTER);
pub const VEC12: Reg = Reg::new(28, 12, 64, VECTOR_REGISTER);
pub const VEC13: Reg = Reg::new(29, 13, 64, VECTOR_REGISTER);
pub const VEC14: Reg = Reg::new(30, 14, 64, VECTOR_REGISTER);
pub const VEC15: Reg = Reg::new(31, 15, 64, VECTOR_REGISTER);
pub const VEC16: Reg = Reg::new(32, 16, 64, VECTOR_REGISTER);
pub const VEC17: Reg = Reg::new(33, 17, 64, VECTOR_REGISTER);
pub const VEC18: Reg = Reg::new(34, 18, 64, VECTOR_REGISTER);
pub const VEC19: Reg = Reg::new(35, 19, 64, VECTOR_REGISTER);
pub const VEC20: Reg = Reg::new(36, 20, 64, VECTOR_REGISTER);
pub const VEC21: Reg = Reg::new(37, 21, 64, VECTOR_REGISTER);
pub const VEC22: Reg = Reg::new(38, 22, 64, VECTOR_REGISTER);
pub const VEC23: Reg = Reg::new(39, 23, 64, VECTOR_REGISTER);
pub const VEC24: Reg = Reg::new(40, 24, 64, VECTOR_REGISTER);
pub const VEC25: Reg = Reg::new(41, 25, 64, VECTOR_REGISTER);
pub const VEC26: Reg = Reg::new(42, 26, 64, VECTOR_REGISTER);
pub const VEC27: Reg = Reg::new(43, 27, 64, VECTOR_REGISTER);
pub const VEC28: Reg = Reg::new(44, 28, 64, VECTOR_REGISTER);
pub const VEC29: Reg = Reg::new(45, 29, 64, VECTOR_REGISTER);
pub const VEC30: Reg = Reg::new(46, 30, 64, VECTOR_REGISTER);
pub const VEC31: Reg = Reg::new(47, 31, 64, VECTOR_REGISTER);

pub const K0: Reg = Reg::new(48, 01, 1, MASK_REGISTER);
pub const K1: Reg = Reg::new(49, 01, 1, MASK_REGISTER);
pub const K2: Reg = Reg::new(50, 02, 1, MASK_REGISTER);
pub const K3: Reg = Reg::new(51, 03, 1, MASK_REGISTER);
pub const K4: Reg = Reg::new(52, 04, 1, MASK_REGISTER);
pub const K5: Reg = Reg::new(53, 05, 1, MASK_REGISTER);
pub const K6: Reg = Reg::new(54, 06, 1, MASK_REGISTER);
pub const K7: Reg = Reg::new(55, 07, 1, MASK_REGISTER);

impl Reg {
  const SIB_RM: u8 = 0b100;

  pub(crate) fn index(&self) -> u8 {
    match *self {
      R8 | RAX | VEC0 => 0x00,
      R9 | RCX | VEC1 => 0x01,
      R10 | RDX | VEC2 => 0x02,
      R11 | RBX | VEC3 => 0x03,
      R12 | RSP | VEC4 => 0x04,
      R13 | RBP | VEC5 => 0x05,
      R14 | RSI | VEC6 => 0x06,
      R15 | RDI | VEC7 => 0x07,
      _ => Self::SIB_RM, // uses SIB byte
    }
  }

  pub(crate) fn is_general_purpose(&self) -> bool {
    (self.flags() & (VECTOR_REGISTER | MASK_REGISTER)) == 0
  }

  pub(crate) fn is_xmm(&self) -> bool {
    (self.flags() & (VECTOR_REGISTER)) > 0
  }

  pub(crate) fn is_mask(&self) -> bool {
    (self.flags() & (MASK_REGISTER)) > 0
  }

  /// The register is one of R8-R15
  pub(crate) fn is_upper_8_reg(&self) -> bool {
    (self.real_index() & 0b1111_1000) >= 8
  }

  pub(crate) fn is_upper_16_reg(&self) -> bool {
    self.real_index() >= 16
  }

  /// Returns an Arg::Reg op for the given register. Panics if the Graphid is
  /// not a register
  pub(crate) fn as_reg_op(&self) -> Arg {
    Arg::Reg(*self)
  }

  /// Returns an Arg::Mem op for the given register. Panics if the Graphid is
  /// not a register
  pub(crate) fn as_mem_op(&self) -> Arg {
    Arg::Mem(*self)
  }
}

#[derive(PartialEq, Debug, Hash)]
pub(crate) enum OperandType {
  REG,
  MEM,
  XMM,
  IMM_INT,
  MSK,
  NONE,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arg {
  Reg(Reg),
  Mem(Reg),
  RSP_REL(i64),
  RIP_REL(i64),
  MemRel(Reg, i64),
  SIBAddress { base: Reg, index: Reg, scale: u8, disp: i32 },
  Imm_Int(i64),
  OpExt(u8),
  None,
}

impl Arg {
  pub(crate) fn ty(&self) -> OperandType {
    match self {
      Arg::Imm_Int(..) => OperandType::IMM_INT,
      Arg::Reg(reg) => {
        if reg.is_xmm() {
          OperandType::XMM
        } else if reg.is_mask() {
          OperandType::MSK
        } else {
          OperandType::REG
        }
      }
      Arg::Mem(..) => OperandType::MEM,
      Arg::MemRel(..) => OperandType::MEM,
      Arg::RSP_REL(..) => OperandType::MEM,
      Arg::RIP_REL(..) => OperandType::MEM,
      Arg::SIBAddress { .. } => OperandType::MEM,
      _ => OperandType::NONE,
    }
  }

  /// Converts the argument from an operation on a value stored in a register to
  /// an operation performed on the memory location resolved from the registers
  /// value
  pub(crate) fn to_mem(&self) -> Arg {
    match self {
      Arg::Reg(reg) => Arg::Mem(*reg),
      arg => *arg,
    }
  }

  pub(crate) fn is_immediate(&self) -> bool {
    matches!(self, Arg::Imm_Int(..))
  }

  pub(crate) fn is_reg(&self) -> bool {
    matches!(self, Arg::Reg(..))
  }

  pub(crate) fn reg_index(&self) -> u8 {
    match self {
      Arg::RIP_REL(_) => 0x5,
      Arg::RSP_REL(_) => 0x4,
      Arg::Reg(reg) | Arg::Mem(reg) | Arg::MemRel(reg, ..) => reg.real_index() as u8,
      Self::OpExt(index) => *index,
      arg => unreachable!("{arg:?}"),
    }
  }

  pub(crate) fn is_mask_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.flags() & MASK_REGISTER) > 0,
      _ => false,
    }
  }

  pub(crate) fn is_upper_8_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) | Arg::MemRel(reg, _) => reg.is_upper_8_reg(),
      _ => false,
    }
  }

  pub(crate) fn is_upper_16_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) | Arg::MemRel(reg, _) => reg.is_upper_16_reg(),
      _ => false,
    }
  }
}
