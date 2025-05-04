use crate::targets::reg::Reg;

#[derive(Debug, Hash, Clone, Copy)]
pub(crate) enum OpEncoding {
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

const GENERAL_REGISTER: u8 = 1;
const VECTOR_REGISTER: u8 = 2;
const MASK_REGISTER: u8 = 4;
const RELATIVE: u8 = 8;
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

pub const XMM0: Reg = Reg::new(16, 00, 16, VECTOR_REGISTER);
pub const XMM1: Reg = Reg::new(17, 01, 16, VECTOR_REGISTER);
pub const XMM2: Reg = Reg::new(18, 02, 16, VECTOR_REGISTER);
pub const XMM3: Reg = Reg::new(19, 03, 16, VECTOR_REGISTER);
pub const XMM4: Reg = Reg::new(20, 04, 16, VECTOR_REGISTER);
pub const XMM5: Reg = Reg::new(21, 05, 16, VECTOR_REGISTER);
pub const XMM6: Reg = Reg::new(22, 06, 16, VECTOR_REGISTER);
pub const XMM7: Reg = Reg::new(23, 07, 16, VECTOR_REGISTER);
pub const XMM8: Reg = Reg::new(24, 08, 16, VECTOR_REGISTER);
pub const XMM9: Reg = Reg::new(25, 09, 16, VECTOR_REGISTER);
pub const XMM10: Reg = Reg::new(26, 10, 16, VECTOR_REGISTER);
pub const XMM11: Reg = Reg::new(27, 11, 16, VECTOR_REGISTER);
pub const XMM12: Reg = Reg::new(28, 12, 16, VECTOR_REGISTER);
pub const XMM13: Reg = Reg::new(29, 13, 16, VECTOR_REGISTER);
pub const XMM14: Reg = Reg::new(30, 14, 16, VECTOR_REGISTER);
pub const XMM15: Reg = Reg::new(31, 15, 16, VECTOR_REGISTER);

pub const YMM0: Reg = Reg::new(32, 00, 32, VECTOR_REGISTER);
pub const YMM1: Reg = Reg::new(33, 01, 32, VECTOR_REGISTER);
pub const YMM2: Reg = Reg::new(34, 02, 32, VECTOR_REGISTER);
pub const YMM3: Reg = Reg::new(35, 03, 32, VECTOR_REGISTER);
pub const YMM4: Reg = Reg::new(36, 04, 32, VECTOR_REGISTER);
pub const YMM5: Reg = Reg::new(37, 05, 32, VECTOR_REGISTER);
pub const YMM6: Reg = Reg::new(38, 06, 32, VECTOR_REGISTER);
pub const YMM7: Reg = Reg::new(39, 07, 32, VECTOR_REGISTER);
pub const YMM8: Reg = Reg::new(40, 08, 32, VECTOR_REGISTER);
pub const YMM9: Reg = Reg::new(41, 09, 32, VECTOR_REGISTER);
pub const YMM10: Reg = Reg::new(42, 10, 32, VECTOR_REGISTER);
pub const YMM11: Reg = Reg::new(43, 11, 32, VECTOR_REGISTER);
pub const YMM12: Reg = Reg::new(44, 12, 32, VECTOR_REGISTER);
pub const YMM13: Reg = Reg::new(45, 13, 32, VECTOR_REGISTER);
pub const YMM14: Reg = Reg::new(46, 14, 32, VECTOR_REGISTER);
pub const YMM15: Reg = Reg::new(47, 15, 32, VECTOR_REGISTER);

pub const ZMM0: Reg = Reg::new(48, 00, 64, VECTOR_REGISTER);
pub const ZMM1: Reg = Reg::new(49, 01, 64, VECTOR_REGISTER);
pub const ZMM2: Reg = Reg::new(50, 02, 64, VECTOR_REGISTER);
pub const ZMM3: Reg = Reg::new(51, 03, 64, VECTOR_REGISTER);
pub const ZMM4: Reg = Reg::new(52, 04, 64, VECTOR_REGISTER);
pub const ZMM5: Reg = Reg::new(53, 05, 64, VECTOR_REGISTER);
pub const ZMM6: Reg = Reg::new(54, 06, 64, VECTOR_REGISTER);
pub const ZMM7: Reg = Reg::new(55, 07, 64, VECTOR_REGISTER);
pub const ZMM8: Reg = Reg::new(56, 08, 64, VECTOR_REGISTER);
pub const ZMM9: Reg = Reg::new(57, 09, 64, VECTOR_REGISTER);
pub const ZMM10: Reg = Reg::new(58, 10, 64, VECTOR_REGISTER);
pub const ZMM11: Reg = Reg::new(59, 11, 64, VECTOR_REGISTER);
pub const ZMM12: Reg = Reg::new(60, 12, 64, VECTOR_REGISTER);
pub const ZMM13: Reg = Reg::new(61, 13, 64, VECTOR_REGISTER);
pub const ZMM14: Reg = Reg::new(62, 14, 64, VECTOR_REGISTER);
pub const ZMM15: Reg = Reg::new(63, 15, 64, VECTOR_REGISTER);
pub const ZMM16: Reg = Reg::new(64, 16, 64, VECTOR_REGISTER);
pub const ZMM17: Reg = Reg::new(65, 17, 64, VECTOR_REGISTER);
pub const ZMM18: Reg = Reg::new(66, 18, 64, VECTOR_REGISTER);
pub const ZMM19: Reg = Reg::new(67, 19, 64, VECTOR_REGISTER);
pub const ZMM20: Reg = Reg::new(68, 20, 64, VECTOR_REGISTER);
pub const ZMM21: Reg = Reg::new(69, 21, 64, VECTOR_REGISTER);
pub const ZMM22: Reg = Reg::new(70, 22, 64, VECTOR_REGISTER);
pub const ZMM23: Reg = Reg::new(71, 23, 64, VECTOR_REGISTER);
pub const ZMM24: Reg = Reg::new(72, 24, 64, VECTOR_REGISTER);
pub const ZMM25: Reg = Reg::new(73, 25, 64, VECTOR_REGISTER);
pub const ZMM26: Reg = Reg::new(74, 26, 64, VECTOR_REGISTER);
pub const ZMM27: Reg = Reg::new(75, 27, 64, VECTOR_REGISTER);
pub const ZMM28: Reg = Reg::new(76, 28, 64, VECTOR_REGISTER);
pub const ZMM29: Reg = Reg::new(77, 29, 64, VECTOR_REGISTER);
pub const ZMM30: Reg = Reg::new(78, 30, 64, VECTOR_REGISTER);
pub const ZMM31: Reg = Reg::new(79, 31, 64, VECTOR_REGISTER);

pub const K0: Reg = Reg::new(81, 01, 1, MASK_REGISTER);
pub const K1: Reg = Reg::new(81, 01, 1, MASK_REGISTER);
pub const K2: Reg = Reg::new(82, 02, 1, MASK_REGISTER);
pub const K3: Reg = Reg::new(83, 03, 1, MASK_REGISTER);
pub const K4: Reg = Reg::new(84, 04, 1, MASK_REGISTER);
pub const K5: Reg = Reg::new(85, 05, 1, MASK_REGISTER);
pub const K6: Reg = Reg::new(86, 06, 1, MASK_REGISTER);
pub const K7: Reg = Reg::new(87, 07, 1, MASK_REGISTER);

impl Reg {
  const SIB_RM: u8 = 0b100;

  pub(crate) fn index(&self) -> u8 {
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

  pub(crate) fn is_general_purpose(&self) -> bool {
    (self.flags() & (VECTOR_REGISTER | MASK_REGISTER)) == 0
  }

  /// The register is one of R8-R15
  pub(crate) fn is_ext_8_reg(&self) -> bool {
    self.real_index() >= 8
  }

  pub(crate) fn is_upper_16_reg(&self) -> bool {
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
pub(crate) enum OperandType {
  REG,
  MEM,
  IMM_INT,
  NONE,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arg {
  Reg(Reg),
  Mem(Reg),
  RSP_REL(i64),
  RIP_REL(i64),
  MemRel(Reg, i64),
  Imm_Int(i64),
  OpExt(u8),
  None,
}

impl Arg {
  pub(crate) fn ty(&self) -> OperandType {
    match self {
      Arg::Imm_Int(..) => OperandType::IMM_INT,
      Arg::Reg(..) => OperandType::REG,
      Arg::Mem(..) => OperandType::MEM,
      Arg::MemRel(..) => OperandType::MEM,
      Arg::RSP_REL(..) => OperandType::MEM,
      Arg::RIP_REL(..) => OperandType::MEM,
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

  pub(crate) fn is_reg(&self) -> bool {
    matches!(self, Arg::Reg(..))
  }

  pub(crate) fn reg_index(&self) -> u8 {
    match self {
      Arg::RIP_REL(_) => 0x5,
      Arg::RSP_REL(_) => 0x4,
      Arg::Reg(reg) | Arg::Mem(reg) | Arg::MemRel(reg, ..) => (reg.0 & 7) as u8,
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

  pub(crate) fn is_vector_register(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.flags() & VECTOR_REGISTER) > 0,
      _ => false,
    }
  }

  pub(crate) fn is_general_purpose(&self) -> bool {
    match self {
      Arg::Reg(reg) => (reg.flags() & (VECTOR_REGISTER | MASK_REGISTER)) == 0,
      _ => false,
    }
  }

  pub(crate) fn is_upper_8_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) => reg.is_ext_8_reg(),
      _ => false,
    }
  }

  pub(crate) fn is_upper_16_reg(&self) -> bool {
    match self {
      Arg::Reg(reg) | Arg::Mem(reg) => reg.is_upper_16_reg(),
      _ => false,
    }
  }
}
