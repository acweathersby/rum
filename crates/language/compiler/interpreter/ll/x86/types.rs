use crate::compiler::interpreter::ll::types::{GraphId, SSAFunction, TypeInfo};

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

  RMI,
}

pub const RAX: GraphId = GraphId(00).as_register();
pub const RCX: GraphId = GraphId(01).as_register();
pub const RDX: GraphId = GraphId(02).as_register();
pub const RBX: GraphId = GraphId(03).as_register();
pub const RSP: GraphId = GraphId(04).as_register();
pub const RBP: GraphId = GraphId(05).as_register();
pub const RSI: GraphId = GraphId(06).as_register();
pub const RDI: GraphId = GraphId(07).as_register();

pub const R8: GraphId = GraphId(08).as_register();
pub const R9: GraphId = GraphId(09).as_register();
pub const R10: GraphId = GraphId(10).as_register();
pub const R11: GraphId = GraphId(11).as_register();
pub const R12: GraphId = GraphId(12).as_register();
pub const R13: GraphId = GraphId(13).as_register();
pub const R14: GraphId = GraphId(14).as_register();
pub const R15: GraphId = GraphId(15).as_register();

pub const XMM1: GraphId = GraphId(16).as_register();
pub const XMM2: GraphId = GraphId(17).as_register();
pub const XMM3: GraphId = GraphId(18).as_register();
pub const XMM4: GraphId = GraphId(19).as_register();
pub const XMM5: GraphId = GraphId(20).as_register();
pub const XMM6: GraphId = GraphId(21).as_register();
pub const XMM7: GraphId = GraphId(22).as_register();
pub const XMM8: GraphId = GraphId(23).as_register();

pub const RIP_REL: GraphId = GraphId(62).as_register();

impl GraphId {
  const SIB_RM: u8 = 0b100;

  pub(super) fn displacement(&self) -> Option<u64> {
    let val = self.0 & !Self::FLAGS_MASK;

    if val & 0x5F == 63 {
      Some((val >> 7) as u64)
    } else {
      None
    }
  }

  pub(super) fn index(&self) -> u8 {
    debug_assert!(self.is_register());
    match *self {
      R8 | RAX | XMM1 => 0x00,
      R9 | RCX | XMM2 => 0x01,
      R10 | RDX | XMM3 => 0x02,
      R11 | RBX | XMM4 => 0x03,
      R12 | RSP | XMM5 => 0x04,
      R13 | RBP | XMM6 => 0x05,
      R14 | RSI | XMM7 => 0x06,
      R15 | RDI | XMM8 => 0x07,
      RIP_REL => 0x5,    // uses SIB byte
      _ => Self::SIB_RM, // uses SIB byte
      _ => 0xFF,
    }
  }

  pub(super) fn is_general_purpose(&self) -> bool {
    debug_assert!(self.is_register());
    match *self {
      R8 | RAX | R9 | RCX | R10 | RDX | R11 | RBX | R12 | RSP | R13 | RBP | R14 | RSI | R15
      | RDI => true,
      _ => false,
    }
  }

  /// The register is one of R8-R15
  pub(super) fn is_64_extended(&self) -> bool {
    debug_assert!(self.is_register());
    match *self {
      R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 => true,
      _ => false,
    }
  }

  pub fn into_addr_op(&self, ctx: &SSAFunction) -> Arg {
    if self.is_register() {
      Arg::Mem(*self)
    } else {
      self.into_op(ctx)
    }
  }

  pub fn into_op(&self, ctx: &SSAFunction) -> Arg {
    if self.is_register() {
      Arg::Reg(*self)
    } else if self.is_const() {
      let value = ctx.constants[*self];
      Arg::Imm_Int(value.convert(TypeInfo::Integer | TypeInfo::b64).load().unwrap())
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

#[derive(Clone, Copy, Debug)]
pub(super) enum Arg {
  Reg(GraphId),
  Mem(GraphId),
  RSP_REL(u64),
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
      Arg::RSP_REL(..) => OperandType::MEM,
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
