use crate::compiler::interpreter::raw::ir::{
  ir_types::{GraphId, SSAFunction, TypeInfo},
  GraphIdType,
};

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

pub const RAX: GraphId = GraphId::register(00);
pub const RCX: GraphId = GraphId::register(01);
pub const RDX: GraphId = GraphId::register(02);
pub const RBX: GraphId = GraphId::register(03);
pub const RSP: GraphId = GraphId::register(04);
pub const RBP: GraphId = GraphId::register(05);
pub const RSI: GraphId = GraphId::register(06);
pub const RDI: GraphId = GraphId::register(07);

pub const R8: GraphId = GraphId::register(08);
pub const R9: GraphId = GraphId::register(09);
pub const R10: GraphId = GraphId::register(10);
pub const R11: GraphId = GraphId::register(11);
pub const R12: GraphId = GraphId::register(12);
pub const R13: GraphId = GraphId::register(13);
pub const R14: GraphId = GraphId::register(14);
pub const R15: GraphId = GraphId::register(15);

pub const XMM0: GraphId = GraphId::register(16);
pub const XMM1: GraphId = GraphId::register(17);
pub const XMM2: GraphId = GraphId::register(18);
pub const XMM3: GraphId = GraphId::register(19);
pub const XMM4: GraphId = GraphId::register(20);
pub const XMM5: GraphId = GraphId::register(21);
pub const XMM6: GraphId = GraphId::register(22);
pub const XMM7: GraphId = GraphId::register(23);
pub const XMM8: GraphId = GraphId::register(24);
pub const XMM9: GraphId = GraphId::register(25);
pub const XMM10: GraphId = GraphId::register(26);
pub const XMM11: GraphId = GraphId::register(27);
pub const XMM12: GraphId = GraphId::register(28);
pub const XMM13: GraphId = GraphId::register(29);
pub const XMM14: GraphId = GraphId::register(30);
pub const XMM15: GraphId = GraphId::register(31);

pub const YMM0: GraphId = GraphId::register(16);
pub const YMM1: GraphId = GraphId::register(17);
pub const YMM2: GraphId = GraphId::register(18);
pub const YMM3: GraphId = GraphId::register(19);
pub const YMM4: GraphId = GraphId::register(20);
pub const YMM5: GraphId = GraphId::register(21);
pub const YMM6: GraphId = GraphId::register(22);
pub const YMM7: GraphId = GraphId::register(23);
pub const YMM8: GraphId = GraphId::register(24);
pub const YMM9: GraphId = GraphId::register(25);
pub const YMM10: GraphId = GraphId::register(26);
pub const YMM11: GraphId = GraphId::register(27);
pub const YMM12: GraphId = GraphId::register(28);
pub const YMM13: GraphId = GraphId::register(29);
pub const YMM14: GraphId = GraphId::register(30);
pub const YMM15: GraphId = GraphId::register(31);

pub const ZMM0: GraphId = GraphId::register(16);
pub const ZMM1: GraphId = GraphId::register(17);
pub const ZMM2: GraphId = GraphId::register(18);
pub const ZMM3: GraphId = GraphId::register(19);
pub const ZMM4: GraphId = GraphId::register(20);
pub const ZMM5: GraphId = GraphId::register(21);
pub const ZMM6: GraphId = GraphId::register(22);
pub const ZMM7: GraphId = GraphId::register(23);
pub const ZMM8: GraphId = GraphId::register(24);
pub const ZMM9: GraphId = GraphId::register(25);
pub const ZMM10: GraphId = GraphId::register(26);
pub const ZMM11: GraphId = GraphId::register(27);
pub const ZMM12: GraphId = GraphId::register(28);
pub const ZMM13: GraphId = GraphId::register(29);
pub const ZMM14: GraphId = GraphId::register(30);
pub const ZMM15: GraphId = GraphId::register(31);
pub const ZMM16: GraphId = GraphId::register(16);
pub const ZMM17: GraphId = GraphId::register(17);
pub const ZMM18: GraphId = GraphId::register(18);
pub const ZMM19: GraphId = GraphId::register(19);
pub const ZMM20: GraphId = GraphId::register(20);
pub const ZMM21: GraphId = GraphId::register(21);
pub const ZMM22: GraphId = GraphId::register(22);
pub const ZMM23: GraphId = GraphId::register(23);
pub const ZMM24: GraphId = GraphId::register(24);
pub const ZMM25: GraphId = GraphId::register(25);
pub const ZMM26: GraphId = GraphId::register(26);
pub const ZMM27: GraphId = GraphId::register(27);
pub const ZMM28: GraphId = GraphId::register(28);
pub const ZMM29: GraphId = GraphId::register(29);
pub const ZMM30: GraphId = GraphId::register(30);
pub const ZMM31: GraphId = GraphId::register(31);

impl GraphId {
  const SIB_RM: u8 = 0b100;

  pub(super) fn displacement(&self) -> Option<u64> {
    let val = self.0 & !Self::TY_MASK;

    if val & 0x5F == 63 {
      Some((val >> 7) as u64)
    } else {
      None
    }
  }

  pub(super) fn index(&self) -> u8 {
    debug_assert!(self.is(GraphIdType::REGISTER));
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
    debug_assert!(self.is(GraphIdType::REGISTER));
    match *self {
      R8 | RAX | R9 | RCX | R10 | RDX | R11 | RBX | R12 | RSP | R13 | RBP | R14 | RSI | R15
      | RDI => true,
      _ => false,
    }
  }

  /// The register is one of R8-R15
  pub(super) fn is_64_extended(&self) -> bool {
    debug_assert!(self.is(GraphIdType::REGISTER));
    match *self {
      R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 => true,
      _ => false,
    }
  }

  pub fn into_addr_op(&self, ctx: &SSAFunction, stack_offsets: &[u64]) -> Arg {
    if self.is(GraphIdType::REGISTER) {
      Arg::Mem(*self)
    } else {
      self.into_op(ctx, stack_offsets)
    }
  }

  pub fn into_op(&self, ctx: &SSAFunction, stack_offsets: &[u64]) -> Arg {
    if self.is(GraphIdType::REGISTER) {
      Arg::Reg(*self)
    } else if self.is(GraphIdType::CONST) {
      let value = ctx.constants[self.var_value()];
      Arg::Imm_Int(value.convert(TypeInfo::Integer | TypeInfo::b64).load().unwrap())
    } else if self.is(GraphIdType::VAR_LOAD) {
      Arg::RSP_REL(stack_offsets[self.var_value()])
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
  RIP_REL(u64),
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
