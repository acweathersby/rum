use crate::targets::x86::push_bytes;

use super::{print_instruction, set_bytes, x86_types::*};

pub type OpSignature = (u16, OperandType, OperandType, OperandType);
pub type OpEncoder = fn(binary: &mut InstructionProps, op_code: u32, bit_size: u64, enc: OpEncoding, op1: Arg, op2: Arg, op3: Arg, ext: u8);

pub(crate) struct InstructionProps<'bin> {
  instruction_name:       &'static str,
  bin:                    &'bin mut Vec<u8>,
  pub displacement_index: usize,
  displacement_bit_size:  usize,
}

impl<'bin> InstructionProps<'bin> {
  #[track_caller]
  pub fn displace_too(&mut self, offset: usize) {
    if self.displacement_bit_size > 0 {
      let ip_offset = self.bin.len() as i64;
      let dis = offset as i64 - ip_offset;

      match self.displacement_bit_size {
        8 => set_bytes(self.bin, self.displacement_index, dis as i8),
        16 => set_bytes(self.bin, self.displacement_index, dis as i16),
        32 => set_bytes(self.bin, self.displacement_index, dis as i32),
        64 => set_bytes(self.bin, self.displacement_index, dis as i64),
        size => panic!("Invalid displacement size {size}. {}", self.instruction_name),
      }
    } else {
      panic!("Attempt to adjust displacement of instruction that has no such value. {}", self.instruction_name)
    }
  }
}

pub(crate) fn encode_zero<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
) -> InstructionProps<'bin> {
  encode_x86(binary, table, bit_size, Arg::None, Arg::None, Arg::None)
}

pub(crate) fn encode_unary<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
) -> InstructionProps<'bin> {
  encode_x86(binary, table, bit_size, op1, Arg::None, Arg::None)
}

pub(crate) fn test_enc_uno<'bin>(table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]), bit_size: u64, op1: Arg) -> String {
  let mut bin = vec![];
  encode_x86(&mut bin, table, bit_size, op1, Arg::None, Arg::None);
  print_instruction(&bin)
}

pub(crate) fn test_enc_dos<'bin>(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
  op2: Arg,
) -> String {
  let mut bin = vec![];
  encode_x86(&mut bin, table, bit_size, op1, op2, Arg::None);
  //println!("{:02X?}", &bin);
  print_instruction(&bin)
}

pub(crate) fn test_enc_tres<'bin>(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> String {
  let mut bin = vec![];
  encode_x86(&mut bin, table, bit_size, op1, op2, op3);
  //println!("{:02X?}", &bin);
  print_instruction(&bin)
}

pub(crate) fn encode_binary<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
  op2: Arg,
) -> InstructionProps<'bin> {
  encode_x86(binary, table, bit_size, op1, op2, Arg::None)
}

pub(crate) fn encode_x86<'bin>(
  binary: &'bin mut Vec<u8>,
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> InstructionProps<'bin> {
  debug_assert!(bit_size <= 512);

  if bit_size == 16 {
    // 16bit encoding requires the 0x66 prefix
    binary.push(0x66);
  }

  let signature = (bit_size as u16, op1.ty(), op2.ty(), op3.ty());

  for (sig, (op_code, ext, encoding, encoder)) in &table.1 {
    if *sig == signature {
      let mut props = InstructionProps {
        instruction_name:      table.0,
        bin:                   binary,
        displacement_index:    0,
        displacement_bit_size: 0,
      };
      unsafe { (std::mem::transmute::<_, OpEncoder>(*encoder))(&mut props, *op_code, bit_size, *encoding, op1, op2, op3, *ext) };
      return props;
    }
  }

  panic!(
    "Could not find operation for {signature:?} in encoding table \n\n{}",
    format!("{}:\n{}", table.0, table.1.iter().map(|v| format!("{:?} {:?}", v.0, v.1)).collect::<Vec<_>>().join("\n"))
  );
}

pub(crate) fn encoded_vec(
  table: &(&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]),
  bit_size: u64,
  op1: Arg,
  op2: Arg,
  op3: Arg,
) -> usize {
  let mut bin = vec![];
  encode_x86(&mut bin, table, bit_size, op1, op2, op3);
  bin.len()
}

pub(crate) fn gen_zero_op(props: &mut InstructionProps, op_code: u32, bit_size: u64, enc: OpEncoding, _: Arg, _: Arg, _: Arg, ext: u8) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    Zero => {
      insert_op_code_bytes(props.bin, op_code as u32);
    }
    enc => panic!("{enc:?} not valid for unary operations"),
  }
}

pub(crate) fn gen_unary_op(props: &mut InstructionProps, op_code: u32, bit_size: u64, enc: OpEncoding, op1: Arg, _: Arg, _: Arg, ext: u8) {
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
          8 => push_bytes(props.bin, imm as u8),
          16 => push_bytes(props.bin, imm as u16),
          32 => push_bytes(props.bin, imm as u32),
          64 => push_bytes(props.bin, imm as u64),
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
      props.displacement_index = props.bin.len();
      match op1 {
        Arg::Imm_Int(imm) => match bit_size {
          8 => push_bytes(props.bin, imm as i8),
          16 => push_bytes(props.bin, imm as i16),
          _ => push_bytes(props.bin, imm as i32),
        },
        _ => unreachable!(),
      }
    }
    enc => panic!("{enc:?} not valid for unary operations"),
  }
}

pub(crate) fn gen_multi_op(props: &mut InstructionProps, op_code: u32, bit_size: u64, enc: OpEncoding, op1: Arg, op2: Arg, op3: Arg, ext: u8) {
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
          8 => push_bytes(props.bin, imm as u8),
          16 => push_bytes(props.bin, imm as u16),
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
          8 => push_bytes(props.bin, imm as u8),
          16 => push_bytes(props.bin, imm as u16),
          32 => push_bytes(props.bin, imm as u32),
          64 => push_bytes(props.bin, imm as u64),
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
          8 => push_bytes(props.bin, imm as u8),
          16 => push_bytes(props.bin, imm as u16),
          64 | 32 => push_bytes(props.bin, 3 as u32),
          size => panic!("Invalid immediate size of {size:?} for OI encoding"),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    enc => panic!("{enc:?} not valid for binary operations on 1:{op1:?} 2:{op2:?}"),
  }
}

pub(crate) fn encode_mod_rm_reg(props: &mut InstructionProps, r_m: Arg, reg: Arg) {
  const SIB_SCALE_OFFSET: u8 = 6;
  const SIB_INDEX_OFFSET: u8 = 3;
  const SIB_INDEX_NOT_USED: u8 = 0b100 << SIB_INDEX_OFFSET;
  const SIB_NO_INDEX_SCALE: u8 = 0b00 << SIB_SCALE_OFFSET;
  const DISPLACEMENT_INDEX: u8 = 0b101;
  const SIB_RIP_BASE: u8 = 0b101;

  let mut mem_encoding = 0b00;
  let mut displace_val = 0 as i64;
  let rm_index = r_m.reg_index();

  let sib = match rm_index {
    4 => match r_m {
      Arg::Mem(RSP) | Arg::Mem(R12) => {
        // use sib index to access the RSP register
        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | (RSP.0 & 7) as u8) as u8
      }

      Arg::RSP_REL(val) | Arg::MemRel(_, val) => {
        if (val & !0xFF) > 0 {
          mem_encoding = 0b10
        } else {
          mem_encoding = 0b01;
        }

        displace_val = val;

        (SIB_NO_INDEX_SCALE | SIB_INDEX_NOT_USED | (RSP.0 & 7) as u8) as u8
      }
      _ => 0,
      arg => unreachable!("{arg:?}"),
    },
    5 => match r_m {
      Arg::RIP_REL(val) | Arg::MemRel(_, val) => {
        displace_val = val;
        0
      }
      Arg::Mem(RBP) | Arg::Mem(R13) => {
        // use sib index to access the RSP register
        mem_encoding = 0b01;
        (SIB_NO_INDEX_SCALE | (0b000 << 3) | 0b000) as u8
      }
      Arg::Reg(RBP) | Arg::Reg(R13) => {
        // use sib index to access the RBP register
        0
      }
      _ => unreachable!(),
    },
    _ => match r_m {
      Arg::MemRel(_, val) => {
        if (val & !0xFF) > 0 {
          mem_encoding = 0b10
        } else {
          mem_encoding = 0b01;
        }

        displace_val = val;

        0
      }
      _ => 0,
    },
  };

  let mod_bits = match r_m {
    Arg::RSP_REL(_) | Arg::RIP_REL(_) | Arg::Mem(_) | Arg::MemRel(..) => mem_encoding,
    Arg::Reg(_) => 0b11,
    op => panic!("Invalid r_m operand {op:?}"),
  };

  dbg!(rm_index);

  props.bin.push(((mod_bits & 0b11) << 6) | ((reg.reg_index() & 0x7) << 3) | (rm_index & 0x7));

  if sib != 0 {
    props.bin.push(sib)
  }

  match mod_bits {
    0b01 => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 8;
      push_bytes(props.bin, displace_val as u64 as u8);
    }
    0b00 if rm_index == DISPLACEMENT_INDEX => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 32;
      push_bytes(props.bin, displace_val as u64 as u32);
    }
    0b10 => {
      props.displacement_index = props.bin.len();
      props.displacement_bit_size = 32;
      push_bytes(props.bin, displace_val as u64 as u32);
    }
    _ => {}
  }
}

pub(crate) fn encode_rex(props: &mut InstructionProps, bit_size: u64, r_m: Arg, reg: Arg) {
  const REX_W_64B: u8 = 0b0100_1000;
  const REX_R_REG_EX: u8 = 0b0100_0100;
  const REX_X_SIP: u8 = 0b0100_0010;
  const REX_B_MEM_REG_EX: u8 = 0b0100_0001;

  let mut rex = 0;
  rex |= (bit_size == 64).then_some(REX_W_64B).unwrap_or(0);
  rex |= (r_m.is_upper_8_reg()).then_some(REX_B_MEM_REG_EX).unwrap_or(0);
  rex |= (reg.is_upper_8_reg()).then_some(REX_R_REG_EX).unwrap_or(0);
  if rex > 0 {
    props.bin.push(rex);
  }
}

pub(crate) fn encode_evex(op_code: u32, r_m: Arg, reg: Arg, op3: Arg, bit_size: u64, props: &mut InstructionProps<'_>, w: u8) -> u32 {
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
    Arg::Reg(reg) if !op3.is_mask_register() => (!(reg.0 as u8) & 0xF, ((reg.is_upper_16_reg() as u8) ^ 0x1)),
    _ => (0xF, 0x1),
  };
  let (vvvv, V) = (vvvv << 3, V << 3);

  let z = 0 << 7;
  let ll = match bit_size {
    512 => 2,
    256 => 1,
    128 | _ => 0,
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

pub(crate) fn encode_vex(op_code: u32, op2: Arg, op1: Arg, bit_size: u64, props: &mut InstructionProps<'_>) -> u32 {
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

pub(crate) fn gen_tri_op(props: &mut InstructionProps, op_code: u32, bit_size: u64, enc: OpEncoding, op1: Arg, op2: Arg, op3: Arg, ext: u8) {
  use Arg::*;
  use OpEncoding::*;

  match enc {
    RMI => match op3 {
      Imm_Int(imm) => {
        encode_rex(props, bit_size, op1, op2);
        insert_op_code_bytes(props.bin, op_code);

        encode_mod_rm_reg(props, op1, op2);
        match bit_size {
          8 => push_bytes(props.bin, imm as u8),
          _ => push_bytes(props.bin, imm as u32),
        }
      }
      imm => panic!("Invalid immediate arg op2 of {imm:?} for MI encoding"),
    },
    enc => panic!("{enc:?} not valid for binary operations on {op1:?} on {op2:?}"),
  }
}

pub(crate) fn insert_op_code_bytes(binary: &mut Vec<u8>, op_code: u32) {
  for byte in op_code.to_be_bytes() {
    if byte != 0 {
      push_bytes(binary, byte);
    }
  }
}
