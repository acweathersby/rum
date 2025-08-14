#![allow(non_snake_case)]

use std::{
  collections::HashMap,
  io::{Read, Write},
  os::unix::fs::FileExt,
  path::Path,
};

use rum_common::{CachedString, IString};

use crate::{
  linker::{link, BinaryObject}, targets::x86::print_instructions, types::*
};

#[repr(u8)]
#[derive(Clone, Copy)]
enum ELF64Endianess {
  None,
  Little,
  Big,
}

#[repr(u16)]
#[derive(Clone, Copy)]
enum ELF64FileType {
  None,
  Relocatable,
  Executable,
  Dynamic,
  Core,
}

#[repr(u16)]
#[derive(Clone, Copy)]
enum ELF64MachineType {
  None,
  M32,
  Sparc,
  _386,
  _68K,
  _88K,
  _860,
  MIPS,
  MIPS_RS4_BE,
  AMD64 = 0x3E,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum ELF64SectionType {
  Null,
  ProgramBits,
  SymTable,
  StringTable,
  RelocationEntriesAndAddends,
  SymbolHashTable,
  Dynamic,
  Note,
  NoBits,
  RelocationEntries,
  ShLib,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum ELF64SegmentType {
  Null,
  Load,
  Dynamic,
  Interpreter,
  Note,
  Shlib,
  Phdr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ELF64Header {
  /// Magic number
  EI_MAGIC_NUMBER: [u8; 4],
  /// Bit Class
  EI_CLASS:        u8,
  /// The byte format of the file: Little / Big Endian
  EI_DATA:         ELF64Endianess,
  /// ELF Version; Only Version 1 exists
  EI_VERSION:      u8,
  /// ABI Architecture
  EI_OSABI:        u8,
  ///
  EI_ABI_VERSION:  u8,
  /// Padding. Should be Zeroed
  EI_PAD:          [u8; 7],
  // Object File Type
  ty:              ELF64FileType,
  /// CPU Architecture
  machine:         ELF64MachineType,
  version:         u32,
  /// The virtual address to which the system first transfers control, thus starting the process.
  entry:           u64,
  /// Program Header Table File Offset
  phoff:           u64,
  /// Section Header Table File Offset
  shoff:           u64,
  /// CPU Specific Flags
  flags:           u32,
  /// Elf Header Size; 64 bit elf files have 64 byte headers.
  ehsize:          u16,
  /// Size of a program header table's entry
  phentsize:       u16,
  /// Number of entries in the program header table
  phnum:           u16,
  /// Size of a section header table's entry
  shentsize:       u16,
  /// Number of entries in the section header table
  shnum:           u16,
  /// Index of the string section entry in the section header.
  /// The value is `0` if there is no string section
  shstrndx:        u16,
}

impl ELF64Header {
  pub fn new() -> Self {
    Self {
      EI_MAGIC_NUMBER: [0x7F, 0x45, 0x4C, 0x46],
      EI_CLASS:        2, // 64 bit
      EI_DATA:         ELF64Endianess::Little,
      EI_VERSION:      1,
      EI_OSABI:        0x03, // Linux ABI,
      EI_ABI_VERSION:  0,
      EI_PAD:          Default::default(),
      ty:              ELF64FileType::Executable, // Executable
      machine:         ELF64MachineType::AMD64,   // AMD 64
      version:         1,
      entry:           0,
      phoff:           0,
      shoff:           0,
      flags:           0,
      ehsize:          0x40,
      phentsize:       0x38,
      phnum:           0,
      shentsize:       0x40,
      shnum:           0,
      shstrndx:        0,
    }
  }
}

#[repr(C)]
struct ELF64ProgramHeader {
  p_type:   ELF64SegmentType,
  p_flags:  u32,
  p_offset: u64,
  p_vaddr:  u64,
  p_paddr:  u64,
  p_filesz: u64,
  p_memsz:  u64,
  p_align:  u64,
}

#[repr(C)]
struct ELF64SectionHeader {
  /// Entry in the string header table
  sh_name:       u32,
  sh_type:       ELF64SectionType,
  sh_flags:      u64,
  sh_addr:       u64,
  sh_offset:     u64,
  sh_size:       u64,
  sh_link:       u32,
  sh_info:       u32,
  sh_addrealign: u64,
  sh_entsize:    u64,
}

#[repr(C)]
struct ELF64SymbolEntry {
  st_name:  u32,
  st_info:  u8,
  st_other: u8,
  st_shndx: u16,
  st_value: u64,
  st_size:  u64,
}

const SHF_WRITE: u32 = 1;
const SHF_ALLOC: u32 = 2;
const SHF_EXECINSTR: u32 = 4;

const PF_EXECINSTR: u32 = 1;
const PF_WRITE: u32 = 2;
const PF_READ: u32 = 4;

pub fn elf_link(db: &SolveDatabase<'_>, main: CMPLXId, mut bin_functs: Vec<(CMPLXId, BinaryObject)>, out_dir: &Path, file_name: &str) {
  let (complex_lookups, binary) = link(db, bin_functs);


  print_instructions(&binary, 0);

  let Some(entry_offset) = complex_lookups.get(&main) else { panic!("Could not located entrypoint!") };
  let entry_offset = *entry_offset as u64;

  const size_of_program_header: u64 = std::mem::size_of::<ELF64ProgramHeader>() as _;
  const size_of_section_header: u64 = std::mem::size_of::<ELF64SectionHeader>() as _;
  const size_of_file_header: u64 = std::mem::size_of::<ELF64Header>() as _;
  const size_of_symbol_entry: u64 = std::mem::size_of::<ELF64SymbolEntry>() as _;

  std::fs::create_dir_all(out_dir).expect("Could not create directory path: {out_dir:?}");

  let mut string_data: Vec<u8> = Vec::from_iter([0u8]);
  let mut string_offsets: HashMap<IString, usize> = HashMap::new();

  let file_path = out_dir.join(file_name);

  let Ok(mut file_writer) = std::fs::File::create(file_path.clone()) else { panic!("Could not prepare file {file_path:?} for writing") };

  let mut elf_header = ELF64Header::new();

  //elf_header.s

  //let mut
  let mut byte_counter = 0;
  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 64]>(elf_header) });
  byte_counter += std::mem::size_of::<ELF64Header>();

  // String table. Create section for strings

  // Globals - including type table

  // text segment

  // out_file.

  let program_header_1 = ELF64ProgramHeader {
    p_type:   ELF64SegmentType::Load,
    p_align:  0x1000,
    p_vaddr:  0x400000,
    p_offset: 0,
    p_filesz: (byte_counter as u64 + size_of_program_header + binary.len() as u64),
    p_memsz:  0x1000,
    p_flags:  PF_EXECINSTR | PF_READ,
    p_paddr:  0,
  };

  elf_header.entry = 0x400000 + size_of_file_header + size_of_program_header + entry_offset;
  elf_header.phoff = byte_counter as _;
  elf_header.phnum = 1;
  elf_header.phentsize = (size_of_program_header) as _;

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 56]>(program_header_1) });
  byte_counter += 56;

  file_writer.write(&binary);
  byte_counter += binary.len();

  let reserved_entry = ELF64SymbolEntry { st_info: 0, st_name: 0, st_other: 0, st_shndx: 0, st_size: 0, st_value: 0 };

  let symbol_entry_start = byte_counter;
  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 24]>(reserved_entry) });
  byte_counter += 24;

  let main_entry = ELF64SymbolEntry {
    st_info:  1 << 4 | 2,
    st_name:  get_string_index("main", &mut string_offsets, &mut string_data) as _,
    st_other: 0,
    st_shndx: 1,
    st_size:  binary.len() as _,
    st_value: elf_header.entry,
  };

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 24]>(main_entry) });
  byte_counter += 24;

  elf_header.shoff = byte_counter as _;
  elf_header.shnum = 4;

  let section_one = ELF64SectionHeader {
    sh_addr:       0,
    sh_type:       ELF64SectionType::Null,
    sh_flags:      0,
    sh_addrealign: 0,
    sh_entsize:    0,
    sh_size:       0,
    sh_offset:     0,
    sh_info:       0,
    sh_link:       0,
    sh_name:       0,
  };

  let text_section = ELF64SectionHeader {
    sh_addr:       elf_header.entry as _,
    sh_type:       ELF64SectionType::ProgramBits,
    sh_flags:      0,
    sh_addrealign: 0x1000,
    sh_entsize:    1,
    sh_size:       binary.len() as _,
    sh_offset:     (size_of_file_header + size_of_program_header) as _,
    sh_info:       0,
    sh_link:       0,
    sh_name:       get_string_index(".text", &mut string_offsets, &mut string_data) as _,
  };

  let mut string_table = ELF64SectionHeader {
    sh_addr:       0,
    sh_type:       ELF64SectionType::StringTable,
    sh_flags:      0,
    sh_addrealign: 0,
    sh_entsize:    1,
    sh_size:       0,
    sh_offset:     0,
    sh_info:       0,
    sh_link:       0,
    sh_name:       get_string_index(".strtab", &mut string_offsets, &mut string_data) as _,
  };

  let symbol_table = ELF64SectionHeader {
    sh_addr:       0,
    sh_type:       ELF64SectionType::SymTable,
    sh_flags:      0,
    sh_addrealign: 0,
    sh_entsize:    24,
    sh_size:       48,
    sh_offset:     symbol_entry_start as _,
    sh_info:       1,
    sh_link:       2,
    sh_name:       get_string_index(".symtab", &mut string_offsets, &mut string_data) as _,
  };

  let strings_offset = byte_counter;
  string_table.sh_offset = strings_offset as _;
  string_table.sh_size = string_data.len() as _;
  file_writer.write(&string_data);
  byte_counter += string_data.len();

  elf_header.shoff = byte_counter as _;
  elf_header.shnum = 4;
  elf_header.shstrndx = 2;

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 64]>(section_one) });
  byte_counter += 64;

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 64]>(text_section) });
  byte_counter += 64;

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 64]>(string_table) });
  byte_counter += 64;

  file_writer.write(unsafe { &std::mem::transmute::<_, [u8; 64]>(symbol_table) });
  byte_counter += 64;

  file_writer.write_at(unsafe { &std::mem::transmute::<_, [u8; 64]>(elf_header) }, 0);

  file_writer.flush();
}

fn get_string_index(str: &str, string_offsets: &mut HashMap<IString, usize>, string_buffer: &mut Vec<u8>) -> usize {
  *string_offsets.entry(str.to_token()).or_insert_with(|| {
    let val = string_buffer.len();
    string_buffer.extend(str.as_bytes());
    string_buffer.push(0);
    val
  })
}
