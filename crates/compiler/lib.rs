#![feature(debug_closure_helpers)]
#![feature(box_as_ptr)]
#![feature(unsized_tuple_coercion)]
#![feature(if_let_guard)]
#![feature(str_from_raw_parts)]
#![allow(mixed_script_confusables)]
#![feature(let_chains)]
// Temporary
#![allow(unused)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod bitfield;

pub mod basic_block_compiler;

pub mod ir_compiler;

pub mod _interpreter;

pub mod optimizer;

pub mod finalizer;

pub mod linker;

pub mod elf_link;

pub mod solver;

pub mod types;

pub mod targets;
