#![feature(box_as_ptr)]
#![feature(unsized_tuple_coercion)]
#![feature(if_let_guard)]
#![feature(str_from_raw_parts)]

pub mod bitfield;

pub mod basic_block_compiler;

pub mod ir_compiler;

pub mod _interpreter;

pub mod optimizer;

pub mod finalizer;

pub mod linker;

pub mod solver;

pub mod types;

pub mod targets;
