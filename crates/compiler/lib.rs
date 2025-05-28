#![feature(box_as_ptr)]
#![feature(unsized_tuple_coercion)]
#![feature(if_let_guard)]

pub mod ir_compiler;

pub mod interpreter;

pub mod optimizer;

pub mod solver;

pub mod types;

pub mod targets;
