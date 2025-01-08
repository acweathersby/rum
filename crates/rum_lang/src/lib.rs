#![feature(unsized_tuple_coercion)]
#![feature(allocator_api)]
#![feature(box_patterns)]
#![feature(vec_into_raw_parts)]
#![feature(debug_closure_helpers)]
#![allow(warnings)]

pub mod error;
mod log;
pub mod parser;
//pub mod types;
//pub mod vm;
//pub mod x86;

pub use radlr_rust_runtime::types::Token;

#[test]
fn test() {
  // parser::script_parser::parse_raw_expr("2*8").unwrap();
}

use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
};
