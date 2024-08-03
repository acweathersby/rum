#![feature(unsized_tuple_coercion)]
#![feature(allocator_api)]
#![feature(box_patterns)]

pub mod bitfield;
pub mod container;
pub mod error;
pub mod ir;
mod istring;
mod log;
pub mod parser;
pub mod types;
pub mod x86;

use std::{
  collections::{HashMap, VecDeque},
  f64::consts::PI,
};

use ir::ir_graph;
use ir_graph::BlockId;
pub use radlr_rust_runtime::types::Token;
use types::PrimitiveType;

use crate::{
  ir_graph::{IRGraphId, IRGraphNode},
  //x86::compile_from_ssa_fn,
  parser::script_parser::property_Value,
};

// Get expression type.
type Type = ();

#[test]
fn test() {
  parser::script_parser::parse_raw_expr("2*8").unwrap();
}

use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
};

pub fn create_u64_hash<T: Hash>(t: T) -> u64 {
  let mut s = DefaultHasher::new();

  t.hash(&mut s);

  s.finish()
}
