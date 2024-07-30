#![feature(unsized_tuple_coercion)]
pub mod bitfield;
pub mod parser;
pub mod error;
pub mod ir;
pub mod types;
pub mod x86;
use std::{
  collections::{HashMap, VecDeque},
  f64::consts::PI,
};

use ir::ir_graph;
use ir_graph::BlockId;
pub use radlr_rust_runtime::types::Token;
use rum_container::{get_aligned_value, ArrayVec};
use rum_istring::{CachedString, IString};
use types::PrimitiveType;

use crate::{
  parser::script_parser::property_Value,
  ir_graph::{IRGraphId, IRGraphNode},
  //x86::compile_from_ssa_fn,
};

// Get expression type.
type Type = ();

#[test]
fn test() {
  parser::script_parser::parse_raw_expr("2*8").unwrap();
}
