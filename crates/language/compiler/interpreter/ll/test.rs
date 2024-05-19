use std::path::PathBuf;

use crate::{
  compiler::{interpreter::error::RumResult, parser::parse_ll},
  utils::get_source_file,
};

use super::{ssa_block_compiler::compile_function_blocks, x86::compile_from_ssa_fn};
