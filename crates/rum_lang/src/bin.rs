use rum_lang::{
  ir::{
    db::Database,
    ir_rvsdg::{lower::lower_ast_to_rvsdg, RVSDGNodeType},
    types::{EntryOffsetData, PrimitiveBaseType, Type, TypeDatabase},
  },
  ir_interpreter::value::Value,
  parser::script_parser::parse_raw_module,
};
use std::{collections::VecDeque, path::PathBuf};

fn main() {}
