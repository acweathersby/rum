use std::sync::Arc;

use radlr_rust_runtime::{
  parsers::ast::AstDatabase,
  types::{RuntimeDatabase, StringInput},
};

#[allow(warnings)]
pub mod ast;

#[allow(warnings)]
mod parser;

pub use radlr_rust_runtime::types::Token;

pub use ast::*;

pub type ASTNode = ast::ASTNode<radlr_rust_runtime::types::Token>;

pub fn parse_rs(input: &str) -> Result<ASTNode, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("RS").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(node),
  }
}

/// Parses input based on the LL grammar.
pub fn parse_raw(input: &str) -> Result<Arc<RawRoutine<Token>>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("raw_function").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(node.into_RawRoutine().unwrap()),
  }
}

pub fn parse_raw_call(input: &str) -> Result<Arc<RawCall<Token>>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("raw_call").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(node.into_RawCall().unwrap()),
  }
}

pub fn parse_raw_module(input: &str) -> Result<Arc<RawModule<Token>>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("raw_module").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(node.into_RawModule().unwrap()),
  }
}
