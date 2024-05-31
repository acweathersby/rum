use radlr_rust_runtime::{
  parsers::ast::{AstDatabase, Tk},
  types::{ParserProducer, RuntimeDatabase, StringInput},
};

#[allow(warnings)]
mod ast;

#[allow(warnings)]
mod parser;

pub use radlr_rust_runtime::types::Token;

pub use ast::*;

pub type ASTNode = ast::ASTNode<radlr_rust_runtime::types::Token>;

pub fn parse_RS(input: &str) -> Result<ASTNode, String> {
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
pub fn parse_ll(input: &str) -> Result<LLFunction<Token>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("LL").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(*node.into_LLFunction().unwrap()),
  }
}
