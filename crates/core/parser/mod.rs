use std::sync::Arc;

use radlr_rust_runtime::{
  parsers::ast::AstDatabase,
  types::{RuntimeDatabase, StringInput, Token},
};

#[allow(warnings)]
pub mod ast;

#[allow(warnings)]
mod parser;

/// Parses input based on the LL grammar.
pub fn parse_ops(input: &str) -> Result<Arc<ast::Ops>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("rules").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => {
      let node: ast::ASTNode<Token> = node;
      Ok(node.into_Ops().unwrap())
    }
  }
}
