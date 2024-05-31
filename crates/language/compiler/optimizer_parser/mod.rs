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

pub fn parse(input: &str) -> Result<Box<Optimizations>, String> {
  let parser_db = parser::ParserDB::new();
  match parser_db.build_ast(
    &mut StringInput::from(input),
    parser_db.get_entry_data_from_name("default").unwrap(),
    ast::ReduceRules::<radlr_rust_runtime::types::Token>::new(),
  ) {
    Err(err) => {
      println!("{err:?}");
      Err("Failed to parse input".to_string())
    }
    Ok(node) => Ok(node.into_Optimizations().unwrap()),
  }
}

#[test]
fn build_opti_ast() {
  let input = r##" 
  
  match mov $?( var const ) $? ( const ) { }
  
  "##;

  let output = parse(input).expect("Could not parse input");

  dbg!(output);
}
