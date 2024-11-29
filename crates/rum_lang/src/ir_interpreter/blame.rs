use crate::parser::script_parser::ASTNode;
use radlr_rust_runtime::types::{BlameColor, Token};

pub fn blame(node: &ASTNode, message: &str) -> String {
  let default = Token::default();
  let tok: &Token = {
    use crate::parser::script_parser::ast::ASTNode::*;
    match node {
      NamedMember(node) => &node.tok,
      MemberCompositeAccess(node) => &node.tok,
      RawAggregateMemberInit(node) => &node.tok,
      RawAggregateInstantiation(node) => &node.tok,
      Var(node) => &node.tok,
      RawCall(node) => &node.tok,
      RawNum(node) => &node.tok,
      Expression(node) => &node.tok,
      Add(node) => &node.tok,
      RawParamType(node) => &node.tok,
      RawParamBinding(node) => &node.tok,
      RawMatchClause(node) => &node.tok,
      RawExprMatch(node) => &node.tok,
      None => &default,
      node => panic!("unrecognized node: {node:#?}"),
    }
  };

  tok.blame(0, 0, message, BlameColor::RED)
}
