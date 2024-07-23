#![allow(unused)]

use rum_istring::CachedString;
mod bitsize;
mod primitive;
mod complex;
mod base;

use std::collections::HashMap;
use crate::IString;
pub use bitsize::*;
pub use primitive::*;
pub use complex::*;
pub use base::*;


#[derive(Debug)]
enum VariableEntry {
	WeakCandidates(Vec<Type>, Vec<Type>),
	StrongCandidate(Type),
}

#[derive(Debug)]
struct TypeScope {
	types: HashMap<IString, Type>
}

#[derive(Debug)]
struct TypeScopeEntry {
	parent_index: Option<usize>,
	entries: Vec<Box<Type>>
}

struct VarScope<'v_scope> {
	type_scope: *mut TypeScope,
	parent: Option<&'v_scope mut VarScope< 'v_scope>>,
	variables: HashMap<IString, VariableEntry>
}

impl<'v_scope> VarScope< 'v_scope> {
	pub fn get(&self, name:IString) -> Option<&VariableEntry> {
		if let Some(var) = self.variables.get(&name) {
			Some(var)
		} else if let Some(par) = &self.parent {
			par.get(name)
		} else {
			None
		}
	}

	pub fn set(&mut self, name: IString, var: VariableEntry) {
		self.variables.insert(name, var);
	}

	pub fn create_child_scope<'new_v_scope: 'v_scope>(&'new_v_scope mut self) -> VarScope< 'new_v_scope> {
		Self {
			type_scope: self.type_scope,
			parent: Some(self),
			variables: Default::default()
		}
	}
}


fn test_type_scope() {}


#[test]
fn test_variable_scope() {
	let mut t_scope = TypeScope { types: Default::default() };
	let mut v_scope = VarScope { type_scope: &mut t_scope, parent: None, variables: Default::default() };

	let mut c_v_scope = v_scope.create_child_scope();

	c_v_scope.set("test".intern(), VariableEntry::WeakCandidates(vec![PrimitiveType::i64.into()], vec![]));

	if let Some(var) = c_v_scope.get("test".to_token()) {
		dbg!(var);
	} else {
		panic!("Variable test not found");
	}
}