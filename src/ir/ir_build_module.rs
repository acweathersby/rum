use super::{
  ir_builder::{IRBuilder, SuccessorMode},
  ir_graph::IRGraphId,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  ir::{
    ir_builder::{SMO, SMT},
    ir_graph::{IROp, VarId},
  },
  istring::*,
  parser::script_parser::{
    assignment_var_Value,
    bitfield_element_Value,
    block_expression_group_1_Value,
    expression_Value,
    loop_statement_group_1_Value,
    match_clause_Value,
    match_expression_Value,
    match_scope_Value,
    property_Value,
    raw_module_Value,
    routine_type_Value,
    statement_Value,
    type_Value,
    Add,
    Div,
    Expression,
    Mul,
    Pow,
    RawBlock,
    RawCall,
    RawLoop,
    RawMatch,
    RawMember,
    RawNum,
    RawRoutine,
    RawStructDeclaration,
    Sub,
    BIT_SL,
    BIT_SR,
  },
  types::*,
};
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::{btree_map, BTreeMap},
  sync::Arc,
};

use IROp::*;
use MemberName::*;
use SMO::*;
use SMT::{Inherit, Undef};

pub fn process_types<'a>(module: &'a Vec<raw_module_Value<Token>>, type_scope_index: usize, type_scope: &mut TypeContext) -> Vec<IString> {
  let mut routines = Vec::new();

  for mod_member in module {
    match mod_member {
      raw_module_Value::RawRoutine(routine) => {
        process_routine_signature(routine, type_scope_index, type_scope);
        let name = routine.name.id.intern();
        routines.push(name);
      }
      raw_module_Value::RawUnion(union) => {
        dbg!(union);
        todo!("Build / Union");
      }
      raw_module_Value::RawStruct(strct) => {
        let name = strct.name.id.intern();
        let mut members = Vec::new();
        let mut offset = 0;
        let mut min_alignment = 1;

        for (index, prop) in strct.properties.iter().enumerate() {
          match prop {
            property_Value::RawBitCompositeProp(bitfield) => {
              let max_bit_size = bitfield.bit_count as u64;
              let max_byte_size = max_bit_size.div_ceil(8) as u64;
              let bit_field_size = max_byte_size << 3;

              let mut size = 0;
              let mut bit_offset = 0;

              let prop_offset = get_aligned_value(offset, max_byte_size);
              offset = prop_offset + max_byte_size as u64;

              for prop in &bitfield.props {
                match prop {
                  bitfield_element_Value::BitFieldDescriminator(desc) => {
                    if bit_offset != 0 {
                      panic!("A discriminator must be the first element of a bitfield.")
                    }
                    let bit_size = desc.bit_count as u64;

                    debug_assert!(bit_size <= 128);

                    members.push(StructMemberType {
                      ty:             (PrimitiveType::Discriminant
                        | PrimitiveType::new_bit_size(bit_size)
                        | PrimitiveType::new_bitfield_data(bit_offset as u8, bit_field_size as u8))
                      .into(),
                      original_index: index,
                      name:           "#desc".intern(),
                      offset:         prop_offset,
                    });

                    bit_offset += bit_size;
                  }
                  bitfield_element_Value::BitFieldProp(prop) => {
                    let ty = get_type(&prop.r#type, type_scope_index, type_scope).unwrap();

                    let bit_size = ty.bit_size();
                    let name = prop.name.id.intern();

                    if members.iter().any(|m| m.name == name) {
                      panic!("Name already taken {name:?}")
                    }

                    members.push(StructMemberType {
                      ty: (*ty.as_prim().unwrap() | PrimitiveType::new_bitfield_data(bit_offset as u8, bit_field_size as u8)).into(),
                      original_index: index,
                      name,
                      offset: prop_offset,
                    });

                    bit_offset += bit_size;
                  }
                  bitfield_element_Value::None => {}
                }
              }

              if bit_offset > max_bit_size {
                panic!("Bitfield element size {bit_offset} overflow bitfield size {max_bit_size}")
              }
            }
            property_Value::RawProperty(raw_prop) => {
              let ty = get_type(&raw_prop.r#type, type_scope_index, type_scope).unwrap();

              let prop_offset = get_aligned_value(offset, ty.alignment() as u64);
              offset = prop_offset + ty.byte_size() as u64;

              min_alignment = min_alignment.max(ty.alignment() as u64);

              let name = raw_prop.name.id.intern();

              if members.iter().any(|m| m.name == name) {
                panic!("Name already taken {name:?}")
              }

              members.push(StructMemberType { ty, original_index: index, name, offset: prop_offset })
            }
            _ => {}
          }
        }

        let s = StructType {
          name,
          members: members.into_iter().map(|s| Box::new(ComplexType::StructMember(s))).collect(),
          alignment: min_alignment,
          size: get_aligned_value(offset, min_alignment as u64),
        };

        type_scope.set(type_scope_index, name, crate::types::ComplexType::Struct(s));
      }
      _ => {}
    }
  }

  routines
}

pub fn build_module(module: &Vec<raw_module_Value<Token>>, type_scope_index: usize, type_scope: &mut TypeContext) {
  let routines = process_types(module, type_scope_index, type_scope);
  //let mut types = Vec::new();

  for routine in &routines {
    // Gather function type information.
    process_routine(*routine, type_scope_index, type_scope);
  }
}

/// Processes the signature of a routine and stores the result into the type
/// context.
fn process_routine_signature(routine: &Arc<RawRoutine<Token>>, type_scope_index: usize, type_scope: &mut TypeContext) {
  let name = routine.name.id.intern();

  let none = type_Value::None;

  let (params, ret) = match &routine.ty {
    routine_type_Value::RawFunctionType(fn_ty) => (fn_ty.params.as_ref(), &fn_ty.return_type),
    routine_type_Value::RawProcedureType(proc_ty) => (proc_ty.params.as_ref(), &none),
    _ => unreachable!(),
  };

  let mut parameters = Vec::new();

  for (index, param) in params.params.iter().enumerate() {
    let name = param.var.id.intern();
    dbg!(param);
    if param.inferred {
      parameters.push((name, index, Type::UNRESOLVED));
    } else if let Some(ty) = get_type(&param.ty, type_scope_index, type_scope) {
      if ty.is_unresolved() {
        panic!("Could not resolve type!");
      }

      parameters.push((name, index, ty));
    } else {
      panic!("Could not resolve type!");
    }
  }

  let returns = match ret {
    type_Value::None => vec![],
    ty => {
      let return_type = get_type(ty, type_scope_index, type_scope);
      vec![return_type.unwrap()]
    }
  };

  let ty = RoutineType {
    name,
    variables: Default::default(),
    body: Default::default(),
    parameters,
    returns,
    ast: routine.clone(),
  };

  type_scope.set(type_scope_index, name, crate::types::ComplexType::Routine(ty));
}

fn process_routine(routine_name: IString, type_scope_index: usize, type_scope: &TypeContext) {
  if let Some(ComplexType::Routine(rt)) = type_scope.get_mut(type_scope_index, routine_name) {
    rt.body.resolved = false;
    let RoutineType { name, parameters, returns, body, variables: vars, ast } = rt;

    let mut ir_builder = IRBuilder::new(body, vars, type_scope_index, type_scope);

    for (name, index, ty) in parameters.iter() {
      let var = ir_builder.push_para_var(IdMember(*name), *ty, VarId::new(*index as u32));
      ir_builder.push_ssa(PARAM_VAL, var.ty.into(), &[var.store.into()], var.id);
    }

    process_expression(&ast.expression.expr, &mut ir_builder);

    if rt.returns.len() > 0 {
      todo!("Check that return types match!");
    }

    dbg!(rt);
  } else {
    panic!("Could not find type definition for {}", routine_name.to_str().as_str());
  }
}

fn process_expression(expr: &expression_Value<Token>, ib: &mut IRBuilder) {
  match expr {
    expression_Value::RawCall(call) => process_call(call, ib),
    expression_Value::RawNum(num) => process_const_number(num, ib),
    expression_Value::AddressOf(addr) => process_address_of(ib, addr),
    expression_Value::RawStructDeclaration(struct_decl) => process_struct_instantiation(struct_decl, ib),
    expression_Value::RawMember(mem) => process_member_load(mem, ib),
    expression_Value::RawBlock(ast_block) => process_block(ast_block, ib),
    expression_Value::Add(add) => process_add(add, ib),
    expression_Value::Sub(sub) => process_sub(sub, ib),
    expression_Value::Mul(mul) => process_mul(mul, ib),
    expression_Value::Div(div) => process_div(div, ib),
    expression_Value::Pow(pow) => process_pow(pow, ib),
    expression_Value::BIT_SL(sl) => process_sl(sl, ib),
    expression_Value::BIT_SR(sr) => process_sr(sr, ib),
    d => todo!("expression: {d:#?}"),
  }
}

fn process_address_of(ib: &mut IRBuilder<'_, '_, '_>, addr: &std::sync::Arc<crate::parser::script_parser::AddressOf<Token>>) {
  if let Some(var) = ib.get_variable(IdMember(addr.id.id.intern())) {
    ib.push_ssa(ADDR, var.ty.as_pointer().into(), &[SMO::IROp(var.store)], var.id)
  } else {
    panic!("Variable not found")
  }
}

fn process_member_load(mem: &std::sync::Arc<RawMember<Token>>, ib: &mut IRBuilder<'_, '_, '_>) {
  println!("{}", mem.tok.blame(2, 2, "", None));
  if let Some(var) = resolve_variable(mem, ib) {
    if var.is_member_pointer {
      // Loading the variable into a register creates a temporary variable
      match var.ty.base_type() {
        BaseType::Prim(prim) => {
          if prim.bitfield_offset() > 0 {
            let ty: Type = prim.mask_out_bitfield().into();
            ib.push_ssa(MEM_LOAD, ty.into(), &[var.store.into()], Default::default());
          } else {
            ib.push_ssa(MEM_LOAD, var.ty.into(), &[var.store.into()], Default::default());
          }
        }
        _ => {
          ib.push_ssa(MEM_LOAD, var.ty.into(), &[var.store.into()], Default::default());
        }
      }
    } else if var.ty.is_pointer() {
      ib.push_node(var.store)
    } else {
      match var.ty.base_type() {
        BaseType::Prim(..) => ib.push_node(var.store),
        BaseType::UNRESOLVED => {
          todo!("Handle UNRESOLVED reference")
        }
        BaseType::Complex(..) => {
          todo!("Handle complex reference")
        }
      }
    }
  } else {
    let blame_string = mem.tok.blame(1, 1, "could not find variable", None);
    panic!("{blame_string}",)
  }
}

fn process_add(add: &Add<Token>, ib: &mut IRBuilder) {
  process_expression(&(add.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(add.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(ADD, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_sub(sub: &Sub<Token>, ib: &mut IRBuilder) {
  process_expression(&(sub.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sub.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SUB, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_mul(mul: &Mul<Token>, ib: &mut IRBuilder) {
  process_expression(&(mul.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(mul.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(MUL, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_div(div: &Div<Token>, ib: &mut IRBuilder) {
  process_expression(&(div.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(div.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(DIV, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_pow(pow: &Pow<Token>, ib: &mut IRBuilder) {
  process_expression(&(pow.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(pow.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(POW, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_sl(sl: &BIT_SL<Token>, ib: &mut IRBuilder) {
  process_expression(&(sl.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sl.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SHL, Inherit, &[StackOp, StackOp], Default::default());
}

fn process_sr(sr: &BIT_SR<Token>, ib: &mut IRBuilder) {
  process_expression(&(sr.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sr.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SHR, Inherit, &[StackOp, StackOp], Default::default());
}

fn resolve_variable(mem: &RawMember<Token>, ib: &mut IRBuilder) -> Option<ExternalVData> {
  let base = &mem.members[0];

  let var_name = base.id.intern();

  if let Some(var_data) = ib.get_variable(IdMember(var_name)) {
    match var_data.ty.base_type() {
      BaseType::Prim(..) => {
        return Some(var_data);
      }
      BaseType::UNRESOLVED => {
        if mem.members.len() == 1 {
          Some(var_data)
        } else {
          let sub_name = mem.members[1].id.intern();
          if let Some(mut sub_var) = ib.get_variable_member(&var_data, IdMember(sub_name)) {
            ib.push_ssa(PTR_MEM_CALC, sub_var.ty.as_pointer().into(), &[SMO::IROp(var_data.store)], sub_var.id);
            sub_var.store = ib.pop_stack().unwrap();

            Some(sub_var)
          } else {
            None
          }
        }
      }
      BaseType::Complex(ir_type) => match &ir_type {
        ComplexType::Struct(..) => {
          if mem.members.len() == 1 {
            Some(var_data)
          } else {
            let sub_name = mem.members[1].id.intern();
            if let Some(mut sub_var) = ib.get_variable_member(&var_data, IdMember(sub_name)) {
              ib.push_ssa(PTR_MEM_CALC, sub_var.ty.as_pointer().into(), &[SMO::IROp(var_data.store)], sub_var.id);
              sub_var.store = ib.pop_stack().unwrap();

              Some(sub_var)
            } else {
              None
            }
          }
        }
        _ => unreachable!(),
      },
    }
  } else {
    None
  }
}

fn process_const_number(num: &RawNum<Token>, ib: &mut IRBuilder) {
  let string_val = num.tok.to_string();

  ib.push_const(if string_val.contains(".") {
    ConstVal::new(PrimitiveType::Float | PrimitiveType::b64, num.val)
  } else {
    ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b64, string_val.parse::<u64>().unwrap())
  });
}

enum InitResult {
  /// A set of MEM_STORE or MPTR_CAL nodes that need to have their base ptr
  /// variable set.
  StructInit(IRGraphId, IRGraphId),
  None,
}

fn process_call(call: &RawCall<Token>, ib: &mut IRBuilder) {
  let call_name = call.id.id.intern();

  let call_var = if let Some(complex) = ib.get_type(call_name) {
    let mut args = Vec::new();
    for arg in &call.args {
      process_expression(&arg.expr, ib);
      ib.push_ssa(CALL_ARG, Inherit, &[StackOp], Default::default());
      args.push(ib.pop_stack().unwrap());
    }

    match complex {
      ComplexType::Routine(routine) => {
        dbg!(routine);
        if routine.body.resolved {
          ib.push_variable(IdMember(routine.name), Type::from(complex));

          panic!("call resolved routine")
        } else {
          let mut new_params = Vec::new();
          let mut new_returns = routine.returns.clone();

          let mut new_name: String = routine.name.to_string();
          let mut generic_args = Vec::new();

          // Find matching resolved types.
          for var in &routine.variables.entries {
            let param_index = var.parameter_index.usize();
            // Find matching variable args
            if var.parameter_index.is_valid() {
              let MemberName::IdMember(name) = var.name else { unreachable!() };

              if let Some(arg_id) = args.get(param_index) {
                if var.ty.is_unresolved() {
                  if let Some(node_ty) = ib.get_node_type(*arg_id) {
                    generic_args.push(node_ty.to_string());
                    new_params.push((name, param_index, node_ty))
                  }
                  // Match
                } else if let Some(node_ty) = ib.get_node_type(*arg_id) {
                  if node_ty != var.ty {
                    panic!("Mismatched types in call expression! arg({}) != param({})", node_ty, var.ty);
                  } else {
                    new_params.push((name, param_index, var.ty))
                  }
                } else {
                  panic!("could not resolve type.")
                }
              }
            }
          }

          let name = (new_name + "<" + &generic_args.join(", ") + ">").intern();
          if let Some(ty) = ib.type_scopes.get(0, name) {
            ib.push_variable(IdMember(name), Type::from(ib.type_scopes.get(0, name).unwrap()))
          } else {
            let new_routine = RoutineType {
              name,
              parameters: new_params,
              body: Default::default(),
              variables: Default::default(),
              returns: new_returns,
              ast: routine.ast.clone(),
            };

            ib.type_scopes.set(0, name, ComplexType::Routine(new_routine));

            process_routine(name, 0, &ib.type_scopes);

            ib.push_variable(IdMember(name), Type::from(ib.type_scopes.get(0, name).unwrap()))
          }
        }
      }
      _ => panic!("Invalid type for calling"),
    }
  } else {
    panic!("Could not find routine {}", call_name.to_str().as_str())
  };

  ib.push_ssa(CALL, call_var.ty.into(), &[StackOp], call_var.id);
  ib.pop_stack();
}

fn process_struct_instantiation(struct_decl: &RawStructDeclaration<Token>, ib: &mut IRBuilder) {
  let struct_type_name = struct_decl.name.id.intern();

  if let Some(s_type @ ComplexType::Struct(struct_definition)) = ib.get_type(struct_type_name) {
    let s_type: Type = s_type.into();

    let struct_var = ib.push_variable(IdMember(struct_type_name), s_type.into());

    struct StructEntry {
      ty:    Type,
      value: IRGraphId,
      name:  IString,
    }

    let mut value_maps = BTreeMap::<u64, StructEntry>::new();

    //var_ctx.set_variable(struct_type_name, s_type, struct_id,
    // struct_id.graph_id());

    for init_expression in &struct_decl.inits {
      let member_name = init_expression.name.id.intern();

      if let Some(box ComplexType::StructMember(member)) = struct_definition.members.iter().find(|i| i.name() == member_name) {
        process_expression(&init_expression.expression.expr, ib);

        if Some(member.ty) != ib.get_top_type() {
          if member.ty.is_primitive() {
            //
          } else {
            todo!("Handle non-primitive struct member");
          }
        }

        if member.ty.is_primitive() && member.ty.as_prim().unwrap().bitfield_size() > 0 {
          let ty: &PrimitiveType = member.ty.as_prim().unwrap();
          let bit_size = ty.bit_size();
          let bit_offset = ty.bitfield_offset();
          let bit_field_size = ty.bitfield_size();
          let bit_mask = ((1 << bit_size) - 1) << bit_offset;

          let bf_type: Type = (PrimitiveType::Unsigned | PrimitiveType::new_bit_size(bit_field_size)).into();

          // bitfield initializers must be combined into one value and then
          // submitted at the end of this section. So we make or retrieve the
          // temporary variable for this field.

          ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_offset as u32));
          ib.push_ssa(SHL, bf_type.into(), &[StackOp, StackOp], Default::default());

          ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_mask as u32));
          ib.push_ssa(AND, bf_type.into(), &[StackOp, StackOp], Default::default());

          match value_maps.entry(member.offset) {
            btree_map::Entry::Occupied(mut entry) => {
              let val = entry.get_mut().value;
              ib.push_ssa(OR, bf_type.into(), &[val.into(), StackOp], Default::default());
              entry.get_mut().value = ib.pop_stack().unwrap();
            }
            btree_map::Entry::Vacant(val) => {
              val.insert(StructEntry { ty: bf_type, value: ib.pop_stack().unwrap(), name: member_name });
            }
          }

          // Depending the scope invariants, may need to throw error if value is
          // truncated.

          // Mask out the expression.
        } else {
          value_maps.insert(member.offset, StructEntry { ty: member.ty, value: ib.pop_stack().unwrap(), name: member_name });
        };

        // Calculate the offset to the member within the struct
      } else {
        panic!("Member name not found.");
      }
    }

    ib.push_ssa(ADDR, s_type.as_pointer().into(), &[struct_var.store.into()], struct_var.id);
    let ptr_id = ib.pop_stack().unwrap();

    for (offset, StructEntry { ty, value, name }) in value_maps {
      if let Some(var) = ib.get_variable_member(&struct_var, IdMember(name)) {
        ib.push_ssa(PTR_MEM_CALC, ty.as_pointer().into(), &[ptr_id.into()], var.id);
        ib.push_ssa(MEM_STORE, Inherit, &[StackOp, value.into()], var.id);
        ib.pop_stack();
      } else {
        todo!("TBD");
      }
    }
  } else {
    panic!("Could not find struct definition for {struct_type_name:?}")
  }
}

fn process_block(ast_block: &RawBlock<Token>, ib: &mut IRBuilder) {
  ib.push_var_scope();

  let len = ast_block.statements.len();

  for (i, stmt) in ast_block.statements.iter().enumerate() {
    process_statement(stmt, ib, i == len - 1);
  }

  let returns = match &ast_block.exit {
    block_expression_group_1_Value::None => false,
    block_expression_group_1_Value::RawBreak(brk) => {
      dbg!(brk);
      if !brk.label.id.is_empty() {
        let name = brk.label.id.to_token();
        if let Some((.., end_block)) = ib.loop_stack.iter().find(|(_, head, _)| ib.body.blocks[*head].name == name) {
          ib.set_successor(*end_block, SuccessorMode::Default);
        } else {
          panic!("Could not find successor block!");
        }
      } else {
        if let Some((.., end_block)) = ib.loop_stack.last() {
          ib.set_successor(*end_block, SuccessorMode::Default);
        }
      }
      true
    }
    d => {
      todo!("process block exit: {d:#?}");
      true
    }
  };

  ib.pop_var_scope();
}

fn process_statement(stmt: &statement_Value<Token>, ib: &mut IRBuilder, last_value: bool) {
  match stmt {
    statement_Value::RawAssignment(assign) => process_assign_statement(assign, ib),
    statement_Value::Expression(expr) => process_expression_statement(expr, ib, last_value),
    statement_Value::RawLoop(loop_) => process_loop(loop_, ib),
    statement_Value::RawMatch(match_) => process_match(match_, ib),

    d => todo!("process statement: {d:#?}"),
  }
}

fn process_match(match_: &RawMatch<Token>, ib: &mut IRBuilder<'_, '_, '_>) {
  process_expression(&match_.expression.expr, ib);

  let expr_id = ib.pop_stack().unwrap();

  let end = ib.create_block();

  for clause in &match_.clauses {
    match clause {
      match_clause_Value::RawDefaultClause(def) => {
        process_match_scope(&def.scope, ib);
      }
      match_clause_Value::RawMatchClause(mtch) => {
        match &mtch.expr {
          match_expression_Value::RawExprMatch(expr_match) => {
            process_expression(&expr_match.expr.expr, ib);
            let op = match expr_match.op.as_str() {
              "<" => LS,
              ">" => GR,
              "<=" => LE,
              ">=" => GE,
              "==" => EQ,
              "!=" => NE,
              _ => unreachable!(),
            };

            ib.push_ssa(op, Inherit, &[expr_id.into(), StackOp], Default::default());
            ib.pop_stack();
          }
          _ => todo!(),
        }

        let (succeed, failed) = ib.create_branch();

        ib.set_active(succeed);

        process_match_scope(&mtch.scope, ib);

        ib.set_successor(end, SuccessorMode::Default);
        ib.set_active(failed);
      }
      _ => {}
    }
  }

  ib.set_successor(end, SuccessorMode::Default);
  ib.set_active(end);
}

fn process_match_scope(scope: &match_scope_Value<Token>, ib: &mut IRBuilder<'_, '_, '_>) {
  match &scope {
    match_scope_Value::Expression(expr) => process_expression(&expr.expr, ib),
    match_scope_Value::RawBlock(block) => process_block(block, ib),
    _ => unreachable!(),
  }
}

fn process_loop(loop_: &RawLoop<Token>, ib: &mut IRBuilder<'_, '_, '_>) {
  let (loop_head, loop_exit) = ib.push_loop(loop_.label.id.intern());

  match &loop_.scope {
    loop_statement_group_1_Value::RawBlock(block) => process_block(block, ib),
    loop_statement_group_1_Value::RawMatch(match_) => process_match(match_, ib),
    _ => unreachable!(),
  }

  ib.set_successor(loop_head, SuccessorMode::Default);
  ib.pop_loop();
}

fn process_expression_statement(expr: &std::sync::Arc<Expression<Token>>, ib: &mut IRBuilder<'_, '_, '_>, last_value: bool) {
  process_expression(&expr.expr, ib);
  if !last_value {
    ib.pop_stack();
  }
}

fn process_assign_statement(assign: &std::sync::Arc<crate::parser::script_parser::RawAssignment<Token>>, ib: &mut IRBuilder<'_, '_, '_>) {
  // Process assignments.
  for expression in &assign.expressions {
    process_expression(&expression.expr, ib)
  }

  let mut expression_data = (0..assign.expressions.len()).into_iter().map(|_| ib.pop_stack().unwrap()).collect::<ArrayVec<6, _>>();
  expression_data.reverse();

  // Process assignment targets.
  for (variable, expr_id) in assign.vars.iter().zip(&mut expression_data.iter().cloned()) {
    let expr_ty = ib.get_node_type(expr_id).unwrap();
    match variable {
      assignment_var_Value::RawArrayAccess(..) => {
        todo!("Array access")
      }
      assignment_var_Value::RawAssignmentVariable(var_assign) => {
        if let Some(var_data) = resolve_variable(&var_assign.var, ib) {
          if var_data.is_member_pointer {
            match var_data.ty.base_type() {
              BaseType::Prim(prim) => {
                if prim.bitfield_size() > 0 {
                  let ty: &PrimitiveType = &prim;
                  let bit_size = ty.bit_size();
                  let bit_offset = ty.bitfield_offset();
                  let bit_field_size = ty.bitfield_size();
                  let bit_mask = ((1 << bit_size) - 1) << bit_offset;
                  let bf_type: Type = (PrimitiveType::Unsigned | PrimitiveType::new_bit_size(bit_field_size)).into();

                  // bitfield initializers must be combined into one value and then
                  // submitted at the end of this section. So we make or retrieve the
                  // temporary variable for this field.

                  // Offset variable
                  ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_offset as u32));
                  ib.push_ssa(SHL, var_data.ty.into(), &[StackOp, StackOp], Default::default());

                  // Mask out unwanted bits
                  ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_mask as u32));
                  ib.push_ssa(AND, Inherit, &[StackOp, StackOp], Default::default());

                  // Load the base value from the structure and mask out target bitfield
                  ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, !bit_mask as u32));
                  ib.push_ssa(MEM_LOAD, var_data.ty.into(), &[var_data.store.into()], var_data.id);
                  ib.push_ssa(AND, var_data.ty.into(), &[StackOp, StackOp], var_data.id);

                  // Combine the original value and the new bitfield value.
                  ib.push_ssa(OR, var_data.ty.into(), &[StackOp, StackOp], var_data.id);

                  // Store value back in the structure.
                  ib.push_ssa(MEM_STORE, var_data.ty.into(), &[var_data.store.into(), StackOp], var_data.id);
                  ib.pop_stack();
                } else {
                  dbg!(&ib);
                  ib.push_ssa(MEM_STORE, var_data.ty.into(), &[var_data.store.into(), expr_id.into()], var_data.id);
                  ib.pop_stack();
                }
              }
              BaseType::Complex(ComplexType::StructMember(mem)) => {
                ib.push_ssa(MEM_STORE, var_data.ty.into(), &[var_data.store.into(), expr_id.into()], var_data.id);
                ib.pop_stack();
              }
              BaseType::UNRESOLVED | _ => {
                ib.push_ssa(MEM_STORE, var_data.ty.into(), &[var_data.store.into(), expr_id.into()], var_data.id);
                ib.pop_stack();
              }
              ty => unreachable!("{ty:#?}"),
            }
          } else {
            ib.push_ssa(STORE, var_data.ty.into(), &[var_data.decl.into(), expr_id.into()], var_data.id);
            ib.pop_stack();
          }
        } else if var_assign.var.members.len() == 1 {
          // Create and assign a new variable based on the expression.
          let var_name = var_assign.var.members[0].id.intern();

          match expr_ty.base_type() {
            BaseType::Complex(ty) => match &ty {
              ComplexType::Struct(..) => ib.rename_var(expr_id, IdMember(var_name)),
              _ => unreachable!(),
            },
            _ => {
              let var = ib.push_variable(IdMember(var_name), expr_ty);
              ib.pop_stack();

              ib.push_ssa(STORE, var.ty.into(), &[var.decl.into(), expr_id.into()], var.id);
              ib.pop_stack();
            }
          }
        } else {
          let blame_string = var_assign.tok.blame(1, 1, "could not find variable", None);
          panic!("{blame_string}",)
        }
      }
      assignment_var_Value::RawAssignmentDeclaration(var_decl) => {
        let var_name = var_decl.var.id.intern();
        let expected_ty = get_type_from_sm(&var_decl.ty, ib).unwrap();

        match expr_ty.base_type() {
          BaseType::Complex(ty) => match &ty {
            ComplexType::Struct(..) => ib.rename_var(expr_id, IdMember(var_name)),
            _ => unreachable!(),
          },
          _ => {
            if expected_ty != expr_ty {
              match (expected_ty.base_type(), expr_ty.base_type()) {
                (BaseType::Prim(prim_ty), BaseType::Prim(prim_expr_ty)) => {
                  let var = ib.push_variable(IdMember(var_name), expr_ty);
                  ib.pop_stack();

                  ib.push_ssa(STORE, var.ty.into(), &[var.decl.into(), expr_id.into()], var.id);
                  ib.pop_stack();
                }
                _ => panic!("Miss matched types ty:{expected_ty:?} expr_ty:{expr_ty:?}"),
              }
            } else {
              let var = ib.push_variable(IdMember(var_name), expr_ty);
              ib.pop_stack();

              ib.push_ssa(STORE, var.ty.into(), &[var.decl.into(), expr_id.into()], var.id);
              ib.pop_stack();
            }
          }
        }
      }
      _ => unreachable!(),
    }
  }

  // Match assignments to targets.
}

pub fn get_type_from_sm(ir_type: &type_Value<Token>, ib: &mut IRBuilder) -> Option<crate::types::Type> {
  get_type(ir_type, ib.type_context_index, ib.type_scopes)
}

pub fn get_type(ir_type: &type_Value<Token>, scope_index: usize, type_context: &TypeContext) -> Option<crate::types::Type> {
  match ir_type {
    type_Value::Type_Flag(_) => Some((PrimitiveType::Flag | PrimitiveType::b1).into()),
    type_Value::Type_u8(_) => Some((PrimitiveType::Unsigned | PrimitiveType::b8).into()),
    type_Value::Type_u16(_) => Some((PrimitiveType::Unsigned | PrimitiveType::b16).into()),
    type_Value::Type_u32(_) => Some((PrimitiveType::Unsigned | PrimitiveType::b32).into()),
    type_Value::Type_u64(_) => Some((PrimitiveType::Unsigned | PrimitiveType::b64).into()),
    type_Value::Type_i8(_) => Some((PrimitiveType::Signed | PrimitiveType::b8).into()),
    type_Value::Type_i16(_) => Some((PrimitiveType::Signed | PrimitiveType::b16).into()),
    type_Value::Type_i32(_) => Some((PrimitiveType::Signed | PrimitiveType::b32).into()),
    type_Value::Type_i64(_) => Some((PrimitiveType::Signed | PrimitiveType::b64).into()),
    type_Value::ReferenceType(name) => {
      if let Some(ty) = type_context.get(scope_index, name.name.id.intern()) {
        let t: Type = ty.into();
        Some(t.as_pointer())
      } else {
        Some(Type::UNRESOLVED)
      }
    }
    type_Value::NamedType(name) => {
      if let Some(ty) = type_context.get(scope_index, name.name.id.intern()) {
        let t: Type = ty.into();
        Some(t)
      } else {
        Some(Type::UNRESOLVED)
      }
    }
    _t => None,
  }
}

#[cfg(test)]
mod test;
