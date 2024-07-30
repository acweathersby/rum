use super::{
  ir_graph::IRGraphId,
  ir_state_machine::{ExternalVData, StateMachine},
};
use crate::{
  compiler::script_parser::{
    assignment_var_Value,
    bitfield_element_Value,
    block_expression_group_1_Value,
    expression_Value,
    property_Value,
    raw_module_Value,
    statement_Value,
    type_Value,
    Expression,
    RawBlock,
    RawCall,
    RawFunction,
    RawNum,
    RawStructDeclaration,
  },
  ir::{
    ir_graph::{IRGraphNode, IROp},
    ir_state_machine::{SuccessorMode, SMO, SMT},
  },
  types::{BaseType, ComplexType, ConstVal, PrimitiveType, ProcedureBody, ProcedureType, StructMemberType, StructType, Type, TypeScopes},
  IString,
};
pub use radlr_rust_runtime::types::Token;
use rum_container::get_aligned_value;
use rum_istring::CachedString;
use std::collections::{hash_map, BTreeMap, HashMap};

pub fn process_types(
  module: &Vec<raw_module_Value<Token>>,
  type_scope_index: usize,
  type_scope: &mut TypeScopes,
) -> Vec<std::rc::Rc<RawFunction<Token>>> {
  let mut functions = Vec::new();

  for mod_member in module {
    match mod_member {
      raw_module_Value::RawFunction(funct) => {
        functions.push(funct.clone());
      }
      raw_module_Value::RawUnion(union) => {
        dbg!(union);
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

            property_Value::RawBitCompositeProp(bit_composition) => {
              println!("Bit composite property type not supported yet");
            }
            _ => {}
          }
        }

        let s = StructType {
          name,
          members,
          alignment: min_alignment,
          size: get_aligned_value(offset, min_alignment as u64),
        };

        type_scope.set(type_scope_index, name, crate::types::ComplexType::Struct(s));
      }
      _ => {}
    }
  }

  functions
}

pub fn build_module(module: &Vec<raw_module_Value<Token>>, type_scope_index: usize, type_scope: &mut TypeScopes) -> () {
  let mut functions = process_types(module, type_scope_index, type_scope);
  //let mut types = Vec::new();

  for function in &functions {
    // Gather function type information.

    let name = function.name.id.intern();
    let params = &function.ty.params;
    let return_ty = &function.ty.return_type;

    let body = process_function(function, type_scope_index, &type_scope);

    let ty = ProcedureType {
      name,
      body,
      parameters: Default::default(),
      returns: Default::default(),
    };

    type_scope.set(type_scope_index, name, crate::types::ComplexType::Procedure(ty));
  }

  ()
}

fn process_function(function: &RawFunction<Token>, type_ctx_index: usize, type_context: &TypeScopes) -> ProcedureBody {
  // Ensure the return type is present in our type context.
  println!("TODO: Ensure the return type is present in our type context");

  let mut pb = ProcedureBody { graph: Default::default(), blocks: Default::default() };

  let mut state_machine = StateMachine::new(&mut pb, type_ctx_index, type_context);

  process_expression(&function.expression, &mut state_machine);

  dbg!(state_machine);

  pb
}

fn process_expression(expr: &Expression<Token>, sm: &mut StateMachine) {
  match &expr.expr {
    expression_Value::RawCall(call) => process_call(call, sm),
    expression_Value::RawNum(num) => process_const_number(num, sm),
    expression_Value::AddressOf(addr) => {
      if let Some(var) = sm.get_variable(addr.id.id.intern()) {
        sm.push_ssa(IROp::ADDR, var.ty.as_pointer().into(), &[SMO::IROp(var.store)], var.decl.graph_id())
      } else {
        panic!("Variable not found")
      }
    }
    expression_Value::RawStructDeclaration(struct_decl) => process_struct_instantiation(struct_decl, sm),
    expression_Value::RawMember(mem) => {
      if let Some(var) = resolve_variable(mem, sm) {
        if var.is_member_pointer {
          // Loading the variable into a register creates a temporary variable
          sm.push_ssa(IROp::MEM_LOAD, var.ty.into(), &[SMO::IROp(var.store)], var.decl.graph_id());
        } else {
          unreachable!("Could not locate variable")
        }
      } else {
        let blame_string = mem.tok.blame(1, 1, "could not find variable", None);
        panic!("{blame_string}",)
      }
    }
    expression_Value::RawBlock(ast_block) => {
      sm.push_block(SuccessorMode::Default);
      process_block(ast_block, sm);
      sm.swap_block(SuccessorMode::Default);
      println!("Return the graph id value of a raw block");
    }
    d => todo!("process expression: {d:#?}"),
  }
}

fn resolve_variable(mem: &std::rc::Rc<crate::compiler::script_parser::RawMember<Token>>, sm: &mut StateMachine) -> Option<ExternalVData> {
  let base = &mem.members[0];

  let var_name = base.id.intern();

  if let Some(var_data) = sm.get_variable(var_name) {
    match var_data.ty.base_type() {
      BaseType::Prim(prim) => {
        return Some(var_data);
      }
      BaseType::Complex(ir_type) => match &ir_type {
        ComplexType::Struct(strct) => {
          let sub_name = mem.members[1].id.intern();
          if let Some(mut sub_var) = sm.get_variable_member(&var_data, sub_name) {
            sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(sub_var.offset as u32));
            sm.push_ssa(
              IROp::PTR_MEM_CALC,
              sub_var.ty.as_pointer().into(),
              &[SMO::IROp(var_data.store), SMO::StackOp],
              sub_var.decl.graph_id(),
            );
            sub_var.store = sm.pop_stack().unwrap();
            Some(sub_var)
          } else {
            None
          }
        }
        _ => unreachable!(),
      },
    }
  } else {
    None
  }
}

fn process_const_number(num: &RawNum<Token>, sm: &mut StateMachine) {
  let string_val = num.tok.to_string();

  sm.push_const(if string_val.contains(".") {
    ConstVal::new(PrimitiveType::Float | PrimitiveType::b64).store(num.val)
  } else {
    ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b64).store::<u64>(string_val.parse::<u64>().unwrap())
  });
}

pub fn remap_primitive_type(desired_type: PrimitiveType, node_id: IRGraphId, sm: &mut StateMachine) {
  // Add some rules to say whether the type can be coerced or converted into the
  // desired type.

  if node_id.is_invalid() {
    return;
  }

  match &mut sm.graph[node_id.graph_id()] {
    IRGraphNode::Const { id: ssa_id, val } => {
      if val.is_lit() {
        *val = val.convert(desired_type);
      } else {
        *val = ConstVal::new(desired_type);
      }
    }
    IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } => {
      if out_ty.is_primitive() {
        *out_ty = desired_type.into();
        let operands = *operands;
        remap_primitive_type(desired_type, operands[0], sm);
        remap_primitive_type(desired_type, operands[1], sm);
      } else if *out_ty != desired_type.into() {
        panic!("Can't convert type \n ",);
      }
    }
    _ => {}
  }
}

enum InitResult {
  /// A set of MEM_STORE or MPTR_CAL nodes that need to have their base ptr
  /// variable set.
  StructInit(IRGraphId, IRGraphId),
  None,
}

fn process_call(call: &RawCall<Token>, sm: &mut StateMachine) {
  let call_name = &call.id.id;

  if call_name.contains("sys_") {
    for _ in call.args.iter().map(|arg| process_expression(&arg, sm)).collect::<Vec<_>>() {
      sm.push_ssa(IROp::CALL_ARG, SMT::Inherit, &[SMO::StackOp], usize::MAX);
      println!("TODO: match type_info with arg type");
    }
  }

  sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b64).store(1 as u64));
  sm.push_ssa(IROp::CALL, SMT::None, &[SMO::StackOp], usize::MAX);
  sm.pop_stack();
}

fn process_struct_instantiation(struct_decl: &RawStructDeclaration<Token>, sm: &mut StateMachine) {
  let struct_type_name = struct_decl.name.id.intern();

  if let Some(s_type @ ComplexType::Struct(struct_definition)) = sm.get_type(struct_type_name) {
    let s_type: Type = s_type.into();

    let struct_var = sm.push_variable(struct_type_name, s_type.into());

    struct StructEntry {
      ty:    Type,
      value: IRGraphId,
      name:  IString,
    }

    let mut value_maps = HashMap::<u64, StructEntry>::new();

    //var_ctx.set_variable(struct_type_name, s_type, struct_id,
    // struct_id.graph_id());

    for init_expression in &struct_decl.inits {
      let member_name = init_expression.name.id.intern();

      if let Some(member) = struct_definition.members.iter().find(|i| i.name == member_name) {
        process_expression(&init_expression.expression, sm);

        if Some(member.ty) != sm.get_top_type() {
          if member.ty.is_primitive() {
            remap_primitive_type(*member.ty.as_prim().unwrap(), sm.get_top_id().unwrap(), sm)
          } else {
            //let node = &block.ctx().graph[expr_id.graph_id()];
            //panic!("handle type coercions of member:{:?} expr:{:?}",
            // member.ty, node.ty());
          }
        }

        if member.ty.is_primitive() && member.ty.as_prim().unwrap().bitfield_size() > 0 {
          let ty: &PrimitiveType = member.ty.as_prim().unwrap();
          let bit_size = ty.bit_size();
          let bit_offset = ty.bitfield_offset();
          let bit_field_size = ty.bitfield_size();
          let bit_mask = ((1 << bit_size) - 1) << bit_offset;

          println!("{bit_mask:032b}");

          let bf_type: Type = (PrimitiveType::Unsigned | PrimitiveType::new_bit_size(bit_field_size)).into();

          // bitfield initializers must be combined into one value and then
          // submitted at the end of this section. So we make or retrieve the
          // temporary variable for this field.

          remap_primitive_type(*bf_type.as_prim().unwrap(), sm.get_top_id().unwrap(), sm);

          sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(bit_offset as u32));
          sm.push_ssa(IROp::SHIFT_L, bf_type.into(), &[SMO::StackOp, SMO::StackOp], usize::MAX);

          sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(bit_mask as u32));
          sm.push_ssa(IROp::AND, bf_type.into(), &[SMO::StackOp, SMO::StackOp], usize::MAX);

          match value_maps.entry(member.offset) {
            hash_map::Entry::Occupied(mut entry) => {
              let val = entry.get_mut().value;
              sm.push_ssa(IROp::OR, bf_type.into(), &[val.into(), SMO::StackOp], usize::MAX);
              entry.get_mut().value = sm.pop_stack().unwrap();
            }
            hash_map::Entry::Vacant(val) => {
              val.insert(StructEntry { ty: bf_type, value: sm.pop_stack().unwrap(), name: member_name });
            }
          }

          // Depending the scope invariants, may need to throw error if value is
          // truncated.

          // Mask out the expression.
        } else {
          value_maps.insert(member.offset, StructEntry { ty: member.ty, value: sm.pop_stack().unwrap(), name: member_name });
        };

        // Calculate the offset to the member within the struct
      } else {
        panic!("Member name not found.");
      }
    }

    sm.push_ssa(IROp::ADDR, s_type.as_pointer().into(), &[struct_var.store.into()], struct_var.var_index);
    let ptr_id = sm.pop_stack().unwrap();

    for (offset, StructEntry { ty, value, name }) in value_maps {
      if let Some(var) = sm.get_variable_member(&struct_var, name) {
        let member_ptr = ty.as_pointer();

        sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(offset as u32));

        sm.push_ssa(IROp::PTR_MEM_CALC, member_ptr.into(), &[ptr_id.into(), SMO::StackOp], var.var_index);

        sm.push_ssa(IROp::MEM_STORE, SMT::Inherit, &[SMO::StackOp, value.into()], var.var_index);

        sm.pop_stack();
      } else {
        panic!("AAAAA");
      }
    }
  } else {
    panic!("Could not find struct definition for {struct_type_name:?}")
  }
}

fn process_block(ast_block: &RawBlock<Token>, sm: &mut StateMachine) {
  let len = ast_block.statements.len();
  for (i, stmt) in ast_block.statements.iter().enumerate() {
    process_statement(stmt, sm, i == len - 1);
  }

  match &ast_block.exit {
    block_expression_group_1_Value::None => {
      println!("Resolve return expression of a block exit");
    }
    d => todo!("process block exit: {d:#?}"),
  }
}

fn process_statement(stmt: &statement_Value<Token>, sm: &mut StateMachine, last_value: bool) {
  match stmt {
    statement_Value::RawAssignment(assign) => {
      // Process assignments.
      for expression in &assign.expressions {
        process_expression(&expression, sm)
      }

      // Process assignment targets.
      for variable in assign.vars.iter() {
        match variable {
          assignment_var_Value::RawArrayAccess(array) => {
            todo!("Array access")
          }
          assignment_var_Value::RawAssignmentVariable(var_assign) => {
            let expr_ty = sm.get_top_type().unwrap();

            if let Some(var_data) = resolve_variable(&var_assign.var, sm) {
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

                      remap_primitive_type(*bf_type.as_prim().unwrap(), sm.get_top_id().unwrap(), sm);

                      // Offset variable
                      sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(bit_offset as u32));
                      sm.push_ssa(IROp::SHIFT_L, var_data.ty.into(), &[SMO::StackOp, SMO::StackOp], usize::MAX);

                      // Mask out unwanted bits
                      sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(bit_mask as u32));
                      sm.push_ssa(IROp::AND, SMT::Inherit, &[SMO::StackOp, SMO::StackOp], usize::MAX);

                      // Load the base value from the structure and mask out target bitfield
                      sm.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32).store(!bit_mask as u32));
                      sm.push_ssa(IROp::MEM_LOAD, var_data.ty.into(), &[var_data.store.into()], var_data.var_index);
                      sm.push_ssa(IROp::AND, var_data.ty.into(), &[SMO::StackOp, SMO::StackOp], var_data.var_index);

                      // Combine the original value and the new bitfield value.
                      sm.push_ssa(IROp::OR, var_data.ty.into(), &[SMO::StackOp, SMO::StackOp], var_data.var_index);

                      // Store value back in the structure.
                      sm.push_ssa(IROp::MEM_STORE, var_data.ty.into(), &[var_data.store.into(), SMO::StackOp], var_data.var_index);
                      sm.pop_stack();
                    } else {
                      remap_primitive_type(*var_data.ty.as_prim().unwrap(), sm.get_top_id().unwrap(), sm);
                      sm.push_ssa(IROp::MEM_STORE, var_data.ty.into(), &[var_data.store.into(), SMO::StackOp], var_data.var_index);
                      sm.pop_stack();
                    }
                  }
                  _ => unreachable!(),
                }
              } else {
                sm.push_ssa(IROp::STORE, var_data.ty.into(), &[var_data.decl.into(), SMO::StackOp], var_data.var_index);
              }
            } else if var_assign.var.members.len() == 1 {
              // Create and assign a new variable based on the expression.
              let var_name = var_assign.var.members[0].id.intern();

              match expr_ty.base_type() {
                BaseType::Complex(ty) => match &ty {
                  ComplexType::Struct(strc) => {
                    let node = sm.pop_stack().unwrap();
                    sm.rename_var(node, var_name);
                  }
                  _ => unreachable!(),
                },
                _ => {
                  let var = sm.push_variable(var_name, expr_ty);

                  sm.push_ssa(IROp::STORE, var.ty.into(), &[var.decl.into(), SMO::StackOp], var.var_index);
                }
              }
            } else {
              let blame_string = var_assign.tok.blame(1, 1, "could not find variable", None);
              panic!("{blame_string}",)
            }
          }
          assignment_var_Value::RawAssignmentDeclaration(var_decl) => {
            let var_name = var_decl.var.id.intern();
            let expected_ty = get_type_from_sm(&var_decl.ty, sm).unwrap();

            let expr_ty = sm.get_top_type().unwrap();

            match expr_ty.base_type() {
              BaseType::Complex(ty) => match &ty {
                ComplexType::Struct(strc) => {
                  todo!("Rename struct variable");
                }
                _ => unreachable!(),
              },
              _ => {
                if expected_ty != expr_ty {
                  match (expected_ty.base_type(), expr_ty.base_type()) {
                    (BaseType::Prim(prim_ty), BaseType::Prim(prim_expr_ty)) => {
                      remap_primitive_type(prim_ty, sm.get_top_id().unwrap(), sm);
                      let var = sm.push_variable(var_name, expr_ty);
                      sm.push_ssa(IROp::STORE, var.ty.into(), &[var.decl.into(), SMO::StackOp], var.var_index);
                    }
                    _ => panic!("Miss matched types ty:{expected_ty:?} expr_ty:{expr_ty:?}"),
                  }
                } else {
                  let var = sm.push_variable(var_name, expr_ty);
                  sm.push_ssa(IROp::STORE, var.ty.into(), &[var.decl.into(), SMO::StackOp], var.var_index);
                }
              }
            }
          }
          _ => unreachable!(),
        }
      }

      // Match assignments to targets.
    }
    statement_Value::Expression(expr) => {
      process_expression(expr, sm);
      if !last_value {
        sm.pop_stack();
      }
    }
    d => todo!("process statement: {d:#?}"),
  }
}

pub fn get_type_from_sm(ir_type: &type_Value<Token>, sm: &mut StateMachine) -> Option<crate::types::Type> {
  get_type(ir_type, sm.type_context_index, sm.type_scopes)
}

pub fn get_type(ir_type: &type_Value<Token>, scope_index: usize, type_context: &TypeScopes) -> Option<crate::types::Type> {
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
    type_Value::NamedType(name) => {
      let type_name = name.name.id.intern();
      if let Some(ty) = type_context.get(scope_index, name.name.id.intern()) {
        Some(ty.into())
      } else {
        panic!("{type_name:?} not found")
      }
    }
    _t => None,
  }
}
