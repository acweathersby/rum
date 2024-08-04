use super::{
  ir_builder::{IRBuilder, SuccessorMode},
  ir_graph::IRGraphId,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  ir::{
    ir_builder::{SMO, SMT},
    ir_graph::{IROp, TypeVar, VarId},
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
    RawParamType,
    RawRoutine,
    RawStructDeclaration,
    ReferenceType,
    Sub,
    BIT_SL,
    BIT_SR,
  },
  types::*,
};
use core::panic;
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::{btree_map, BTreeMap, HashMap, VecDeque},
  rc::Rc,
  sync::{Arc, Mutex},
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
          members: members.into_iter().map(|s| Rc::new(ComplexType::StructMember(s))).collect(),
          alignment: min_alignment,
          size: get_aligned_value(offset, min_alignment as u64),
        };

        let _ = type_scope.set(type_scope_index, name, Rc::new(ComplexType::Struct(s)).into());
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
  let type_context = TypeContext::new();

  let (params, ret) = match &routine.ty {
    routine_type_Value::RawFunctionType(fn_ty) => (fn_ty.params.as_ref(), Some(fn_ty.return_type.as_ref())),
    routine_type_Value::RawProcedureType(proc_ty) => (proc_ty.params.as_ref(), None),
    _ => unreachable!(),
  };

  let mut parameters = Vec::new();

  for (index, param) in params.params.iter().enumerate() {
    let param_name = param.var.id.intern();
    let param_type = &param.ty;
    if param_type.inferred {
      let type_name = get_type_name(&param_type.ty);

      let ty = match type_context.set(0, type_name, Rc::new(ComplexType::UNRESOLVED { name: type_name }).into()) {
        Err(rt) => {
          debug_assert!(rt.is_unresolved());
          rt
        }
        Ok(ty) => ty,
      };

      let ty = if matches!(&param_type.ty, ReferenceType) { ty.as_pointer() } else { ty.clone() };

      parameters.push((param_name, index, ty));
    } else if let Some(ty) = get_type(&param_type.ty, type_scope_index, type_scope) {
      if ty.is_unresolved() {
        panic!("Could not resolve type!");
      }
      parameters.push((param_name, index, ty));
    } else {
      panic!("Could not resolve type!");
    }
  }

  let returns = match ret {
    Some(param_type) => {
      if param_type.inferred {
        let type_name = get_type_name(&param_type.ty);

        let ty = match type_context.set(0, type_name, Rc::new(ComplexType::UNRESOLVED { name: type_name }).into()) {
          Err(rt) => {
            debug_assert!(rt.is_unresolved());
            rt
          }
          Ok(ty) => ty,
        };

        vec![ty.clone().into()]
      } else if let Some(ty) = get_type(&param_type.ty, type_scope_index, type_scope) {
        if ty.is_unresolved() {
          panic!("Could not resolve type!");
        }
        vec![ty.into()]
      } else {
        panic!("Could not resolve return type")
      }
    }
    _ => vec![],
  };

  let ty = RoutineType { name, body: Default::default(), parameters, returns, ast: routine.clone(), type_context };

  if let Err(existing) = type_scope.set(type_scope_index, name, Rc::new(ComplexType::Routine(Mutex::new(ty))).into()) {
    panic!("Type name {name:?} has already been defined as [{existing}]\n{}", routine.name.tok.blame(1, 1, "", None))
  }
}

fn process_routine(routine_name: IString, type_scope_index: usize, type_scope: &TypeContext) {
  if let Some(ComplexType::Routine(old_rt)) = type_scope.get(type_scope_index, routine_name).and_then(|t| t.as_cplx_ref()) {
    let mut rt = old_rt.lock().unwrap();

    let rt: &mut RoutineType = &mut rt;

    rt.body.resolved = false;

    let RoutineType { name, parameters, returns, body, ast, type_context } = rt;

    let mut ir_builder = IRBuilder::new(body, type_context, type_scope_index, type_scope);

    for (name, index, ty) in parameters.iter() {
      match ty.base_type() {
        BaseType::Complex(ComplexType::UNRESOLVED { name: ty_name }) => {
          if let Some(resolved_ty) = ir_builder.get_type(*ty_name) {
            let r_ty: Type = resolved_ty.clone();

            let r_ty = if ty.is_pointer() { r_ty.as_pointer() } else { r_ty };

            let var = ir_builder.push_para_var(*name, r_ty, VarId::new(*index as u32));
            ir_builder.push_ssa(PARAM_VAL, var.ty_var.into(), &[var.store.into()]);
          } else {
            let var = ir_builder.push_para_var(*name, ty.clone(), VarId::new(*index as u32));
            ir_builder.push_ssa(PARAM_VAL, var.ty_var.into(), &[var.store.into()]);
          }
        }
        _ => {
          let var = ir_builder.push_para_var(*name, ty.clone(), VarId::new(*index as u32));
          ir_builder.push_ssa(PARAM_VAL, var.ty_var.into(), &[var.store.into()]);
        }
      }
    }

    // Seal the internal signature scope and create the main body scope
    ir_builder.push_lexical_scope();

    process_expression(&ast.expression.expr, &mut ir_builder);

    if rt.returns.len() > 0 {
      // todo!("Check that return types match!");
    }

    dbg!(ir_builder);
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

fn process_address_of(ib: &mut IRBuilder<'_, '_>, addr: &std::sync::Arc<crate::parser::script_parser::AddressOf<Token>>) {
  if let Some(var) = ib.get_variable_from_id(IdMember(addr.id.id.intern())) {
    let ptr = ib.get_variable_ptr(&var, "---".intern()).unwrap();
    ib.push_ssa(ADDR, ptr.ty_var.into(), &[SMO::IROp(var.store)])
  } else {
    panic!("Variable not found")
  }
}

fn resolve_variable(mem: &RawMember<Token>, ib: &mut IRBuilder) -> Option<ExternalVData> {
  let base = &mem.members[0];

  let var_name = base.id.intern();

  if let Some(var_data) = ib.get_variable_from_id(IdMember(var_name)) {
    match var_data.ty.base_type() {
      BaseType::Prim(..) => {
        return Some(var_data);
      }
      BaseType::Complex(ir_type) => match &ir_type {
        ComplexType::UNRESOLVED { name } => {
          if mem.members.len() == 1 {
            Some(var_data)
          } else {
            let sub_name = mem.members[1].id.intern();
            if let Some(mut sub_var) = ib.get_variable_member(&var_data, IdMember(sub_name)) {
              let ptr_var = ib.get_variable_ptr(&sub_var, "000".intern()).unwrap();
              let ptr_par_var = ib.get_variable_ptr(&var_data, "000".intern()).unwrap();
              ib.push_ssa(PTR_MEM_CALC, ptr_var.ty_var.into(), &[SMO::IROp(ptr_par_var.store)]);
              dbg!(&ib);
              sub_var.store = ib.pop_stack().unwrap();

              Some(sub_var)
            } else {
              None
            }
          }
        }
        ComplexType::Struct(..) => {
          if mem.members.len() == 1 {
            Some(var_data)
          } else {
            let sub_name = mem.members[1].id.intern();
            if let Some(mut sub_var) = ib.get_variable_member(&var_data, IdMember(sub_name)) {
              let ptr_var = ib.get_variable_ptr(&sub_var, "000".intern()).unwrap();
              let ptr_par_var = ib.get_variable_ptr(&var_data, "000".intern()).unwrap();
              ib.push_ssa(PTR_MEM_CALC, ptr_var.ty_var.into(), &[SMO::IROp(ptr_par_var.store)]);
              dbg!(&ib);
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

fn process_member_load(mem: &std::sync::Arc<RawMember<Token>>, ib: &mut IRBuilder<'_, '_>) {
  if let Some(var) = resolve_variable(mem, ib) {
    if var.is_member_pointer {
      // Loading the variable into a register creates a temporary variable
      dbg!(&var);
      match var.ty.as_cplx().map(|t| t.as_ref()) {
        Some(ComplexType::UNRESOLVED { name }) => {
          ib.push_ssa(MEM_LOAD, var.ty_var.into(), &[var.store.into()]);
        }
        Some(ComplexType::StructMember(mem)) => match mem.ty.base_type() {
          BaseType::Prim(prim) => {
            let ty = prim;
            if prim.bitfield_offset() > 0 {
              ib.push_ssa(MEM_LOAD, TypeVar::from_prim(ty.mask_out_bitfield()).into(), &[var.store.into()]);
            } else {
              ib.push_ssa(MEM_LOAD, TypeVar::from_prim(ty).into(), &[var.store.into()]);
            }
          }
          d => {
            dbg!(d);
            ib.push_ssa(MEM_LOAD, var.ty_var.into(), &[var.store.into()]);
          }
        },
        _ => {
          panic!("Could not resolve type of member {}", var.ty)
        }
      }
    } else if var.ty.is_pointer() {
      ib.push_node(var.store)
    } else {
      match var.ty.base_type() {
        BaseType::Prim(..) => ib.push_node(var.store),
        BaseType::Complex(..) => {
          let tok = &mem.tok;
          println!(
            "ACCESSING A COMPLEX TYPE DIRECTLY IS INVALID IR SEMANTIC. This should be typed as pointer.\n!!!This is an internal compiler error.!!!\n{}",
            tok.blame(1, 1, "", None)
          );
          ib.push_node(var.store);
        }
      }
    }
  } else {
    let blame_string = mem.tok.blame(1, 1, "could not find variable", None);
    panic!("{blame_string} \n",)
  }
}

fn process_add(add: &Add<Token>, ib: &mut IRBuilder) {
  process_expression(&(add.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(add.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(ADD, Inherit, &[StackOp, StackOp]);
}

fn process_sub(sub: &Sub<Token>, ib: &mut IRBuilder) {
  process_expression(&(sub.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sub.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SUB, Inherit, &[StackOp, StackOp]);
}

fn process_mul(mul: &Mul<Token>, ib: &mut IRBuilder) {
  process_expression(&(mul.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(mul.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(MUL, Inherit, &[StackOp, StackOp]);
}

fn process_div(div: &Div<Token>, ib: &mut IRBuilder) {
  process_expression(&(div.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(div.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(DIV, Inherit, &[StackOp, StackOp]);
}

fn process_pow(pow: &Pow<Token>, ib: &mut IRBuilder) {
  process_expression(&(pow.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(pow.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(POW, Inherit, &[StackOp, StackOp]);
}

fn process_sl(sl: &BIT_SL<Token>, ib: &mut IRBuilder) {
  process_expression(&(sl.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sl.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SHL, Inherit, &[StackOp, StackOp]);
}

fn process_sr(sr: &BIT_SR<Token>, ib: &mut IRBuilder) {
  process_expression(&(sr.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sr.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SHR, Inherit, &[StackOp, StackOp]);
}

fn process_const_number(num: &RawNum<Token>, ib: &mut IRBuilder) {
  let string_val = num.tok.to_string();

  ib.push_const(if string_val.contains(".") {
    ConstVal::new(PrimitiveType::Float | PrimitiveType::b64, num.val)
  } else {
    ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b64, string_val.parse::<u64>().unwrap())
  });
}

fn process_call(call: &RawCall<Token>, ib: &mut IRBuilder) {
  let call_name = call.id.id.intern();

  let call_var = if let Some(complex) = ib.get_type(call_name) {
    let mut args = Vec::new();
    for arg in &call.args {
      process_expression(&arg.expr, ib);

      let var_id = ib.get_top_var_id();

      ib.push_ssa(CALL_ARG, Inherit, &[StackOp]);

      args.push(ib.pop_stack().unwrap());
    }

    match complex.as_cplx_ref() {
      Some(ComplexType::Routine(routine)) => {
        let routine = routine.lock().unwrap();
        let routine: &RoutineType = &routine;

        if routine.body.resolved {
          ib.push_variable(routine.name, complex.clone().into());
          todo!("call resolved routine")
        } else {
          let mut _new_name: String = routine.name.to_string();
          let mut generic_args = Vec::new();

          let new_type_context = TypeContext::new();

          let mut param_queue: VecDeque<(usize, usize)> = routine.body.vars.lex_scopes[0].iter().map(|i| (0, *i)).collect();

          // Find matching resolved types.
          while let Some((tries, param_var_index)) = param_queue.pop_front() {
            let param_var = &routine.body.vars.entries[param_var_index];
            let param_pos = param_var.parameter_index;

            if !param_pos.is_valid() {
              continue;
            };

            let param_index = param_pos.usize();

            if let Some(arg_id) = args.get(param_index) {
              let arg_var = ib.get_base_variable_from_node_mut(*arg_id).expect("Could not extract variable data");
              let MemberName::IdMember(param_name) = param_var.name else { unreachable!() };

              let param_ty: Type = { param_var.ty.clone() };

              use BaseType::*;
              use ComplexType::*;

              match (param_ty.base_type(), arg_var.ty.base_type()) {
                (Complex(UNRESOLVED { name: param_type_name }), Complex(UNRESOLVED { name: own_name })) => {
                  let own_name = *own_name;
                  if let Some(ty) = new_type_context.get(0, *param_type_name) {
                    arg_var.ty = ty.clone();
                    let _ = ib.local_ty_ctx.replace(0, own_name, ty.clone());
                  } else if tries == 0 && !param_queue.is_empty() {
                    // Try to match this after other types have been defined.
                    param_queue.push_back((tries + 1, param_var_index));
                  } else {
                    panic!("Could not resolve this type");
                  }
                }
                (_, Complex(UNRESOLVED { name: own_name })) => {
                  let own_name = *own_name;
                  let ty = param_ty;
                  arg_var.ty = ty.clone();
                  let _ = ib.local_ty_ctx.replace(0, own_name, ty.clone());
                }
                (Complex(UNRESOLVED { name }), _) => {
                  generic_args.push(name.to_string() + ":" + &arg_var.ty.to_string());
                  let _ = new_type_context.set(0, *name, arg_var.ty.clone());
                }
                _ => {
                  if arg_var.ty != param_ty {
                    println!("Type missmatch {param_index} arg({}) != param({})", param_ty, arg_var.ty)
                  }
                }
              }
            } else {
              panic!("No matching argument for parameter [{param_index}] in function ");
            };
          }

          let name = (_new_name + "<" + &generic_args.join(", ") + ">").intern();

          let mut _new_returns = routine
            .returns
            .iter()
            .map(|r| match r.base_type() {
              BaseType::Complex(cplx) => match cplx {
                ComplexType::UNRESOLVED { name } => {
                  if let Some(ty) = new_type_context.get(0, *name) {
                    if r.is_pointer() {
                      ty.as_pointer()
                    } else {
                      ty.clone()
                    }
                  } else {
                    r.clone()
                  }
                }
                _ => r.clone(),
              },
              _ => r.clone(),
            })
            .collect();

          if let Some(_) = ib.global_ty_ctx.get(0, name) {
            ib.push_variable(name, ib.global_ty_ctx.get(0, name).unwrap().clone())
          } else {
            let new_routine = RoutineType {
              name,
              parameters: routine.parameters.clone(),
              body: Default::default(),
              returns: _new_returns,
              ast: routine.ast.clone(),
              type_context: new_type_context,
            };

            drop(routine);

            let _ = ib.global_ty_ctx.set(0, name, Rc::new(ComplexType::Routine(Mutex::new(new_routine))).into());

            process_routine(name, 0, &ib.global_ty_ctx);

            ib.push_variable(name, ib.global_ty_ctx.get(0, name).unwrap().clone())
          }
        }
      }
      _ => panic!("Invalid type for calling"),
    }
  } else {
    panic!("Could not find routine {}", call_name.to_str().as_str())
  };

  ib.push_ssa(CALL, call_var.ty_var.into(), &[]);
  ib.pop_stack();
}

fn process_struct_instantiation(struct_decl: &RawStructDeclaration<Token>, ib: &mut IRBuilder) {
  let struct_type_name = struct_decl.name.id.intern();

  if let Some(s_type) = ib.get_type(struct_type_name) {
    if let Some(ComplexType::Struct(StructType { name, members, size, alignment })) = s_type.as_cplx_ref() {
      let struct_var = ib.push_variable(struct_type_name, s_type.clone());
      let s_type = struct_var.ty_var;

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

        if let Some(ComplexType::StructMember(member)) = members.iter().find(|i| i.name() == member_name).map(|t| t.as_ref()) {
          process_expression(&init_expression.expression.expr, ib);

          if Some(member.ty.clone()) != ib.get_top_type() {
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

            let var_id = TypeVar::from_prim((PrimitiveType::Unsigned | PrimitiveType::new_bit_size(bit_field_size)));

            // bitfield initializers must be combined into one value and then
            // submitted at the end of this section. So we make or retrieve the
            // temporary variable for this field.

            ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_offset as u32));
            ib.push_ssa(SHL, var_id.into(), &[StackOp, StackOp]);

            ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_mask as u32));
            ib.push_ssa(AND, var_id.into(), &[StackOp, StackOp]);

            match value_maps.entry(member.offset) {
              btree_map::Entry::Occupied(mut entry) => {
                let val = entry.get_mut().value;
                ib.push_ssa(OR, var_id.into(), &[val.into(), StackOp]);
                entry.get_mut().value = ib.pop_stack().unwrap();
              }
              btree_map::Entry::Vacant(val) => {
                val.insert(StructEntry { ty: bf_type, value: ib.pop_stack().unwrap(), name: member_name });
              }
            }

            // Depending the scope invariants, may need to throw error if value
            // is truncated.

            // Mask out the expression.
          } else {
            value_maps.insert(member.offset, StructEntry { ty: member.ty.clone(), value: ib.pop_stack().unwrap(), name: member_name });
          };

          // Calculate the offset to the member within the struct
        } else {
          panic!("Member name not found.");
        }
      }

      let ptr_var = ib.get_variable_ptr(&struct_var, "---".intern()).unwrap();

      ib.push_ssa(ADDR, ptr_var.ty_var.into(), &[struct_var.store.into()]);

      let ptr_id = ib.pop_stack().unwrap();

      for (_, StructEntry { ty, value, name }) in value_maps {
        if let Some(var) = ib.get_variable_member(&struct_var, IdMember(name)) {
          ib.push_ssa(PTR_MEM_CALC, var.ty_var.into(), &[ptr_id.into()]);
          ib.push_ssa(MEM_STORE, Inherit, &[StackOp, value.into()]);
          ib.pop_stack();
        } else {
          todo!("TBD");
        }
      }

      ib.push_ssa(IROp::VAR, struct_var.ty_var.into(), &[]);
    }
  } else {
    panic!("Could not find struct definition for {struct_type_name:?}")
  }

  dbg!(ib);
}

fn process_block(ast_block: &RawBlock<Token>, ib: &mut IRBuilder) {
  ib.push_lexical_scope();

  let len = ast_block.statements.len();

  for (i, stmt) in ast_block.statements.iter().enumerate() {
    process_statement(stmt, ib, i == len - 1);
  }

  let _returns = match &ast_block.exit {
    block_expression_group_1_Value::None => false,
    block_expression_group_1_Value::RawBreak(brk) => {
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

  ib.pop_lexical_scope();
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

fn process_match(match_: &RawMatch<Token>, ib: &mut IRBuilder<'_, '_>) {
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

            ib.push_ssa(op, Inherit, &[expr_id.into(), StackOp]);
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

fn process_match_scope(scope: &match_scope_Value<Token>, ib: &mut IRBuilder<'_, '_>) {
  match &scope {
    match_scope_Value::Expression(expr) => process_expression(&expr.expr, ib),
    match_scope_Value::RawBlock(block) => process_block(block, ib),
    _ => unreachable!(),
  }
}

fn process_loop(loop_: &RawLoop<Token>, ib: &mut IRBuilder<'_, '_>) {
  let (loop_head, loop_exit) = ib.push_loop(loop_.label.id.intern());

  match &loop_.scope {
    loop_statement_group_1_Value::RawBlock(block) => process_block(block, ib),
    loop_statement_group_1_Value::RawMatch(match_) => process_match(match_, ib),
    _ => unreachable!(),
  }

  ib.set_successor(loop_head, SuccessorMode::Default);
  ib.pop_loop();
}

fn process_expression_statement(expr: &std::sync::Arc<Expression<Token>>, ib: &mut IRBuilder<'_, '_>, last_value: bool) {
  process_expression(&expr.expr, ib);
  if !last_value {
    ib.pop_stack();
  }
}

fn process_assign_statement(assign: &std::sync::Arc<crate::parser::script_parser::RawAssignment<Token>>, ib: &mut IRBuilder<'_, '_>) {
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
                  ib.push_ssa(SHL, var_data.ty_var.into(), &[StackOp, StackOp]);

                  // Mask out unwanted bits
                  ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, bit_mask as u32));
                  ib.push_ssa(AND, Inherit, &[StackOp, StackOp]);

                  // Load the base value from the structure and mask out target bitfield
                  ib.push_const(ConstVal::new(PrimitiveType::Unsigned | PrimitiveType::b32, !bit_mask as u32));
                  ib.push_ssa(MEM_LOAD, var_data.ty_var.into(), &[var_data.store.into()]);
                  ib.push_ssa(AND, var_data.ty_var.into(), &[StackOp, StackOp]);

                  // Combine the original value and the new bitfield value.
                  ib.push_ssa(OR, var_data.ty_var.into(), &[StackOp, StackOp]);

                  // Store value back in the structure.
                  ib.push_ssa(MEM_STORE, var_data.ty_var.into(), &[var_data.store.into(), StackOp]);
                  ib.pop_stack();
                } else {
                  ib.push_ssa(MEM_STORE, var_data.ty_var.into(), &[var_data.store.into(), expr_id.into()]);
                  ib.pop_stack();
                }
              }
              BaseType::Complex(ComplexType::StructMember(mem)) => {
                ib.push_ssa(MEM_STORE, var_data.ty_var.into(), &[var_data.store.into(), expr_id.into()]);
                ib.pop_stack();
              }
              BaseType::Complex(ComplexType::UNRESOLVED { .. }) | _ => {
                ib.push_ssa(MEM_STORE, var_data.ty_var.into(), &[var_data.store.into(), expr_id.into()]);
                ib.pop_stack();
              }
              ty => unreachable!("{ty:#?}"),
            }
          } else {
            ib.push_ssa(STORE, var_data.ty_var.into(), &[var_data.store.into(), expr_id.into()]);
            ib.pop_stack();
          }
        } else if var_assign.var.members.len() == 1 {
          dbg!(expr_ty.base_type());
          // Create and assign a new variable based on the expression.
          let var_name = var_assign.var.members[0].id.intern();
          let var_id = ib.body.graph[expr_id].ty_var();

          dbg!(&ib, var_name);

          match expr_ty.base_type() {
            BaseType::Complex(ty) => match &ty {
              ComplexType::Struct(..) => ib.rename_var(var_id, IdMember(var_name)),
              _ => unreachable!(),
            },
            _ => {
              let var = ib.push_variable(var_name, expr_ty);
              ib.pop_stack();

              ib.push_ssa(STORE, var.ty_var.into(), &[var.store.into(), expr_id.into()]);
              ib.pop_stack();
            }
          }
        } else {
          let blame_string = var_assign.tok.blame(1, 1, "could not find variable @", None);
          panic!("{blame_string} \n{ib:?}",)
        }
      }
      assignment_var_Value::RawAssignmentDeclaration(var_decl) => {
        dbg!(expr_ty.base_type());
        let var_name = var_decl.var.id.intern();
        let expected_ty = get_type_from_sm(&var_decl.ty, ib).unwrap();
        let var_id = ib.body.graph[expr_id].ty_var();

        match expr_ty.base_type() {
          BaseType::Complex(ty) => match &ty {
            ComplexType::Struct(..) => ib.rename_var(var_id, IdMember(var_name)),
            _ => unreachable!(),
          },
          _ => {
            if expected_ty != expr_ty {
              match (expected_ty.base_type(), expr_ty.base_type()) {
                (BaseType::Prim(prim_ty), BaseType::Prim(prim_expr_ty)) => {
                  let var = ib.push_variable(var_name, expected_ty);
                  ib.pop_stack();

                  ib.push_ssa(STORE, var.ty_var.into(), &[var.store.into(), expr_id.into()]);
                  ib.pop_stack();
                }
                _ => panic!("Miss matched types ty:{expected_ty:?} expr_ty:{expr_ty:?}"),
              }
            } else {
              let var = ib.push_variable(var_name, expr_ty);
              ib.pop_stack();

              ib.push_ssa(STORE, var.ty_var.into(), &[var.store.into(), expr_id.into()]);
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
  get_type(ir_type, ib.g_ty_ctx_index, ib.global_ty_ctx)
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
    type_Value::Type_f32(_) => Some((PrimitiveType::Float | PrimitiveType::b32).into()),
    type_Value::Type_f64(_) => Some((PrimitiveType::Float | PrimitiveType::b64).into()),
    type_Value::Type_f32v2(_) => Some((PrimitiveType::Float | PrimitiveType::b32 | PrimitiveType::v2).into()),
    type_Value::Type_f32v4(_) => Some((PrimitiveType::Float | PrimitiveType::b32 | PrimitiveType::v4).into()),
    type_Value::Type_f64v2(_) => Some((PrimitiveType::Float | PrimitiveType::b64 | PrimitiveType::v2).into()),
    type_Value::Type_f64v4(_) => Some((PrimitiveType::Float | PrimitiveType::b64 | PrimitiveType::v4).into()),
    type_Value::ReferenceType(name) => {
      if let Some(ty) = type_context.get(scope_index, name.name.id.intern()) {
        Some(ty.as_pointer())
      } else {
        None
      }
    }
    type_Value::NamedType(name) => {
      if let Some(ty) = type_context.get(scope_index, name.name.id.intern()) {
        Some(ty.clone())
      } else {
        None
      }
    }
    _t => None,
  }
}

pub fn get_type_name(ir_type: &type_Value<Token>) -> IString {
  match ir_type {
    type_Value::Type_Flag(_) => "Flag".intern(),
    type_Value::Type_u8(_) => "u8".intern(),
    type_Value::Type_u16(_) => "u16".intern(),
    type_Value::Type_u32(_) => "u32".intern(),
    type_Value::Type_u64(_) => "u64".intern(),
    type_Value::Type_i8(_) => "i8".intern(),
    type_Value::Type_i16(_) => "i16".intern(),
    type_Value::Type_i32(_) => "i32".intern(),
    type_Value::Type_i64(_) => "i64".intern(),
    type_Value::Type_f32(_) => "f32".intern(),
    type_Value::Type_f64(_) => "f64".intern(),
    type_Value::Type_f32v2(_) => "f32v2".intern(),
    type_Value::Type_f32v4(_) => "f32v4".intern(),
    type_Value::Type_f64v2(_) => "f64v2".intern(),
    type_Value::Type_f64v4(_) => "f64v4".intern(),
    type_Value::ReferenceType(name) => name.name.id.intern(),
    type_Value::NamedType(name) => name.name.id.intern(),
    _t => Default::default(),
  }
}

#[cfg(test)]
mod test;
