use super::{
  ir_builder::{IRBuilder, SuccessorMode},
  ir_graph::TyData,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  ir::{
    ir_builder::{SMO, SMT},
    ir_graph::IROp,
  },
  istring::*,
  parser::script_parser::{
    assignment_var_Value,
    bitfield_element_Value,
    block_expression_group_3_Value,
    expression_Value,
    lifetime_Value,
    loop_statement_group_1_Value,
    match_clause_Value,
    match_expression_Value,
    match_scope_Value,
    module_member_Value,
    module_members_group_Value,
    pointer_type_group_Value,
    property_Value,
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
    RawModMembers,
    RawModule,
    RawNum,
    RawRoutine,
    RawStruct,
    RawStructInstantiation,
    Sub,
    BIT_SL,
    BIT_SR,
  },
  types::*,
};
use core::panic;
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::VecDeque,
  ops::{Deref, DerefMut},
  sync::Arc,
};
use IROp::*;
use SMO::*;
use SMT::Inherit;

pub fn build_module(module: &Arc<RawModule<Token>>) {
  // Step 1: Scan and reserve slots for types. Also do this for any scope, and so
  // on and so forth.
  // Step 2: Build definitions for all types, in this order: 0:zero_types
  // 1:arrays 2:structs 3:unions  4:routines

  // Step 3: Build routine bodies.

  // Step 4: Validate - All relaxed routines and types should have proper
  // semantic behavior. This step may be deferred until such a time when
  // either debug, reflection, or compiler information is needed for a given
  // type.

  let mut ty_db = TypeDatabase::new();

  let routines = process_module_members(&module.members, &mut ty_db);

  for routine in &routines {
    // Gather function type information.
    process_routine(*routine, &mut ty_db);
  }
}

pub fn process_module_members<'a>(members: &'a RawModMembers<Token>, ty_db: &mut TypeDatabase) -> Vec<IString> {
  let routines = declare_types(members, "./".intern(), ty_db);

  return routines;
}

fn declare_types(members: &RawModMembers<Token>, scope_name: IString, ty_db: &mut TypeDatabase) -> Vec<IString> {
  // TODO, get hash of node. If hash already exists in system there is no
  // need to rebuild this node. Otherwise we evict any existing
  // type mapped to this name and proceed to evaluate and proceed this
  // node as a replacement type.

  use module_members_group_Value::*;

  let mut routines = vec![];

  for mod_member in &members.members {
    match &mod_member {
      AnnotationVariable(anno_var) => {
        let name = anno_var.name.val.intern();

        todo!("Annotation");
        /*
        let ty = ty_db.insert_ty(name, Type::UNRESOLVED { name: MemberName::IdMember(name) });
        */
        println!("Declaring ANNOTATION_VAR [{scope_name:<2}{name}]");
      }
      LifetimeVariable(lt_var) => {
        let name = match &lt_var.name {
          lifetime_Value::GlobalLifetime(_) => "*".to_string(),
          lifetime_Value::ScopedLifetime(scope) => scope.val.to_string() + "*",
          _ => unreachable!(),
        };
        println!("Declaring LIFETIME_VAR [{scope_name:<2}{name}]");
      }
      AnnotatedModMember(member) => {
        use module_member_Value::*;

        if !member.annotation.val.is_empty() {
          println!("-- {}", member.annotation.val);
        }

        match &member.member {
          RawScope(scope) => {
            let name = scope.name.id.intern();

            println!("Declaring SCOPE     [{scope_name:<2}{name}]");

            /*             let mut sub_type_scope = TypeDatabase::new(Some(ty_db));
            declare_types(&scope.members, (scope_name.to_string() + &name.to_string() + "/").intern(), &mut sub_type_scope);

            let ty = ty_db.insert_ty(name, ScopeType { name, ctx: sub_type_scope }.into()); */
          }
          RawEnum(array) => {
            let name = array.name.id.intern();

            /* let ty = ty_db.insert_ty(name, Type::UNRESOLVED { name:
             * MemberName::IdMember(name) }); */

            println!("Declaring ENUM      [{scope_name:2}{name}]")
          }
          RawBitFlagEnum(flag) => {
            let name = flag.name.id.intern();
            /* let ty = ty_db.insert_ty(name, Type::UNRESOLVED { name:
             * MemberName::IdMember(name) }); */

            println!("Declaring FLAG_ENUM [{scope_name:2}{name}]")
          }
          RawArray(array) => {
            let name = array.name.id.intern();
            /* let ty = ty_db.insert_ty(name, Type::UNRESOLVED { name:
             * MemberName::IdMember(name) }); */

            println!("Declaring ARRAY     [{scope_name:2}{name}]")
          }
          RawUnion(union) => {
            let name = union.name.id.intern();
            /* let ty = ty_db.insert_ty(name, Type::UNRESOLVED { name:
             * MemberName::IdMember(name) }); */

            println!("Declaring UNION     [{scope_name:2}{name}]")
          }
          RawStruct(strct) => {
            let name = strct.name.id.intern();
            ty_db.insert_type(name, StructType { name, members: Default::default(), size: 0, alignment: 0 }.into());

            println!("Declaring STRUCTURE [{scope_name:2}{name}]")
          }
          RawRoutine(routine) => {
            let name = routine.name.id.intern();
            routines.push(name);

            let ty = RoutineType {
              name,
              parameters: Default::default(),
              returns: Default::default(),
              body: RoutineBody::new(ty_db),
              ast: routine.clone(),
            };

            let ty = ty_db.insert_type(name, ty.into());

            use routine_type_Value::*;
            match &routine.ty {
              RawFunctionType(..) => println!("Declaring FUNCTION  [{scope_name:2}{name}]"),
              RawProcedureType(..) => println!("Declaring PROCEDURE [{scope_name:2}{name}]",),
              _ => {}
            }
          }
          ty => unreachable!("Type not recognized {ty:#?}"),
        }
      }
      ty => unreachable!("Type not recognized {ty:#?}"),
    }
  }

  for mod_member in &members.members {
    match &mod_member {
      AnnotationVariable(anno_var) => {}
      LifetimeVariable(lt_var) => {}
      AnnotatedModMember(member) => {
        use module_member_Value::*;

        if !member.annotation.val.is_empty() {
          println!("-- {}", member.annotation.val);
        }

        match &member.member {
          RawScope(scope) => {}
          RawEnum(array) => {}
          RawBitFlagEnum(flag) => {}
          RawArray(array) => {}
          RawUnion(union) => {}
          RawStruct(strct) => {
            process_struct(strct, ty_db);
          }
          RawRoutine(routine) => {
            process_routine_signature(routine, ty_db);
          }
          ty => unreachable!("Type not recognized {ty:#?}"),
        }
      }
      ty => unreachable!("Type not recognized {ty:#?}"),
    }
  }

  // Resolve super_type

  return routines;
}

fn process_struct(strct: &RawStruct<Token>, ty_db: &mut TypeDatabase) {
  let name = strct.name.id.intern();
  let Some(mut ty_ref) = ty_db.get_type_mut(name) else {
    panic!("Could not find Struct type: {name}",);
  };

  match ty_ref {
    Type::Struct(st) => {
      let name = strct.name.id.intern();
      let mut offset = 0;
      let mut min_alignment = 1;

      use property_Value::*;
      for (index, prop) in strct.properties.iter().enumerate() {
        match prop {
          RawBitCompositeProp(bitfield) => {
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

                  let ty = (PrimitiveType::Discriminant | PrimitiveType::new_bit_size(bit_size) | PrimitiveType::new_bitfield_data(bit_offset as u8, bit_field_size as u8));

                  let ty_name = format!("{ty}").intern();

                  let ty = TypeSlot::Primitive(ty);

                  st.members.push(StructMemberType { ty, original_index: index, name: "#desc".intern(), offset: prop_offset });

                  bit_offset += bit_size;
                }
                bitfield_element_Value::BitFieldProp(prop) => {
                  todo!("AAAA");
                  /*         use type_Value::*;
                  match &prop.r#type {
                    Type_u32(_) => {
                      if let Some(ty) = type_scope.get_type_entry("u32".intern()) {
                        dbg!(ty);
                      }
                    }
                    Type_Variable(ty) => {
                      let name = ty.name.id.intern();
                      if let Some(ty) = type_scope.get_type_entry(name) {
                        dbg!(ty);
                      }
                    }
                    _ => unreachable!(),
                  } */
                }
                bitfield_element_Value::None => {}
              }
            }

            if bit_offset > max_bit_size {
              panic!("Bitfield element size {bit_offset} overflow bitfield size {max_bit_size}")
            }
          }
          RawProperty(raw_prop) => {
            if let Some(ty) = get_type(&raw_prop.r#type, ty_db, false) {
              let name = raw_prop.name.id.intern();
              /*        let prop_offset = get_aligned_value(offset, ty.alignment() as u64);
              offset = prop_offset + ty.byte_size() as u64;

              min_alignment = min_alignment.max(ty.alignment() as u64);


              if st.members.iter().any(|m| m.name == name) {
                panic!("Name already taken {name:?}")z
              } */

              st.members.push(StructMemberType { ty, original_index: index, name, offset: 0 })
            } else {
              panic!("Could not resolve type");
            }
          }
          node => panic!("Unhandled property type {node:#?}"),
        }
      }
    }
    _ => unreachable!(),
  };
}

fn process_routine_signature(routine: &Arc<RawRoutine<Token>>, ty_db: &mut TypeDatabase) {
  let name = routine.name.id.intern();

  let Some(mut ty_ref) = ty_db.get_type_mut(name) else {
    panic!("Could not find Struct type: {name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      let (params, ret) = match &routine.ty {
        routine_type_Value::RawFunctionType(fn_ty) => (fn_ty.params.as_ref(), Some(fn_ty.return_type.as_ref())),
        routine_type_Value::RawProcedureType(proc_ty) => (proc_ty.params.as_ref(), Option::None),
        _ => unreachable!(),
      };

      let parameters = &mut rt.parameters;
      let body = &mut rt.body;

      for (index, param) in params.params.iter().enumerate() {
        let param_name = param.var.id.intern();
        let param_type = &param.ty;
        if param_type.inferred {
          let gen_ty_name = get_type_name(&param.ty.ty);
          let slot = body.ctx.insert_generic(gen_ty_name);
          debug_assert!(!matches!(slot, TypeSlot::UNRESOLVED(..)));
          parameters.push((param_name, index, slot));
        } else if let Some(ty) = get_type(&param_type.ty, ty_db, false) {
          parameters.push((param_name, index, ty));
        } else {
          panic!("Could not resolve type! {param_name}");
        }
      }

      // Seal parameter scope.
      body.ctx.ty_var_scopes.push_scope();

      rt.returns = match ret {
        Some(ret_type) => {
          if ret_type.inferred {
            let gen_ty_name = get_type_name(&ret_type.ty);
            let slot = body.ctx.insert_generic(gen_ty_name);
            vec![slot.clone().into()]
          } else if let Some(ty) = get_type(&ret_type.ty, ty_db, false) {
            vec![ty.into()]
          } else {
            panic!("Could not resolve return type")
          }
        }
        _ => vec![],
      };

      // Seal the return signature scope.
      body.ctx.ty_var_scopes.push_scope();
    }
    _ => unreachable!(),
  }
}

fn process_routine(routine_name: IString, type_scope: &mut TypeDatabase) {
  let Some(mut ty_ref) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find Struct type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      rt.body.resolved = false;

      let RoutineType { name, parameters, returns, body, ast, .. } = rt;

      let mut ib = IRBuilder::new(body);

      for (name, index, ty) in parameters.iter() {
        let mut var = ib.body.ctx.insert_var(*name, *ty).clone();

        ib.push_ssa(PARAM_DECL, var.id.into(), &[]);
        let var_data = ib.pop_stack().unwrap();
        var.store = var_data;

        ib.push_ssa(MEMB_PTR_CALC, SMT::Data(TyData::PtrVar(var.id.into())), &[SMO::IROp(var.store)]);
        var.reference = ib.pop_stack().unwrap();

        let reference = var.reference;
        ib.push_ssa(IROp::MEMB_PTR_LOAD, SMT::Data(TyData::DerefVar(var.id.into())), &[SMO::IROp(reference)]);
        var.pointer = ib.pop_stack().unwrap();

        ib.body.ctx.vars[var.id] = var;
      }

      process_expression(&ast.expression.expr, &mut ib);

      if returns.len() > 0 {
        // todo!("Check that return types match!");
      }

      dbg!(ib);
    }
    _ => unreachable!(),
  }
}

/// Processes the signature of a routine and stores the result into the type
/// context.

fn process_expression(expr: &expression_Value<Token>, ib: &mut IRBuilder) {
  // Create type for the expression.

  match expr {
    expression_Value::RawCall(call) => process_call(call, ib),
    expression_Value::RawNum(num) => process_const_number(num, ib),
    // expression_Value::AddressOf(addr) => process_address_of(ib, addr),
    expression_Value::RawStructInstantiation(struct_instance) => process_struct_instantiation(struct_instance, ib),
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

/* fn process_address_of(ib: &mut IRBuilder<'_>, addr: &std::sync::Arc<crate::parser::script_parser::AddressOf<Token>>) {
  if let Some(var) = ib.get_variable_from_id(IdMember(addr.id.id.intern())) {
    let ptr = ib.get_variable_ptr(&var, "---".intern()).unwrap();
    ib.push_ssa(ADDR, ptr.ty_var.into(), &[SMO::IROp(var.store)])
  } else {
    panic!("Variable not found")
  }
} */

fn resolve_variable(mem: &RawMember<Token>, ib: &mut IRBuilder) -> Option<Variable> {
  let base = &mem.members[0];

  let var_name = base.id.intern();

  if let Some(var_data) = ib.get_variable(var_name) {
    let mut var_data = var_data.clone();

    for var_name in &mem.members[1..] {
      let name = var_name.id.intern();

      if let Some(var) = ib.get_var_member(&var_data, name) {
        let mut var = var.clone();

        ib.push_ssa(MEMB_PTR_CALC, SMT::Data(TyData::PtrVar(var.id.into())), &[SMO::IROp(var_data.store)]);
        var.reference = ib.pop_stack().unwrap();

        let reference = var.reference;
        ib.push_ssa(IROp::MEMB_PTR_LOAD, SMT::Data(TyData::DerefVar(var.id.into())), &[SMO::IROp(reference)]);
        var.pointer = ib.pop_stack().unwrap();

        ib.body.ctx.vars[var.id.usize()] = var;

        var_data = var;
      } else {
        panic!("Could not resolve variable");
      }
    }

    Some(var_data)
  } else {
    Option::None
  }
}

fn process_member_load(mem: &std::sync::Arc<RawMember<Token>>, ib: &mut IRBuilder<'_>) {
  //todo!("Handle Member Load");
  if let Some(var) = resolve_variable(mem, ib) {
    ib.push_ssa(LOAD, var.id.into(), &[var.reference.into()]);
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
  // Member call resolution may involve lookup on compatible functions in the
  // type scope stack.

  let name = if call.member.members.len() == 1 {
    let name = call.member.members[0].clone();
    name.id.intern()
  } else {
    todo!("Handle multi member call");
  };

  let call = if let Some(routine_entry) = ib.body.ctx.db_mut().get_type_mut(name).as_ref() {
    let mut args = Vec::new();

    for arg in &call.args {
      process_expression(&arg.expr, ib);

      let var_id = ib.get_top_var_id();

      ib.push_ssa(CALL_ARG, Inherit, &[StackOp]);

      args.push(ib.pop_stack().unwrap());
    }

    match routine_entry {
      Type::Routine(routine) => {
        if routine.body.resolved {
          todo!("call resolved routine")
          //ib.push_var(routine.name, routine_entry);
        } else {
          let mut _new_name: String = routine.name.to_string();
          let mut generic_args = Vec::new();

          let mut routine_body = RoutineBody::new(&mut ib.body.ctx.db_mut());

          let mut param_queue: VecDeque<_> = routine.parameters.iter().enumerate().map(|(index, (.., ty))| (0, index, ty)).collect();
          let mut return_queue: VecDeque<_> = routine.body.ctx.ty_var_scopes.scopes[1].1.iter().enumerate().map(|(index, i)| (0, index, i.0, i.1)).collect();

          // Find matching resolved types.
          while let Some((tries, param_index, param_ty_slot)) = param_queue.pop_front() {
            if let Some(arg_id) = args.get(param_index) {
              let arg_var = ib.get_node_variable(*arg_id).expect("Could not extract variable data").clone();

              dbg!(arg_var.ty_slot);
              let arg_var_ty = arg_var.ty_slot.ty_base(&ib.body.ctx);
              let param_ty = param_ty_slot.ty_base(&routine.body.ctx);

              println!("\n ---  {} {} \n --- {ib:#?} \n ---", param_ty, arg_var_ty);

              match (param_ty, arg_var_ty) {
                (TypeRef::UNRESOLVED(p_name, p_index), TypeRef::UNRESOLVED(a_name, a_index)) => {
                  if let Some(ty) = routine_body.ctx.get_type_local(p_name) {
                    debug_assert!(!matches!(ty, TypeSlot::CtxIndex(_)));
                    let _ = ib.body.ctx.type_slots[a_index as usize] = ty;
                  } else if tries == 0 && !param_queue.is_empty() {
                    // Try to match this after other types have been defined.
                    param_queue.push_back((tries + 1, param_index, param_ty_slot));
                  } else {
                    panic!("Could not resolve this type {arg_var_ty}");
                  }
                }
                (ty @ TypeRef::UNRESOLVED(p_name, p_index), _) => {
                  generic_args.push(p_name.to_string() + ":" + &arg_var_ty.to_string());
                  if let TypeSlot::CtxIndex(index) = routine_body.ctx.insert_generic(p_name) {
                    let slot = arg_var.ty_slot.resolve_to_outer_slot(&ib.body.ctx);
                    debug_assert!(!matches!(slot, TypeSlot::CtxIndex(_)));
                    routine_body.ctx.type_slots[index as usize] = slot;
                  }
                  // routine
                }
                (ty, TypeRef::UNRESOLVED(a_name, a_index)) => {
                  debug_assert!(!matches!(param_ty_slot, TypeSlot::CtxIndex(_)));
                  ib.body.ctx.type_slots[a_index as usize] = *param_ty_slot;
                  println!("AAAAAA\n {ty} \n {ib:#?} \n ---");
                }

                _ => {
                  //  if arg_var_ty != param_ty {
                  println!("Is type matching? {param_index} arg({}) != param({})", param_ty, arg_var_ty)
                  //  }
                }
              }
            } else {
              panic!("No matching argument for parameter [{param_index}] in function ");
            };
          }

          let name = (_new_name + "<" + &generic_args.join(", ") + ">").intern();

          /*          let mut _new_returns = routine
          .returns
          .iter()
          .map(|r| match r.ty_enum() {
            Type::Complex(cplx) => match &cplx.as_ref().ty {
              ComplexType::UNRESOLVED { name } => {
                if let Some(ty) = routine_body.type_context.get(0, *name) {
                  ty.clone()
                } else {
                  r.clone()
                }
              }
              _ => r.clone(),
            },
            _ => r.clone(),
          })
          .collect(); */

          let mut _new_returns = Default::default();

          if let Some(ty) = ib.body.ctx.get_type(name) {
            ib.body.ctx.insert_var(name, ty).clone()
          } else {
            dbg!(&routine_body);
            let new_routine = RoutineType {
              name,
              parameters: routine.parameters.clone(),
              body: routine_body,
              returns: _new_returns,
              ast: routine.ast.clone(),
            };

            let _ = routine;

            let entry = ib.body.ctx.db_mut().insert_type(name, Type::Routine(new_routine));

            process_routine(name, ib.body.ctx.db_mut());

            ib.body.ctx.insert_var(name, TypeSlot::GlobalIndex(entry as u32)).clone()
          }
        }
      }
      _ => panic!("Invalid type for calling"),
    }
  } else {
    todo!("Report call resolution error")
    //panic!("Could not find routine {}", call_name.to_str().as_str())
  };

  ib.push_ssa(CALL, call.id.into(), &[]);
  ib.pop_stack();

  dbg!(ib);
}

fn process_struct_instantiation(struct_decl: &RawStructInstantiation<Token>, ib: &mut IRBuilder) {
  /*  let struct_type_name = struct_decl.name.id.intern();

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

        if let Some(member) = members.iter().find(|i| i.name == member_name) {
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
  } */

  dbg!(ib);
}

fn process_block(ast_block: &RawBlock<Token>, ib: &mut IRBuilder) {
  ib.body.ctx.push_scope();

  let len = ast_block.statements.len();

  for (i, stmt) in ast_block.statements.iter().enumerate() {
    process_statement(stmt, ib, i == len - 1);
  }

  use block_expression_group_3_Value::*;
  let _returns = match &ast_block.exit {
    None => false,
    RawBreak(brk) => {
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

  ib.body.ctx.pop_scope();
}

fn process_statement(stmt: &statement_Value<Token>, ib: &mut IRBuilder, last_value: bool) {
  use statement_Value::*;
  match stmt {
    RawAssignment(assign) => process_assign_statement(assign, ib),
    Expression(expr) => process_expression_statement(expr, ib, last_value),
    RawLoop(loop_) => process_loop(loop_, ib),
    RawMatch(match_) => process_match(match_, ib),
    d => todo!("process statement: {d:#?}"),
  }
}

fn process_match(match_: &RawMatch<Token>, ib: &mut IRBuilder<'_>) {
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

fn process_match_scope(scope: &match_scope_Value<Token>, ib: &mut IRBuilder<'_>) {
  match &scope {
    match_scope_Value::Expression(expr) => process_expression(&expr.expr, ib),
    match_scope_Value::RawBlock(block) => process_block(block, ib),
    _ => unreachable!(),
  }
}

fn process_loop(loop_: &RawLoop<Token>, ib: &mut IRBuilder<'_>) {
  let (loop_head, loop_exit) = ib.push_loop(loop_.label.id.intern());

  match &loop_.scope {
    loop_statement_group_1_Value::RawBlock(block) => process_block(block, ib),
    loop_statement_group_1_Value::RawMatch(match_) => process_match(match_, ib),
    _ => unreachable!(),
  }

  ib.set_successor(loop_head, SuccessorMode::Default);
  ib.pop_loop();
}

fn process_expression_statement(expr: &std::sync::Arc<Expression<Token>>, ib: &mut IRBuilder<'_>, last_value: bool) {
  process_expression(&expr.expr, ib);
  if !last_value {
    ib.pop_stack();
  }
}

fn process_assign_statement(assign: &std::sync::Arc<crate::parser::script_parser::RawAssignment<Token>>, ib: &mut IRBuilder<'_>) {
  let db = ib.body.ctx.db();
  // Process assignments.
  for expression in &assign.expressions {
    process_expression(&expression.expr, ib)
  }

  let mut expression_data = (0..assign.expressions.len()).into_iter().map(|_| ib.pop_stack().unwrap()).collect::<ArrayVec<6, _>>();
  expression_data.reverse();

  // Process assignment targets.
  for (variable, expr_id) in assign.vars.iter().zip(&mut expression_data.iter().cloned()) {
    let expr_ty = ib.get_node_ty(expr_id).unwrap();
    let expr_ty_slot = expr_ty.ty_slot(&ib.body.ctx);

    use assignment_var_Value::*;
    match variable {
      RawArrayAccess(..) => {
        todo!("RawArrayAccess")
      }
      RawAssignmentVariable(var_assign) => {
        if let Some(var_data) = resolve_variable(&var_assign.var, ib) {
          dbg!(&var_data);
          ib.push_ssa(STORE, Inherit, &[var_data.reference.into(), expr_id.into()]);
          ib.pop_stack();
        } else if var_assign.var.members.len() == 1 {
          // Create and assign a new variable based on the expression.
          let var_name = var_assign.var.members[0].id.intern();
          let var_id = ib.body.graph[expr_id].ty_var();

          if let Some(var) = ib.get_node_variable(expr_id) {
            if let TypeRef::Struct(_) = var.clone().ty_slot.ty_base(&ib.body.ctx) {
              //ib.rename_var(var.id, IdMember(var_name));
            }
          }

          let var = ib.declare_variable(var_name, expr_ty_slot).clone();

          ib.push_ssa(STORE, var.id.into(), &[var.reference.into(), expr_id.into()]);

          ib.pop_stack();
        } else {
          let blame_string = var_assign.tok.blame(1, 1, "could not find variable @", Option::None);
          panic!("{blame_string} \n{ib:?}",)
        }
      }
      RawAssignmentDeclaration(var_decl) => {
        todo!("process assignment_var_Value::");
        /*
        dbg!(expr_ty.ty_enum());
        let var_name = var_decl.var.id.intern();
        let (expected_ty, _ptr) = get_type_from_sm(&var_decl.ty, ib).unwrap();
        let var_id = ib.body.graph[expr_id].ty_var();

        match expr_ty.ty_enum() {
          BaseType::Complex(ty) => match &ty.as_ref().ty {
            ComplexType::Struct(..) => ib.rename_var(var_id, IdMember(var_name)),
            _ => unreachable!(),
          },
          _ => {
            if expected_ty != expr_ty {
              match (expected_ty.ty_enum(), expr_ty.ty_enum()) {
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
        } */
      }
      _ => unreachable!(),
    }
  }

  // Match assignments to targets.
}

pub fn get_type_from_sm(ir_type: &type_Value<Token>, ib: &mut IRBuilder) -> Option<(Type, PtrType)> {
  todo!("get_type_from_sm");
  //get_type(ir_type, ib.global_ty_ctx)
}

#[derive(Default)]
pub enum PtrType {
  #[default]
  None,
  Reference,
  Global,
  Name(IString),
}

pub fn get_pointer_type(ir_type: &type_Value<Token>) -> PtrType {
  use pointer_type_group_Value::*;
  use type_Value::*;
  match ir_type {
    Type_Pointer(ptr) => match &ptr.ptr_type {
      GlobalLifetime(_) => PtrType::Global,
      ScopedLifetime(name) => PtrType::Name(name.val.intern()),
      Reference(_) => PtrType::Reference,
      _ => unreachable!(),
    },
    _t => PtrType::None,
  }
}

pub fn get_type(ir_type: &type_Value<Token>, type_db: &mut TypeDatabase, insert_unresolved: bool) -> Option<TypeSlot> {
  use type_Value::*;
  match ir_type {
    Type_Flag(_) => Some(TypeSlot::Primitive(PrimitiveType::Flag)),
    Type_u8(_) => Some(TypeSlot::Primitive(PrimitiveType::u8)),
    Type_u16(_) => Some(TypeSlot::Primitive(PrimitiveType::u16)),
    Type_u32(_) => Some(TypeSlot::Primitive(PrimitiveType::u32)),
    Type_u64(_) => Some(TypeSlot::Primitive(PrimitiveType::u64)),
    Type_i8(_) => Some(TypeSlot::Primitive(PrimitiveType::i8)),
    Type_i16(_) => Some(TypeSlot::Primitive(PrimitiveType::i16)),
    Type_i32(_) => Some(TypeSlot::Primitive(PrimitiveType::i32)),
    Type_i64(_) => Some(TypeSlot::Primitive(PrimitiveType::i64)),
    Type_f32(_) => Some(TypeSlot::Primitive(PrimitiveType::f32)),
    Type_f64(_) => Some(TypeSlot::Primitive(PrimitiveType::f64)),
    Type_f32v2(_) => Some(TypeSlot::Primitive(PrimitiveType::f32v2)),
    Type_f32v4(_) => Some(TypeSlot::Primitive(PrimitiveType::f32v4)),
    Type_f64v2(_) => Some(TypeSlot::Primitive(PrimitiveType::f64v2)),
    Type_f64v4(_) => Some(TypeSlot::Primitive(PrimitiveType::f64v4)),
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), type_db, insert_unresolved) {
        use pointer_type_group_Value::*;
        match &ptr.ptr_type {
          Reference(_) | GlobalLifetime(_) => {
            Some(TypeSlot::GlobalIndex(type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), base_type)) as u32))
          }
          ScopedLifetime(scope) => Some(TypeSlot::GlobalIndex(
            type_db.get_or_add_type_index(format!("{}*{}", scope.val, base_type.ty_gb(type_db)).intern(), Type::Pointer(scope.val.intern(), base_type)) as u32,
          )),
          _ => unreachable!(),
        }
      } else {
        Option::None
      }
    }
    Type_Variable(type_var) => {
      if let Some(ty) = type_db.get_type_index(type_var.name.id.intern()) {
        Some(TypeSlot::GlobalIndex(ty as u32))
      } else {
        Option::None
      }
    }
    _t => Option::None,
  }
}

pub fn get_type_name(ir_type: &type_Value<Token>) -> IString {
  use type_Value::*;
  match ir_type {
    Type_Flag(_) => "Flag".intern(),
    Type_u8(_) => "u8".intern(),
    Type_u16(_) => "u16".intern(),
    Type_u32(_) => "u32".intern(),
    Type_u64(_) => "u64".intern(),
    Type_i8(_) => "i8".intern(),
    Type_i16(_) => "i16".intern(),
    Type_i32(_) => "i32".intern(),
    Type_i64(_) => "i64".intern(),
    Type_f32(_) => "f32".intern(),
    Type_f64(_) => "f64".intern(),
    Type_f32v2(_) => "f32v2".intern(),
    Type_f32v4(_) => "f32v4".intern(),
    Type_f64v2(_) => "f64v2".intern(),
    Type_f64v4(_) => "f64v4".intern(),
    Type_Pointer(ptr) => get_type_name(&ptr.ty.clone().to_ast().into_type_Value().unwrap()),
    Type_Variable(type_var) => type_var.name.id.intern(),
    _t => Default::default(),
  }
}

#[cfg(test)]
mod test;
