use super::ir_builder::{IRBuilder, SuccessorMode};
use crate::{
  container::get_aligned_value,
  ir::{
    ir_builder::{IterStack, SMO, SMT},
    ir_graph::{IRGraphId, IROp, VarId},
  },
  istring::*,
  parser::script_parser::{
    assignment_var_Value,
    bitfield_element_group_Value,
    block_expression_group_3_Value,
    expression_Value,
    lifetime_Value,
    loop_statement_group_1_Value,
    module_member_Value,
    module_members_group_Value,
    property_Value,
    routine_type_Value,
    statement_Value,
    type_Value,
    Add,
    Div,
    Expression,
    IterReentrance,
    MemberCompositeAccess,
    Mul,
    Pow,
    RawAggregateInstantiation,
    RawAssignment,
    RawBlock,
    RawCall,
    RawIterStatement,
    RawLoop,
    RawMatch,
    RawModMembers,
    RawModule,
    RawNum,
    RawRoutine,
    Sub,
    Type_Array,
    Type_Enum,
    Type_Flag,
    Type_Struct,
    BIT_AND,
    BIT_OR,
    BIT_SL,
    BIT_SR,
    BIT_XOR,
  },
  types::*,
};
use core::panic;
pub use radlr_rust_runtime::types::Token;
use std::{collections::BTreeMap, sync::Arc};
use IROp::*;
use SMO::*;
use SMT::Inherit;

pub fn build_module(module: &Arc<RawModule<Token>>) -> Box<TypeDatabase> {
  // Step 1: Scan and reserve slots for types. Also do this for any scope, and so
  // on and so forth.
  // Step 2: Build definitions for all types, in this order: 0:zero_types
  // 1:arrays 2:structs 3:unions  4:routines

  // Step 3: Build routine bodies.

  // Step 4: Validate - All relaxed routines and types should have proper
  // semantic behavior. This step may be deferred until such a time when
  // either debug, reflection, or compiler information is needed for a given
  // type.

  let mut ty_db = Box::new(TypeDatabase::new());

  let routines = process_module_members(&module.members, &mut ty_db);

  for routine in &routines {
    // Gather function type information.
    process_routine(*routine, &mut ty_db);
  }

  ty_db
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

          RawBoundType(bound_type) => {
            let name = bound_type.name.id.intern();

            use type_Value::*;
            match &bound_type.ty {
              Type_Flag(flag_enum) => process_flag_enum(flag_enum, ty_db, name),
              Type_Enum(enumerator) => process_enum(enumerator, ty_db, name),
              Type_Array(array) => process_array(array, ty_db, name),
              Type_Struct(strct) => {
                ty_db.insert_type(name, StructType { name, members: Default::default(), size: 0, alignment: 0 }.into());

                println!("Declaring STRUCTURE [{scope_name:2}{name}]")
              }
              ty => println!("Declaring {ty:?} bound to {name}"),
            }
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

            ty_db.insert_type(name, ty.into());

            use routine_type_Value::*;
            match &routine.ty {
              RawFunctionType(..) => println!("Declaring FUNCTION  [{scope_name:2}{name}]"),
              RawProcedureType(..) => println!("Declaring PROCEDURE [{scope_name:2}{name}]",),

              _ => {}
            }
          }
          ty => unreachable!(":: Type not recognized {ty:#?}"),
        }
      }
      ty => unreachable!(": Type not recognized {ty:#?}"),
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

          RawBoundType(bound_type) => {
            let name = bound_type.name.id.intern();

            use type_Value::*;
            match &bound_type.ty {
              Type_Struct(strct) => {
                process_struct(strct, ty_db, name);
              }
              Type_Flag(flag_enum) => {}
              ty => println!("@ Type not recognized {ty:#?}"),
            }
          }
          RawRoutine(routine) => {
            process_routine_signature(routine, ty_db);
          }
          ty => unreachable!("& Type not recognized {ty:#?}"),
        }
      }
      ty => unreachable!("* Type not recognized {ty:#?}"),
    }
  }

  // Resolve super_type

  return routines;
}

fn process_array(array: &Type_Array<Token>, ty_db: &mut TypeDatabase, name: IString) {
  if let Some(base_type) = get_type(&array.base_type.clone().into(), ty_db, true) {
    let size = array.size as usize;

    let ty = ArrayType { name, element_type: base_type, size: size };

    ty_db.insert_type(name, Type::Array(ty));
  } else {
    panic!("Invalid type")
  }
}

fn process_enum(enumarator: &Type_Enum<Token>, ty_db: &mut TypeDatabase, name: IString) {
  if let Some(base_type) = get_type(&enumarator.base_type.clone().into(), ty_db, true) {
    if base_type.is_primitive() {
      let members = enumarator.values.iter().map(|v| v.name.id.intern()).collect();

      let ty = EnumType { name, base_type, members };

      ty_db.insert_type(name, Type::Enum(ty));
    } else {
      panic!("Invalid type")
    }
  } else {
    panic!("Invalid type")
  }
}

fn process_flag_enum(flag: &Type_Flag<Token>, ty_db: &mut TypeDatabase, name: IString) {
  let size = flag.flag_size;

  match size {
    8 | 16 | 32 | 64 => {}
    size => panic!("Flag enum size {size} not supported:\n{}", flag.tok.blame(1, 1, "must be one of 8 | 16 | 32 | 64", None)),
  }

  let members = flag.values.iter().map(|v| v.id.intern()).collect();

  let ty = FlagEnumType { name, bit_size: size as u64, members };

  ty_db.insert_type(name, Type::Flag(ty));
}

fn process_struct(strct: &Type_Struct<Token>, ty_db: &mut TypeDatabase, name: IString) {
  let Some((ty_ref, _)) = ty_db.get_type_mut(name) else {
    panic!("Could not find Struct type: {name}",);
  };

  match ty_ref {
    Type::Structure(st) => {
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
              match &prop.r#type {
                bitfield_element_group_Value::Discriminator(desc) => {
                  if bit_offset != 0 {
                    panic!("A discriminator must be the first element of a bitfield.")
                  }
                  let bit_size = desc.bit_count as u64;

                  debug_assert!(bit_size <= 128);

                  let ty = (RumType::Discriminant | RumType::new_bit_size(bit_size) | RumType::new_bitfield_data(bit_offset as u8, bit_field_size as u8));

                  let ty_name = format!("{ty}").intern();

                  st.members.push(StructMemberType { ty, original_index: index, name: "#desc".intern(), offset: prop_offset });

                  bit_offset += bit_size;
                }
                ty => { /*         use type_Value::*;
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
              }
            }

            if bit_offset > max_bit_size {
              panic!("Bitfield element size {bit_offset} overflow bitfield size {max_bit_size}")
            }
          }
          Property(raw_prop) => {
            if let Some(ty) = get_type(&raw_prop.ty, ty_db, false) {
              let name = raw_prop.name.id.intern();
              /*        let prop_offset = get_aligned_value(offset, ty.alignment() as u64);
              offset = prop_offset + ty.byte_size() as u64;

              min_alignment = min_alignment.max(ty.alignment() as u64);

              if st.members.iter().any(|m| m.name == name) {
                panic!("Name already taken {name:?}")z
              } */

              st.members.push(StructMemberType { ty, original_index: index, name, offset: 0 })
            } else {
              let name = get_type_name(&raw_prop.ty);
              panic!("Could not resolve type: {name}");
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

  let Some((ty_ref, _)) = ty_db.get_type_mut(name) else {
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

      let mut ib = IRBuilder::new(body);

      for (index, param) in params.params.iter().enumerate() {
        let param_type = &param.ty;
        let param_name = param.var.id.intern();

        let ty = if param_type.inferred {
          let gen_ty_name = get_type_name(&param.ty.ty);
          ib.get_generic_type(gen_ty_name)
        } else {
          get_type(&param_type.ty, ty_db, false).expect("Could not resolve parameter type")
        };

        let param_ty = ty.increment_pointer();
        let var = ib.declare_variable(param_name, param_ty, param.tok.clone(), PARAM_DECL).clone();

        ib.push_ssa(PARAM_VAL, ty.into(), &[], Default::default(), param.tok.clone());
        ib.push_ssa(STORE, param_ty.into(), &[var.declaration.into(), StackOp], var.id, param.tok.clone());

        parameters.push((param_name, index, ty, var.id, param.tok.clone()));
      }

      // Seal parameter scope.
      ib.body.ctx.ty_var_scopes.push_scope();

      let returns = match ret {
        Some(ret_type) => {
          let ty = if ret_type.inferred {
            let gen_ty_name = get_type_name(&ret_type.ty);
            ib.get_generic_type(gen_ty_name)
          } else {
            get_type(&ret_type.ty, ty_db, false).expect("Could not resolve return")
          };

          let var = ib.declare_variable(format!("ret_1").intern(), ty, ret_type.tok.clone(), VAR_DECL);
          vec![(ty.into(), ret_type.tok.clone(), var.id)]
        }
        _ => vec![],
      };

      // Seal the return signature scope.
      ib.body.ctx.ty_var_scopes.push_scope();

      rt.returns = returns;
    }
    _ => unreachable!(),
  }
}

fn process_routine(routine_name: IString, type_scope: &mut TypeDatabase) {
  let Some((ty_ref, _)) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find Struct type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      rt.body.resolved = false;

      let RoutineType { returns, body, ast, .. } = rt;

      let mut ib = IRBuilder::attach(body);

      process_expression(&ast.expression.expr, &mut ib);

      if returns.len() > 0 {
        /*         debug_assert_eq!(
          ib.ssa_stack.len(),
          returns.len(),
          "Invalid number of return expression {} in {routine_name}. Expected {} \n{ib:?} {:?}",
          ib.ssa_stack.len(),
          returns.len(),
          ib.ssa_stack
        ); */
        let mut i: usize = 0;
        while let Some(stack) = ib.pop_stack() {
          let (ty, t, var_id) = &returns[i];
          ib.push_ssa(RET_VAL, (*ty).into(), &[stack.into()], *var_id, t.clone());

          //ib.push_ssa(STORE, (*ty).into(), &[ib.body.ctx.vars[var_id.usize()].declaration.into(), stack.into()], *var_id, t.clone());
          break;
        }
        i += 1;
      }

      println!("{ib:?}");
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
    //expression_Value::RawStructInstantiation(struct_instance) => process_struct_instantiation(struct_instance, ib),
    expression_Value::MemberCompositeAccess(mem) => process_member_load(mem, ib),
    expression_Value::RawBlock(ast_block) => process_block(ast_block, ib),
    expression_Value::Add(add) => process_add(add, ib),
    expression_Value::Sub(sub) => process_sub(sub, ib),
    expression_Value::Mul(mul) => process_mul(mul, ib),
    expression_Value::Div(div) => process_div(div, ib),
    expression_Value::Pow(pow) => process_pow(pow, ib),
    expression_Value::BIT_SL(sl) => process_sl(sl, ib),
    expression_Value::BIT_SR(sr) => process_sr(sr, ib),
    expression_Value::BIT_OR(or) => process_or(or, ib),
    expression_Value::BIT_XOR(xor) => process_xor(xor, ib),
    expression_Value::BIT_AND(and) => process_and(and, ib),
    expression_Value::RawAggregateInstantiation(agg) => process_aggregate_instantiation(agg, ib),
    expression_Value::RawMatch(agg) => process_match(agg, ib),
    d => todo!("expression: {d:#?}"),
  }
}

fn resolve_variable(mem: &MemberCompositeAccess<Token>, ib: &mut IRBuilder) -> Option<Variable> {
  let base_var = &mem.root;

  let var_name = base_var.name.id.intern();

  if let Some(var_data) = ib.get_variable(var_name) {
    let mut var_data = var_data.clone();

    if mem.sub_members.len() == 0 {
      ib.push_node(var_data.declaration);
    }

    for sub_member in &mem.sub_members {
      ib.push_ssa(LOAD, var_data.ty.decrement_pointer().into(), &[var_data.declaration.into()], var_data.id, mem.tok.clone());

      use crate::parser::script_parser::member_group_Value::*;

      match sub_member {
        NamedMember(mem) => {
          let name = mem.name.id.intern();

          if let Some(var) = ib.get_member_var(var_data.id, MemberName::String(name)) {
            let mut var = var.clone();

            ib.push_ssa(MEMB_PTR_CALC, var.ty.into(), &[StackOp], var.id, mem.tok.clone());

            ib.body.ctx.vars[var.id.usize()] = var;

            var_data = var;
          } else {
            panic!("Could not resolve variable");
          }
        }
        IndexedMember(mem) => {
          use crate::parser::script_parser::pointer_offset_Value::*;
          match &mem.expression {
            RawInt(int) => {
              let index = int.val as isize;
              if index < 0 {
                panic!("{}", int.tok.blame(1, 1, "Sub-index literals not supported", Option::None));
              } else {
                if let Some(var) = ib.get_member_var(var_data.id, MemberName::Index(index as usize)) {
                  let mut var = var.clone();

                  ib.push_ssa(MEMB_PTR_CALC, var.ty.into(), &[StackOp], var.id, mem.tok.clone());

                  ib.body.ctx.vars[var.id.usize()] = var;

                  var_data = var;
                }
              }
            }
            expr => {
              let blame_string = mem.tok.blame(1, 1, "Index by expression not supported yet.", Option::None);
              panic!("\n{blame_string} ",)
            }
          }
        }
        _ => unreachable!(),
      }
    }

    Some(var_data)
  } else {
    Option::None
  }
}

fn process_member_load(mem: &MemberCompositeAccess<Token>, ib: &mut IRBuilder<'_>) {
  if let Some(var) = resolve_variable(mem, ib) {
    ib.push_ssa(LOAD, var.ty.decrement_pointer().into(), &[StackOp], VarId::NONE, mem.tok.clone());
  } else {
    // Create error for undefined variable.
  }
}

fn process_add(add: &Add<Token>, ib: &mut IRBuilder) {
  process_expression(&(add.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(add.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(ADD, Inherit, &[StackOp, StackOp], VarId::NONE, add.tok.clone());
}

fn process_sub(sub: &Sub<Token>, ib: &mut IRBuilder) {
  process_expression(&(sub.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sub.right.clone().to_ast().into_expression_Value().unwrap()), ib);

  ib.push_ssa(SUB, Inherit, &[StackOp, StackOp], VarId::NONE, sub.tok.clone());
}

fn process_mul(mul: &Mul<Token>, ib: &mut IRBuilder) {
  process_expression(&(mul.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(mul.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(MUL, Inherit, &[StackOp, StackOp], VarId::NONE, mul.tok.clone());
}

fn process_div(div: &Div<Token>, ib: &mut IRBuilder) {
  process_expression(&(div.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(div.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(DIV, Inherit, &[StackOp, StackOp], VarId::NONE, div.tok.clone());
}

fn process_pow(pow: &Pow<Token>, ib: &mut IRBuilder) {
  process_expression(&(pow.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(pow.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(POW, Inherit, &[StackOp, StackOp], VarId::NONE, pow.tok.clone());
}

fn process_sl(sl: &BIT_SL<Token>, ib: &mut IRBuilder) {
  process_expression(&(sl.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sl.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(SHL, Inherit, &[StackOp, StackOp], VarId::NONE, Default::default());
}

fn process_sr(sr: &BIT_SR<Token>, ib: &mut IRBuilder) {
  process_expression(&(sr.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(sr.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(SHR, Inherit, &[StackOp, StackOp], VarId::NONE, Default::default());
}

fn process_or(or: &BIT_OR<Token>, ib: &mut IRBuilder) {
  process_expression(&(or.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(or.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(OR, Inherit, &[StackOp, StackOp], VarId::NONE, Default::default());
}

fn process_xor(xor: &BIT_XOR<Token>, ib: &mut IRBuilder) {
  process_expression(&(xor.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(xor.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(XOR, Inherit, &[StackOp, StackOp], VarId::NONE, Default::default());
}

fn process_and(and: &BIT_AND<Token>, ib: &mut IRBuilder) {
  process_expression(&(and.left.clone().to_ast().into_expression_Value().unwrap()), ib);
  process_expression(&(and.right.clone().to_ast().into_expression_Value().unwrap()), ib);
  ib.push_ssa(AND, Inherit, &[StackOp, StackOp], VarId::NONE, Default::default());
}

fn process_const_number(num: &RawNum<Token>, ib: &mut IRBuilder) {
  let string_val = num.tok.to_string();

  ib.push_const(
    if string_val.contains(".") {
      ConstVal::new(RumType::Float | RumType::b64, num.val)
    } else {
      ConstVal::new(RumType::Unsigned | RumType::b64, string_val.parse::<u64>().unwrap())
    },
    num.tok.clone(),
  );
}

const SYS_CALL_TARGETS: [&'static str; 2] = ["_sys_allocate", "_sys_free"];
const DBG_CALL_TARGETS: [&'static str; 1] = ["_malloc"];

fn process_call(call_node: &RawCall<Token>, ib: &mut IRBuilder) {
  // Member call resolution may involve lookup on compatible functions in the
  // type scope stack.

  let name = {
    if call_node.member.sub_members.is_empty() {
      call_node.member.root.name.id.intern()
    } else {
      if let Some(var) = resolve_variable(&call_node.member, ib) {
        var.mem_name
      } else {
        panic!("Not a call")
      }
    }
  };

  let mut args = Vec::new();
  for arg in &call_node.args {
    process_expression(&arg.expr, ib);

    ib.push_ssa(CALL_ARG, Inherit, &[StackOp], VarId::NONE, arg.tok.clone());

    args.push(ib.pop_stack().unwrap());
  }

  let tok = &call_node.tok;

  if let Some(sys_call_name) = SYS_CALL_TARGETS.iter().find(|d| (**d).cmp(name.to_str().as_str()).is_eq()) {
    let call = sys_call_name.intern();
    let call_target_id = ib.body.ctx.db_mut().get_or_add_type_index(sys_call_name.intern(), Type::Syscall(call));
    let call_slot = RumType::Undefined.to_aggregate_id(call_target_id);

    ib.push_ssa(CALL, call_slot.into(), &[], VarId::NONE, tok.clone());
    ib.pop_stack();

    let var = ib.declare_variable(Default::default(), call_slot, tok.clone(), VAR_DECL).clone();
    ib.push_ssa(CALL_RET, var.ty.into(), &[], VarId::NONE, tok.clone());
  } else if let Some(sys_call_name) = DBG_CALL_TARGETS.iter().find(|d| (**d).cmp(name.to_str().as_str()).is_eq()) {
    let call_name = sys_call_name.intern();
    let call_target_id = ib.body.ctx.db_mut().get_or_add_type_index(call_name, Type::DebugCall(call_name));
    let call_slot = RumType::Undefined.to_aggregate_id(call_target_id);

    ib.push_ssa(DBG_CALL, call_slot.into(), &[], VarId::NONE, tok.clone());
    ib.pop_stack();

    ib.push_ssa(CALL_RET, RumType::u8.increment_pointer().into(), &[], VarId::NONE, tok.clone());
  } else {
    let call = if let Some((routine_entry, _)) = ib.body.ctx.db_mut().get_type_mut(name).as_ref() {
      match routine_entry {
        Type::Routine(routine) => {
          if routine.body.resolved {
            todo!("call resolved routine")
            //ib.push_var(routine.name, routine_entry);
          } else {
            let mut _new_name: String = routine.name.to_string();

            let mut routine_body = RoutineBody::new(&mut ib.body.ctx.db_mut());

            let mut _new_returns = Default::default();

            if let Some((ty, var)) = ib.body.ctx.get_type(name) {
              ib.body.ctx.insert_new_var(name, ty).clone()
            } else {
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

              println!("{name}");

              ib.body.ctx.insert_new_var(name, RumType::Undefined.to_aggregate_id(entry)).clone()
            }
          }
        }
        _ => panic!("Invalid type for calling"),
      }
    } else {
      todo!("Report call resolution error")
      //panic!("Could not find routine {}", call_name.to_str().as_str())
    };
    ib.push_ssa(CALL, call.ty.into(), &[], VarId::NONE, Default::default());
    ib.pop_stack();
  }
}

fn process_aggregate_instantiation(struct_decl: &RawAggregateInstantiation<Token>, ib: &mut IRBuilder) {
  // note(Anthony): All struct instantiations should be on known types. If the struct to be instantiate is incomplete, this should signal that
  // downstream processing of this routine should not continue.

  //let struct_type_name = struct_decl.name.id.intern();

  //  if let Some((struct_entry, struct_slot)) = ib.body.ctx.db_mut().get_type_mut(struct_type_name).as_ref() {

  //if let Type::Structure(StructType { name, members, size, alignment }) = &struct_entry {
  //let base_type = *struct_slot;

  let name = format!("TEMP_{}", VarId::new(ib.body.graph.len() as u32));

  let gen_ty = ib.get_generic_type(Default::default());
  let mut agg_var = ib.declare_variable(name.intern(), gen_ty, struct_decl.tok.clone(), IROp::AGG_DECL).clone();

  struct StructEntry {
    value: IRGraphId,
    name:  MemberName,
    tok:   Token,
  }

  let mut value_maps = BTreeMap::<MemberName, StructEntry>::new();

  for (index, init_expression) in struct_decl.inits.iter().enumerate() {
    let member_name = init_expression.name.id.intern();
    process_expression(&init_expression.expression.expr, ib);

    let name = if member_name.is_empty() { MemberName::Index(index) } else { MemberName::String(member_name) };

    value_maps.insert(name, StructEntry { value: ib.pop_stack().unwrap(), name, tok: init_expression.tok.clone() });

    // Calculate the offset to the member within the struct
  }

  ib.body.ctx.vars[agg_var.id] = agg_var;

  let par_id = agg_var.id;

  for (_, StructEntry { value, name, tok }) in value_maps {
    if let Some(mut var) = ib.get_member_var(par_id, name).cloned() {
      ib.push_ssa(MEMB_PTR_CALC, var.ty.into(), &[SMO::IROp(agg_var.declaration)], var.id, tok.clone());
      var.declaration = ib.get_top_id().unwrap();
      ib.body.ctx.vars[var.id] = var;

      ib.push_ssa(STORE, var.ty.into(), &[StackOp, value.into()], var.id, tok);
      ib.pop_stack();
    } else {
      todo!("TBD");
    }
  }

  ib.push_node(agg_var.declaration);
  // }
  /*   } else {
    panic!("Could not find struct definition for {struct_type_name:?}")
  } */
}

fn process_block(ast_block: &RawBlock<Token>, ib: &mut IRBuilder) {
  ib.body.ctx.push_scope();

  let len = ast_block.statements.len();
  let last_index = len - 1;

  for (i, stmt) in ast_block.statements.iter().enumerate() {
    process_statement(stmt, ib);

    if i < last_index {
      ib.pop_stack().unwrap();
    }
  }

  use block_expression_group_3_Value::*;

  match &ast_block.exit {
    None => {}
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
    }
    RawYield(yield_) => {
      if let Some(IterStack { body_block, output_vars }) = ib.get_iter_stack().cloned() {
        {
          let var_id = output_vars[0];
          let mut var = ib.body.ctx.vars[var_id].clone();

          match &yield_.expr.expr {
            expression_Value::MemberCompositeAccess(mem) => {
              if let Some(temp_var) = resolve_variable(mem, ib) {
                ib.push_ssa(ASSIGN, var.ty.into(), &[var.declaration.into(), StackOp], var.id, mem.tok.clone());
              /*   if var.ty.is_generic() {
                var.ty = temp_var.ty;
                ib.set_variable(var);
              } */
              } else {
                unreachable!();
              }
            }
            other => {
              process_expression(other, ib);
              ib.push_ssa(STORE, Inherit, &[StackOp], var.id, yield_.tok.clone());
            }
          }

          let expr_id = ib.pop_stack().unwrap();

          //ib.pop_stack().unwrap();
        }
        ib.set_successor(body_block, SuccessorMode::Default)
      }
    }
    d => {
      todo!("process block exit: {d:#?}");
    }
  };

  ib.body.ctx.pop_scope();
}

fn process_statement(stmt: &statement_Value<Token>, ib: &mut IRBuilder) {
  use statement_Value::*;
  match stmt {
    RawAssignment(assign) => process_assign_statement(assign, ib),
    Expression(expr) => process_expression_statement(expr, ib),
    RawLoop(loop_) => process_loop(loop_, ib),
    IterReentrance(iter) => process_iter_block(iter, ib),
    d => todo!("process statement: {d:#?}"),
  }
}

fn process_match(match_: &RawMatch<Token>, ib: &mut IRBuilder<'_>) {
  process_expression(&match_.expression.clone().into(), ib);

  let expr_id = ib.pop_stack().unwrap();
  let end = ib.create_block();

  //let var = ib.declare_generic(format!("var::{}", match_.tok.get_start()).intern(), Default::default(), match_.tok.clone()).clone();

  // ib.push_ssa(MATCH_LOC, var.ty.decrement_pointer().into(), &[], match_.tok.clone());
  //
  // // var.declaration = ib.pop_stack().unwrap();
  // ib.set_variable(var);

  //ib.push_ssa(MEMB_PTR_CALC, SMT::Data(TyData::Var(1, var.id.into())), &[], match_.tok.clone());
  //let store_target = ib.pop_stack().unwrap();
  //let store_target = var.declaration;

  for clause in &match_.clauses {
    if clause.default {
      process_block(&clause.scope, ib);

      let expression_result = ib.pop_stack().unwrap();

      if !expression_result.is_invalid() {
        //  ib.push_ssa(STORE, Inherit, &[store_target.into(), expression_result.into()], var.id, clause.expr.tok.clone());
        //  ib.pop_stack();
      }

      break;
    } else {
      process_expression(&clause.expr.expr.clone().to_ast().into_expression_Value().unwrap(), ib);
      let op = match clause.expr.op.as_str() {
        "<" => LS,
        ">" => GR,
        "<=" => LE,
        ">=" => GE,
        "==" => EQ,
        "!=" => NE,
        _ => unreachable!(),
      };

      ib.push_ssa(op, SMT::Undef, &[expr_id.into(), StackOp], VarId::NONE, clause.expr.tok.clone());
      ib.pop_stack();

      let (succeed, failed) = ib.create_branch();

      ib.set_active(succeed);

      process_block(&clause.scope, ib);

      let expression_result = ib.pop_stack().unwrap();

      if !expression_result.is_invalid() {
        // ib.push_ssa(STORE, Inherit, &[store_target.into(), expression_result.into()], var.id, clause.expr.tok.clone());
        // ib.pop_stack();
      }

      ib.set_successor(end, SuccessorMode::Default);
      ib.set_active(failed);
    }
  }

  ib.set_successor(end, SuccessorMode::Default);
  ib.set_active(end);

  //ib.push_node(store_target);
}

fn process_loop(loop_: &RawLoop<Token>, ib: &mut IRBuilder<'_>) {
  match &loop_.scope {
    loop_statement_group_1_Value::RawIterStatement(iter_stmt) => process_iter_loop(iter_stmt, loop_.label.id.intern(), ib),
    _ => {
      let (loop_head, loop_exit) = ib.push_loop(loop_.label.id.intern());
      ib.body.blocks[loop_head].name = "LOOP_HEAD".intern();

      match &loop_.scope {
        loop_statement_group_1_Value::RawBlock(block) => process_block(block, ib),
        loop_statement_group_1_Value::RawMatch(match_) => process_match(match_, ib),
        _ => unreachable!(),
      }

      ib.set_successor(loop_head, SuccessorMode::Default);
      ib.set_successor(loop_exit, SuccessorMode::Fail);
      ib.pop_loop();
    }
  }
}

fn process_iter_block(iter: &IterReentrance<Token>, ib: &mut IRBuilder<'_>) {
  let IterReentrance { tok, expr } = iter;

  process_expression(&expr.clone().to_ast().into_expression_Value().unwrap(), ib);
}

fn process_iter_loop(iter_stmt: &RawIterStatement<Token>, loop_name: IString, ib: &mut IRBuilder<'_>) {
  let RawIterStatement { var, iter, block, tok } = iter_stmt;
  ib.push_lexical_scope();
  // Create variables for the yield

  // lookup loop expression

  let iter_preamble = ib.create_block();
  ib.body.blocks[iter_preamble].name = "ITER PREAMBLE".intern();
  ib.set_successor(iter_preamble, SuccessorMode::Default);
  ib.set_active(iter_preamble);

  let call_name = if iter.member.sub_members.is_empty() {
    iter.member.root.name.id.intern()
  } else {
    panic!("Need to handle member based calls");
  };

  let var_name = iter_stmt.var.id.intern();
  let gen_ty = ib.get_generic_type(Default::default());
  let mut var = ib.declare_variable(var_name, gen_ty, iter_stmt.var.tok.clone(), VAR_DECL).clone();

  if let Some((routine_entry, _)) = ib.body.ctx.db_mut().get_type_mut(call_name).as_ref() {
    match routine_entry {
      Type::Routine(routine) => {
        // Map the routines parameters to the expression inputs.
        for (arg, (param_name, _, param_ty, _, param_token)) in iter.args.iter().zip(routine.parameters.iter()) {
          // both the arg and the param will be mapped to the same variable
          process_expression(&arg.expr, ib);

          let val = ib.pop_stack().unwrap();
          let ty = ib.get_node(val.usize()).ty();
          let var = ib.declare_variable(*param_name, ty.increment_pointer(), param_token.clone(), VAR_DECL).clone();

          ib.push_ssa(STORE, var.ty.into(), &[var.declaration.into(), val.into()], var.id, param_token.clone());
        }

        ib.push_lexical_scope();
        match &routine.ast.expression.expr {
          expression_Value::RawBlock(iter_setup_block) => {
            for statement in &iter_setup_block.statements {
              match statement {
                statement_Value::IterReentrance(iter) => {
                  //create the variable for the loop

                  let iter_body = ib.create_block();
                  ib.body.blocks[iter_body].name = "ITER BODY".intern();
                  //ib.set_successor(iter_preamble, SuccessorMode::Default);

                  ib.push_iter_var_stack(IterStack { body_block: iter_body, output_vars: vec![var.id] });

                  let (loop_head, loop_exit) = ib.push_loop(loop_name);

                  process_expression(&iter.expr.clone().to_ast().into_expression_Value().unwrap(), ib);

                  ib.set_successor(loop_exit, SuccessorMode::Default);

                  ib.set_active(iter_body);

                  process_block(block, ib);

                  ib.set_successor(loop_head, SuccessorMode::Default);

                  ib.pop_loop();
                  break;
                }
                stmt => process_statement(stmt, ib),
              }
            }
          }
          _ => panic!("Invalid expression for an iterator"),
        }
        ib.pop_lexical_scope()
      }
      _ => {
        panic!("Could not find routine type!");
      }
    }
  };
  ib.pop_lexical_scope()
}

fn process_expression_statement(expr: &Expression<Token>, ib: &mut IRBuilder<'_>) {
  process_expression(&expr.expr, ib);
}

fn process_assign_statement(assign: &RawAssignment<Token>, ib: &mut IRBuilder<'_>) {
  let db = ib.body.ctx.db();
  // Process assignments.

  process_expression(&assign.expression.expr, ib);

  // Process assignment targets.
  let (variable, expr_id) = (&assign.var, ib.pop_stack().unwrap());

  use assignment_var_Value::*;
  match variable {
    MemberCompositeAccess(var_assign) => {
      if let Some(var) = resolve_variable(&var_assign, ib) {
        ib.push_ssa(STORE, var.ty.into(), &[var.declaration.into(), expr_id.into()], var.id, var_assign.tok.clone());
        ib.pop_stack().unwrap();
      } else if var_assign.sub_members.len() == 0 {
        // Create and assign a new variable based on the expression.
        let var_name = var_assign.root.name.id.intern();

        if let Some(var) = ib.get_node_variable(expr_id).cloned() {
          if let Some(Type::Structure(..)) = var.clone().ty.aggregate(db) {
            ib.body.ctx.rename_var(var.mem_name, var_name);
            return;
          }
        }

        let mut var = ib.declare_variable(var_name, ib.get_node(expr_id.usize()).ty().increment_pointer(), var_assign.tok.clone(), VAR_DECL).clone();

        ib.push_ssa(STORE, var.ty.into(), &[var.declaration.into(), expr_id.into()], var.id, var_assign.tok.clone());
        ib.pop_stack().unwrap();
      } else {
        let blame_string = var_assign.tok.blame(1, 1, "could not find variable @", Option::None);
        panic!("\n{blame_string} \n{ib:?}",)
      }
    }
    PointerCastToAddress(cast) => {
      todo!("process PointerCastToAddress::");
    }
    RawAssignmentDeclaration(var_decl) => {
      let var_name = var_decl.var.id.intern();
      if let Some(mut ty) = get_type(&var_decl.ty, &mut ib.body.ctx.db_mut(), false) {
        /*     if let Some(mut var) = ib.get_node_variable(expr_id).cloned() {
          var.ty = ty;
          ib.set_variable(var);
          ib.body.ctx.alias_variable(&var, var_name);
        } else { */
        if ty.is_aggregate() {
          ty = ty.to_ptr_depth(2);
        } else {
          ty = ty.to_ptr_depth(1);
        }

        let var = ib.declare_variable(var_name, ty, var_decl.tok.clone(), VAR_DECL).clone();
        ib.push_ssa(STORE, var.ty.into(), &[var.declaration.into(), expr_id.into()], var.id, var_decl.tok.clone());
        ib.pop_stack().unwrap();
        //}
      } else {
        todo!("Handle no type")
      }
    }
    _ => unreachable!(),
  }

  ib.push_node(IRGraphId::INVALID);

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

pub fn get_type(ir_type: &type_Value<Token>, type_db: &mut TypeDatabase, insert_unresolved: bool) -> Option<RumType> {
  use type_Value::*;
  match ir_type {
    Type_Flag(_) => Some(RumType::Flag),
    Type_u8(_) => Some(RumType::u8),
    Type_u16(_) => Some(RumType::u16),
    Type_u32(_) => Some(RumType::u32),
    Type_u64(_) => Some(RumType::u64),
    Type_i8(_) => Some(RumType::i8),
    Type_i16(_) => Some(RumType::i16),
    Type_i32(_) => Some(RumType::i32),
    Type_i64(_) => Some(RumType::i64),
    Type_f32(_) => Some(RumType::f32),
    Type_f64(_) => Some(RumType::f64),
    Type_f32v2(_) => Some(RumType::f32v2),
    Type_f32v4(_) => Some(RumType::f32v4),
    Type_f64v2(_) => Some(RumType::f64v2),
    Type_f64v4(_) => Some(RumType::f64v4),

    Type_Reference(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), type_db, insert_unresolved) {
        Some(base_type.increment_pointer())
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), type_db, insert_unresolved) {
        use lifetime_Value::*;
        match &ptr.ptr_type {
          GlobalLifetime(_) => {
            Some(base_type.increment_pointer())
            //Some(TypeSlot::GlobalIndex(0, type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), 0, base_type)) as u32))
          }
          ScopedLifetime(scope)  =>

          Some(base_type.increment_pointer())/* Some(TypeSlot::GlobalIndex(
            0,
            type_db.get_or_add_type_index(format!("{}*{}", scope.val, base_type.ty_gb(type_db)).intern(), Type::Pointer(scope.val.intern(), 0, base_type)) as u32,
          )) */,
          _ => unreachable!(),
        }
      } else {
        Option::None
      }
    }
    Type_Variable(type_var) => {
      if let Some(ty) = type_db.get_type_index(type_var.name.id.intern()) {
        Some(RumType::Undefined.to_aggregate_id(ty))
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
