#![allow(unused)]

use super::{ir_const_val::ConstVal, ir_types::*};
use crate::compiler::{
  self,
  interpreter::{
    error::RumResult,
    raw::ir::ir_types::{RawType, RawVal, TypeInfo},
  },
  script_parser::*,
};
use num_traits::{Num, NumCast};
use radlr_rust_runtime::types::Token;
use rum_container::ArrayVec;
use rum_istring::{CachedString, IString};
use rum_logger::todo_note;
use std::{
  collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
  default,
  fmt::{Debug, Write},
  process::Output,
};

pub fn compile_function_blocks(funct: &RawFunction<Token>) -> RumResult<SSAFunction> {
  let mut ctx = IRContextBuilder::default();
  let root_block = ctx.push_block(None);
  root_block.inner.name = "entry".to_token();

  if let Err(err) = process_params(&funct.params, root_block) {
    panic!("{err:>?}");
  }

  let mut leaf_blocks = vec![];

  let active_block =
    match process_block(&funct.block, root_block, BlockId::default(), &mut leaf_blocks) {
      Err(err) => {
        panic!("failed {err:?}");
      }
      Ok(Some(block)) => block,
      _ => unreachable!(),
    };

  for leaf_block_id in leaf_blocks {
    let leaf_block = unsafe { ctx.blocks[leaf_block_id].as_mut() }.unwrap();
    leaf_block.inner.branch_unconditional = Some(active_block.inner.id);
  }

  match &funct.block.exit {
    block_statement_group_1_Value::None => {
      active_block.push_zero_op(IROp::RET_VAL, Default::default());
    }
    block_statement_group_1_Value::RawBreak(_) => {
      panic!("Invalid break!");
    }
    block_statement_group_1_Value::BlockExitExpressions(exit_expr) => {
      let expressions = &exit_expr.expressions;
      let return_values = &funct.return_types;

      assert_eq!(expressions.len(), return_values.len(), "mismatched return types!");

      let block = active_block.create_successor();
      active_block.inner.branch_unconditional = Some(block.inner.id);

      for (expression, return_val) in expressions.iter().zip(return_values.iter()) {
        let ty = ast_ty_to_ssa_ty(&return_val);
        let val = process_arithmetic_expression(&expression, block, ty, true)?;

        if let IRGraphNode::SSA { out_ty, .. } = ctx.graph[val.graph_id()] {
          block.push_unary_op(IROp::RET_VAL, out_ty, val);
        } else {
          panic!("Invalid operation")
        }
      }
    }
  }

  let funct = SSAFunction {
    blocks:    ctx
      .blocks
      .into_iter()
      .map(|b| unsafe {
        {
          let b = Box::from_raw(b);
          b.inner
        }
      })
      .collect(),
    variables: ctx.variables,
    graph:     ctx.graph,
    calls:     ctx.calls,
  };

  Ok(funct)
}

fn process_params(
  params: &Vec<Box<RawParamBinding<Token>>>,
  block: &mut IRBlockConstructor,
) -> RumResult<()> {
  for (index, param) in params.iter().enumerate() {
    create_binding(&param.var, &param.ty, &param.tok, block)?;
  }

  Ok(())
}

fn process_block<'a>(
  ast_block: &RawBlock<Token>,
  block: &'a mut IRBlockConstructor,
  scope_block: BlockId,
  leaf_blocks: &mut Vec<BlockId>,
) -> RumResult<Option<&'a mut IRBlockConstructor>> {
  let mut block = block;

  for stmt in &ast_block.statements {
    if let Some(new_block) = process_statements(stmt, block, scope_block, leaf_blocks)? {
      block = new_block
    } else {
      return Ok(None);
    }
  }

  match &ast_block.exit {
    block_statement_group_1_Value::RawBreak(t) => {
      block.break_id = Some("".to_token());
      leaf_blocks.push(block.inner.id);
      return Ok(None);
    }
    _ => {}
  }

  Ok(Some(block))
}

/* fn process_assignment(
  block: &mut IRBlockConstructor,
  location: &assignment_Value<Token>,
  expression: &arithmetic_Value<Token>,
  tok: &Token,
) -> RumResult<GraphId> {
  match location {
    assignment_Value::Id(id) => {
      let name = id.id.intern();

      if let Some(((decl, ty))) = block.get_binding(name, true) {
        let decl = decl.clone();
        let value = process_arithmetic_expression(&expression, block, ty.unstacked(), false)?;

        block.debug_op(tok.clone());

        Ok(block.push_store(ty, decl, value))
      } else {
        panic!("{}", id.tok.blame(1, 1, "not found", Option::None))
      }
    }

    assignment_Value::RawMemLocation(location) => {
      let base_ptr = resolve_base_ptr(&location.base_ptr, block);
      let offset = resolve_mem_offset(&location.expression, block);

      let mut mem_ptr_ty = block.get_type(base_ptr);
      let base_type = mem_ptr_ty.deref();
      let mem_ptr_ty = mem_ptr_ty.mask_out_elements() | TypeInfo::unknown_ele_count();

      let expression = process_arithmetic_expression(&expression, block, base_type, false)?;

      let (temp_ptr, temp_ty) = if !offset.is_invalid() {
        let element_byte_size = base_type.ele_byte_size() as u32;
        let (temp_ptr, temp_ty) = block.create_anonymous_binding(mem_ptr_ty, tok.clone())?;
        let temp_ptr = block.push_unary_op(IROp::V_DEF, temp_ty, base_ptr);

        let multiple = block
          .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(element_byte_size));

        let offset =
          block.push_binary_op(IROp::MUL, TypeInfo::Integer | TypeInfo::b64, offset, multiple);

        (block.push_binary_op(IROp::ADD, temp_ty.mask_out_var_id(), temp_ptr, offset), temp_ty)
      } else {
        (base_ptr, mem_ptr_ty)
      };

      block.debug_op(tok.clone());
      Ok(block.push_binary_op(
        IROp::MEM_STORE,
        temp_ty.mask_out_var_id().deref(),
        temp_ptr,
        expression,
      ))
    }
    _ => unreachable!(),
  }
} */

/* fn create_allocation(
  block: &mut IRBlockConstructor,
  binding: &MemBinding<Token>,
  is_heap: bool,
  base_ty: TypeInfo,
) -> GraphId {
  match binding.byte_count {
    mem_binding_group_Value::RawUint(int) => {
      let val = int.val * base_ty.deref().ele_byte_size() as i64;

      let constant = if (64 - val.leading_zeros()).div_ceil(8) <= 4 {
        ConstVal::new(TypeInfo::Unsigned | TypeInfo::b32).store(val as i64)
      } else {
        ConstVal::new(TypeInfo::Unsigned | TypeInfo::b64).store(val as i64)
      };

      if val < u16::MAX as i64 {
        base_ty |= TypeInfo::elements(val as u16);
      } else {
        base_ty |= TypeInfo::unknown_ele_count();
      }

      if is_heap {
        let arg = block.push_constant(constant);
        //block.push_binary_op(SSAOp::MALLOC, ty, target, arg);

        let malloc =
          block.push_call(base_ty.unstacked(), "malloc".intern(), ArrayVec::from_iter([arg]));

        let def = block.push_unary_op(IROp::V_DEF, base_ty, malloc);

        let name = block.binding_name(base_ty.var_id().unwrap()).unwrap();

        block.scope_ssa.insert(name, (def, base_ty));
      }
    }

    mem_binding_group_Value::Var(id) => {
      base_ty |= TypeInfo::unknown_ele_count();

      if let Some(((ptr_id, ptr_ty))) = block.get_binding(id.id.to_token(), true) {
        if is_heap {
          block.debug_op(id.tok.clone());
          let id = block.push_call(ptr_ty, "malloc".intern(), ArrayVec::from_iter([ptr_id]));
          let name = block.binding_name(ptr_ty.var_id().unwrap()).unwrap();
          block.scope_ssa.insert(name, (id, base_ty));
        }
      } else {
        panic!()
      }
    }

    val => unreachable!("{val:?}"),
  };

  GraphId::default()
}*/

fn process_statements<'a>(
  statement: &statement_Value<Token>,
  block: &'a mut IRBlockConstructor,
  scope_block: BlockId,
  leaf_blocks: &mut Vec<BlockId>,
) -> RumResult<Option<&'a mut IRBlockConstructor>> {
  use IROp::*;

  match statement {
    statement_Value::RawAssignment(assignment) => {
      let vars = &assignment.vars;

      let expressions = &assignment.expressions;

      let mut processed_expressions =
        ArrayVec::<32, GraphId>::from_iter(expressions.iter().map(|_| GraphId::default()));

      for expression in expressions {
        match expression {
          assignment_statement_list_1_Value::AssignmentExprVal(expr) => {
            let value =
              process_arithmetic_expression(&expr.expr, block, TypeInfo::default(), false)?;
          }
          assignment_statement_list_1_Value::MemBinding(mem_binding) => mem_binding.byte_count,
          _ => unreachable!(),
        }
      }

      for (index, var) in vars.iter().enumerate() {
        let var_name = var.var.id.intern();
        let tok: &Token = &var.tok;

        match var.ty {
          assignment_var_Value::None => {
            if let Some(((decl, ty))) = block.get_binding(var_name, true) {
              let graph_id = &mut processed_expressions[index];
              let expression = &expressions[index];

              let decl = decl.clone();

              let value = process_arithmetic_expression(&expression, block, ty.unstacked(), false)?;

              block.debug_op(tok.clone());

              block.push_store(ty, decl, value);
            } else {
              panic!("{}", tok.blame(1, 1, "not found", Option::None))
            }
            // use existing variable
          }
          ty => {
            process_ty
            // New declaration for this variable
          }
        }
      }

      return Ok(Some(block));
    }
    /*     statement_Value::RawPtrDeclaration(ptr_decl) => {
      create_binding(&ptr_decl.id, &(ptr_decl.ty.clone().into()), &ptr_decl.tok, block)?;
      let target = &ptr_decl.id;
      let target_name = target.id.to_token();

      create_allocation(
        block,
        target_name,
        ptr_decl.expression.heap,
        &ptr_decl.expression.byte_count,
      );
    }
    statement_Value::RawPrimitiveDeclaration(prim_decl) => {
      create_binding(&prim_decl.id, &(prim_decl.ty.clone().into()), &prim_decl.tok, block)?;
      let target = &prim_decl.id;
      let target_name = target.id.to_token();
      let location = (prim_decl.id.clone().into());

      match &prim_decl.expression {
        arithmetic_Value::None => {}
        expression => {
          process_assignment(block, &location, &expression, &prim_decl.tok)?;
        }
      }
    }
    /// Binds a variable to a type.
    statement_Value::RawAssign(assign) => {
      process_assignment(block, &assign.location, &assign.expression, &assign.tok)?;
    } */
    statement_Value::RawLoop(loop_) => {
      todo!("Test");
      /*       let loop_head = block.create_successor();
      let loop_head_id = loop_head.inner.id;
      block.inner.branch_unconditional = Some(loop_head_id);
      loop_head.inner.name = "l_head".to_token();

      let mut new_leaf_blocks = vec![];

      match &loop_.scope {
        loop_statement_group_Value::RawBlock(block) => {
          if let Some(block) = process_block(block, loop_head, loop_head_id, &mut new_leaf_blocks)?
          {
            new_leaf_blocks.push(block.inner.id);
          }
        }
        loop_statement_group_Value::RawMatch(m) => {
          let block = process_match(m, loop_head, loop_head_id, &mut new_leaf_blocks)?;
          new_leaf_blocks.push(block.inner.id);
        }
        _ => unreachable!(),
      }

      let loop_exit = block.create_successor();
      let id = loop_exit.inner.id;
      loop_exit.inner.name = "l_exit".to_token();

      let blocks = &mut block.ctx().blocks;

      for leaf_block_id in new_leaf_blocks {
        let leaf_block = unsafe { blocks[leaf_block_id].as_mut() }.unwrap();
        if leaf_block.break_id == Some("".to_token()) {
          leaf_blocks.push(leaf_block_id);
        } else {
          leaf_block.inner.branch_unconditional = Some(loop_head_id);
        }
      }

      return Ok(Some(loop_exit)); */
    }
    statement_Value::RawMatch(m) => {
      return Ok(Some(process_match(m, block, scope_block, leaf_blocks)?))
    }
    val => {
      todo!("Process statement {val:#?} {block:?}")
    }
  }

  Ok(Some(block))
}

fn process_match<'a>(
  m: &Box<RawMatch<Token>>,
  block: &mut IRBlockConstructor,
  scope_block: BlockId,
  leaf_blocks: &mut Vec<BlockId>,
) -> RumResult<&'a mut IRBlockConstructor> {
  let expression = process_arithmetic_expression(&m.expression, block, Default::default(), true)?;

  todo!("build match expressions")

  /*   match  {
    LogicalExprType::Arithmatic(op_arg, _) => {
      todo!("Arithmetic based matching")
    }
    LogicalExprType::Boolean(bool_block) => {
      let mut default_case = None;
      let mut bool_success_case = None;

      for match_block in &m.statements {
        match match_block {
          match_statement_Value::DefaultMatchCase(block) => {
            let start_block = bool_block.create_successor();
            let start_block_id = start_block.inner.id;

            let end_block = process_block(&block.block, start_block, scope_block, leaf_blocks)?;

            default_case = Some((start_block_id, end_block));
          }

          match_statement_Value::RawMatchCase(case) => {
            let start_block = bool_block.create_successor();
            let start_block_id = start_block.inner.id;

            let end_block = process_block(&case.block, start_block, scope_block, leaf_blocks)?;

            match &case.val {
              compiler::script_parser::match_case_Value::RawFalse(_) => {
                default_case = Some((start_block_id, end_block));
              }
              compiler::script_parser::match_case_Value::RawTrue(_) => {
                bool_success_case = Some((start_block_id, end_block));
              }
              compiler::script_parser::match_case_Value::RawNum(val) => {
                panic!("Incorrect expression for logical type");
              }
              _ => unreachable!(),
            }
          }
          _ => unreachable!(),
        }
      }

      let join_block = bool_block.create_successor();
      join_block.inner.name = "m_join".to_token();

      if let Some((start_block_id, end_block)) = default_case {
        bool_block.inner.branch_default = Some(start_block_id);
        if let Some(block) = end_block {
          block.inner.branch_unconditional = Some(join_block.inner.id);
        }
      } else {
        bool_block.inner.branch_default = Some(join_block.inner.id);
      }

      if let Some((start_block_id, end_block)) = bool_success_case {
        bool_block.inner.branch_succeed = Some(start_block_id);
        if let Some(block) = end_block {
          block.inner.branch_unconditional = Some(join_block.inner.id);
        }
      } else {
        bool_block.inner.branch_succeed = Some(join_block.inner.id);
      }

      return Ok(join_block);
    }
  } */
}

fn resolve_base_ptr(
  base_ptr: &pointer_offset_group_Value<Token>,
  block: &mut IRBlockConstructor,
) -> GraphId {
  match base_ptr {
    pointer_offset_group_Value::Var(id) => {
      if let Some((decl, _)) = block.get_binding(id.id.to_token(), true) {
        decl
      } else {
        panic!("Couldn't find base pointer");
      }
    }
    pointer_offset_group_Value::RawPointerCast(cast) => todo!(),
    pointer_offset_group_Value::None => unreachable!(),
  }
}

fn resolve_mem_offset(expr: &mem_expr_Value<Token>, block: &mut IRBlockConstructor) -> GraphId {
  match expr {
    mem_expr_Value::Var(id) => {
      if let Some((val, _)) = block.get_binding(id.id.to_token(), true) {
        let op = resolve_base_ptr(&id.clone().into(), block);
        block.debug_op(id.tok.clone());
        val
      } else {
        panic!()
      }
    }
    mem_expr_Value::RawInt(int) => block
      .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store::<u64>(int.val as u64)),
    mem_expr_Value::RawMemAdd(add) => {
      let left = resolve_mem_offset(&add.left, block);
      let right = resolve_mem_offset(&add.right, block);

      block.debug_op(add.tok.clone());
      block.push_binary_op(IROp::ADD, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::RawMemMul(mul) => {
      let left = resolve_mem_offset(&mul.left, block);
      let right = resolve_mem_offset(&mul.right, block);
      block.debug_op(mul.tok.clone());
      block.push_binary_op(IROp::MUL, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::None => GraphId::INVALID,
    ast => unreachable!("{ast:?}"),
  }
}

enum LogicalExprType<'a> {
  /// Index of the boolean expression block.
  Boolean(&'a mut IRBlockConstructor),
  Arithmatic(GraphId, &'a mut IRBlockConstructor),
}

impl<'a> LogicalExprType<'a> {
  pub fn map_arith_result(
    result: RumResult<GraphId>,
    block: &'a mut IRBlockConstructor,
  ) -> RumResult<Self> {
    match result {
      Err(err) => Err(err),
      Ok(op_arg) => Ok(Self::Arithmatic(op_arg, block)),
    }
  }
}

macro_rules! boolean_op {
  ($name: ident, $node_type:ty, $ir_op: expr) => {
    fn $name<'a>(
      val: &$node_type,
      block: &'a mut IRBlockConstructor,
      e_val: TypeInfo,
    ) -> RumResult<LogicalExprType<'a>> {
      let left = process_arithmetic_expression(&val.left, block, e_val, false)?;
      let right = process_arithmetic_expression(&val.right, block, e_val, false)?;

      let l_val = block.get_type(left);
      let r_val = block.get_type(right);

      block.debug_op(val.tok.clone());
      block.push_binary_op($ir_op, l_val, left, right);

      Ok(LogicalExprType::Boolean(block))
    }
  };
}

/* boolean_op!(handle_ge, compiler::script_parser::GE<Token>, IROp::GE);
boolean_op!(handle_gr, compiler::script_parser::GR<Token>, IROp::GR);
boolean_op!(handle_le, compiler::script_parser::LE<Token>, IROp::LE);
boolean_op!(handle_ls, compiler::script_parser::LS<Token>, IROp::LS);
boolean_op!(handle_eq, compiler::script_parser::EQ<Token>, IROp::EQ);
boolean_op!(handle_ne, compiler::script_parser::NE<Token>, IROp::NE); */

macro_rules! arithmetic_op {
  ($name: ident, $node_type:ty, $ir_op: expr) => {
    fn $name<'a>(
      node: &$node_type,
      block: &mut IRBlockConstructor,
      e_val: TypeInfo,
      ret_val: bool,
    ) -> RumResult<GraphId> {
      let left = process_arithmetic_expression(&node.left, block, e_val, ret_val)?;
      let right = process_arithmetic_expression(&node.right, block, e_val, ret_val)?;
      //let right = convert_val(right, left.ll_val().info, block, ret_val);

      let l_val = block.get_type(left);
      let r_val = block.get_type(right);

      block.debug_op(node.tok.clone());
      Ok(block.push_binary_op($ir_op, l_val, left, right))
    }
  };
}

arithmetic_op!(handle_sub, compiler::script_parser::Sub<Token>, IROp::SUB);
arithmetic_op!(handle_add, compiler::script_parser::Add<Token>, IROp::ADD);
arithmetic_op!(handle_mul, compiler::script_parser::Mul<Token>, IROp::MUL);
arithmetic_op!(handle_div, compiler::script_parser::Div<Token>, IROp::DIV);

/* fn process_match_expression<'b, 'a: 'b>(
  expression: &boolean_Value<Token>,
  block: &'a mut IRBlockConstructor,
  e_val: TypeInfo,
) -> RumResult<LogicalExprType<'a>> {
  use LogicalExprType as LET;

  match expression {
    boolean_Value::EQ(val) => handle_eq(val, block, e_val),
    boolean_Value::LE(val) => handle_le(val, block, e_val),
    boolean_Value::LS(val) => handle_ls(val, block, e_val),
    boolean_Value::GR(val) => handle_gr(val, block, e_val),
    boolean_Value::GE(val) => handle_ge(val, block, e_val),
    boolean_Value::NE(val) => handle_ne(val, block, e_val),
    //match_group_Value::AND(val) => handle_and(val, block),
    //match_group_Value::OR(val) => handle_or(val, block),
    //match_group_Value::XOR(val) => handle_xor(val, block),
    //match_group_Value::NOT(val) => handle_not(val, block),
    //match_group_Value::RawMember(mem) => {
    //  LET::map_arith_result(handle_member(mem, block), block)
    //}
    //match_group_Value::RawSelfMember(mem) => {
    //  LET::map_arith_result(handle_self_member(mem, block), block)
    //}
    //match_group_Value::RawMemLocation(mem) => {
    //  LET::map_arith_result(handle_mem_location(mem, block), block)
    //}
    //match_group_Value::RawCall(val) => {
    //  LET::map_arith_result(handle_call(val, block), block)
    //}
    //match_group_Value::RawPrimitiveCast(prim) => {
    //  LET::map_arith_result(handle_primitive_cast(prim, block), block)
    //}
    //match_group_Value::RawStr(..) => todo!(),
    //match_group_Value::RawNum(val) => LET::map_arith_result(handle_num(val, e_val), block),
    /*     match_group_Value::Add(val) => {
      LET::map_arith_result(handle_add(val, block, Default::default(), false), block)
    }
    match_group_Value::Div(val) => {
      LET::map_arith_result(handle_div(val, block, Default::default(), false), block)
    }
    //match_group_Value::Log(val) => LET::map_arith_result(handle_log(val, block), block),
    match_group_Value::Mul(val) => {
      LET::map_arith_result(handle_mul(val, block, Default::default(), false), block)
    }
    //match_group_Value::Mod(val) => LET::map_arith_result(handle_mod(val, block), block),
    //match_group_Value::Pow(val) => LET::map_arith_result(handle_pow(val, block), block),
    match_group_Value::Sub(val) => {
      LET::map_arith_result(handle_sub(val, block, e_val, false), block)
    } */
    //match_group_Value::Root(..) => todo!(),
    //match_group_Value::None => unreachable!(),
    exp => unreachable!("{exp:#?}"),
  }
}*/

fn process_arithmetic_expression(
  expression: &bitwise_Value<Token>,
  block: &mut IRBlockConstructor,
  expected_ty: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  match expression {
    /*
    //bitwise_Value::RawSelfMember(mem) => handle_self_member(mem, block),
    //bitwise_Value::RawMemLocation(mem) => handle_mem_location(mem, block),
    //bitwise_Value::RawCall(val) => handle_call(val, block),
    bitwise_Value::RawPrimitiveCast(prim) => {
      handle_primitive_cast(prim, block, expected_ty, ret_val)
    } */
    //bitwise_Value::RawStr(..) => todo!(),
    bitwise_Value::RawNum(val) => handle_num(val, block, expected_ty),
    bitwise_Value::Add(val) => handle_add(val, block, expected_ty, ret_val),
    bitwise_Value::Div(val) => handle_div(val, block, expected_ty, ret_val),
    //bitwise_Value::Log(val) => handle_log(val, block),
    bitwise_Value::Mul(val) => handle_mul(val, block, expected_ty, ret_val),
    //bitwise_Value::Mod(val) => handle_mod(val, block),
    //bitwise_Value::Pow(val) => handle_pow(val, block),
    bitwise_Value::Sub(val) => handle_sub(val, block, expected_ty, ret_val),
    //arithmetic_Value::Root(..) => todo!(), */
    exp => unreachable!("{exp:#?}"),
  }
}

fn handle_num(
  num: &compiler::script_parser::RawNum<Token>,
  block: &mut IRBlockConstructor,
  expected_val: TypeInfo,
) -> RumResult<GraphId> {
  let val = num.val;
  let bit_size: BitSize = expected_val.into();

  let constant = match expected_val.ty() {
    RawType::Integer => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Integer | TypeInfo::b8).store(val as i8),
      BitSize::b16 => ConstVal::new(TypeInfo::Integer | TypeInfo::b16).store(val as i16),
      BitSize::b32 => ConstVal::new(TypeInfo::Integer | TypeInfo::b32).store(val as i32),
      BitSize::b64 => ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(val as i64),
      BitSize::b128 => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
      _ => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
    },
    RawType::Unsigned => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b8).store(val as u8),
      BitSize::b16 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b16).store(val as u16),
      BitSize::b32 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b32).store(val as u32),
      BitSize::b64 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b64).store(val as u64),
      BitSize::b128 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
      _ => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
    },
    RawType::Float | _ => match bit_size {
      BitSize::b32 => ConstVal::new(TypeInfo::Float | TypeInfo::b32).store(val as f32),
      _ | BitSize::b64 => ConstVal::new(TypeInfo::Float | TypeInfo::b64).store(val as f64),
    },
  };

  Ok(block.push_constant(constant))
}

fn create_binding(
  id: &Var<Token>,
  ty: &type_Value,
  tok: &Token,
  block: &mut IRBlockConstructor,
) -> Result<(), compiler::interpreter::error::RumScriptError> {
  let ty = ast_ty_to_ssa_ty(&ty);
  let name = id.id.intern();
  if ty.is_undefined() {
    return Err(format!("Invalid function parameter: \n{}", tok.blame(1, 1, "", None)).into());
  }

  block.create_binding(name, ty, tok.clone());

  Ok(())
}

fn ast_ty_to_ssa_ty(val: &type_Value) -> TypeInfo {
  let val = match val {
    type_Value::Type_u64(..) => TypeInfo::Unsigned | TypeInfo::b64,
    type_Value::Type_u32(..) => TypeInfo::Unsigned | TypeInfo::b32,
    type_Value::Type_u16(..) => TypeInfo::Unsigned | TypeInfo::b16,
    type_Value::Type_u8(..) => TypeInfo::Unsigned | TypeInfo::b8,
    type_Value::Type_i64(..) => TypeInfo::Integer | TypeInfo::b64,
    type_Value::Type_i32(..) => TypeInfo::Integer | TypeInfo::b32,
    type_Value::Type_i16(..) => TypeInfo::Integer | TypeInfo::b16,
    type_Value::Type_i8(..) => TypeInfo::Integer | TypeInfo::b8,
    type_Value::Type_f64(..) => TypeInfo::Float | TypeInfo::b64,
    type_Value::Type_f32(..) => TypeInfo::Float | TypeInfo::b32,
    type_Value::Type_8BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b8,
    type_Value::Type_16BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b16,
    type_Value::Type_32BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b32,
    type_Value::Type_64BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b64,
    type_Value::Type_128BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b128,
    _ => Default::default(),
  };

  val
}
