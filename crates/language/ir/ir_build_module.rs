use super::{
  ir_const_val::ConstVal,
  ir_context::{graph_actions, IRBlockConstructor, IRCallable, IRModule, IRStruct, IRStructMember, OptimizerContext, VariableContext},
  ir_types::{IRGraphId, IRTypeInfo, SSAFunction, TypeInfoResult},
};
use crate::{
  compiler::script_parser::{
    assignment_statement_list_Value,
    block_expression_group_1_Value,
    expression_Value,
    raw_module_Value,
    statement_Value,
    type_Value,
    Expression,
    RawBlock,
    RawCall,
    RawNum,
    RawStructDeclaration,
  },
  ir::{
    ir_context::{IRFunctionBuilder, IRSubType, IRType, TypeContext},
    ir_types::{IRGraphNode, IROp, IRPointerState},
  },
};

pub use radlr_rust_runtime::types::Token;
use rum_container::{get_aligned_value, ArrayVec};
use rum_istring::CachedString;

use IRGraphNode as GN;
use IRPointerState as Ptr;
use IRPrimitiveType as Prim;

use crate::{compiler::script_parser::property_Value, ir_types::IRPrimitiveType};

pub fn build_module(module: &Vec<raw_module_Value<Token>>) -> IRModule {
  let mut type_context = TypeContext { parent_context: std::ptr::null_mut(), local_types: Vec::new() };

  let mut functions = Vec::new();
  //let mut types = Vec::new();

  for mod_member in module {
    match mod_member {
      raw_module_Value::RawFunction(funct) => {
        functions.push(funct);
      }
      raw_module_Value::RawStruct(strct) => {
        let name = strct.name.id.intern();
        let mut members = Vec::new();
        let mut offset = 0;
        let mut min_alignment = 1;

        for (index, prop) in strct.properties.iter().enumerate() {
          match prop {
            property_Value::RawProperty(raw_prop) => {
              let ty = get_type(&raw_prop.r#type, &type_context);

              let prop_offset = get_aligned_value(offset, ty.alignment() as u64);
              offset = prop_offset + ty.byte_size() as u64;

              min_alignment = min_alignment.max(ty.alignment() as u64);

              members.push(IRStructMember {
                ty,
                original_index: index,
                name: raw_prop.name.id.intern(),
                offset: prop_offset as usize,
              })
            }

            property_Value::RawBitCompositeProp(bit_composition) => {
              println!("Bit composite property type not supported yet");
            }
            _ => {}
          }
        }

        let s = IRStruct {
          name,
          module: "default".intern(),
          members: ArrayVec::from_iter(members),
          alignment: min_alignment,
          size: get_aligned_value(offset, min_alignment as u64),
        };

        type_context.local_types.push(IRType {
          name,
          alignment: min_alignment as usize,
          byte_size: get_aligned_value(offset, min_alignment as u64) as usize,
          sub_type: IRSubType::Struct(s),
        });
      }
      _ => {}
    }
  }

  for function in &functions {
    // Gather function type information.

    let name = function.name.id.intern();
    let params = &function.ty.params;
    let return_ty = &function.ty.return_type;

    type_context.local_types.push(IRType {
      name,
      alignment: 8, // pointer size
      byte_size: 8, // pointer size
      sub_type: IRSubType::Function,
    })
  }

  let mut out_functions = Vec::new();

  // All types should have been resolved.
  for function in functions {
    let funct = process_function(function, &type_context);
    out_functions.push(funct);
  }

  IRModule { type_context, functions: out_functions }
}

#[test]
fn test() {
  build_module(
    &crate::compiler::script_parser::parse_raw_module(
      &r##"

Temp => [
  root:u32
  val:u32
]

main => () {
  data  =  Temp[ 
    root = 72 
    val = 0
  ]

  mango = data.root

  stdcout_fd  :u32 = 0
  buffer      :u32 = 72
  size        :u64 = 2

  sys_write(stdcout_fd, &buffer, size)
}
    
  
  
  "##,
    )
    .unwrap(),
  );
}

fn process_function(function: &std::rc::Rc<crate::compiler::script_parser::RawFunction<Token>>, type_context: &TypeContext) -> IRCallable {
  // Ensure the return type is present in our type context.
  println!("TODO: Ensure the return type is present in our type context");

  let name = function.name.id.intern();

  let mut f_ctx = IRFunctionBuilder::default();
  let mut head_block = f_ctx.push_block(None);

  let mut base_var_context = VariableContext {
    local_variables: Default::default(),
    parent_context:  std::ptr::null_mut(),
  };

  //Module scope

  // initial_block
  // block-scope
  // function_scope

  process_expression(&function.expression, type_context, &mut head_block, &mut base_var_context);

  IRCallable {
    blocks: f_ctx.blocks.into_iter().map(|b| unsafe { Box::from_raw(b).into() }).collect(),
    graph: f_ctx.graph,
    module: Default::default(),
    name,
    signature: (),
  }
}

fn process_expression(
  expr: &Expression<Token>,
  type_ctx: &TypeContext,
  block: &mut IRBlockConstructor,
  var_ctx: &mut VariableContext,
) -> (IRTypeInfo, IRGraphId, InitResult) {
  match &expr.expr {
    expression_Value::RawCall(call) => process_call(call, type_ctx, block, var_ctx),
    expression_Value::RawNum(num) => process_const_number(num, block),
    expression_Value::AddressOf(addr) => {
      let var_name = addr.id.id.intern();

      if let Some((type_, graph_id, _)) = var_ctx.get_variable(var_name) {
        let ptr = type_.as_ptr(Ptr::Stack);

        let graph_id = block.push_node(GN::create_ssa(IROp::ADDR, ptr, &[graph_id], graph_id.graph_id()));

        (ptr, graph_id, InitResult::None)
      } else {
        panic!()
      }
    }
    expression_Value::RawStructDeclaration(struct_decl) => process_struct_instantiation(struct_decl, type_ctx, block, var_ctx),
    expression_Value::RawMember(mem) => {
      let base = &mem.members[0];

      let var_name = base.id.intern();

      if let Some((type_, graph_id, var_index)) = var_ctx.get_variable(var_name) {
        match type_.base_type() {
          TypeInfoResult::IRPrimitive(prim) => {
            return ((*prim).into(), graph_id, InitResult::None);
          }
          TypeInfoResult::IRType(ir_type) => match &ir_type.sub_type {
            IRSubType::Struct(strct) => {
              let sub_name = mem.members[1].id.intern();

              if let Some(sub_type) = strct.members.iter().find(|d| d.name == sub_name) {
                let offset = block.push_node(GN::create_const(ConstVal::new(Prim::Unsigned | Prim::b32).store(sub_type.offset as u32)));

                // Acquire a pointer to this type.

                // create a ptr to this type
                let index = block.ctx().graph.len();
                let ptr = block.push_node(GN::create_ssa(
                  IROp::PTR_MEM_CALC,
                  sub_type.ty.as_ptr(IRPointerState::Temporary),
                  &[graph_id, offset],
                  index,
                ));

                let val = block.push_node(GN::create_ssa(IROp::ADDR, sub_type.ty, &[ptr], graph_id.graph_id()));

                return (sub_type.ty, val, InitResult::None);
              } else {
                todo!("")
              }
            }
            _ => unreachable!(),
          },
        }
      } else {
        panic!("Type {var_name:?} not found");
      }
    }
    expression_Value::RawBlock(ast_block) => {
      process_block(ast_block, type_ctx, block, var_ctx);
      println!("Return the graph id value of a raw block");
      (IRTypeInfo::default(), IRGraphId::default(), InitResult::None)
    }
    d => todo!("process expression: {d:#?}"),
  }
}

fn process_const_number(num: &RawNum<Token>, block: &mut IRBlockConstructor) -> (IRTypeInfo, IRGraphId, InitResult) {
  let string_val = num.tok.to_string();
  let graph_id = block.push_node(IRGraphNode::create_const(if string_val.contains(".") {
    ConstVal::new(Prim::Float | Prim::b64).store(num.val)
  } else {
    ConstVal::new(Prim::Unsigned | Prim::b64).store::<u64>(string_val.parse::<u64>().unwrap())
  }));

  (block.ctx().graph[graph_id.graph_id()].ty(), graph_id, InitResult::None)
}

pub fn remap_primitive_type(desired_type: IRPrimitiveType, node_id: IRGraphId, graph: &mut Vec<IRGraphNode>) {
  // Add some rules to say whether the type can be coerced or converted into the
  // desired type.

  if node_id.is_invalid() {
    return;
  }

  match &mut graph[node_id.graph_id()] {
    IRGraphNode::Const { id: ssa_id, val } => {
      if val.is_lit() {
        *val = val.convert(desired_type);
      } else {
        *val = ConstVal::new(desired_type);
      }
    }
    IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } => {
      if out_ty.is_numeric() {
        *out_ty = desired_type.into();
        let operands = *operands;

        remap_primitive_type(desired_type, operands[0], graph);

        remap_primitive_type(desired_type, operands[1], graph);
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

fn process_call(
  call: &RawCall<Token>,
  type_ctx: &TypeContext,
  block: &mut IRBlockConstructor,
  var_ctx: &mut VariableContext,
) -> (IRTypeInfo, IRGraphId, InitResult) {
  let call_name = &call.id.id;

  if call_name.contains("sys_") {
    for (type_info, graph_id, ..) in call.args.iter().map(|arg| process_expression(&arg, type_ctx, block, var_ctx)).collect::<Vec<_>>() {
      block.push_node(IRGraphNode::create_ssa(IROp::CALL_ARG, type_info, &[graph_id], usize::MAX));

      println!("TODO: match type_info with arg type");
    }
  }

  let graph_id = block.push_node(IRGraphNode::create_const(ConstVal::new(Prim::Unsigned | Prim::b64).store(1 as u64)));

  let graph_id = block.push_node(IRGraphNode::create_ssa(IROp::CALL, Default::default(), &[graph_id], usize::MAX));

  (Default::default(), graph_id, InitResult::None)
}

fn process_struct_instantiation(
  struct_decl: &RawStructDeclaration<Token>,
  type_ctx: &TypeContext,
  block: &mut IRBlockConstructor,
  var_ctx: &mut VariableContext,
) -> (IRTypeInfo, IRGraphId, InitResult) {
  // locate the type

  let struct_type_name = struct_decl.name.id.intern();

  let mut base_ptr_resolutions = vec![];

  if let Some(struct_type) = type_ctx.local_types.iter().find(|ty| ty.name == struct_type_name) {
    if let IRSubType::Struct(struct_definition) = &struct_type.sub_type {
      let s_type = struct_type.into();

      let struct_id = block.push_node(GN::create_var(struct_type_name, s_type));
      var_ctx.set_variable(struct_type_name, s_type, struct_id, struct_id.graph_id());

      let struct_ptr_id = block.push_node(GN::create_ssa(IROp::ADDR, s_type.as_ptr(Ptr::Stack), &[struct_id], block.ctx().graph.len()));

      for init_expression in &struct_decl.inits {
        let member_name = init_expression.name.id.intern();

        if let Some(member) = struct_definition.members.iter().find(|i| i.name == member_name) {
          let (expr_type, id, _) = process_expression(&init_expression.expression, type_ctx, block, var_ctx);

          if member.ty != expr_type {
            if member.ty.is_numeric() {
              remap_primitive_type(member.ty.as_prim(), id, &mut block.ctx().graph)
            } else {
              let node = &block.ctx().graph[id.graph_id()];
              panic!("handle type coercions of member:{:?} expr:{:?}", member.ty, node.ty());
            }
          }

          // Calculate the offset to the member within the struct
          let member_ptr = member.ty.as_ptr(Ptr::Temporary);

          if member.offset > 0 {
            let offset = block.push_node(GN::create_const(ConstVal::new(Prim::Unsigned | Prim::b32).store(member.offset as u32)));
            let index = block.ctx().graph.len();
            let ptr = block.push_node(GN::create_ssa(IROp::PTR_MEM_CALC, member_ptr, &[struct_ptr_id, offset], index));

            block.ctx().variables.push((member_ptr, ptr));

            base_ptr_resolutions.push(ptr);

            block.push_node(GN::create_ssa(IROp::MEM_STORE, member_ptr, &[ptr, id], usize::MAX));
          } else {
            let node = block.push_node(GN::create_ssa(IROp::MEM_STORE, member_ptr, &[struct_ptr_id, id], usize::MAX));
            base_ptr_resolutions.push(node);
          }
        } else {
          panic!("Member name not found.");
        }
      }

      return (s_type.into(), struct_ptr_id, InitResult::StructInit(struct_id, struct_ptr_id));
    }
  }

  (IRTypeInfo::default(), IRGraphId::default(), InitResult::None)
}

fn process_block(ast_block: &RawBlock<Token>, type_ctx: &TypeContext, block: &mut IRBlockConstructor, var_ctx: &mut VariableContext) {
  for stmt in &ast_block.statements {
    process_statement(stmt, type_ctx, block, var_ctx);
  }

  match &ast_block.exit {
    block_expression_group_1_Value::None => {
      println!("Resolve return expression of a block exit");
    }
    d => todo!("process block exit: {d:#?}"),
  }
}

fn process_statement(stmt: &statement_Value<Token>, type_ctx: &TypeContext, block: &mut IRBlockConstructor, var_ctx: &mut VariableContext) {
  match stmt {
    statement_Value::RawAssignment(assign) => {
      // Process assignments.
      let expression = assign.expressions.iter().map(|expr| process_expression(expr, type_ctx, block, var_ctx)).collect::<Vec<_>>();

      // Process assignment targets.
      for (variable_index, variable) in assign.vars.iter().enumerate() {
        match variable {
          assignment_statement_list_Value::RawArrayAccess(array) => {
            todo!("Array access")
          }
          assignment_statement_list_Value::RawAssignmentLabel(var_assign) => {
            let var_name = var_assign.var.id.intern();
            let mut ty = get_type(&var_assign.ty, type_ctx);
            let existing_variable = var_ctx.get_variable(var_name);

            if ty.is_undefined() && existing_variable.is_some() && !existing_variable.unwrap().0.is_undefined() {
              panic!("Variable already exists. Use := expression to reassign this variable to a new slot.");
            }

            let (expr_ty, expr_graph_id, init_result) = &expression[variable_index];

            match expr_ty.base_type() {
              TypeInfoResult::IRType(ty) => match &ty.sub_type {
                IRSubType::Struct(strc) => {
                  if let InitResult::StructInit(strct_var_id, ptr_id) = init_result {
                    var_ctx.set_variable(var_name, ty.into(), *ptr_id, strct_var_id.graph_id());
                    // graph_id is a VAR set to temp. We should commandeer it
                    // for our own use.
                  } else {
                    panic!("Invalid struct declaration")
                  }
                }
                _ => unreachable!(),
              },
              _ => {
                let (var_index, store_type, type_ptr) = if ty.is_undefined() {
                  // This may be a new definition for var_name, or it may be a new
                  // assignment to an existing var_name.

                  if let Some((var_ty, graph_node, decl_index)) = var_ctx.get_variable(var_name) {
                    ty = var_ty;

                    if false {
                      todo!("Assert the expression type is compatible with the var type");
                    }

                    (decl_index, ty, graph_node)
                  } else {
                    ty = *expr_ty;
                    let store_type = ty.as_ptr(IRPointerState::Stack);

                    let type_ptr = block.push_node(GN::create_var(var_name, ty));

                    var_ctx.set_variable(var_name, ty, type_ptr, type_ptr.graph_id());

                    (type_ptr.graph_id(), store_type, type_ptr)
                  }
                } else if ty != *expr_ty {
                  match (ty.base_type(), expr_ty.base_type()) {
                    (TypeInfoResult::IRPrimitive(prim_ty), TypeInfoResult::IRPrimitive(prim_expr_ty)) => {
                      remap_primitive_type(*prim_ty, *expr_graph_id, &mut block.ctx().graph);

                      let store_type = ty.as_ptr(IRPointerState::Stack);
                      let type_ptr = block.push_node(GN::create_var(var_name, ty));
                      var_ctx.set_variable(var_name, ty, type_ptr, type_ptr.graph_id());
                      (type_ptr.graph_id(), store_type, type_ptr)
                    }
                    _ => panic!("Miss matched types ty:{ty:?} expr_ty:{expr_ty:?}"),
                  }
                } else {
                  let store_type = ty.as_ptr(IRPointerState::Stack);
                  let type_ptr = block.push_node(GN::create_var(var_name, ty));
                  var_ctx.set_variable(var_name, ty, type_ptr, type_ptr.graph_id());
                  (type_ptr.graph_id(), store_type, type_ptr)
                };

                let graph_id =
                  block.push_node(GN::create_ssa(IROp::STORE, store_type.as_ptr(Ptr::None), &[type_ptr, *expr_graph_id], var_index));

                var_ctx.set_id(var_name, graph_id);
              }
            }
          }
          _ => unreachable!(),
        }
      }

      // Match assignments to targets.
    }
    statement_Value::Expression(expr) => {
      let (ty, graph_actions, _) = process_expression(expr, type_ctx, block, var_ctx);
    }
    d => todo!("process statement: {d:#?}"),
  }
}

pub fn get_type(ir_type: &type_Value<Token>, type_context: &TypeContext) -> IRTypeInfo {
  match ir_type {
    type_Value::Type_u8(_) => (IRPrimitiveType::Unsigned | IRPrimitiveType::b8).into(),
    type_Value::Type_u16(_) => (IRPrimitiveType::Unsigned | IRPrimitiveType::b16).into(),
    type_Value::Type_u32(_) => (IRPrimitiveType::Unsigned | IRPrimitiveType::b32).into(),
    type_Value::Type_u64(_) => (IRPrimitiveType::Unsigned | IRPrimitiveType::b64).into(),
    type_Value::Type_i8(_) => (IRPrimitiveType::Integer | IRPrimitiveType::b8).into(),
    type_Value::Type_i16(_) => (IRPrimitiveType::Integer | IRPrimitiveType::b16).into(),
    type_Value::Type_i32(_) => (IRPrimitiveType::Integer | IRPrimitiveType::b32).into(),
    type_Value::Type_i64(_) => (IRPrimitiveType::Integer | IRPrimitiveType::b64).into(),
    type_Value::NamedType(name) => {
      let type_name = name.name.id.intern();
      if let Some(ty) = type_context.local_types.iter().find(|t| t.name == type_name) {
        ty.into()
      } else {
        Default::default()
      }
    }
    t => IRTypeInfo::from(Prim::default()),
  }
}
