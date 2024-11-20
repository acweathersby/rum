use libc::shm_open;
use solve_pipeline::{solve_node, OPConstraint};
use type_solve::VarConstraint;

use crate::{
  ir::{
    db::{self, Solver},
    ir_rvsdg::{IROp, *},
    types::{ty_u16, ty_u32, PrimitiveType, TypeDatabase},
  },
  istring::CachedString,
  parser::script_parser::*,
};
use std::{
  any::Any,
  collections::{HashMap, VecDeque},
  usize,
};

const crazy_ty: Type = Type::Primitive(PrimitiveType {
  base_index: 0,
  base_ty:    crate::ir::types::PrimitiveBaseType::Float,
  byte_size:  20,
  ele_count:  20,
});

#[derive(Clone, Copy, Debug)]
enum Write {
  IntraNodeBind { node_index: usize, output_index: usize },
}

#[derive(Clone, Copy, Debug)]
struct ThreadedGraphId(IRGraphId, u32);

#[derive(Clone, Debug)]
struct Var {
  id:             VarId,
  origin_node_id: u32,
  op:             IRGraphId,
  ast:            ASTNode,
  writes:         Vec<Write>,
  nonce:          usize,
  ty:             Type,
}

#[derive(Debug)]
struct Builder {
  id:         u32,
  id_counter: *mut u32,

  node:           Box<RVSDGNode>,
  /// Monotonically increasing identifier for the node stored in this builder
  node_id:        usize,
  label_lookup:   HashMap<IString, IRGraphId>,
  const_lookup:   HashMap<ConstVal, IRGraphId>,
  /// Stores variable names that have been declared in the current scope, or
  /// a previous scope.
  var_lookup:     HashMap<VarId, Var>,
  op_constraints: *mut Vec<OPConstraint>,
  type_vars:      *mut Vec<TypeVar>,
}

type WIPNode = Builder;

impl Builder {
  pub fn new(id: u32, counter: *mut u32, op_constraints: *mut Vec<OPConstraint>, type_vars: *mut Vec<TypeVar>) -> Self {
    Builder {
      id,
      id_counter: counter,
      node: Default::default(),
      node_id: Default::default(),
      label_lookup: Default::default(),
      const_lookup: Default::default(),
      var_lookup: Default::default(),
      op_constraints,
      type_vars,
    }
  }

  fn add_op_constraint(&mut self, op_constraint: OPConstraint) {
    (unsafe { &mut *self.op_constraints }).push(op_constraint);
  }

  fn inc_counter(&self) -> u32 {
    unsafe {
      let id = *self.id_counter;
      *self.id_counter += 1;
      *self.id_counter
    }
  }

  fn create_id(node_id: &mut usize) -> IRGraphId {
    let id = IRGraphId::new(*node_id);
    *node_id += 1;
    id
  }

  pub fn type_vars<'a>(&self) -> &'a mut Vec<TypeVar> {
    unsafe { &mut *self.type_vars }
  }

  pub fn add_simple(&mut self, op: IROp, operands: [IRGraphId; 3], ty: Type, ast: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_node(RVSDGInternalNode::Simple { op, operands }, ast, ty);
    id
  }

  pub fn add_binding(&mut self, mut binding: RSDVGBinding, node: ASTNode, ty: Type, b_ty: BindingType) -> RSDVGBinding {
    let id = Self::create_id(&mut self.node_id);
    binding.out_op = id;
    self.node.inputs.push(binding);
    self.add_binding_node_internal(id, ty, node, b_ty);
    binding
  }

  pub fn add_input_node(&mut self, ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_binding_node_internal(id, ty, node, BindingType::IntraBinding)
  }

  pub fn add_output_node(&mut self, ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_binding_node_internal(id, ty, node, BindingType::IntraBinding)
  }

  fn add_binding_node_internal(&mut self, id: IRGraphId, ty: Type, ast: ASTNode, b_ty: BindingType) -> IRGraphId {
    self.add_node(RVSDGInternalNode::Binding { ty: b_ty }, ast, ty);
    id
  }

  pub fn add_output(&mut self, mut binding: RSDVGBinding) -> RSDVGBinding {
    self.node.outputs.push(binding);
    binding
  }

  pub fn add_const(&mut self, const_val: ConstVal, node: ASTNode) -> IRGraphId {
    let const_id = self.get_const(const_val);
    let id = Self::create_id(&mut self.node_id);

    self.add_node(RVSDGInternalNode::Simple { op: IROp::CONST_DECL, operands: [const_id, Default::default(), Default::default()] }, node, Default::default());

    id
  }

  fn bump_gen_type_vars(&mut self) -> Type {
    let ty_vars = self.type_vars();

    let id = ty_vars.len();

    let ty = Type::Generic { ptr_count: 0, gen_index: id as u32 };

    let mut var = TypeVar::new(id as u32);

    debug_assert!(ty.is_generic());

    var.ty = ty;

    ty_vars.push(var);

    ty
  }

  fn add_other(&mut self, node: RVSDGInternalNode, ast: ASTNode, ty: Type) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_node(node, ast, ty);
    id
  }

  fn add_node(&mut self, node: RVSDGInternalNode, ast: ASTNode, ty: Type) {
    self.node.nodes.push(node);
    self.node.source_nodes.push(ast);
    self.node.types.push(ty);
  }

  pub fn get_label(&mut self, name: IString) -> IRGraphId {
    let Self { label_lookup, node_id, node, .. } = self;
    match label_lookup.entry(name) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        *val.insert(id);

        self.add_node(RVSDGInternalNode::Label(name), Default::default(), Type::NoUse);

        id
      }
    }
  }

  pub fn set_type_if_undefined(&mut self, op: IRGraphId, ty: Type) -> Type {
    self.node.set_type_if_undefined(op, ty)
  }

  pub fn create_gen_var(&mut self, var_id: VarId, op: IRGraphId, ast: ASTNode, outer_ty: Type) -> Type {
    let var_ty = self.bump_gen_type_vars();

    let op = if op.is_valid() {
      self.set_type_if_undefined(op, var_ty);
      if !outer_ty.is_open() {
        self.add_op_constraint(OPConstraint::GenVarToTy(var_ty.generic_id().unwrap(), outer_ty));
        op
      } else {
        op
      }
    } else {
      op
    };

    self.create_var_lookup(var_id, op, var_ty, ast);

    var_ty
  }

  fn create_var_lookup(&mut self, var_id: VarId, op: IRGraphId, var_ty: Type, ast: ASTNode) {
    debug_assert!(var_ty.is_generic());

    self.var_lookup.insert(var_id, Var { id: var_id, origin_node_id: self.id, op, ast, writes: Default::default(), nonce: 0, ty: var_ty });
  }

  pub fn update_var(&mut self, var_id: VarId, op: IRGraphId, ast: ASTNode) -> bool {
    if let Some(var) = self.var_lookup.get_mut(&var_id) {
      var.op = op;
      var.ast = ast;
      var.nonce += 1;
      let var_ty = var.ty;

      if op.is_valid() {
        self.set_type_if_undefined(op, var_ty);
      }

      true
    } else {
      false
    }
  }

  pub fn get_const(&mut self, const_val: ConstVal) -> IRGraphId {
    let Self { const_lookup: val_lookup, node_id, node, .. } = self;
    match val_lookup.entry(const_val) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        *val.insert(id);

        self.add_node(RVSDGInternalNode::Const(const_val), Default::default(), Default::default());

        id
      }
    }
  }
}

pub fn lower_ast_to_rvsdg(module: &std::sync::Arc<RawModule<Token>>, ty_db: &mut TypeDatabase) -> Solver {
  let mut solver = Solver::new();

  let members = &module.members;

  for mem in &members.members {
    match mem {
      module_members_group_Value::AnnotatedModMember(annotation) => match &annotation.member {
        module_member_Value::RawBoundType(bound_type) => match &bound_type.ty {
          type_Value::Type_Struct(strct) => {
            let name = bound_type.name.id.intern();
            let (struct_, constraints) = lower_struct_to_rvsdg(strct, ty_db);

            if let Some(ty) = ty_db.add_ty(bound_type.name.id.intern(), struct_.clone()) {
              solver.add_type(ty, constraints);
            }
          }
          type_Value::Type_Array(array) => {
            let name = bound_type.name.id.intern();
            let (array_, constraints) = lower_array_to_rvsdg(array, ty_db);

            if let Some(ty) = ty_db.add_ty(bound_type.name.id.intern(), array_.clone()) {
              solver.add_type(ty, constraints);
            }
          }
          _ => unreachable!(),
        },
        module_member_Value::RawRoutine(rt) => {
          let name = rt.name.id.intern();

          let (mut funct, mut global_constraints) = lower_routine_to_rvsdg(&rt.def, ty_db);

          if let Some(ty) = ty_db.add_ty(name, funct) {
            solver.add_type(ty, global_constraints);
          } else {
            todo!("Construct an error about invalid overload {name} ");
          }
        }
        module_member_Value::RawScope(scope) => {}
        _ => unreachable!(),
      },
      module_members_group_Value::AnnotationVariable(var) => {}
      module_members_group_Value::LifetimeVariable(var) => {}
      node => unreachable!("Unrecognized node type {node:#?}"),
    }
  }

  solver
}

fn lower_array_to_rvsdg(array_: &Type_Array<Token>, ty_db: &mut TypeDatabase) -> (Box<RVSDGNode>, Vec<OPConstraint>) {
  let mut node = RVSDGNode::default();
  let mut constraints = Vec::new();

  node.ty = RVSDGNodeType::Array;
  node.id = 0;

  let ty = get_type(&array_.base_type, false, ty_db).unwrap_or_default();

  let mut output_binding = RSDVGBinding::default();
  let id = Builder::create_id(&mut 0);

  output_binding.id = VarId::ArrayType;
  output_binding.in_op = id;

  node.outputs.push(output_binding);

  node.nodes.push(RVSDGInternalNode::Binding { ty: BindingType::IntraBinding });
  node.types.push(Type::default());
  node.source_nodes.push(array_.base_type.clone().into());

  constraints.push(OPConstraint::OpToTy(IRGraphId::new(0), ty));

  if array_.size > 0 {
    let mut output_binding = RSDVGBinding::default();
    let id = Builder::create_id(&mut 2);

    output_binding.id = VarId::ArraySize;
    output_binding.in_op = id;

    node.outputs.push(output_binding);

    node.nodes.push(RVSDGInternalNode::Const(ConstVal::new(ty_u32.to_primitive().unwrap(), array_.size)));
    node.types.push(Type::NoUse);
    node.source_nodes.push(Default::default());

    node.nodes.push(RVSDGInternalNode::Simple { op: IROp::CONST_DECL, operands: [IRGraphId::new(1), Default::default(), Default::default()] });
    node.types.push(Type::default());
    node.source_nodes.push(Default::default());

    constraints.push(OPConstraint::OpToTy(IRGraphId::new(2), ty_u32));
  }

  (Box::new(node), constraints)
}

fn lower_struct_to_rvsdg(struct_: &Type_Struct<Token>, ty_db: &mut TypeDatabase) -> (Box<RVSDGNode>, Vec<OPConstraint>) {
  let mut node = RVSDGNode::default();
  let mut constraints = Vec::new();

  node.ty = RVSDGNodeType::Struct;
  node.id = 0;

  let mut node_id = 0;

  for prop in &struct_.properties {
    match prop {
      property_Value::Property(prop) => {
        let name = prop.name.id.intern();
        let ty = get_type(&prop.ty, false, ty_db).unwrap_or_default();

        let mut output_binding = RSDVGBinding::default();
        let id = Builder::create_id(&mut node_id);

        output_binding.id = VarId::VarName(name);
        output_binding.in_op = id;

        node.outputs.push(output_binding);

        node.nodes.push(RVSDGInternalNode::Binding { ty: BindingType::IntraBinding });
        node.types.push(Type::default());
        node.source_nodes.push(ASTNode::Property(prop.clone()));

        if !ty.is_open() {
          constraints.push(OPConstraint::OpToTy(IRGraphId::new(node.nodes.len() - 1), ty));
        }
      }
      n => todo!("Handle construction of {n:?}"),
    }
  }

  (Box::new(node), constraints)
}

pub fn lower_routine_to_rvsdg(routine_def: &RawRoutineDefinition<Token>, ty_db: &mut TypeDatabase) -> (Box<RVSDGNode>, Vec<(OPConstraint)>) {
  let params = match &routine_def.ty {
    routine_type_Value::RawFunctionType(ty) => &ty.params,
    routine_type_Value::RawProcedureType(ty) => &ty.params,
    _ => unreachable!(),
  };

  let expr = &routine_def.expression.expr;

  let mut node_stack = VecDeque::new();
  let mut counter = 0;

  let mut op_constraints = Vec::new();
  let mut type_vars = Vec::new();

  let mut builder = Builder::new(0, &mut counter, &mut op_constraints as *mut _, &mut type_vars as *mut _);
  builder.node.ty = RVSDGNodeType::Undefined;
  builder.create_gen_var(VarId::Return, Default::default(), Default::default(), Default::default());

  node_stack.push_front(builder);

  push_new_builder(&mut node_stack, RVSDGNodeType::Routine);

  let fn_builder = node_stack.front_mut().unwrap();

  insert_params(params, fn_builder, ty_db);

  let ret_val = process_expression(expr, &mut node_stack, ty_db);

  let ret_ty = match &routine_def.ty {
    routine_type_Value::RawFunctionType(fn_ty) => {
      insert_returns(resolve_binding(ret_val, &mut node_stack, Default::default()), fn_ty, &mut node_stack, ty_db, &mut Vec::new())
    }
    _ => Default::default(),
  };

  while node_stack.len() > 2 {
    pop_and_merge_single_node_with_return(&mut node_stack, Default::default());
  }

  let vars = node_stack.front_mut().unwrap().var_lookup.keys().cloned().collect::<Vec<_>>();

  for (var_id) in vars {
    match var_id {
      VarId::Return | VarId::SideEffect(..) => {
        seal_var(var_id, &mut node_stack, ret_ty);
      }
      _ => {}
    }
  }

  seal_var(VarId::HeapContext, &mut node_stack, Default::default());

  pop_and_merge_single_node(&mut node_stack, Default::default());

  let mut fn_node = node_stack.pop_back().unwrap().node;

  if let RVSDGInternalNode::Complex(mut node) = fn_node.nodes.remove(0) {
    node.ty_vars = type_vars;
    debug_assert_eq!(node.nodes.len(), node.types.len());

    (node, op_constraints)
  } else {
    panic!()
  }
}

fn insert_returns(
  ret_val: IRGraphId,
  fn_ty: &std::sync::Arc<RawFunctionType<Token>>,
  node_stack: &mut VecDeque<WIPNode>,
  ty_db: &mut TypeDatabase,
  errors: &mut Vec<String>,
) -> Type {
  let ret = &fn_ty.return_type;
  let mut input = RSDVGBinding::default();
  let ty = get_type(&ret.ty, false, ty_db).unwrap_or_default();

  create_return_sink(node_stack, ret_val);

  ty
}

fn create_return_sink(node_stack: &mut VecDeque<Builder>, ret_val: IRGraphId) {
  if ret_val.is_valid() {
    let prev_ret = commit_writes_to_var(VarId::Return, node_stack).unwrap_or_default();
    let builder = node_stack.front_mut().unwrap();
    let ty = builder.var_lookup.get(&VarId::Return).unwrap().ty;

    let heap = builder.var_lookup.get(&VarId::HeapContext).unwrap().op;

    let sink_id = builder.add_simple(IROp::RET_VAL, [ret_val, prev_ret, heap], ty, Default::default());
    //let sink_id = builder.add_other(RVSDGInternalNode:: { src: ret_val, ty: BindingType::Return }, Default::default(), ty);

    builder.update_var(VarId::Return, sink_id, Default::default());
  }
}

fn insert_params(params: &std::sync::Arc<Params<Token>>, builder: &mut Builder, ty_db: &mut TypeDatabase) {
  for (param_index, param) in params.params.iter().enumerate() {
    let ty = get_type(&param.ty.ty, false, ty_db).unwrap_or_default();

    let inner_id = VarId::VarName(param.var.id.intern());
    let outer_id = VarId::Param(param_index);

    let ast = ASTNode::RawParamBinding(param.clone());

    let input = builder.add_binding(
      RSDVGBinding { id: outer_id, in_op: IRGraphId::new(param_index), ..Default::default() },
      ast,
      Default::default(),
      BindingType::ParamBinding,
    );

    let var = builder.create_gen_var(inner_id, input.out_op, ASTNode::RawParamBinding(param.clone()), ty);
    if !ty.is_open() {
      builder.add_op_constraint(OPConstraint::GenVarToTy(var.generic_id().unwrap(), ty));
    }
  }

  // Add allocator context.

  {
    let allocator_index = params.params.len();
    let ctx_id = VarId::HeapContext;
    let ctx_input = builder.add_binding(
      RSDVGBinding { id: ctx_id, in_op: IRGraphId::new(allocator_index), ..Default::default() },
      Default::default(),
      Default::default(),
      BindingType::ParamBinding,
    );

    let var = builder.create_gen_var(ctx_id, ctx_input.out_op, Default::default(), Default::default());
    builder.add_op_constraint(OPConstraint::GenVarToTy(var.generic_id().unwrap(), Type::MemContext));
  }
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> ThreadedGraphId {
  match expr {
    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      let val = node_stack.front_mut().unwrap().add_const(
        if string_val.contains(".") {
          ConstVal::new(ty_db.get_ty("f64").expect("f64 should exist").to_primitive().unwrap(), num.val)
        } else {
          ConstVal::new(ty_db.get_ty("i64").expect("i64 should exist").to_primitive().unwrap(), string_val.parse::<i64>().unwrap())
        },
        num.clone().into(),
      );

      ThreadedGraphId(val, node_stack.front().unwrap().id)
    }

    expression_Value::Add(op) => binary_expr(
      IROp::ADD,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Add(op.clone()),
      ty_db,
    ),

    expression_Value::Mul(op) => binary_expr(
      IROp::MUL,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Mul(op.clone()),
      ty_db,
    ),

    expression_Value::Div(op) => binary_expr(
      IROp::DIV,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Div(op.clone()),
      ty_db,
    ),

    expression_Value::Sub(op) => binary_expr(
      IROp::SUB,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Sub(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_XOR(op) => binary_expr(
      IROp::XOR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_XOR(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_OR(op) => binary_expr(
      IROp::OR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_OR(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_AND(op) => binary_expr(
      IROp::AND,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_AND(op.clone()),
      ty_db,
    ),

    expression_Value::RawBlock(block) => {
      let mut last_id = ThreadedGraphId(Default::default(), node_stack.front().unwrap().id);
      for expr in &block.statements {
        match expr {
          statement_Value::Expression(expr) => {
            let li = process_expression(&expr.expr, node_stack, ty_db);
            last_id = li;
          }
          statement_Value::RawAssignment(assign) => {
            process_assign(&assign, node_stack, ty_db);

            last_id = ThreadedGraphId(Default::default(), node_stack.front().unwrap().id)
          }
          statement_Value::RawLoop(loop_expr) => {
            push_new_builder(node_stack, RVSDGNodeType::Loop);

            let builder = node_stack.front_mut().unwrap();
            builder.create_gen_var(VarId::LoopActivation, Default::default(), Default::default(), Default::default());

            {
              match &loop_expr.scope {
                loop_statement_group_1_Value::RawMatch(mtch) => {
                  process_match(node_stack, mtch, ty_db, false);
                  if let Some(match_activation_op) = commit_writes_to_var(VarId::MatchActivation, node_stack) {
                    debug_assert!(match_activation_op.is_valid());
                    let builder = node_stack.front_mut().unwrap();
                    builder.update_var(VarId::LoopActivation, match_activation_op, Default::default());
                  }
                }
                loop_statement_group_1_Value::RawBlock(block) => {
                  process_expression(&expression_Value::RawBlock(block.clone().into()), node_stack, ty_db);
                }
                ast => todo!("{ast:#?}"),
              }
            }

            seal_var(VarId::LoopActivation, node_stack, Default::default());

            //panic!("{:#?}", node_stack.front().unwrap());
            pop_and_merge_single_node(node_stack, loop_expr.clone().into());
          }
          ty => todo!("{ty:#?}"),
        }
      }

      match &block.exit {
        Some(block_expression_group_3_Value::BlockExitExpressions(e)) => {
          let out_id = process_expression(&e.expression.expr, node_stack, ty_db);
          let out_id = resolve_binding(out_id, node_stack, Default::default());

          create_return_sink(node_stack, out_id);

          ThreadedGraphId(Default::default(), node_stack.front().unwrap().id)
        }
        _ => last_id,
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      let call_id = match lookup_var(&call.member, node_stack, true, ty_db) {
        VarLookup::Mem { ref_id: mem_id, id } => mem_id,
        VarLookup::Var(var) => var,
        VarLookup::None(name) => node_stack.front_mut().unwrap().get_label(name),
      };

      for arg in &call.args {
        args.push((process_expression(&arg.expr, node_stack, ty_db), ASTNode::expression_Value(arg.expr.clone())));
      }

      push_new_builder(node_stack, RVSDGNodeType::Call);

      {
        let mut input = RSDVGBinding::default();
        input.in_op = call_id;
        input.id = VarId::CallRef;
        let call_builder = node_stack.front_mut().unwrap();
        call_builder.add_binding(input, ASTNode::MemberCompositeAccess(call.member.clone()), Default::default(), BindingType::IntraBinding);
      }

      let heap_id = {
        let mut input = RSDVGBinding::default();
        input.in_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();
        input.id = VarId::HeapContext;
        let call_builder = node_stack.front_mut().unwrap();
        //call_builder.update_var(VarId::HeapContext, input.in_op, Default::default());
        //call_builder.add_binding(input, Default::default(), Default::default(), BindingType::IntraBinding);
        input.in_op
      };

      for (index, (arg, node)) in args.iter().enumerate() {
        resolve_binding(*arg, node_stack, VarId::Param(index));
      }

      let call_builder = node_stack.front_mut().unwrap();

      let mut output = RSDVGBinding::default();
      output.id = VarId::Return;
      //input.out_id = ret_id;
      call_builder.add_output(output);

      // The return value should now be mapped to a new_node id in the parent scope.

      let mut call_builder = pop_builder(node_stack);

      let par_builder: &mut Builder = node_stack.front_mut().unwrap();

      let call_index = Builder::create_id(&mut par_builder.node_id).usize();

      par_builder.add_node(RVSDGInternalNode::PlaceHolder, call.clone().into(), Default::default());

      {
        let out_op = par_builder.add_output_node(Default::default(), ASTNode::None);
        par_builder.update_var(VarId::HeapContext, out_op, Default::default());
        let mut input = RSDVGBinding::default();
        input.in_op = heap_id;
        input.out_op = out_op;
        input.id = VarId::HeapContext;
        call_builder.add_output(input);
      }

      let outputs = &mut call_builder.node.outputs;
      for output_index in 0..outputs.len() {
        let id = par_builder.add_output_node(Default::default(), ASTNode::None);
        outputs[output_index].out_op = id;
        par_builder.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);
        return ThreadedGraphId(id, node_stack.front().unwrap().id);
      }

      par_builder.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);

      ThreadedGraphId(Default::default(), node_stack.front().unwrap().id)
    }

    expression_Value::RawMatch(mtch) => process_match(node_stack, mtch, ty_db, true),

    expression_Value::RawAggregateInstantiation(agg) => {
      let allocator_ctx_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();

      let builder = node_stack.front_mut().unwrap();

      let agg_ty = builder.create_gen_var(VarId::Generic, Default::default(), agg.clone().into(), Default::default());

      let mut agg_op =
        builder.add_simple(IROp::AGG_DECL, [allocator_ctx_op, Default::default(), Default::default()], agg_ty, ASTNode::RawAggregateInstantiation(agg.clone()));
      builder.update_var(VarId::HeapContext, agg_op, Default::default());

      let ty_vars = builder.type_vars();
      let agg_index = agg_ty.generic_id().unwrap();

      for (index, init) in agg.inits.iter().enumerate() {
        let expr_id = process_expression(&init.expression.expr, node_stack, ty_db);
        let expr_id = resolve_binding(expr_id, node_stack, Default::default());

        if let Some(name) = &init.name {
          let builder = node_stack.front_mut().unwrap();
          let name_id = name.id.intern();
          let ty_index = ty_vars.len();
          let mem_ty = builder.create_gen_var(VarId::MemRef(ty_index), Default::default(), init.clone().into(), Default::default());

          let label_id = builder.get_label(name_id);

          let ref_id = builder.add_simple(IROp::REF, [agg_op, label_id, Default::default()], mem_ty, name.clone().into());

          let allocator_ctx_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();

          let builder = node_stack.front_mut().unwrap();
          let store_op =
            builder.add_simple(IROp::STORE, [ref_id, expr_id, allocator_ctx_op], Default::default(), ASTNode::RawAggregateMemberInit(init.clone()));

          builder.update_var(VarId::HeapContext, store_op, Default::default());

          ty_vars[agg_index].add_mem(name_id, mem_ty, Default::default());
        } else {
          let allocator_ctx_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();
          let builder = node_stack.front_mut().unwrap();

          let const_id = builder.add_const(ConstVal::new(ty_u32.to_primitive().unwrap(), index as u32), init.clone().into());

          let ref_id = builder.add_simple(IROp::REF, [agg_op, const_id, Default::default()], Default::default(), Default::default());

          let store_op =
            builder.add_simple(IROp::STORE, [ref_id, expr_id, allocator_ctx_op], Default::default(), ASTNode::RawAggregateMemberInit(init.clone()));

          builder.update_var(VarId::HeapContext, store_op, Default::default());

          ty_vars[agg_index].add(type_solve::VarConstraint::Indexable);
        }
      }

      ThreadedGraphId(agg_op, node_stack.front().unwrap().id)
    }

    expression_Value::MemberCompositeAccess(mem) => match lookup_var(mem, node_stack, true, ty_db) {
      VarLookup::Mem { ref_id: output, id } => {
        //let se_op = builder.var_lookup.get(&se).unwrap().op;

        let allocator_ctx_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();
        let builder = node_stack.front_mut().unwrap();
        let load_id = builder.add_simple(IROp::LOAD, [output, allocator_ctx_op, Default::default()], Default::default(), Default::default());
        ThreadedGraphId(load_id, node_stack.front().unwrap().id)
      }
      VarLookup::Var(var) => ThreadedGraphId(var, node_stack.front().unwrap().id),
      VarLookup::None(name) => {
        todo!("report error on lookup of {name} \n{} \n {node_stack:#?}", mem.tok.blame(0, 0, "", None))
      }
    },
    node => todo!("{node:#?}"),
  }
}

fn process_match(node_stack: &mut VecDeque<Builder>, mtch: &Arc<RawMatch<Token>>, ty_db: &mut TypeDatabase, seal_activation: bool) -> ThreadedGraphId {
  // create the match entry
  let builder = node_stack.front_mut().unwrap();
  builder.create_gen_var(VarId::MatchOutputVal, Default::default(), Default::default(), Default::default());

  let eval_id = process_expression(&mtch.expression.clone().into(), node_stack, ty_db);

  let eval_id = resolve_binding(eval_id, node_stack, Default::default());
  let builder = node_stack.front_mut().unwrap();

  builder.create_gen_var(VarId::MatchInputExpr, eval_id, Default::default(), Default::default());
  let id = builder.create_gen_var(VarId::MatchActivation, Default::default(), Default::default(), Default::default());

  push_new_builder(node_stack, RVSDGNodeType::Match);
  push_new_builder(node_stack, RVSDGNodeType::MatchHead);

  {
    let match_builder = node_stack.front_mut().unwrap();
    let mut merges = vec![];
    for clause in mtch.clauses.iter().chain(mtch.default_clause.iter()) {
      push_new_builder(node_stack, RVSDGNodeType::MatchActivation);

      let activation_op = if clause.default {
        let builder = node_stack.front_mut().unwrap();
        builder.add_const(ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), 1 as u64), Default::default())
      } else if let Some(expr) = clause.expr.as_ref() {
        let builder = node_stack.front_mut().unwrap();
        let v = process_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db);

        let v = resolve_binding(v, node_stack, Default::default());

        let op_ty = match expr.op.as_str() {
          ">" => IROp::GR,
          "<" => IROp::LS,
          ">=" => IROp::GE,
          "<=" => IROp::LE,
          "==" => IROp::EQ,
          "!=" => IROp::NE,
          _ => todo!(),
        };

        let eval_id = commit_writes_to_var(VarId::MatchInputExpr, node_stack).unwrap();
        let builder = node_stack.front_mut().unwrap();
        builder.add_simple(op_ty, [eval_id, v, Default::default()], Default::default(), ASTNode::RawExprMatch(expr.clone()))
      } else {
        unreachable!()
      };

      ignore_writes_to_var(VarId::MatchActivation, node_stack);

      let builder = node_stack.front_mut().unwrap();
      builder.update_var(VarId::MatchActivation, activation_op, Default::default());

      merges.push((pop_builder(node_stack), Default::default()));
    }

    merge_multiple_nodes(node_stack, merges);
  }
  pop_and_merge_single_node(node_stack, Default::default()); // MatchHead

  push_new_builder(node_stack, RVSDGNodeType::MatchBody);
  {
    let mut merges = vec![];
    for clause in mtch.clauses.iter().chain(mtch.default_clause.iter()) {
      push_new_builder(node_stack, RVSDGNodeType::MatchClause);
      let base_len = node_stack.len();

      let def_id = process_expression(&clause.scope.clone().into(), node_stack, ty_db);

      let def_id = resolve_binding(def_id, node_stack, Default::default());

      if def_id.is_valid() {
        ignore_writes_to_var(VarId::MatchOutputVal, node_stack); // Ensure var is present in current scope.

        let builder = node_stack.front_mut().unwrap();
        builder.update_var(VarId::MatchOutputVal, def_id, Default::default());
      }

      while node_stack.len() > base_len {
        pop_and_merge_single_node(node_stack, Default::default());
      }

      merges.push((pop_builder(node_stack), Default::default()));
    }

    merge_multiple_nodes(node_stack, merges);
  }

  pop_and_merge_single_node(node_stack, Default::default()); // MatchBody

  pop_and_merge_single_node(node_stack, Default::default()); // Match

  if seal_activation {
    seal_var(VarId::MatchActivation, node_stack, Default::default());
  }

  remove_var(VarId::MatchInputExpr, node_stack);

  let id = take_var(VarId::MatchOutputVal, node_stack);

  ThreadedGraphId(id, node_stack.front().unwrap().id)
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) {
  let var = &expr.var;
  let expr = &expr.expression;

  let graph_id = process_expression(&expr.expr, node_stack, ty_db);
  let graph_id = resolve_binding(graph_id, node_stack, Default::default());

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = VarId::VarName(decl.var.id.intern());

      let ty = get_type(&decl.ty, false, ty_db).unwrap();

      let builder = node_stack.front_mut().unwrap();

      match &expr.expr {
        expression_Value::RawAggregateInstantiation(..) => {
          let var = builder.node.types[graph_id.usize()];
          builder.create_var_lookup(root_name, graph_id, var, decl.clone().into());
          builder.add_op_constraint(OPConstraint::GenVarToTy(var.generic_id().unwrap(), ty));
        }
        _ => {
          builder.create_gen_var(root_name, graph_id, ASTNode::Var(decl.var.clone()), ty);
        }
      }
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = VarId::VarName(mem.root.name.id.intern());
      let ast = ASTNode::MemberCompositeAccess(mem.clone());

      match lookup_var(&mem, node_stack, true, ty_db) {
        VarLookup::Mem { ref_id: mem_id, id } => {
          let allocator_ctx_op = commit_writes_to_var(VarId::HeapContext, node_stack).unwrap();
          let builder = node_stack.front_mut().unwrap();
          let store_op = builder.add_simple(IROp::STORE, [mem_id, graph_id, allocator_ctx_op], Default::default(), ASTNode::Expression(expr.clone()));
          builder.update_var(VarId::HeapContext, store_op, Default::default());

          //builder.update_var(se, store_op, ast);
        }
        VarLookup::Var(output) => {
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(root_name, graph_id, ast);
        }
        VarLookup::None(..) => {
          let builder = node_stack.front_mut().unwrap();
          builder.create_gen_var(root_name, graph_id, ast, Default::default());
        }
      }
    }
    _ => todo!(),
  }
}

fn binary_expr(
  op: IROp,
  left: expression_Value<Token>,
  right: expression_Value<Token>,
  node_stack: &mut VecDeque<Builder>,
  node: ASTNode,
  ty_db: &mut TypeDatabase,
) -> ThreadedGraphId {
  let left_id = process_expression(&left, node_stack, ty_db);
  let right_id = process_expression(&right, node_stack, ty_db);

  let left_id = resolve_binding(left_id, node_stack, Default::default());
  let right_id = resolve_binding(right_id, node_stack, Default::default());

  let builder = node_stack.front_mut().unwrap();

  let id = (builder.add_simple(op, [left_id, right_id, Default::default()], Default::default(), node));

  ThreadedGraphId(id, builder.id)
}

enum VarLookup {
  Var(IRGraphId),
  Mem { ref_id: IRGraphId, id: VarId },
  None(IString),
}

fn lookup_var<'a>(mem: &MemberCompositeAccess<Token>, node_stack: &'a mut VecDeque<Builder>, read: bool, ty_db: &mut TypeDatabase) -> VarLookup {
  let mut name = mem.root.name.id.clone();

  let mut var_id = VarId::VarName(name.intern());

  if mem.sub_members.len() > 0 {
    if let Some(mut prev_ref) = if read { commit_writes_to_var(var_id, node_stack) } else { ignore_writes_to_var(var_id, node_stack) } {
      let ty = node_stack.front().unwrap().var_lookup.get(&var_id).unwrap().ty;

      let ty_vars = node_stack.front().unwrap().type_vars();
      let mut index = ty.generic_id().unwrap();

      for sub_member in &mem.sub_members {
        match sub_member {
          member_group_Value::NamedMember(name) => {
            let name_id: IString = name.name.id.intern();

            let ty: Type = if let Some((_, ty)) = ty_vars[index].get_mem(name_id) {
              index = ty.generic_id().unwrap();
              var_id = VarId::MemRef(index);
              commit_writes_to_var(var_id, node_stack);
              ty
            } else {
              let builder = node_stack.front_mut().unwrap();
              let ty_index = ty_vars.len();
              let ty: Type = builder.create_gen_var(VarId::MemRef(ty_index), Default::default(), sub_member.clone().into(), Default::default());
              let mut var = &mut ty_vars[index];
              var.add_mem(name_id, ty, Default::default());

              index = ty_index;

              var_id = VarId::MemRef(ty_index);

              ty
            };
            let builder = node_stack.front_mut().unwrap();

            let var = builder.var_lookup.get((&var_id)).unwrap();

            if var.op.is_valid() {
              prev_ref = var.op;
            } else {
              let label_id = builder.get_label(name_id);
              prev_ref = builder.add_simple(IROp::REF, [prev_ref, label_id, Default::default()], ty, ASTNode::NamedMember(name.clone()));
              builder.var_lookup.get_mut((&var_id)).unwrap().op = prev_ref;
            }
          }
          member_group_Value::IndexedMember(index_lookup) => {
            let index_op = process_expression(&index_lookup.expression.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db).0;
            let builder = node_stack.front_mut().unwrap();
            prev_ref = builder.add_simple(IROp::REF, [prev_ref, index_op, Default::default()], Default::default(), index_lookup.clone().into());
          }
          _ => unreachable!(),
        }
      }

      VarLookup::Mem { ref_id: prev_ref, id: var_id }
    } else {
      panic!("AAA");
      VarLookup::None(Default::default())
    }
  } else if let Some(var) = if read { commit_writes_to_var(var_id, node_stack) } else { ignore_writes_to_var(var_id, node_stack) } {
    VarLookup::Var(var)
  } else {
    VarLookup::None(var_id.to_string())
  }
}

fn pop_builder(node_stack: &mut VecDeque<Builder>) -> Builder {
  node_stack.pop_front().unwrap()
}

fn pop_and_merge_single_node(node_stack: &mut VecDeque<Builder>, ast: ASTNode) {
  let child_builder = pop_builder(node_stack);

  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn pop_and_merge_single_node_with_return(node_stack: &mut VecDeque<Builder>, ast: ASTNode) {
  let lu = node_stack.front_mut().unwrap().var_lookup.clone();

  for (var_id, var) in lu {
    match var_id {
      VarId::Return => {
        commit_writes_to_var(VarId::Return, node_stack);
        let child_builder = node_stack.front_mut().unwrap();

        if child_builder.node.nodes.len() > 0 {
          let op = IRGraphId::new(child_builder.node.nodes.len() - 1);
          child_builder.update_var(VarId::Return, op, Default::default());
        }
      }
      VarId::SideEffect(_) => {
        commit_writes_to_var(VarId::Return, node_stack);
        let child_builder = node_stack.front_mut().unwrap();

        if child_builder.node.nodes.len() > 0 {
          let op = IRGraphId::new(child_builder.node.nodes.len() - 1);
          child_builder.update_var(var_id, op, Default::default());
        }
      }
      _ => {}
    }
  }

  let child_builder = node_stack.front_mut().unwrap();
  if !child_builder.var_lookup.contains_key(&VarId::Return) {}

  let child_builder = pop_builder(node_stack);

  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn push_new_builder(node_stack: &mut VecDeque<Builder>, ty: RVSDGNodeType) {
  let top_builder = node_stack.front().unwrap();
  let id = top_builder.inc_counter();
  let mut entry = Builder::new(id, top_builder.id_counter, top_builder.op_constraints, top_builder.type_vars);
  entry.node.id = id;

  entry.node.ty = ty;
  node_stack.push_front(entry);
}

fn merge_multiple_nodes(node_stack: &mut VecDeque<Builder>, mut children: Vec<(Builder, ASTNode)>) {
  let par_builder = node_stack.front_mut().unwrap();
  let par_ty = par_builder.node.ty;

  for (name, par_var) in par_builder.var_lookup.iter_mut() {
    let par_var_nonce = par_var.nonce;

    for (i, (c_b, _)) in children.iter_mut().enumerate() {
      let c_i = i + par_builder.node_id;

      if let Some(mut child_var) = c_b.var_lookup.remove(name) {
        let var_is_input_only = matches!(name, VarId::MatchInputExpr);

        if child_var.origin_node_id < c_b.id && !var_is_input_only && par_var_nonce != child_var.nonce
        /*  && child_var.op.is_valid()  */
        {
          let child_var_ty = child_var.ty;
          // Merge the var id into the parent scope

          let var_id = child_var.id;
          let mut in_out_link = None;

          if c_b.node.ty == RVSDGNodeType::Loop {
            for i in 0..c_b.node.inputs.len() {
              if c_b.node.inputs[i].id == var_id {
                in_out_link = Some(i as u16);
                break;
              }
            }
          }

          let output_index = c_b.node.outputs.len();

          let op = commit_writes(c_b, &mut child_var);

          c_b.node.outputs.push(RSDVGBinding { in_op: child_var.op, out_op: Default::default(), id: var_id, in_out_link });

          par_var.writes.push(Write::IntraNodeBind { node_index: c_i, output_index });

          par_var.nonce = par_var_nonce + 1;
        }
      }
    }
  }

  for (child, ast) in children.into_iter() {
    par_builder.add_node(RVSDGInternalNode::Complex(child.node), ast, Default::default());
    par_builder.node_id += 1
  }
}

fn commit_writes(builder: &mut Builder, var: &mut Var) -> IRGraphId {
  let num_of_writes = var.writes.len();

  let v = var.clone();

  let mut input_op = IRGraphId::default();

  if num_of_writes > 0 {
    for write in var.writes.drain(0..) {
      match write {
        Write::IntraNodeBind { node_index, output_index } => {
          if input_op.is_invalid() {
            input_op = builder.add_input_node(var.ty, Default::default());
            var.op = input_op;
          }

          match &mut builder.node.nodes[node_index] {
            RVSDGInternalNode::Complex(cmplx) => {
              cmplx.outputs[output_index].out_op = input_op;
            }
            _ => panic!("Invalid"),
          }
        }
      }
    }

    input_op
  } else {
    var.op
  }
}

fn commit_writes_to_var<'a>(var_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_id) {
      found_in_index = i as i32;
      let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
      let par_var = par_data.var_lookup.get_mut(&var_id).unwrap();
      let par_data_2 = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();

      par_var.op = commit_writes(par_data_2, par_var);

      let mut var = par_var.clone();

      for curr_index in (0..found_in_index).rev() {
        let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();
        if !var.op.is_invalid() {
          var.op = par_data
            .add_binding(
              RSDVGBinding { id: var.id, in_op: var.op, out_op: Default::default(), in_out_link: None },
              var.ast.clone(),
              var.ty,
              BindingType::ParamBinding,
            )
            .out_op;
        }

        par_data.var_lookup.insert(var_id, var.clone());
      }

      return Some(var.op);
    }
  }

  None
}

fn ignore_writes_to_var<'a>(var_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_id) {
      found_in_index = i as i32;
      let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
      let par_var = par_data.var_lookup.get_mut(&var_id).unwrap();
      let mut var = par_var.clone();

      var.writes.clear();

      for curr_index in (0..found_in_index).rev() {
        let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

        par_data.var_lookup.insert(var_id, var.clone());
      }

      return Some(var.op);
    }
  }

  None
}

fn import_binding(binding: ThreadedGraphId, node_stack: &mut VecDeque<Builder>) -> IRGraphId {
  let ThreadedGraphId(op, node_id) = binding;

  let builder = node_stack.front_mut().unwrap();
  return builder.add_binding(RSDVGBinding { in_op: op, ..Default::default() }, Default::default(), Default::default(), BindingType::IntraBinding).out_op;
}

fn resolve_binding(binding: ThreadedGraphId, node_stack: &mut VecDeque<Builder>, var_id: VarId) -> IRGraphId {
  let ThreadedGraphId(op, node_id) = binding;

  if op.is_valid() {
    let builder = node_stack.front_mut().unwrap();

    if builder.id != node_id {
      return builder
        .add_binding(RSDVGBinding { id: var_id, in_op: op, ..Default::default() }, Default::default(), Default::default(), BindingType::IntraBinding)
        .out_op;

      // The binding id should be found as a child of the penultimate node.
      todo!(" Resolve binding from {node_id} {builder:#?}")
    } else {
      op
    }
  } else {
    op
  }
}

fn seal_var(var_id: VarId, node_stack: &mut VecDeque<Builder>, ty: Type) {
  let builder = node_stack.front_mut().unwrap();
  if let Some(mut var) = builder.var_lookup.remove(&var_id) {
    let var_op = commit_writes(builder, &mut var);

    if var_op.is_valid() {
      let var_op = if !ty.is_open() {
        commit_writes_to_var(var_id, node_stack);
        let builder = node_stack.front_mut().unwrap();
        builder.add_op_constraint(OPConstraint::GenVarToTy(var.ty.generic_id().unwrap(), ty));
        var_op
      } else {
        var_op
      };

      let builder = node_stack.front_mut().unwrap();
      builder.set_type_if_undefined(var_op, var.ty);

      builder.add_output(RSDVGBinding { in_op: var_op, id: var_id, out_op: Default::default(), in_out_link: None });
    }
  }
}

fn take_var<'a>(var_name_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> IRGraphId {
  let builder = node_stack.front_mut().unwrap();
  if let Some(mut var) = builder.var_lookup.remove(&var_name_id) {
    let var_id = commit_writes(builder, &mut var);

    if var_id.is_valid() {
      builder.set_type_if_undefined(var_id, var.ty);
    }

    var_id
  } else {
    Default::default()
  }
}

fn remove_var<'a>(var_name_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) {
  let mut id = IRGraphId::default();
  let mut ty = Type::Undefined;

  let builder = node_stack.front_mut().unwrap();

  builder.var_lookup.remove(&var_name_id);

  for (i, entry) in builder.node.outputs.iter_mut().enumerate() {
    if entry.id == var_name_id {
      builder.node.outputs.remove(i);
      return;
    }
  }
}

pub fn get_type(ir_type: &type_Value<Token>, insert_unresolved: bool, ty_db: &mut TypeDatabase) -> Option<Type> {
  use type_Value::*;
  match ir_type {
    Type_Flag(_) => Option::None,
    Type_u8(_) => ty_db.get_ty("u8"),
    Type_u16(_) => ty_db.get_ty("u16"),
    Type_u32(_) => ty_db.get_ty("u32"),
    Type_u64(_) => ty_db.get_ty("u64"),
    Type_i8(_) => ty_db.get_ty("i8"),
    Type_i16(_) => ty_db.get_ty("i16"),
    Type_i32(_) => ty_db.get_ty("i32"),
    Type_i64(_) => ty_db.get_ty("i64"),
    Type_f32(_) => ty_db.get_ty("f32"),
    Type_f64(_) => ty_db.get_ty("f64"),
    /* Type_f32v2(_) => ty_db.get_ty("f32v2"),
    Type_f32v4(_) => ty_db.get_ty("f32v4"),
    Type_f64v2(_) => ty_db.get_ty("f64v2"),
    Type_f64v4(_) => ty_db.get_ty("f64v4"), */
    Type_Generic(_) => Some(Type::Generic { ptr_count: 0, gen_index: 0 }),
    Type_Reference(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        ty_db.to_ptr(base_type)
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        use lifetime_Value::*;
        match &ptr.ptr_type {
          GlobalLifetime(_) => {
            ty_db.to_ptr(base_type)
            //Some(TypeSlot::GlobalIndex(0, type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), 0, base_type)) as u32))
          }
          ScopedLifetime(scope)  =>

          ty_db.to_ptr(base_type)/* Some(TypeSlot::GlobalIndex(
            0,
            type_db.get_or_add_type_index(format!("{}*{}", scope.val, base_type.ty_gb(type_db)).intern(), Type::Pointer(scope.val.intern(), 0, base_type)) as u32,
          )) */,
          _ => unreachable!(),
        }
      } else {
        Option::None
      }
    }
    Type_Variable(type_var) => Some(ty_db.get_or_insert_complex_type(type_var.name.id.as_str())),
    _t => Option::None,
  }
}
