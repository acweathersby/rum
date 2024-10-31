use crate::{
  ir::{
    ir_rvsdg::{IROp, *},
    types::TypeDatabase,
  },
  istring::CachedString,
  parser::script_parser::*,
};
use std::{
  any::Any,
  collections::{HashMap, VecDeque},
};

#[derive(Clone, Debug)]
struct Var {
  name:           IString,
  origin_node_id: u32,
  op:             IRGraphId,
  ast:            ASTNode,
  input_only:     bool,
  writes:         Vec<(usize, IRGraphId)>,
  nonce:          usize,
}

#[derive(Debug)]
struct Builder {
  id:         u32,
  id_counter: *mut u32,

  node:         Box<RVSDGNode>,
  /// Monotonically increasing identifier for the node stored in this builder
  node_id:      usize,
  label_lookup: HashMap<IString, IRGraphId>,
  const_lookup: HashMap<ConstVal, IRGraphId>,
  /// Stores variable names that have been declared in the current scope, or
  /// a previous scope.
  var_lookup:   HashMap<IString, Var>,
}

type WIPNode = Builder;

impl Builder {
  pub fn new(id: u32, counter: *mut u32) -> Self {
    Builder {
      id,
      id_counter: counter,
      node: Default::default(),
      node_id: Default::default(),
      label_lookup: Default::default(),
      const_lookup: Default::default(),
      var_lookup: Default::default(),
    }
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

  pub fn add_simple(&mut self, op: IROp, operands: [IRGraphId; 2], ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Simple { id, op, operands });
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_ty_binding(&mut self, in_id: IRGraphId, ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::TypeBinding(in_id, ty));
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_input(&mut self, mut binding: RSDVGBinding, node: ASTNode) -> RSDVGBinding {
    let id = Self::create_id(&mut self.node_id);
    binding.out_id = id;
    binding.input_index = self.node.inputs.len() as u32;
    self.node.inputs.push(binding);
    self.add_input_node_internal(id, binding.ty, binding.input_index as usize, node);
    binding
  }

  pub fn add_input_node(&mut self, ty: Type, input_index: usize, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_input_node_internal(id, ty, input_index, node)
  }

  fn add_input_node_internal(&mut self, id: IRGraphId, ty: Type, input_index: usize, node: ASTNode) -> IRGraphId {
    self.node.nodes.push(RVSDGInternalNode::Input { id, ty, input_index });
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_output(&mut self, binding: RSDVGBinding) -> RSDVGBinding {
    self.node.outputs.push(binding);
    binding
  }

  pub fn add_const(&mut self, const_val: ConstVal, node: ASTNode) -> IRGraphId {
    let const_id = self.get_const(const_val);
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Simple { id, op: IROp::CONST_DECL, operands: [const_id, Default::default()] });
    self.node.source_nodes.push(node);
    id
  }

  pub fn get_label(&mut self, name: IString) -> IRGraphId {
    let Self { label_lookup, node_id, node, .. } = self;
    match label_lookup.entry(name) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        node.nodes.push(RVSDGInternalNode::Label(id, name));
        node.source_nodes.push(Default::default());
        *val.insert(id)
      }
    }
  }

  pub fn create_var(&mut self, name: IString, op: IRGraphId, ast: ASTNode, input_only: bool) {
    println!("new var: {:?}", (self.id, name));
    self.var_lookup.insert(name, Var { name, origin_node_id: self.id, op, ast, input_only, writes: Default::default(), nonce: 0 });
  }

  pub fn update_var(&mut self, name: IString, op: IRGraphId, ast: ASTNode) -> bool {
    if let Some(var) = self.var_lookup.get_mut(&name) {
      var.op = op;
      var.ast = ast;
      var.nonce += 1;
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
        node.nodes.push(RVSDGInternalNode::Const(id.0, const_val));
        node.source_nodes.push(Default::default());
        *val.insert(id)
      }
    }
  }
}

pub fn lower_ast_to_rvsdg(module: &std::sync::Arc<RawModule<Token>>, ty_db: &mut TypeDatabase) -> Vec<IString> {
  let members = &module.members;

  let mut output_names = Vec::new();

  for mem in &members.members {
    match mem {
      module_members_group_Value::AnnotatedModMember(annotation) => match &annotation.member {
        module_member_Value::RawBoundType(bound_type) => match &bound_type.ty {
          type_Value::Type_Struct(strct) => {
            let name = bound_type.name.id.intern();
            let struct_: Box<RVSDGNode> = lower_struct_to_rsvdg(bound_type.name.id.intern(), strct, ty_db);

            ty_db.add_ty(struct_.id, struct_.clone());
          }
          _ => unreachable!(),
        },
        module_member_Value::RawRoutine(rt) => {
          let funct = lower_fn_to_rsvdg(rt, ty_db);

          let name = funct.id;

          ty_db.add_ty(funct.id, funct.clone());
        }
        module_member_Value::RawScope(scope) => {}
        _ => unreachable!(),
      },
      module_members_group_Value::AnnotationVariable(var) => {}
      module_members_group_Value::LifetimeVariable(var) => {}
      _ => unreachable!(),
    }
  }

  output_names
}

fn lower_struct_to_rsvdg(binding_name: IString, struct_: &Type_Struct<Token>, ty_db: &mut TypeDatabase) -> Box<RVSDGNode> {
  let mut node = RVSDGNode::default();

  node.ty = RVSDGNodeType::Struct;
  node.id = binding_name;

  let mut node_id = 0;

  for prop in &struct_.properties {
    match prop {
      property_Value::Property(prop) => {
        let name = prop.name.id.intern();
        let ty = get_type(&prop.ty, false, ty_db).unwrap_or_default();

        let mut output_binding = RSDVGBinding::default();
        let id = Builder::create_id(&mut node_id);

        output_binding.name = name;
        output_binding.ty = ty;
        output_binding.in_id = id;
        output_binding.out_id = Default::default();
        output_binding.input_index = node.inputs.len() as u32;

        node.outputs.push(output_binding);

        node.nodes.push(RVSDGInternalNode::Input { id, ty: output_binding.ty, input_index: output_binding.input_index as usize });
        node.source_nodes.push(ASTNode::Property(prop.clone()));
      }
      n => todo!("Handle construction of {n:?}"),
    }
  }

  Box::new(node)
}

fn lower_fn_to_rsvdg(fn_decl: &RawRoutine<Token>, ty_db: &mut TypeDatabase) -> Box<RVSDGNode> {
  let params = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(ty) => &ty.params,
    routine_type_Value::RawProcedureType(ty) => &ty.params,
    _ => unreachable!(),
  };

  let expr = &fn_decl.expression.expr;

  let mut node_stack = VecDeque::new();
  let mut counter = 0;

  let mut builder = Builder::new(0, &mut counter);
  builder.node.ty = RVSDGNodeType::Undefined;
  builder.create_var("RET".intern(), Default::default(), Default::default(), false);

  node_stack.push_front(builder);

  push_new_builder(&mut node_stack, RVSDGNodeType::Function, fn_decl.name.id.intern());

  let fn_builder = node_stack.front_mut().unwrap();

  insert_params(params, fn_builder, ty_db);

  let ret_val = process_expression(expr, &mut node_stack, ty_db);

  let ret_ty = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(fn_ty) => insert_returns(ret_val, fn_ty, &mut node_stack, ty_db, &mut Vec::new()),
    _ => Default::default(),
  };

  dbg!(ret_ty);

  while node_stack.len() > 2 {
    pop_and_merge_single_node_with_return(&mut node_stack, Default::default());
  }

  seal_var("RET".intern(), &mut node_stack, ret_ty);

  pop_and_merge_single_node(&mut node_stack, Default::default());

  let mut fn_node = node_stack.pop_back().unwrap().node;

  if let RVSDGInternalNode::Complex(node) = fn_node.nodes.remove(0) {
    dbg!(&node);
    //panic!("");
    node
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

  if !ret_val.is_invalid() {
    dbg!((ret, ty));
    let builder = node_stack.front_mut().unwrap();

    let ret_id = "RET".intern();
    let out_id = write_var(ret_id, node_stack).unwrap();
    let builder = node_stack.front_mut().unwrap();
    builder.update_var(ret_id, ret_val, Default::default());

    ty
  } else {
    ty
  }
}

fn insert_params(params: &std::sync::Arc<Params<Token>>, builder: &mut Builder, ty_db: &mut TypeDatabase) {
  for param in &params.params {
    let ty = get_type(&param.ty.ty, false, ty_db).unwrap_or_default();

    let name = param.var.id.intern();

    let ast = ASTNode::RawParamBinding(param.clone());

    let input = builder.add_input(RSDVGBinding { name, in_id: Default::default(), ty, ..Default::default() }, ast);

    builder.create_var(name, input.out_id, ASTNode::RawParamBinding(param.clone()), true);
  }
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> IRGraphId {
  match expr {
    expression_Value::MemberCompositeAccess(mem) => match lookup_var(mem, node_stack) {
      VarLookup::Mem(output) => output,
      VarLookup::Var(var) => var,
      VarLookup::None(name) => {
        todo!("report error on lookup of {name} \n{} \n {node_stack:#?}", mem.tok.blame(0, 0, "", None))
      }
    },

    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      node_stack.front_mut().unwrap().add_const(
        if string_val.contains(".") {
          ConstVal::new(ty_db.get_ty("f64").expect("f64 should exist").to_primitive().unwrap(), num.val)
        } else {
          ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), string_val.parse::<u64>().unwrap())
        },
        ASTNode::RawNum(num.clone()),
      )
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
      let mut last_id = IRGraphId::default();
      for expr in &block.statements {
        match expr {
          statement_Value::Expression(expr) => {
            last_id = process_expression(&expr.expr, node_stack, ty_db);
          }
          statement_Value::RawAssignment(assign) => {
            process_assign(&assign, node_stack, ty_db);
            last_id = IRGraphId::default()
          }
          statement_Value::RawLoop(loop_expr) => {
            let loop_val_id = "__loop_val__".intern();
            let loop_val_id = "__side_effect__".intern();
            let loop_val_id = "__return__".intern();
            let loop_val_id = "__break__".intern();

            push_new_builder(node_stack, RVSDGNodeType::Loop, Default::default());

            {
              match &loop_expr.scope {
                loop_statement_group_1_Value::RawMatch(mtch) => {
                  process_expression(&expression_Value::RawMatch(mtch.clone().into()), node_stack, ty_db);
                }
                loop_statement_group_1_Value::RawBlock(block) => {
                  process_expression(&expression_Value::RawBlock(block.clone().into()), node_stack, ty_db);
                }
                ast => todo!("{ast:#?}"),
              }
            }

            pop_and_merge_single_node(node_stack, loop_expr.clone().into());

            let block = node_stack.front().unwrap();

            //todo!("{block:#?} ");
          }
          ty => todo!("{ty:#?}"),
        }
      }

      match &block.exit {
        Some(block_expression_group_3_Value::BlockExitExpressions(e)) => {
          let out_id = process_expression(&e.expression.expr, node_stack, ty_db);
          let ret_id = "RET".intern();

          let _ = write_var(ret_id, node_stack).unwrap();
          let builder = node_stack.front_mut().unwrap();

          builder.update_var(ret_id, out_id, Default::default());

          Default::default()
        }
        _ => last_id,
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      let call_id = match lookup_var(&call.member, node_stack) {
        VarLookup::Mem(mem_id) => mem_id,
        VarLookup::Var(var) => var,
        VarLookup::None(name) => node_stack.front_mut().unwrap().get_label(name),
      };

      for arg in &call.args {
        args.push((process_expression(&arg.expr, node_stack, ty_db), ASTNode::expression_Value(arg.expr.clone())));
      }

      push_new_builder(node_stack, RVSDGNodeType::Call, Default::default());

      let call_builder = node_stack.front_mut().unwrap();

      {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = call_id;
        input.name = "__NAME__".intern();
        call_builder.add_input(input, ASTNode::MemberCompositeAccess(call.member.clone()));
      }

      for (arg, node) in args {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = arg;
        call_builder.add_input(input, node);
      }

      let mut output = RSDVGBinding::default();
      output.name = "RET".intern();
      //input.out_id = ret_id;
      call_builder.add_output(output);

      // The return value should now be mapped to a new_node id in the parent scope.

      let mut call_builder = pop_builder(node_stack);

      let par_node = node_stack.front_mut().unwrap();

      let outputs = &mut call_builder.node.outputs;

      let call_index = Builder::create_id(&mut par_node.node_id).usize();
      par_node.node.nodes.push(RVSDGInternalNode::PlaceHolder);
      par_node.node.source_nodes.push(ASTNode::RawCall(call.clone()));

      for output_index in 0..outputs.len() {
        let id = par_node.add_input_node(output.ty, 0, ASTNode::None);
        outputs[output_index].out_id = id;
        par_node.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);
        return id;
      }

      par_node.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);

      Default::default()
    }

    expression_Value::RawMatch(mtch) => {
      let match_expression_id = "__match__expr__".intern();
      let match_value_id = "__match__val__".intern();

      // create the match entry
      let builder = node_stack.front_mut().unwrap();
      builder.create_var(match_value_id, Default::default(), Default::default(), false);

      let eval_id = process_expression(&mtch.expression.clone().into(), node_stack, ty_db);

      let builder = node_stack.front_mut().unwrap();
      builder.create_var(match_expression_id, eval_id, Default::default(), true);

      let match_activation_id = "__activation_val__".intern();
      let builder = node_stack.front_mut().unwrap();
      builder.create_var(match_activation_id, Default::default(), Default::default(), false);

      push_new_builder(node_stack, RVSDGNodeType::MatchHead, Default::default());
      {
        let mut merges = vec![];

        let match_builder = node_stack.front_mut().unwrap();

        for clause in mtch.clauses.iter().chain(mtch.default_clause.iter()) {
          push_new_builder(node_stack, RVSDGNodeType::MatchClause, Default::default());

          push_new_builder(node_stack, RVSDGNodeType::MatchActivation, Default::default());

          let activation_op = if clause.default {
            let builder = node_stack.front_mut().unwrap();
            builder.add_const(ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), 1 as u64), Default::default())
          } else if let Some(expr) = clause.expr.as_ref() {
            let builder = node_stack.front_mut().unwrap();
            let v = process_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db);

            let op_ty = match expr.op.as_str() {
              ">" => IROp::GR,
              "<" => IROp::LS,
              ">=" => IROp::GE,
              "<=" => IROp::LE,
              "==" => IROp::EQ,
              "!=" => IROp::NE,
              _ => todo!(),
            };

            let eval_id = read_var(match_expression_id, node_stack).unwrap();
            let builder = node_stack.front_mut().unwrap();
            builder.add_simple(op_ty, [eval_id, v], Default::default(), ASTNode::RawExprMatch(expr.clone()))
          } else {
            unreachable!()
          };

          write_var(match_activation_id, node_stack);
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(match_activation_id, activation_op, Default::default());

          pop_and_merge_single_node(node_stack, Default::default());

          push_new_builder(node_stack, RVSDGNodeType::MatchBody, Default::default());

          let def_id = process_expression(&clause.scope.clone().into(), node_stack, ty_db);

          if def_id.is_valid() {
            write_var(match_value_id, node_stack); // Ensure var is present in current scope.

            let builder = node_stack.front_mut().unwrap();
            builder.update_var(match_value_id, def_id, Default::default());
          }

          pop_and_merge_single_node(node_stack, Default::default());

          merges.push((pop_builder(node_stack), Default::default()));
        }

        merge_multiple_nodes(node_stack, merges);

        seal_var(match_activation_id, node_stack, Default::default());
      }

      pop_and_merge_single_node(node_stack, Default::default());
      remove_var(match_expression_id, node_stack);

      push_new_builder(node_stack, RVSDGNodeType::GenericBlock, Default::default());

      take_var(match_value_id, node_stack)
    }

    expression_Value::RawAggregateInstantiation(agg) => {
      let builder = node_stack.front_mut().unwrap();
      let agg_id =
        builder.add_simple(IROp::AGG_DECL, [Default::default(), Default::default()], Default::default(), ASTNode::RawAggregateInstantiation(agg.clone()));

      for init in &agg.inits {
        let expr_id = process_expression(&init.expression.expr, node_stack, ty_db);

        let builder = node_stack.front_mut().unwrap();

        if let Some(name) = &init.name {
          let label_id = builder.get_label(name.id.intern());
          let ref_id = builder.add_simple(IROp::REF, [agg_id, label_id], Default::default(), ASTNode::Var(name.clone()));
          builder.add_simple(IROp::ASSIGN, [ref_id, expr_id], Default::default(), ASTNode::RawAggregateMemberInit(init.clone()));
        } else {
          todo!("Index based initializers are not supported yet {}", blame(&ASTNode::from(init.clone()), "prefix this with <name> ="))
        }
      }

      agg_id
    }
    node => todo!("{node:#?}"),
  }
}

enum VarLookup {
  Var(IRGraphId),
  Mem(IRGraphId),
  None(IString),
}

fn lookup_var<'a>(mem: &MemberCompositeAccess<Token>, node_stack: &'a mut VecDeque<Builder>) -> VarLookup {
  let name = mem.root.name.id.intern();

  if mem.sub_members.len() > 0 {
    if let Some(mut prev_ref) = read_var(name, node_stack) {
      for sub_member in &mem.sub_members {
        match sub_member {
          member_group_Value::NamedMember(name) => {
            let name_id = name.name.id.intern();

            let builder = node_stack.front_mut().unwrap();

            let label_id = builder.get_label(name_id);

            prev_ref = builder.add_simple(IROp::REF, [prev_ref, label_id], Default::default(), ASTNode::NamedMember(name.clone()));
          }
          _ => unreachable!(),
        }
      }

      VarLookup::Mem(prev_ref)
    } else {
      VarLookup::None(Default::default())
    }
  } else if let Some(var) = read_var(name, node_stack) {
    VarLookup::Var(var)
  } else {
    VarLookup::None(name)
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
  let child_builder = node_stack.front_mut().unwrap();

  let ret_id = "RET".intern();

  if !child_builder.var_lookup.contains_key(&ret_id) {
    write_var(ret_id, node_stack);
    let child_builder = node_stack.front_mut().unwrap();

    if child_builder.node.nodes.len() > 0 {
      let op = IRGraphId::new(child_builder.node.nodes.len() - 1);
      child_builder.update_var(ret_id, op, Default::default());
    }
  }

  let child_builder = pop_builder(node_stack);

  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn binary_expr(
  op: IROp,
  left: expression_Value<Token>,
  right: expression_Value<Token>,
  node_stack: &mut VecDeque<Builder>,
  node: ASTNode,
  ty_db: &mut TypeDatabase,
) -> IRGraphId {
  let left_id = process_expression(&left, node_stack, ty_db);
  let right_id = process_expression(&right, node_stack, ty_db);

  let builder = node_stack.front_mut().unwrap();

  builder.add_simple(op, [left_id, right_id], Default::default(), node)
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) {
  let var = &expr.var;
  let expr = &expr.expression;
  let graph_id = process_expression(&expr.expr, node_stack, ty_db);

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = decl.var.id.intern();
      let ty = get_type(&decl.ty, false, ty_db).unwrap();

      let graph_id = if !ty.is_generic() {
        let builder = node_stack.front_mut().unwrap();
        builder.add_ty_binding(graph_id, ty, ASTNode::Var(decl.var.clone()))
      } else {
        graph_id
      };

      let builder = node_stack.front_mut().unwrap();
      builder.create_var(root_name, graph_id, ASTNode::Var(decl.var.clone()), false);
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = mem.root.name.id.intern();
      let ast = ASTNode::MemberCompositeAccess(mem.clone());

      match lookup_var(&mem, node_stack) {
        VarLookup::Mem(mem_id) => {
          let builder = node_stack.front_mut().unwrap();
          builder.add_simple(IROp::ASSIGN, [mem_id, graph_id], Default::default(), ASTNode::Expression(expr.clone()));
        }
        VarLookup::Var(output) => {
          write_var(root_name, node_stack);
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(root_name, graph_id, ast);
        }
        VarLookup::None(..) => {
          let builder = node_stack.front_mut().unwrap();
          builder.create_var(root_name, graph_id, ast, false);
        }
      }
    }
    _ => todo!(),
  }
}

fn push_new_builder(node_stack: &mut VecDeque<Builder>, ty: RVSDGNodeType, id: IString) {
  let top_builder = node_stack.front().unwrap();
  let mut entry = Builder::new(top_builder.inc_counter(), top_builder.id_counter);
  entry.node.id = id;
  entry.node.ty = ty;
  node_stack.push_front(entry);
}

fn merge_multiple_nodes(node_stack: &mut VecDeque<Builder>, mut children: Vec<(Builder, ASTNode)>) {
  let par_builder = node_stack.front_mut().unwrap();

  for (name, par_var) in par_builder.var_lookup.iter_mut() {
    let par_var_nonce = par_var.nonce;
    for (index, (child, _)) in children.iter_mut().enumerate() {
      let child_index = index + par_builder.node_id;

      if let Some(mut child_var) = child.var_lookup.remove(name) {
        if child_var.origin_node_id < child.id && !child_var.input_only && par_var_nonce != child_var.nonce {
          let var_id = child_var.op;
          // Merge the var id into the parent scope

          let var_id = commit_writes(child, &mut child_var);

          par_var.writes.push((child_index, var_id));
          par_var.nonce = par_var_nonce + 1;
        }
      }
    }
  }

  for (child, ast) in children.into_iter() {
    par_builder.node.nodes.push(RVSDGInternalNode::Complex(child.node));
    par_builder.node.source_nodes.push(ast.clone());
    par_builder.node_id += 1
  }
}

fn commit_writes(builder: &mut Builder, var: &mut Var) -> IRGraphId {
  if var.writes.len() > 0 {
    let input_op = builder.add_input_node(Default::default(), 0, Default::default());

    for (node_index, in_id) in var.writes.drain(0..) {
      match &mut builder.node.nodes[node_index] {
        RVSDGInternalNode::Complex(cmplx) => {
          cmplx.outputs.push(RSDVGBinding { in_id, out_id: input_op, input_index: 0, name: var.name, ty: Default::default() });
        }
        _ => {}
      }
    }

    input_op
  } else {
    var.op
  }
}

fn seal_var(var_name: IString, node_stack: &mut VecDeque<Builder>, ty: Type) {
  let builder = node_stack.front_mut().unwrap();
  if let Some(mut var) = builder.var_lookup.remove(&var_name) {
    let var_id = commit_writes(builder, &mut var);

    if var_id.is_valid() {
      let var_id = if !ty.is_undefined() { builder.add_ty_binding(var_id, ty, Default::default()) } else { var_id };

      builder.add_output(RSDVGBinding { in_id: var_id, input_index: 0, name: var_name, out_id: Default::default(), ty });
    }
  }
}

fn write_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  'outer: for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_name) {
      found_in_index = i as i32;
      if found_in_index == 0 {
        return Some(v.op);
      } else {
        break 'outer;
      }
    }
  }

  if found_in_index < 0 {
    return None;
  } else {
    let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
    let par_var = par_data.var_lookup.get_mut(&var_name).unwrap();
    let mut var = par_var.clone();
    var.writes.clear();

    for curr_index in (0..found_in_index).rev() {
      let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

      par_data.var_lookup.insert(var_name, var.clone());

      if curr_index == 0 {
        return Some(var.op);
      }
    }

    unreachable!("-- {found_in_index}");
  }
}

fn read_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  'outer: for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_name) {
      found_in_index = i as i32;
      if found_in_index == 0 {
        return Some(v.op);
      } else {
        break 'outer;
      }
    }
  }

  if found_in_index < 0 {
    return None;
  } else {
    let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
    let par_var = par_data.var_lookup.get_mut(&var_name).unwrap();

    if !par_var.writes.is_empty() {
      let input_op = par_data.add_input_node(Default::default(), 0, Default::default());
      let par_var = par_data.var_lookup.get_mut(&var_name).unwrap();
      par_var.op = input_op;
      for (node_index, in_id) in par_var.writes.drain(0..) {
        match &mut par_data.node.nodes[node_index] {
          RVSDGInternalNode::Complex(cmplx) => {
            cmplx.outputs.push(RSDVGBinding { in_id, out_id: input_op, input_index: 0, name: par_var.name, ty: Default::default() });
          }
          _ => {}
        }
      }
    }

    let par_var = par_data.var_lookup.get_mut(&var_name).unwrap();
    let mut var = par_var.clone();

    for curr_index in (0..found_in_index).rev() {
      let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

      if !var.op.is_invalid() {
        var.op = par_data
          .add_input(
            RSDVGBinding {
              name:        var.name,
              in_id:       var.op,
              out_id:      Default::default(),
              ty:          Default::default(),
              input_index: 0,
            },
            var.ast.clone(),
          )
          .out_id;
      }

      par_data.var_lookup.insert(var_name, var.clone());

      if curr_index == 0 {
        return Some(var.op);
      }
    }

    unreachable!("-- {found_in_index}");
  }
}

fn take_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> IRGraphId {
  let mut id = IRGraphId::default();
  let mut ty = Type::Undefined;

  let id = read_var(var_name, node_stack).unwrap();

  remove_var(var_name, node_stack);

  id
}

fn remove_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) {
  let mut id = IRGraphId::default();
  let mut ty = Type::Undefined;

  let builder = node_stack.front_mut().unwrap();

  builder.var_lookup.remove(&var_name);

  for (i, entry) in builder.node.outputs.iter_mut().enumerate() {
    if entry.name == var_name {
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
        ty_db.get_ptr(base_type)
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        use lifetime_Value::*;
        match &ptr.ptr_type {
          GlobalLifetime(_) => {
            ty_db.get_ptr(base_type)
            //Some(TypeSlot::GlobalIndex(0, type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), 0, base_type)) as u32))
          }
          ScopedLifetime(scope)  =>

          ty_db.get_ptr(base_type)/* Some(TypeSlot::GlobalIndex(
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
