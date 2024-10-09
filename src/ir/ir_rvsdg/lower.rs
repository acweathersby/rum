type WIPNode = Builder;
use crate::{
  ir::{ir_graph::IRGraphId, ir_rvsdg::*},
  istring::CachedString,
  parser::script_parser::*,
  types::TypeDatabase,
};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct RumIRModule {
  pub structs: HashMap<IString, Box<RVSDGNode>>,
  pub functs:  HashMap<IString, Box<RVSDGNode>>,
}

struct Builder {
  node:         Box<RVSDGNode>,
  node_id:      usize,
  label_lookup: HashMap<IString, IRGraphId>,
}

impl Builder {
  pub fn new() -> Self {
    Builder { node: Default::default(), node_id: Default::default(), label_lookup: Default::default() }
  }

  fn create_id(node_id: &mut usize) -> IRGraphId {
    let id = IRGraphId::new(*node_id);
    *node_id += 1;
    id
  }

  pub fn add_simple(&mut self, op: IROp, operands: [IRGraphId; 2], ty: RumType, tok: &Token) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Simple { id, op, operands, ty });
    self.node.source_tokens.push(tok.clone());
    id
  }

  pub fn add_input(&mut self, mut binding: RSDVGBinding, tok: &Token) -> RSDVGBinding {
    let id = Self::create_id(&mut self.node_id);
    binding.out_id = id;
    binding.input_index = self.node.inputs.len() as u32;
    self.node.inputs.push(binding);
    self.add_input_node_internal(id, binding.ty, binding.input_index as usize, tok);
    binding
  }

  pub fn add_input_node(&mut self, ty: RumType, input_index: usize, tok: &Token) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_input_node_internal(id, ty, input_index, tok)
  }

  fn add_input_node_internal(&mut self, id: IRGraphId, ty: RumType, input_index: usize, tok: &Token) -> IRGraphId {
    self.node.nodes.push(RVSDGInternalNode::Input { id, ty, input_index });
    self.node.source_tokens.push(tok.clone());
    id
  }

  pub fn add_output(&mut self, binding: RSDVGBinding) -> RSDVGBinding {
    self.node.outputs.push(binding);
    binding
  }

  pub fn add_const(&mut self, const_val: ConstVal, tok: &Token) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Const(id.0, const_val));
    self.node.source_tokens.push(tok.clone());
    id
  }

  pub fn get_label(&mut self, name: IString) -> IRGraphId {
    let Self { label_lookup, node_id, node } = self;
    match label_lookup.entry(name) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        node.nodes.push(RVSDGInternalNode::Label(id, name));
        node.source_tokens.push(Default::default());
        *val.insert(id)
      }
    }
  }
}

pub fn lower_ast_to_rvsdg(module: &std::sync::Arc<RawModule<Token>>, mut ty_db: TypeDatabase) -> RumIRModule {
  let members = &module.members;

  let mut module = RumIRModule { functs: Default::default(), structs: Default::default() };

  for mem in &members.members {
    match mem {
      module_members_group_Value::AnnotatedModMember(annotation) => match &annotation.member {
        module_member_Value::RawBoundType(bound_type) => match &bound_type.ty {
          type_Value::Type_Struct(strct) => {
            let funct: Box<RVSDGNode> = lower_struct_to_rsvdg(bound_type.name.id.intern(), strct);
            module.structs.insert(funct.id, funct);
          }
          _ => unreachable!(),
        },
        module_member_Value::RawRoutine(rt) => {
          let funct = lower_fn_to_rsvdg(rt);
          module.functs.insert(funct.id, funct);
        }
        module_member_Value::RawScope(scope) => {}
        _ => unreachable!(),
      },
      module_members_group_Value::AnnotationVariable(var) => {}
      module_members_group_Value::LifetimeVariable(var) => {}
      _ => unreachable!(),
    }
  }

  module
}

fn lower_struct_to_rsvdg(binding_name: IString, struct_: &Type_Struct<Token>) -> Box<RVSDGNode> {
  let mut node = RVSDGNode::default();

  node.ty = RVSDGNodeType::Struct;
  node.id = binding_name;

  let mut node_id = 0;

  for prop in &struct_.properties {
    match prop {
      property_Value::Property(prop) => {
        let name = prop.name.id.intern();
        let ty = get_type(&prop.ty, false).unwrap_or_default();

        let mut output_binding = RSDVGBinding::default();
        let id = Builder::create_id(&mut node_id);

        output_binding.name = name;
        output_binding.ty = ty;
        output_binding.in_id = id;
        output_binding.out_id = Default::default();
        output_binding.input_index = node.inputs.len() as u32;

        node.outputs.push(output_binding);

        node.nodes.push(RVSDGInternalNode::Input { id, ty: output_binding.ty, input_index: output_binding.input_index as usize });
        node.source_tokens.push(prop.tok.clone());
      }
      n => todo!("Handle construction of {n:?}"),
    }
  }

  Box::new(node)
}

fn lower_fn_to_rsvdg(fn_decl: &RawRoutine<Token>) -> Box<RVSDGNode> {
  let params = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(ty) => &ty.params,
    routine_type_Value::RawProcedureType(ty) => &ty.params,
    _ => unreachable!(),
  };

  let expr = &fn_decl.expression.expr;

  let mut node_stack = VecDeque::new();

  let mut builder = Builder::new();
  builder.node.ty = RVSDGNodeType::Function;
  builder.node.id = fn_decl.name.id.intern();

  node_stack.push_front(builder);

  insert_params(params, &mut node_stack);

  let ret_val = process_expression(expr, &mut node_stack);

  let mut fn_node = node_stack.pop_front().unwrap().node;

  match &fn_decl.ty {
    routine_type_Value::RawFunctionType(fn_ty) => {
      insert_returns(ret_val, fn_ty, &mut fn_node);
    }
    _ => {}
  };

  // remove non return values from input

  let mut new_outputs = ArrayVec::new();

  for output in fn_node.outputs.iter() {
    if output.name == "RET".intern() {
      new_outputs.push(*output);
    }
  }

  fn_node.outputs = new_outputs;

  fn_node
}

fn insert_returns(ret_val: IRGraphId, fn_ty: &std::sync::Arc<RawFunctionType<Token>>, fn_node: &mut Box<RVSDGNode>) {
  if !ret_val.is_invalid() {
    let ret = &fn_ty.return_type;
    let mut input = RSDVGBinding::default();
    let ty = get_type(&ret.ty, false).unwrap_or_default();
    input.name = "RET".intern();
    input.ty = ty;
    input.in_id = ret_val;
    fn_node.outputs.push(input);
  } else {
    panic!("Create error reporting an expected return value is not found");
  }
}

fn insert_params(params: &std::sync::Arc<Params<Token>>, node_stack: &mut VecDeque<Builder>) {
  for param in &params.params {
    let ty = get_type(&param.ty.ty, false).unwrap_or_default();
    let name = param.var.id.intern();
    create_input_binding(node_stack.front_mut().unwrap(), name, ty, Default::default(), param.tok.clone());
  }
}

fn create_input_binding(node_data: &mut Builder, name: IString, ty: RumType, in_id: IRGraphId, tok: Token) -> IRGraphId {
  let builder = node_data;

  let mut output_binding = builder.add_input(RSDVGBinding { name, in_id, ty, ..Default::default() }, &tok);

  output_binding.in_id = output_binding.out_id;

  builder.add_output(output_binding);

  output_binding.in_id
}

fn create_var_binding(node_stack: &mut VecDeque<Builder>, name: IString, in_id: IRGraphId, ty: RumType) {
  let mut output_binding = RSDVGBinding::default();
  output_binding.name = name;
  output_binding.ty = ty;
  output_binding.in_id = in_id;
  output_binding.out_id = in_id;
  output_binding.input_index = 0;
  node_stack.front_mut().unwrap().add_output(output_binding);
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>) -> IRGraphId {
  match expr {
    expression_Value::MemberCompositeAccess(mem) => match lookup_var(mem, node_stack) {
      VarLookup::Mem(output) => output,
      VarLookup::Var(var) => var.in_id,
      VarLookup::None => {
        todo!("report error on lookup of {}", mem.tok.blame(1, 1, "", None))
      }
    },

    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      node_stack.front_mut().unwrap().add_const(
        if string_val.contains(".") {
          ConstVal::new(RumType::Float | RumType::b64, num.val)
        } else {
          ConstVal::new(RumType::Unsigned | RumType::b64, string_val.parse::<u64>().unwrap())
        },
        &num.tok,
      )
    }

    expression_Value::Add(op) => binary_expr(
      IROp::ADD,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::Mul(op) => binary_expr(
      IROp::MUL,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::Div(op) => binary_expr(
      IROp::DIV,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::Sub(op) => binary_expr(
      IROp::SUB,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::BIT_XOR(op) => binary_expr(
      IROp::XOR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::BIT_OR(op) => binary_expr(
      IROp::OR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::BIT_AND(op) => binary_expr(
      IROp::AND,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      op.tok.clone(),
    ),

    expression_Value::RawBlock(block) => {
      let mut last_id = IRGraphId::default();
      for expr in &block.statements {
        match expr {
          statement_Value::Expression(expr) => {
            last_id = process_expression(&expr.expr, node_stack);
          }
          statement_Value::RawAssignment(assign) => {
            process_assign(&assign, node_stack);
            last_id = IRGraphId::default()
          }
          _ => todo!(),
        }
      }

      match &block.exit {
        block_expression_group_3_Value::BlockExitExpressions(e) => process_expression(&e.expression.expr, node_stack),
        _ => last_id,
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      for arg in &call.args {
        args.push(process_expression(&arg.expr, node_stack));
      }

      let builder = node_stack.front_mut().unwrap();

      push_node(node_stack, RVSDGNodeType::Call, Default::default());

      let call_builder = node_stack.front_mut().unwrap();

      for arg in args {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = arg;
        call_builder.add_input(input, &Default::default());
      }

      let mut output = RSDVGBinding::default();
      output.name = "RET".intern();
      //input.out_id = ret_id;
      call_builder.add_output(output);

      // The return value should now be mapped to a new_node id in the parent scope.

      let node_index = pop_node(node_stack);

      let par_node = node_stack.front_mut().unwrap();

      let outputs = match &mut par_node.node.nodes.as_mut_slice()[node_index] {
        RVSDGInternalNode::Complex(n) => n.outputs.iter().map(|o| o.ty).collect::<Vec<_>>(),
        n => Default::default(),
      };

      for ty in outputs {
        let id = par_node.add_input_node(ty, 0, &Default::default());

        match &mut par_node.node.nodes.as_mut_slice()[node_index] {
          RVSDGInternalNode::Complex(n) => {
            let last = n.outputs.len() - 1;
            let output = &mut n.outputs.as_mut_slice()[last];
            output.out_id = id;
          }
          n => unreachable!("{n}"),
        };

        return id;
      }

      Default::default()
    }

    expression_Value::RawMatch(mtch) => {
      //todo!("Match");

      // create the match entry
      push_node(node_stack, RVSDGNodeType::MatchHead, Default::default());

      let eval_id = process_expression(&mtch.expression.clone().into(), node_stack);

      let temp_binding_name = "__match__expr__".intern();

      create_var_binding(node_stack, temp_binding_name, eval_id, Default::default());

      let mut merges = vec![];

      for clause in &mtch.clauses {
        push_node(node_stack, RVSDGNodeType::Switch, Default::default());

        if clause.default {
          push_node(node_stack, RVSDGNodeType::SwitchBody, Default::default());

          process_expression(&clause.scope.clone().into(), node_stack);

          pop_and_merge_single_node(node_stack);
        } else {
          let v = process_expression(&clause.expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack);

          let op_ty = match clause.expr.op.as_str() {
            ">" => crate::ir::ir_graph::IROp::GR,
            "<" => crate::ir::ir_graph::IROp::LS,
            ">=" => crate::ir::ir_graph::IROp::GE,
            "<=" => crate::ir::ir_graph::IROp::LE,
            "==" => crate::ir::ir_graph::IROp::EQ,
            "!=" => crate::ir::ir_graph::IROp::NE,
            _ => todo!(),
          };

          {
            let eval_id = get_var(temp_binding_name, node_stack).unwrap().in_id;
            let builder = node_stack.front_mut().unwrap();
            builder.add_simple(op_ty, [eval_id, v], Default::default(), &Default::default());
          }

          push_node(node_stack, RVSDGNodeType::SwitchBody, Default::default());

          process_expression(&clause.scope.clone().into(), node_stack);

          pop_and_merge_single_node(node_stack);
        }

        merges.push(pop_node(node_stack));
      }

      // Merge match nodes - this is the only place we need to do such a complex mapping of inputs.
      {
        let builder = node_stack.front_mut().unwrap();

        let pending_nodes = {
          let Builder { node, node_id, label_lookup } = builder;
          let RVSDGNode { id, ty, inputs, outputs, nodes, .. } = node.as_mut();

          let mut children = nodes
            .iter_mut()
            .enumerate()
            .filter_map(|(i, n)| match (merges.contains(&i), n) {
              (true, RVSDGInternalNode::Complex(node)) => Some(node.as_mut()),
              _ => None,
            })
            .collect::<Vec<_>>();

          let mut unchanged_vars = ArrayVec::<8, _>::new();
          let mut changed_vars = ArrayVec::<8, _>::new();
          let mut pending_nodes = vec![];

          for child in children.iter() {
            for output in child.outputs.iter() {
              if output.in_id != output.out_id {
                changed_vars.push_unique(output.name).expect("");
              } else {
                unchanged_vars.push_unique(output.name).expect("");
              }
            }
          }

          for var_name in changed_vars.iter().cloned() {
            let mut resolved = false;

            for (_, par_out) in outputs.iter_mut().enumerate() {
              if par_out.name == var_name {
                let id = Builder::create_id(node_id);
                pending_nodes.push((id, par_out.ty));
                par_out.in_id = id;

                for child in children.iter_mut() {
                  for output in child.outputs.iter_mut() {
                    if output.name == var_name {
                      output.out_id = id;
                      break;
                    }
                  }
                }
                resolved = true;
                break;
              }
            }

            if !resolved {
              for child in children.iter_mut() {
                for output in child.outputs.iter_mut() {
                  if output.name == var_name {
                    output.out_id = Default::default();
                    break;
                  }
                }
              }
            }
          }

          for var_name in changed_vars.iter().cloned() {
            if changed_vars.contains(&var_name) {
              continue;
            }

            for (_, par_out) in outputs.iter_mut().enumerate() {
              if par_out.name == var_name {
                for child in children.iter_mut() {
                  for output in child.outputs.iter_mut() {
                    if output.name == var_name {
                      output.out_id = par_out.in_id;
                      break;
                    }
                  }
                }
              }
            }
          }
          pending_nodes
        };

        for (id, ty) in pending_nodes {
          builder.add_input_node_internal(id, ty, 0, &Default::default());
        }
      }

      pop_and_merge_single_node(node_stack);

      Default::default()
    }
    _ => todo!(),
  }
}

enum VarLookup<'a> {
  Var(&'a mut RSDVGBinding),
  Mem(IRGraphId),
  None,
}

fn lookup_var<'a>(mem: &MemberCompositeAccess<Token>, node_stack: &'a mut VecDeque<Builder>) -> VarLookup<'a> {
  let name = mem.root.name.id.intern();

  if mem.sub_members.len() > 0 {
    if let Some(output) = get_var(name, node_stack) {
      let mut prev_ref = output.in_id;
      let _ = output;
      for sub_member in &mem.sub_members {
        match sub_member {
          member_group_Value::NamedMember(name) => {
            let name_id = name.name.id.intern();

            let builder = node_stack.front_mut().unwrap();

            let label_id = builder.get_label(name_id);

            prev_ref = builder.add_simple(IROp::REF, [prev_ref, label_id], Default::default(), &name.tok);
          }
          _ => unreachable!(),
        }
      }

      VarLookup::Mem(prev_ref)
    } else {
      VarLookup::None
    }
  } else if let Some(output) = get_var(name, node_stack) {
    VarLookup::Var(output)
  } else {
    VarLookup::None
  }
}

fn pop_and_merge_single_node(node_stack: &mut VecDeque<Builder>) {
  let child_index = pop_node(node_stack);

  let builder = node_stack.front_mut().unwrap();
  let RVSDGNode { outputs, nodes, .. } = builder.node.as_mut();
  let RVSDGInternalNode::Complex(child) = &mut nodes[child_index] else { unreachable!() };

  let mut pending_nodes = vec![];

  for output in child.outputs.iter_mut() {
    let mut resolved = false;

    for (_, par_out) in outputs.iter_mut().enumerate() {
      if par_out.name == output.name {
        let internal_id_has_changed = output.out_id != output.in_id;
        let parent_id_hash_changed = par_out.out_id != par_out.in_id;

        if internal_id_has_changed && !parent_id_hash_changed {
          let id = Builder::create_id(&mut builder.node_id);
          pending_nodes.push((id, par_out.ty));
          par_out.in_id = id;
          output.out_id = id;
        } else {
          output.out_id = par_out.in_id;
        }

        resolved = true;

        break;
      }
    }

    if !resolved {
      output.out_id = Default::default()
    }
  }

  for (id, ty) in pending_nodes {
    builder.add_input_node_internal(id, ty, 0, &Default::default());
  }
}

fn binary_expr(op: IROp, left: expression_Value<Token>, right: expression_Value<Token>, node_stack: &mut VecDeque<Builder>, tok: Token) -> IRGraphId {
  let left_id = process_expression(&left, node_stack);
  let right_id = process_expression(&right, node_stack);

  let builder = node_stack.front_mut().unwrap();

  builder.add_simple(op, [left_id, right_id], Default::default(), &tok)
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>) {
  let var = &expr.var;
  let expr = &expr.expression;
  let graph_id = process_expression(&expr.expr, node_stack);

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = decl.var.id.intern();
      let ty = get_type(&decl.ty, false).unwrap();

      create_var_binding(node_stack, root_name, graph_id, ty);
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = mem.root.name.id.intern();

      match lookup_var(&mem, node_stack) {
        VarLookup::Mem(mem_id) => {
          let builder = node_stack.front_mut().unwrap();
          builder.add_simple(IROp::ASSIGN, [mem_id, graph_id], Default::default(), &expr.tok);
        }
        VarLookup::Var(output) => output.in_id = graph_id,
        VarLookup::None => create_var_binding(node_stack, root_name, graph_id, Default::default()),
      }
    }
    _ => todo!(),
  }
}

fn push_node(node_stack: &mut VecDeque<Builder>, ty: RVSDGNodeType, id: IString) {
  let mut entry = Builder::new();
  entry.node.id = id;
  entry.node.ty = ty;
  node_stack.push_front(entry);
}

fn pop_node(node_stack: &mut VecDeque<Builder>) -> usize {
  let entry = node_stack.pop_front().unwrap();
  let front_nodes = &mut node_stack.front_mut().unwrap().node.nodes;
  front_nodes.push(RVSDGInternalNode::Complex(entry.node));
  front_nodes.len() - 1
}

fn get_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> Option<&'a mut RSDVGBinding> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();
  let mut ty = RumType::Undefined;

  let stack = node_stack as *mut VecDeque<WIPNode>;

  'outer: for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    for entry in builder.node.outputs.iter_mut() {
      if entry.name == var_name {
        found_in_index = i as i32;

        if found_in_index == 0 {
          return Some(entry);
        } else {
          id = entry.in_id;
          ty = entry.ty;
          break 'outer;
        }
      }
    }
  }

  if found_in_index < 0 {
    return None;
  } else {
    for curr_index in (0..found_in_index).rev() {
      let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

      id = create_input_binding(par_data, var_name, ty, id, Default::default());

      if curr_index == 0 {
        let index = par_data.node.outputs.len() - 1;
        return Some(&mut par_data.node.outputs.as_mut_slice()[index]);
      }
    }

    unreachable!("-- {found_in_index}");
  }
}

pub fn get_type(ir_type: &type_Value<Token>, insert_unresolved: bool) -> Option<RumType> {
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
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved) {
        Some(base_type.increment_pointer())
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved) {
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
    Type_Variable(type_var) => Some(RumType::default()),
    _t => Option::None,
  }
}
