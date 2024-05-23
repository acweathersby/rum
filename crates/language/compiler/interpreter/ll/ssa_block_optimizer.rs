use std::{
  collections::{BTreeMap, BTreeSet},
  fmt::Debug,
};

use rum_container::ArrayVec;

use crate::compiler::interpreter::ll::types::OpArg;

use super::types::*;

enum RegisterProperty {
  IntegerArithmwetic,
  FloatingPointArithmetic,
  Vector2,
  Vector4,
  Vector8,
  Vector16,
  Store,
  Load,
}

pub fn optimize_function_blocks(funct: SSAFunction<()>) -> SSAFunction<()> {
  // remove any blocks that are empty.

  let mut funct = funct.clone();

  remove_passive_blocks(&mut funct);

  collect_predecessors(&mut funct);

  let annotations = build_annotations(&mut funct);

  get_loop_regions(&mut funct, &annotations);

  lower_powers(&mut funct, &annotations);

  assign_registers(&mut funct);

  dead_code_elimination(&mut funct);

  // Define registers and loads.

  load_insertion(&mut funct);

  register_assignment(&mut funct);

  funct
}

type ValAssign = LLVal;

#[derive(Debug)]
struct BlockAnnotation {
  dominators: ArrayVec<32, usize>,
  indirects:  ArrayVec<32, usize>,
  ins:        ArrayVec<32, ValAssign>,
  outs:       ArrayVec<32, ValAssign>,
  decls:      ArrayVec<32, u64>,
}

fn get_loop_regions(funct: &mut SSAFunction<()>, annotations: &Vec<BlockAnnotation>) {
  use super::types::{SSAExpr::*, SSAOp::*};
  for block_id in 0..funct.blocks.len() {
    let annotation = &annotations[block_id];
    let block = &funct.blocks[block_id];

    if block.predecessors.iter().any(|i| (*i as usize) >= block_id) {
      // This block is a loop head.

      let mut loop_blocks = vec![block_id];
      loop_blocks.extend(annotation.indirects.iter().filter(|i| **i >= block_id));

      // We can now preform some analysis and optimization on this region
      {
        #[derive(Debug)]
        struct TypeInformation {
          stack_id:      usize,
          ty:            TypeInfo,
          initial_value: Option<LLVal>,
          loop_change:   Option<isize>,
        }

        let mut type_sets: BTreeMap<usize, TypeInformation> = Default::default();

        println!("loop {block_id} dominators ---------------------------");
        for dominator in annotations[loop_blocks[0]].dominators.iter() {
          print_block_with_annotations(*dominator, funct, annotations);

          for ty in annotations[loop_blocks[0]].outs.iter() {
            let stack_id = ty.info.stack_id().unwrap();
            match type_sets.entry(stack_id) {
              std::collections::btree_map::Entry::Occupied(entry) => {}
              std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(TypeInformation {
                  loop_change: None,
                  stack_id,
                  initial_value: Some(ty.clone()),
                  ty: ty.info,
                });
              }
            }
          }
        }

        dbg!(type_sets);

        println!("loop {block_id}---------------------------");
        for i in &loop_blocks {
          let block = &funct.blocks[*i];

          /*      for op in &block.ops {
            match op {
              BinaryOp(SUB, _, left, right) => {
                let mut val = val.ll_val();
                val.info = target.ll_val().info;

                let val = (val, values.len(), 0u64);

                values.push(val);
                annotations[block_id].decls.push(val.1 as u64);
              }
              _ => {}
            }
          } */

          print_block_with_annotations(*i, funct, annotations)
        }
      }
    }
  }
}

fn print_block_with_annotations(
  block_id: usize,
  funct: &SSAFunction<()>,
  annotations: &Vec<BlockAnnotation>,
) {
  let block = &funct.blocks[block_id];
  let annotation = &annotations[block_id];

  println!(
    "{block:?}
  dominators: [{}]
  indirects:  [{}]
  
  decls: [
    {}
  ]

  ins:  [
    {}
  ]

  outs: [
    {}
  ]
",
    annotation.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
    annotation.indirects.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
    annotation.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
  )
}

fn load_insertion(funct: &mut SSAFunction<()>) {}

fn register_assignment(funct: &mut SSAFunction<()>) {}

fn dead_code_elimination(funct: &mut SSAFunction<()>) {}

fn assign_registers(funct: &mut SSAFunction<()>) {}

fn lower_powers(funct: &mut SSAFunction<()>, annotations: &Vec<BlockAnnotation>) {}

fn build_annotations(funct: &mut SSAFunction<()>) -> Vec<BlockAnnotation> {
  use super::types::{SSAExpr::*, SSAOp::*};

  let mut annotations = vec![];

  let mut outs: ArrayVec<32, u64> = Default::default();
  let mut dominators: ArrayVec<32, u64> = Default::default();
  let mut indirect_successors: ArrayVec<32, u64> = Default::default();
  let mut ins: ArrayVec<32, u64> = Default::default();
  let mut values = Vec::new();

  for block_id in 0..funct.blocks.len() {
    annotations.push(BlockAnnotation {
      indirects:  Default::default(),
      dominators: Default::default(),
      ins:        Default::default(),
      outs:       Default::default(),
      decls:      Default::default(),
    });

    ins.push(0);
    outs.push(0);
    dominators.push(0);
    indirect_successors.push(0);

    let block = &funct.blocks[block_id];
    for op in &block.ops {
      match op {
        BinaryOp(ALLOC, _, target, val) => {
          let mut val = val.ll_val();
          val.info = target.ll_val().info;

          let val = (val, values.len(), 0u64);

          values.push(val);
          annotations[block_id].decls.push(val.1 as u64);
        }
        BinaryOp(STORE, _, target, val) => {
          let mut val = val.ll_val();
          val.info = target.ll_val().info;

          let val = (val, values.len(), 0u64);

          values.push(val);
          annotations[block_id].decls.push(val.1 as u64);
        }
        _ => {}
      }
    }
  }

  for val_id in 0..values.len() {
    let ty = values[val_id].0;
    let mut kills = 1u64 << values[val_id].1;

    for val in &values {
      if val.0.info.stack_id() == ty.info.stack_id() {
        kills |= 1 << val.1;
      }
    }

    values[val_id].2 = kills & !((1 as u64) << val_id);
  }

  dbg!(&funct, &values);

  //annotations[0].dominators = u64::MAX;

  loop {
    let mut should_continue = false;
    for block_id in 0..funct.blocks.len() {
      /* dominators */
      {
        let domis = dominators[block_id];
        let new_in = domis | (1u64 << block_id);

        for successor in iter_branch_indices(&funct.blocks[block_id]) {
          let pred_domis = dominators[successor as usize];

          if pred_domis != new_in {
            if pred_domis == 0 {
              dominators[successor as usize] = new_in
            } else {
              dominators[successor as usize] &= new_in;
            }

            should_continue |= dominators[successor as usize] != pred_domis;
          }
        }
      }

      /* indirect successors */
      {
        let mut indirects = indirect_successors[block_id];

        for predecessor in funct.blocks[block_id].predecessors.iter() {
          indirects |= indirect_successors[(*predecessor) as usize];
        }

        should_continue |= indirect_successors[block_id] != indirects;

        indirect_successors[block_id] = indirects | (1 << block_id as u64);
      }

      /* stack type */
      {
        let existing_in = ins[block_id];
        let mut new_in = existing_in;

        for predecessor in funct.blocks[block_id].predecessors.as_slice() {
          new_in |= outs[*predecessor as usize];
        }

        ins[block_id] = new_in;

        should_continue |= new_in != existing_in;

        let (decls, kills) = annotations[block_id]
          .decls
          .as_slice()
          .iter()
          .map(|index| ((1 as u64) << index, values[(*index) as usize].2))
          .fold((0, 0u64), |a, b| (a.0 | b.0, a.1 | b.1));

        outs[block_id] = decls | (new_in & !kills);

        let block = &mut funct.blocks[block_id];

        for op in &block.ops {
          match op.name() {
            STORE => {}
            _ => {}
          }
        }
      }
    }

    if !should_continue {
      break;
    }
  }

  for block_id in 0..funct.blocks.len() {
    let ins = ins[block_id];
    let outs: u64 = outs[block_id];

    let dominator_bits = dominators[block_id];

    for i in 0..64 {
      let mask = 1u64 << i;
      if mask & dominator_bits > 0 {
        annotations[block_id].dominators.push(i as usize)
      }
    }

    let indirect_successsor_bits =
      indirect_successors[block_id] & !(1u64 << block_id) & !dominator_bits;

    for i in 0..64 {
      let mask = 1u64 << i;
      if mask & indirect_successsor_bits > 0 {
        annotations[block_id].indirects.push(i as usize)
      }
    }

    for i in 0..values.len() {
      let mask = 1u64 << i;
      let (a, b, c, ..) = values[i as usize];

      if ins & mask > 0 {
        annotations[block_id].ins.push(a)
      }

      if outs & mask > 0 {
        annotations[block_id].outs.push(a)
      }
    }
  }

  annotations
}

fn iter_branch_indices(block: &SSABlock<()>) -> impl Iterator<Item = usize> {
  get_branch_indices(block).into_iter().filter_map(|d| d)
}

fn get_branch_indices(block: &SSABlock<()>) -> [Option<usize>; 3] {
  [block.branch_succeed, block.branch_fail, block.branch_unconditional]
}

fn collect_predecessors(funct: &mut SSAFunction<()>) {
  let upper_bound = funct.blocks.len() - 1;
  let mut successors = ArrayVec::<2, usize>::new();

  for predecessor in 0..=upper_bound {
    successors.clear();

    if !has_branch(&funct.blocks[predecessor]) {
      // the following block is a natural successor of this block.
      if predecessor < upper_bound {
        funct.blocks[predecessor].branch_unconditional = Some(predecessor + 1);
      }
    }

    let block = &funct.blocks[predecessor];

    if let Some(id) = &block.branch_fail {
      successors.push(*id);
    }

    if let Some(id) = &block.branch_succeed {
      successors.push(*id);
    }

    if let Some(id) = &block.branch_unconditional {
      successors.push(*id);
    }

    for id in successors.iter() {
      funct.blocks[*id].predecessors.push_unique(predecessor as u16).expect("Should be ordered");
    }
  }
}

fn remove_passive_blocks(funct: &mut SSAFunction<()>) {
  'outer: loop {
    let mut block_remaps = (0..funct.blocks.len()).collect::<Vec<_>>();
    for empty_block in 0..funct.blocks.len() {
      let block = &funct.blocks[empty_block];

      if block_is_empty(block) {
        if let Some(target) = &block.branch_unconditional {
          block_remaps[empty_block] = *target;
        }

        funct.blocks.remove(empty_block);

        funct.blocks[empty_block..].iter_mut().for_each(|b| {
          b.id -= 1;
        });

        block_remaps[empty_block + 1..].iter_mut().for_each(|i| {
          *i -= 1;
        });

        for block in &mut funct.blocks {
          update_branch(&mut block.branch_succeed, &block_remaps);
          update_branch(&mut block.branch_fail, &block_remaps);
          update_branch(&mut block.branch_unconditional, &block_remaps);
        }

        continue 'outer;
      }
    }
    break;
  }
}

fn update_branch(patch: &mut Option<usize>, block_remaps: &Vec<usize>) {
  if let Some(branch_block) = patch {
    *branch_block = block_remaps[*branch_block];
  }
}

fn block_is_empty(block: &SSABlock<()>) -> bool {
  block.ops.is_empty() && !has_choice_branch(block)
}

fn has_choice_branch(block: &SSABlock<()>) -> bool {
  block.branch_fail.is_some() || block.branch_succeed.is_some()
}

fn has_branch(block: &SSABlock<()>) -> bool {
  has_choice_branch(block) || !block.branch_unconditional.is_none()
}
