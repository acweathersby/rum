use crate::{
  ir::{
    ir_lowering::lower_into_ssa,
    ir_register_allocator_ssa,
    ir_type_analysis::{resolve_routine, resolve_struct_offset},
  },
  istring::IString,
  types::{Type, TypeDatabase},
};
use std::collections::{HashSet, VecDeque};

#[cfg(test)]
mod test;

pub fn compile_binary_from_entry(entry_routine: IString, _errors: Vec<IString>, db: &mut TypeDatabase) -> (usize, Vec<u8>){
  // extract routine tree from entry routine, and setup a queue to build routine binaries

  let struct_names = get_struct_names(db);

  for struct_name in struct_names {
    resolve_struct_offset(struct_name, db)
  }

  let mut seen = HashSet::<IString>::new();
  let mut pending_routines = VecDeque::from_iter(vec![entry_routine/* , "heap_allocate".intern() */]);
  let mut processed_routines = Vec::new();

  while let Some(pending) = pending_routines.pop_front() {
    println!("#### Building routine {pending}");
    let Some((ty_ref, _)) = db.get_type_mut(pending) else {
      panic!("Could not find Structured Memory type: {pending}",);
    };

    match ty_ref {
      Type::Routine(routine) => {
        for var in &routine.body.ctx.vars {
          if var.ty.is_aggregate() {
            match var.ty.aggregate(db) {
              Some(Type::Routine(r)) => {
                if seen.insert(r.name) {
                  pending_routines.push_back(r.name);
                }
              }
              _ => {}
            }
          }
        }
      }
      _ => unreachable!("Internal error: type is not a routine."),
    }

    resolve_routine(pending, db);

    let (ssa_blocks, ssa_graph) = lower_into_ssa(pending, db);

    let assignements = ir_register_allocator_ssa::generate_register_assignments(&ssa_blocks, &ssa_graph);

    let linkable = compile_from_ssa_fn(pending, &ssa_blocks, &ssa_graph, &assignements,  &[]).expect("Could not create linkable");

    processed_routines.push((0, linkable));
  }

  use crate::x86::*;

  let mut binary: Vec<u8> = vec![];

  const MALLOC: unsafe extern "C" fn(usize) -> *mut libc::c_void = libc::malloc;
  const FREE: unsafe extern "C" fn(*mut libc::c_void) = libc::free;
  push_bytes(&mut binary, MALLOC);
  push_bytes(&mut binary, FREE);

  for (offset, link) in &mut processed_routines {
    *offset = binary.len();
    binary.extend(link.binary.clone());
  }

  for (offset, link) in &processed_routines {
    for rt in &link.link_map {
      match rt.link_type {
        crate::linker::LinkType::DBGRoutine(name) => match name.to_str().as_str() {
          "_malloc" => {
            let diff = (0 as i32) - ((rt.binary_offset + *offset + 4) as i32);
            rt.replace(unsafe { binary.as_mut_ptr().offset(*offset as isize) }, diff);
          }
          name => panic!("could not recognize binary debug function: {name}"),
        },
        crate::linker::LinkType::Routine(name) => {
          if let Some((target_offset, ..)) = processed_routines.iter().find(|(.., l)| l.name == name) {
            let diff = (*target_offset as i32) - ((rt.binary_offset + *offset + 4) as i32);
            rt.replace(unsafe { binary.as_mut_ptr().offset(*offset as isize) }, diff);
          } else {
            panic!("Could not find target {name}");
          }
        }
        _ => {}
      }
    }
  }

  (processed_routines.iter().find(|(.., l)| l.name == entry_routine).map(|(offset, ..)| *offset).unwrap(), binary)
}

fn get_struct_names(db: &mut TypeDatabase) -> Vec<IString>{
  db.types
    .iter()
    .filter_map(|d| match d.as_ref() {
      Type::Structure(s) => Some(s.name),
      _ => None,
    })
    .collect()
}
