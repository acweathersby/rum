#![allow(unused)]
use rum_istring::IString;

use super::{error::RumScriptError, types::Table, *};
use core::num;
use std::{collections::BTreeMap, sync::Arc};

pub type RumScriptResult<T> = Result<T, RumScriptError>;

pub fn interpret_string(input: &str, ctx: &mut Context) -> RumScriptResult<()> {
  let output =
    super::super::parser::parse_RS(input)?.into_RumScript().expect("Could not load script");

  let mut pending_table_generating = rum_container::ArrayVec::<8, _>::new();

  let mut init_block = None;

  for statement in output.statements {
    match statement {
      statement_Value::TableDefinition(global_definition) => {
        integrate_table_definition(&global_definition, ctx)?;
      }

      statement_Value::GlobalTable(global_table) => pending_table_generating.push(global_table),

      statement_Value::InitExeBlock(block) => {
        init_block = Some(block);
      }

      other => {
        println!("Todo: compile this: {other:#?}")
      }
    }
  }

  for table_decl in pending_table_generating.iter() {
    generate_global_table(&table_decl, ctx)?;
  }

  if let Some(init_block) = init_block {
    for stmt in init_block.stmts {
      match stmt {
        table_initialization_Value::TableInit(init) => {
          table_init(&init, ctx);
        }
        table_initialization_Value::TableZeroInit(init) => {
          table_zero_init(&init, ctx);
        }
        _ => {}
      }
    }
  } else {
    panic!("No initialization block");
  }

  dbg!(ctx);

  Ok(())
}

pub fn integrate_table_definition(
  table_def: &TableDefinition<Token>,
  ctx: &mut Context,
  //scope: ...
  //errors: ...
) -> RumScriptResult<()> {
  let name = table_def.id.id.intern();

  println!(
    "TODO(anthony): ensure table is not replaced unless it comes form the same origin module"
  );

  ctx.table_definitions.insert(name, Arc::new(table_def.into()));

  Ok(())
}

pub fn generate_global_table(
  table_decl: &GlobalTable<Token>,
  ctx: &mut Context,
  //scope: ...
  //errors: ...
) -> RumScriptResult<()> {
  println!(
    "TODO(anthony): assert there is a table definition for this global within the active scope"
  );

  let def_name = table_decl.table_def_name.intern();
  let decl_name = table_decl.name.intern();

  match ctx.table_definitions.get(&def_name) {
    Some(table_def) => {
      if let Some(existing_table) = ctx.global_tables.get(&decl_name) {
        panic!("duplicated global definition!")
      }

      ctx
        .global_tables
        .insert(decl_name, std::cell::UnsafeCell::new(Box::new((table_decl, table_def).into())));
    }
    None => {
      panic!(
        "{}",
        table_decl.tok.blame(1, 1, "Could not find a table definition for this declaration", None)
      )
    }
  }

  Ok(())
}

pub fn table_zero_init(
  table_init_node: &TableZeroInit<Token>,
  ctx: &mut Context,
  //scope: ...
  //errors: ...
) -> RumScriptResult<()> {
  let target_table_name = table_init_node.table_name.id.intern();

  if let Some(target_table) = ctx.global_tables.get_mut(&target_table_name) {
    let number_of_rows = table_init_node.rows as usize;

    let target_table = target_table.get_mut();

    if number_of_rows == 0 {
    } else {
      unsafe {
        target_table.rows = number_of_rows;

        let mut offset = 0;

        target_table.definition.get_table_row_offsets(&mut offset);

        let size = number_of_rows * offset as usize;

        let layout = std::alloc::Layout::from_size_align_unchecked(size, 16);

        target_table.byte_size = size;

        target_table.data = std::alloc::alloc(layout);
      }
    }
  } else {
  }

  Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Reference {
  RowCursor(*mut u8),
  ColumnCursor(usize),
}

struct Scope<'b> {
  ctx:         &'b mut Context,
  local_names: BTreeMap<IString, Reference>,
}

pub fn table_init(
  table_init_node: &TableInit<Token>,
  ctx: &mut Context,
  //scope: ...
  //errors: ...
) -> RumScriptResult<()> {
  let target_table_name = table_init_node.table_name.id.intern();
  let cursor_name = table_init_node.cursor_name.id.intern();

  if let Some(target_table_cell) = ctx.global_tables.get(&target_table_name) {
    let number_of_rows = table_init_node.rows as usize;

    let target_table = unsafe { &mut *target_table_cell.get() };

    if number_of_rows == 0 {
    } else {
      unsafe {
        target_table.rows = number_of_rows;

        let mut row_size = 0;

        target_table.definition.get_table_row_offsets(&mut row_size);

        let size = number_of_rows * row_size as usize;

        let layout = std::alloc::Layout::from_size_align_unchecked(size, 16);

        target_table.byte_size = size;

        target_table.row_byte_size = row_size as usize;

        target_table.data = std::alloc::alloc(layout);

        if target_table.data.is_null() {
          panic!("Could not fulfill table request of [{number_of_rows}] for table");
        }

        for i in 0..number_of_rows {
          let target_table = &mut *target_table_cell.get();
          let table_cursors =
            &mut BTreeMap::from_iter(vec![(cursor_name, (i as usize, target_table))]);

          for stmt in &table_init_node.stmts {
            assignment_statement(&stmt, table_cursors)?;
          }
        }

        dbg!(std::slice::from_raw_parts(
          target_table.data as *const f32,
          number_of_rows * target_table.definition.types.len()
        ));
      }
    }
  } else {
  }

  Ok(())
}

pub fn assignment_statement(
  stmt: &AssignStatement,
  tables: &mut BTreeMap<IString, (usize, &mut Box<Table>)>,
  //scope: ...
  //errors: ...
) -> RumScriptResult<()> {
  let name = stmt.left.scopes[0].intern();
  let key = stmt.left.scopes[1].intern();

  if let Some((offset, table)) = tables.get_mut(&name) {
    let cursor_offset = *offset * table.row_byte_size;
    let def = table.definition.clone();

    for column_names in def.col_names.iter() {
      if let Some((col_index, key)) = column_names.iter().enumerate().find(|(_, d)| **d == key) {
        let ty = &def.types[col_index];

        let (_, col_offset) = def.get_table_row_offsets(&mut 0)[col_index];

        let cursor_offset = cursor_offset + col_offset as usize;

        let ptr = unsafe { table.data.offset(cursor_offset as isize) };

        match ty.name.as_str() {
          "f64" => {
            unsafe { *(ptr as *mut f64) = stmt.right.num };
          }
          "f32" => {
            unsafe { *(ptr as *mut f32) = stmt.right.num as f32 };
          }
          _ => {}
        }

        break;
      }
    }
  }

  Ok(())
}
