use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use rum_istring::{CachedString, IString};

use radlr_rust_runtime::types::Token;

use crate::compiler::script_parser::{table_row_types_Value, GlobalTable, RowType};

#[derive(Debug)]
pub(super) struct TableCursor {
  /// The curse mutates the underlying table
  mutates: bool,
  table:   *mut Table,
}

#[derive(Debug)]
pub(super) struct Table {
  pub(super) name:          IString,
  pub(super) definition:    Arc<TableDefinition>,
  pub(super) data:          *mut u8,
  pub(super) byte_size:     usize,
  pub(super) rows:          usize,
  pub(super) row_byte_size: usize,
  pub(super) read_lock:     std::sync::atomic::AtomicU64,
  pub(super) write_lock:    std::sync::atomic::AtomicU64,
  pub(super) indices:       Vec<*const dyn Indice>,
}

impl From<(&GlobalTable<Token>, &Arc<TableDefinition>)> for Table {
  fn from(value: (&GlobalTable<Token>, &Arc<TableDefinition>)) -> Self {
    let (decl, def) = value;

    let name = decl.name.intern();

    Self {
      name,
      definition: def.clone(),
      data: std::ptr::null_mut(),
      byte_size: 0,
      rows: 0,
      row_byte_size: 0,
      read_lock: Default::default(),
      write_lock: Default::default(),
      indices: Default::default(),
    }
  }
}

trait Indice: Debug {}

#[derive(Debug)]
pub(super) struct ExecutionBlock {}

#[allow(unused)]
#[derive(Debug)]
pub struct Context {
  // ============
  pub(super) table_definitions: BTreeMap<IString, Arc<TableDefinition>>,
  pub(super) global_tables:     BTreeMap<IString, std::cell::UnsafeCell<Box<Table>>>,
  pub(super) global_blocks:     BTreeMap<IString, ExecutionBlock>,
}

impl Context {
  pub fn new() -> Context {
    Self {
      table_definitions: Default::default(),
      global_tables:     Default::default(),
      global_blocks:     Default::default(),
    }
  }
}

#[derive(Debug)]
pub(super) struct TableDefinition {
  pub(super) name:       IString,
  pub(super) types:      Vec<RowType>,
  pub(super) col_names:  Vec<Vec<IString>>,
  pub(super) is_uniform: Option<u32>,
}

impl From<&super::super::script_parser::TableDefinition<radlr_rust_runtime::types::Token>>
  for TableDefinition
{
  fn from(
    value: &super::super::script_parser::TableDefinition<radlr_rust_runtime::types::Token>,
  ) -> Self {
    let (column_types, is_uniform) = match &value.types {
      table_row_types_Value::RowTypeDeclaration(rows) => {
        (rows.types.iter().map(|r| r.as_ref().clone()).collect::<Vec<_>>(), None)
      }
      table_row_types_Value::UniformTypeDeclaration(uniform_row) => (
        vec![RowType { name: uniform_row.r#type.clone(), attributes: Default::default() }],
        Some(uniform_row.size),
      ),
      _ => unreachable!(),
    };

    Self {
      name: value.id.id.intern(),
      types: column_types,
      is_uniform,
      col_names: value
        .aliases
        .iter()
        .map(|v| v.names.iter().map(|s| s.intern()).collect())
        .collect(),
    }
  }
}

impl TableDefinition {
  pub fn get_table_row_offsets(&self, offset: &mut u64) -> Vec<(u64, u64)> {
    let mut types = Vec::with_capacity(self.types.len());
    *offset = 0;
    for column in &self.types {
      let size = match column.name.as_str() {
        "u32" => 4,
        "f32" => 4,
        "u64" => 8,
        "f64" => 8,
        "str" => 8,
        "vec2" => 8,
        "vec3" => 16,
        "vec4" => 16,
        "ivec2" => 8,
        "ivec3" => 16,
        "ivec4" => 16,
        "f64" => 8,
        "dvec3" => 16,
        "dvec4" => 16,
        "ptr" => 8,
        ty => panic!("Need to report an invalid type error here {ty}"),
      };

      *offset = rum_container::get_aligned_value(*offset, size as u64);

      let out = (size, *offset);

      *offset += size;

      types.push(out)
    }

    types
  }

  pub fn get_table_col_offsets(&self) -> Vec<u32> {
    let mut types = Vec::with_capacity(self.types.len());
    for column in &self.types {
      let size = match column.name.as_str() {
        "u32" => 4u32,
        "f32" => 4,
        "u64" => 8,
        "f64" => 8,
        "str" => 8,
        _ => panic!("Need to report an invalid type error here"),
      };

      types.push(size)
    }

    types
  }
}

/*
 * INDICES:
 *    SORTED_COLUMN
 *    PRIMARY_KEY
 *
 */
