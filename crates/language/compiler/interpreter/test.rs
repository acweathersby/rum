use super::{
  runner::{interpret_string, RumScriptResult},
  types::Context,
};
use std::path::PathBuf;

fn get_source_path(file_name: &str) -> Result<PathBuf, std::io::Error> {
  PathBuf::from("/home/work/projects/lib_rum_common/crates/language/test_scripts/")
    .canonicalize()?
    .join(file_name)
    .canonicalize()
}

fn get_source_file(file_name: &str) -> Result<(String, PathBuf), std::io::Error> {
  let path = get_source_path(file_name)?;

  Ok((std::fs::read_to_string(&path)?, path))
}

#[test]
fn write_to_simple_table() -> RumScriptResult<()> {
  let mut ctx = Context::new();

  let (input, _) = get_source_file("write_to_simple_table.lang")?;

  interpret_string(&input, &mut ctx)?;

  Ok(())
}
