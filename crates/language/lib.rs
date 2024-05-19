#![feature(unsized_tuple_coercion)]
mod compiler;

#[cfg(test)]
mod utils {
  use std::path::PathBuf;

  pub fn get_source_path(file_name: &str) -> Result<PathBuf, std::io::Error> {
    PathBuf::from("/home/work/projects/lib_rum_common/crates/language/test_scripts/")
      .canonicalize()?
      .join(file_name)
      .canonicalize()
  }

  pub fn get_source_file(file_name: &str) -> Result<(String, PathBuf), std::io::Error> {
    let path = get_source_path(file_name)?;
    Ok((std::fs::read_to_string(&path)?, path))
  }
}
