use radlr_build::*;
use std::path::Path;

const GRAMMAR_PATH: &'static str = "./grammar/";

const GRAMMAR_ROOT: &'static str = "rum_lang.radlr";

const BUILD_OUTPUT_PATH: &'static str = "./compiler/parser/";

fn main() {}
fn main_() -> RadlrResult<()> {
  let workspace_dir = Path::new(GRAMMAR_PATH).parent().unwrap();

  let grammar_root_dir =
    workspace_dir.join(GRAMMAR_PATH).canonicalize().expect("Could not find RADLR grammar dir");
  let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap())
    .canonicalize()
    .expect("Could not find output dir")
    .join(BUILD_OUTPUT_PATH);

  println!(
    "cargo:rerun-if-changed={}",
    grammar_root_dir.to_str().expect("Could not create str from RADLR dir path")
  );

  let radlr_root_file_path = grammar_root_dir.join(GRAMMAR_ROOT);

  let mut build_config = BuildConfig::new(&radlr_root_file_path);
  build_config.source_out = &out_dir;
  build_config.lib_out = &out_dir;

  fs_build(build_config, Default::default(), TargetLanguage::Rust)?;

  Ok(())
}
