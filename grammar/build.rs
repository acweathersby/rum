use std::{
  path::{Path, PathBuf},
  process::Command,
};

use radlr_build::RadlrResult;

const GRAMMAR_PATH: &'static str = "./grammar";
const RUM_SCRIPT_ROOT: &'static str = "rum_lang.radlr";
const RAW_SCRIPT_ROOT: &'static str = "raw/raw.radlr";
const BUILD_OUTPUT_PATH: &'static str = "./compiler/parser/";

fn main() -> RadlrResult<()> {
  return Ok(());
  let workspace_dir = Path::new(GRAMMAR_PATH).parent().unwrap();

  let grammar_root_dir = workspace_dir.join(GRAMMAR_PATH).canonicalize().expect("Could not find RADLR grammar dir");

  println!("cargo:rerun-if-changed={}", grammar_root_dir.to_str().expect("Could not create str from RADLR dir path"));

  println!("cargo:rerun-if-changed={}", grammar_root_dir.join("raw").to_str().expect("Could not create str from RADLR dir path"));

  let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap()).canonicalize().expect("Could not find output dir").join(BUILD_OUTPUT_PATH);

  let process_a = build_rum_script(&grammar_root_dir, &out_dir);

  let output = process_a.wait_with_output().expect("Lost connection to rum_script build process");
  if !output.status.success() {
    panic!("{} {}", String::from_utf8(output.stderr).unwrap(), String::from_utf8(output.stdout).unwrap());
  }

  Ok(())
}

fn build_rum_script(grammar_root_dir: &Path, out_dir: &Path) -> std::process::Child {
  let mut radlr = Command::new("radlr");

  if radlr.get_program().is_empty() {
    panic!("Could not find radlr executable, is this in PATH?");
  }
  radlr.args([
    "build",
    "-o",
    out_dir.as_os_str().to_str().unwrap(),
    "-n",
    "rum_script",
    "-a",
    grammar_root_dir.join(RAW_SCRIPT_ROOT).as_os_str().to_str().unwrap(),
  ]);

  radlr.spawn().expect("Could not spawn rum_script build job")
}
