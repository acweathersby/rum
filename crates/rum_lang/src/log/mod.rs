#[macro_export]
macro_rules! rum_log {
  () => {
    rum_logger::ThreadLoggingBuffer::__log("\n")
  };
  ($($arg:tt)*) => {{
    rum_logger::ThreadLoggingBuffer::__log(&format!($($arg)*));
  }};
}

#[macro_export]
macro_rules! todo_note {
  ($($args: expr),*) => {
    print!("⚠️  TODO: ");
    print!($($args),*);
    print!("\n  in file: {}:{}", file!(), line!());
    println!(""); // to get a new line at the end
  };
}

#[macro_export]
macro_rules! dbg_println {
  ($($args: expr),*) => {
    {
      #[cfg(debug_assertions)]
      println!($($args),*);
    }
  };
}

/// Like `debug_assert!`, except completely compiled out in release
/// builds
#[macro_export]
macro_rules! rum_debug_assert {
  ($expr:expr, $($arg:tt)*) => {
    #[cfg(debug_assertions)]
    assert!($expr, $($arg)*)
  };
}
