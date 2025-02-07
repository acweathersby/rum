pub struct x86Function {
  binary:       *const u8,
  binary_size:  usize,
  entry_offset: usize,
}

impl x86Function {
  pub fn new(binary: &[u8], entry_offset: usize) -> x86Function {
    let allocation_size = binary.len();

    let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
    let flags: i32 = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), allocation_size, prot, flags, -1, 0) as *mut u8 };

    let data = unsafe { std::slice::from_raw_parts_mut(ptr, allocation_size) };

    data.copy_from_slice(&binary);

    Self { binary: ptr, binary_size: allocation_size, entry_offset }
  }

  pub fn access_as_call<'a, F>(&'a self) -> &'a F {
    unsafe {
      let entry_point = &self.binary.offset(self.entry_offset as isize);
      std::mem::transmute(entry_point)
    }
  }
}

impl Drop for x86Function {
  fn drop(&mut self) {
    let result = unsafe { libc::munmap(self.binary as *mut _, self.binary_size) };
    debug_assert_eq!(result, 0);
  }
}
