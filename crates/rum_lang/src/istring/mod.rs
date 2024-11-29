use crate::create_u64_hash;
use once_cell::sync::Lazy;
use std::{
  collections::HashMap,
  fmt::{Debug, Display},
  path::{Path, PathBuf},
  sync::{LockResult, RwLock, RwLockReadGuard},
};

type InnerStringStore = HashMap<IString, String>;

static GLOBAL_STORE: Lazy<IStringStore> = Lazy::new(|| IStringStore { _data: RwLock::new(InnerStringStore::new()) });

#[derive(Default)]
struct IStringStore {
  _data: RwLock<InnerStringStore>,
}

impl Debug for IStringStore {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("IStringStore")
  }
}

impl IStringStore {
  fn intern(&self, string: String, token: IString) -> IString {
    match self._data.read() {
      LockResult::Ok(data) => {
        if data.contains_key(&token) {
          return token;
        }
      }
      _ => panic!(),
    }

    match self._data.write() {
      LockResult::Ok(mut data) => {
        data.insert(token, string);
      }
      _ => panic!(),
    }

    token
  }

  fn get_str<'a>(&'a self, token: IString) -> Option<GuardedStr<'a>> {
    match self._data.read() {
      LockResult::Ok(store) => Some(GuardedStr(token, None, Some(store))),
      _ => panic!(),
    }
  }

  #[cfg(test)]
  /// Returns the number of strings stored within.
  pub fn len<'a>(&'a self) -> usize {
    match self._data.read() {
      LockResult::Ok(store) => store.len(),
      _ => panic!(),
    }
  }
}

/// A reference to a string interned within an [IStringStore]. A read lock on
/// the store is maintained as long as this object is in scope.
///
/// This should never be assigned to any object that outlives its current
/// function context. Should be dropped as soon as possible.
pub struct GuardedStr<'a>(IString, Option<IString>, Option<RwLockReadGuard<'a, InnerStringStore>>);

impl<'a> GuardedStr<'a> {
  pub fn as_str(&'a self) -> &'a str {
    if let Some(istring) = self.1.as_ref() {
      unsafe { istring.small_to_str() }
    } else if cfg!(debug_assertions) {
      self.2.as_ref().unwrap().get(&self.0).unwrap().as_str()
    } else {
      unsafe { self.2.as_ref().unwrap_unchecked().get(&self.0).unwrap_unchecked().as_str() }
    }
  }
}

/// An **I**nterned **String** for fast string operations. Combines a small
/// string type with an interned string type for larger string using
/// [StringStore].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IString(u64);

impl Default for IString {
  fn default() -> Self {
    Self(0)
  }
}

impl Display for IString {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.to_str().as_str())
  }
}

impl std::fmt::Debug for IString {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    unsafe {
      if self.is_large() {
        let mut s = f.debug_tuple("IString::Large");
        // This is an interned string.
        s.field(&self.0);
        s.field(&self.to_str().as_str());
        s.finish()
      } else {
        let mut s = f.debug_tuple("IString::Small");
        s.field(&self.small_to_str());
        s.finish()
      }
    }
  }
}

impl IString {
  pub fn is_empty(&self) -> bool {
    self.0 == 0
  }

  pub fn from_u64(val: u64) -> Self {
    Self(val)
  }

  pub fn as_u64(&self) -> u64 {
    self.0
  }

  pub fn is_small(&self) -> bool {
    unsafe {
      let val_bytes = self as *const IString as *const [u8; 8];
      (*val_bytes)[7] & 0x80 == 0
    }
  }

  pub fn is_large(&self) -> bool {
    !self.is_small()
  }

  pub fn to_string(&self) -> String {
    self.to_str().as_str().to_string()
  }

  pub fn to_path(&self) -> PathBuf {
    PathBuf::from(self.to_str().as_str().to_string())
  }

  /// Returns a [GuardedStr] that can be used to access the `&str` the [IString]
  /// token represents.
  pub fn to_str<'a>(self) -> GuardedStr<'a> {
    if self.is_large() {
      GLOBAL_STORE.get_str(self).unwrap()
    } else {
      GuardedStr(self, Some(self), None)
    }
  }

  /// Returns a [GuardedStr] that can be used to access the `&str` the [IString]
  /// token represents.
  pub fn to_str_global<'a>(self) -> GuardedStr<'a> {
    if self.is_large() {
      GLOBAL_STORE.get_str(self).unwrap()
    } else {
      GuardedStr(self, Some(self), None)
    }
  }

  unsafe fn small_to_str<'a>(&'a self) -> &'a str {
    let val_bytes = (&self.0) as *const u64 as *const [u8; 8];
    // This is a small string. The upper 7 bits comprise the length of the
    // string.

    let mut len = 8;

    for i in 0..8 {
      if (*val_bytes)[i] == 0 {
        len = i;
        break;
      }
    }

    let data = std::slice::from_raw_parts(&((*val_bytes)[0]), len);
    let out = std::str::from_utf8_unchecked(data);

    out
  }

  #[inline]
  fn from_bytes(string: &[u8]) -> Self {
    if string.is_empty() {
      Self(0)
    } else {
      let byte_len = string.len();
      if byte_len > 8 || (byte_len == 8 && (string[7] & 0x80) > 0) {
        let mut val = create_u64_hash(string);
        unsafe {
          let val_bytes = &mut val as *mut u64 as *mut [u8; 8];
          (*val_bytes)[7] |= 0x80;
        }
        Self(val)
      } else {
        let bytes = string;
        let mut val = 0u64;
        unsafe {
          let val_bytes = &mut val as *mut u64 as *mut [u8; 8];
          for (off, byte) in bytes.iter().enumerate() {
            (*val_bytes)[off] = *byte;
          }
          if byte_len < 8 {
            (*val_bytes)[byte_len] = 0;
          }
        }
        Self(val)
      }
    }
  }
}

pub trait CachedString {
  /// Derive the IString representation without interning
  /// the string. This can be useful when needing to compare an already
  /// interned IString with a standard string type.
  fn to_token(&self) -> IString {
    IString::from_bytes(self.get_bytes())
  }
  /// Returns a IString after interning the string within
  /// the given store. Only `LargeString` sub-types are interned.
  fn intern(&self) -> IString {
    let bytes = self.get_bytes();
    let token = IString::from_bytes(bytes);

    if bytes.len() > 7 {
      GLOBAL_STORE.intern(self.get_string(), token)
    } else {
      token
    }
  }

  #[allow(unused)]
  fn intern_global(&self) -> IString {
    let bytes = self.get_bytes();
    let token = IString::from_bytes(bytes);

    if bytes.len() > 7 {
      GLOBAL_STORE.intern(self.get_string(), token)
    } else {
      token
    }
  }

  fn get_string(&self) -> String;

  fn get_bytes(&self) -> &[u8];
}

impl CachedString for String {
  fn get_bytes(&self) -> &[u8] {
    self.as_bytes()
  }

  fn get_string(&self) -> String {
    self.clone()
  }
}

impl CachedString for &[u8] {
  fn get_bytes(&self) -> &[u8] {
    self
  }

  fn get_string(&self) -> String {
    String::from_utf8(self.to_vec()).unwrap()
  }
}

impl CachedString for &String {
  fn get_bytes(&self) -> &[u8] {
    self.as_bytes()
  }

  fn get_string(&self) -> String {
    self.to_string()
  }
}

impl CachedString for &str {
  fn get_bytes(&self) -> &[u8] {
    self.as_bytes()
  }

  fn get_string(&self) -> String {
    self.to_string()
  }
}

impl CachedString for &PathBuf {
  fn get_bytes(&self) -> &[u8] {
    self.to_str().unwrap().as_bytes()
  }

  fn get_string(&self) -> String {
    self.to_str().unwrap().to_owned()
  }
}

impl CachedString for PathBuf {
  fn get_bytes(&self) -> &[u8] {
    self.to_str().unwrap().as_bytes()
  }

  fn get_string(&self) -> String {
    self.to_str().unwrap().to_owned()
  }
}

impl CachedString for &Path {
  fn get_bytes(&self) -> &[u8] {
    self.to_str().unwrap().as_bytes()
  }

  fn get_string(&self) -> String {
    self.to_str().unwrap().to_owned()
  }
}

#[cfg(test)]
mod test;
