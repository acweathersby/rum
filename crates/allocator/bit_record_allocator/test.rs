use std::{
  alloc::Layout,
  hash::{BuildHasher, Hasher},
};

use rum_container::ArrayVec;
use rum_profile::NSReporter;

use super::{block_record::BlockRecord, BlockRecordAllocator};
use crate::bit_record_allocator::allocator::BitState;

const _1GB: usize = 1024 * 1024 * 1024;

#[test]
pub fn block() {
  let mut block = BlockRecord::<u64>::default();
  block.set_nibble(0, BitState::Partial);
  assert_eq!(block.iter_blocks().next().unwrap().0, BitState::Partial);

  let mut block = BlockRecord::<u64>::default();
  block.set_nibble(0, BitState::Full);
  assert_eq!(block.iter_blocks().next().unwrap().0, BitState::Full);

  let mut block = BlockRecord::<u64>::default();
  block.set_nibble(0, BitState::Sub);
  assert_eq!(block.iter_blocks().next().unwrap().0, BitState::Sub);

  let mut block = BlockRecord::<u64>::default();
  block.set_nibble(0, BitState::Empty);
  assert_eq!(block.iter_blocks().next().unwrap().0, BitState::Empty);

  let mut block = BlockRecord::<u64>::default();
  for bit in 0..block.len() {
    block.set_nibble(bit, BitState::Full);
  }
  assert_eq!(block.is_full(), true);

  let mut block = BlockRecord::<u64>::default();
  for bit in 0..block.len() {
    block.set_nibble(bit, BitState::Sub);
  }
  assert_eq!(block.is_full(), true);

  let mut block = BlockRecord::<u64>::default();
  for bit in 0..block.len() {
    block.set_nibble(bit, BitState::Partial);
  }
  assert_eq!(block.is_full(), false);

  let mut block = BlockRecord::<u64>::default();
  for bit in 0..block.len() {
    block.set_nibble(bit, BitState::Empty);
  }
  assert_eq!(block.is_full(), false);

  let mut block = BlockRecord::<u64>::default();
  for bit in 0..block.len() {
    block.set_nibble(bit, BitState::Full);
  }

  for bit in 10..block.len() {
    block.set_nibble(bit, BitState::Partial);
  }

  dbg!(block);

  assert_eq!(block.iter_blocks().collect::<Vec<_>>()[0].0, BitState::Full);
}

#[test]
pub fn basic_allocations() {
  let data = BlockRecordAllocator::<1, u8, _>::new(128).expect("Should be fine!");

  unsafe {
    *data.alloc(1).expect("should have space") = 'h' as u8;
    *data.alloc(1).expect("should have space") = 'e' as u8;
    dbg!(&data);
    *data.alloc(1).expect("should have space") = 'l' as u8;
    *data.alloc(1).expect("should have space") = 'l' as u8;
    dbg!(&data);
    *data.alloc(1).expect("should have space") = 'o' as u8;
    *data.alloc(1).expect("should have space") = ' ' as u8;
    dbg!(&data);
    *data.alloc(1).expect("should have space") = 'w' as u8;
    *data.alloc(1).expect("should have space") = 'o' as u8;
    dbg!(&data);
    *data.alloc(1).expect("should have space") = 'r' as u8;
    *data.alloc(1).expect("should have space") = 'l' as u8;

    let penult = data.alloc(1).expect("should have space");
    *penult = 'd' as u8;

    dbg!(&data);

    let last = data.alloc(1).expect("should have space");
    *last = 'd' as u8;

    let allocatted_string =
      String::from_utf8(std::slice::from_raw_parts(data.data(), "hello world".len()).to_vec())
        .expect("should be utf8 (ASCII)");

    debug_assert_eq!(allocatted_string, "hello world");

    data.free(penult).expect("Corrupted pointer!");
    dbg!(&data);
    data.free(last).expect("Corrupted pointer!");

    dbg!(&data);
  }
}

#[test]
fn unfulfilled_allocation_results_in_none() {
  let allocator = BlockRecordAllocator::<32, u16, _>::new(1024).expect("Should be fine!");
  allocator.alloc(512).expect("Should have space");
  allocator.alloc(512).expect("Should have space");
  assert!(allocator.alloc(512).is_none());

  let allocator = BlockRecordAllocator::<32, u32, _>::new(1024).expect("Should be fine!");
  allocator.alloc(512).expect("Should have space");
  allocator.alloc(512).expect("Should have space");
  assert!(allocator.alloc(512).is_none());
}

#[test]
pub fn test() {
  let data = 0b1111_0011_0110_1011u16;
  let mut x = data;

  let len = 2;
  for _ in 0..(len - 1) {
    x = x & (x << 1);
  }

  let leading = x.leading_zeros();
  println!("{data:0>16b} {x:0>16b} {}", 16 - leading)
}

#[test]
pub fn inter_block_allocations() {
  let allocator = BlockRecordAllocator::<1, u32, _>::new(1024).expect("Should be fine!");
  dbg!(&allocator);
  let r = NSReporter::new("alloc");
  let _ = allocator.alloc(25).expect("should have space");
  r.report();
  dbg!(&allocator);
  let r = NSReporter::new("alloc");
  let ptr = allocator.alloc(8).expect("should have space");
  r.report();
  dbg!(&allocator);
  let r = NSReporter::new("alloc");
  allocator.free(ptr).unwrap();
  r.report();
  dbg!(&allocator);
}

#[test]
pub fn massive_allocation_1_gb() {
  let r = NSReporter::new("create");
  let allocator = BlockRecordAllocator::<4096, u32, _>::new(_1GB).expect("Should be fine!");

  r.report();
  let r = NSReporter::new("alloc");
  let ptr = allocator.alloc(_1GB).expect("should have space");
  r.report();

  let r = NSReporter::new("free");
  assert!(allocator.alloc(_1GB).is_none());
  r.report();
  allocator.free(ptr).expect("Should free ptr");

  for _ in 0..10000 {
    let mut vec = ArrayVec::<2, _>::new();
    vec.push(allocator.alloc(_1GB >> 1).expect("Should allocate"));
    vec.push(allocator.alloc(_1GB >> 1).expect("Should allocate"));

    assert_eq!(vec[1] as usize - vec[0] as usize, _1GB >> 1);

    for ptr in vec.as_slice() {
      allocator.free(*ptr).expect("Should free ptr");
    }

    assert!(allocator.is_empty());
  }

  let mut vec = ArrayVec::<2, _>::new();
  let r = NSReporter::new("alloc");
  vec.push(allocator.alloc(_1GB >> 1).expect("Should allocate"));
  vec.push(allocator.alloc(_1GB >> 1).expect("Should allocate"));
  r.report();

  assert_eq!(vec[1] as usize - vec[0] as usize, _1GB >> 1);

  let r = NSReporter::new("free both");
  for ptr in vec.as_slice() {
    allocator.free(*ptr).expect("Should free ptr");
  }
  r.report();

  assert!(allocator.is_empty());

  dbg!(&allocator);
}

#[test]
pub fn multi_block_allocations() {
  let allocator = BlockRecordAllocator::<4, u16, _>::new(512).expect("Should be fine!");

  let _ = allocator.alloc(1).expect("should have space");
  dbg!(&allocator);
  let ptr_a = allocator.alloc(3).expect("should have space");
  dbg!(&allocator);
  let ptr_b = allocator.alloc(8).expect("should have space");
  dbg!(&allocator);
  let _ = allocator.alloc(3).expect("should have space");
  dbg!(&allocator);
  let _ = allocator.alloc(3).expect("should have space");
  dbg!(&allocator);
  let ptr_c = allocator.alloc(64).expect("should have space");
  dbg!(&allocator);
  allocator.alloc(1).expect("should have space");
  dbg!(&allocator);
  allocator.free(ptr_a).expect("Corrupted pointer!");
  dbg!(&allocator);
  allocator.free(ptr_b).expect("Corrupted pointer!");
  dbg!(&allocator);
  allocator.free(ptr_c).expect("Corrupted pointer!");
  dbg!(&allocator);
}

#[test]
pub fn large_allocation_followed_by_small_allocation() {
  let allocator =
    BlockRecordAllocator::<128, u32, _, true>::new_managed(std::ptr::null_mut(), 8388608)
      .expect("Should be fine!");

  let _ = allocator.alloc(4194304).expect("Should have space");

  let _ = allocator.alloc(4096).expect("Should have space");

  let _ = allocator.alloc(4096).expect("Should have space");

  let ptr = allocator.alloc(128).expect("Should have space");
  dbg!(&allocator);
  allocator.free(ptr).expect("failed to free");
  dbg!(&allocator);
}

#[test]
pub fn tesdst() {
  let allocator =
    BlockRecordAllocator::<4096, u32, _, true>::new_managed(std::ptr::null_mut(), _1GB)
      .expect("Should be fine!");

  let allocations = vec![
    (557056, allocator.alloc(557056).expect("Pointer should be valid")),
    (2408448, allocator.alloc(2408448).expect("Pointer should be valid")),
    (458752, allocator.alloc(458752).expect("Pointer should be valid")),
    (3686400, allocator.alloc(3686400).expect("Pointer should be valid")),
    (2228224, allocator.alloc(2228224).expect("Pointer should be valid")),
    (1064960, allocator.alloc(1064960).expect("Pointer should be valid")),
    (2293760, allocator.alloc(2293760).expect("Pointer should be valid")),
    (1572864, allocator.alloc(1572864).expect("Pointer should be valid")),
    (2113536, allocator.alloc(2113536).expect("Pointer should be valid")),
    (2326528, allocator.alloc(2326528).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (2146304, allocator.alloc(2146304).expect("Pointer should be valid")),
    (2490368, allocator.alloc(2490368).expect("Pointer should be valid")),
    (3162112, allocator.alloc(3162112).expect("Pointer should be valid")),
    (2260992, allocator.alloc(2260992).expect("Pointer should be valid")),
    (1425408, allocator.alloc(1425408).expect("Pointer should be valid")),
    (917504, allocator.alloc(917504).expect("Pointer should be valid")),
    (3145728, allocator.alloc(3145728).expect("Pointer should be valid")),
    (2211840, allocator.alloc(2211840).expect("Pointer should be valid")),
    (3014656, allocator.alloc(3014656).expect("Pointer should be valid")),
    (3162112, allocator.alloc(3162112).expect("Pointer should be valid")),
    (2260992, allocator.alloc(2260992).expect("Pointer should be valid")),
    (1376256, allocator.alloc(1376256).expect("Pointer should be valid")),
    (573440, allocator.alloc(573440).expect("Pointer should be valid")),
    (2588672, allocator.alloc(2588672).expect("Pointer should be valid")),
    (3981312, allocator.alloc(3981312).expect("Pointer should be valid")),
    (458752, allocator.alloc(458752).expect("Pointer should be valid")),
    (3702784, allocator.alloc(3702784).expect("Pointer should be valid")),
    (2392064, allocator.alloc(2392064).expect("Pointer should be valid")),
    (311296, allocator.alloc(311296).expect("Pointer should be valid")),
    (425984, allocator.alloc(425984).expect("Pointer should be valid")),
    (3440640, allocator.alloc(3440640).expect("Pointer should be valid")),
    (294912, allocator.alloc(294912).expect("Pointer should be valid")),
    (360448, allocator.alloc(360448).expect("Pointer should be valid")),
    (835584, allocator.alloc(835584).expect("Pointer should be valid")),
    (475136, allocator.alloc(475136).expect("Pointer should be valid")),
    (3801088, allocator.alloc(3801088).expect("Pointer should be valid")),
    (1146880, allocator.alloc(1146880).expect("Pointer should be valid")),
    (2916352, allocator.alloc(2916352).expect("Pointer should be valid")),
    (262144, allocator.alloc(262144).expect("Pointer should be valid")),
    (98304, allocator.alloc(98304).expect("Pointer should be valid")),
    (2932736, allocator.alloc(2932736).expect("Pointer should be valid")),
    (475136, allocator.alloc(475136).expect("Pointer should be valid")),
    (3817472, allocator.alloc(3817472).expect("Pointer should be valid")),
    (1261568, allocator.alloc(1261568).expect("Pointer should be valid")),
    (1753088, allocator.alloc(1753088).expect("Pointer should be valid")),
    (1474560, allocator.alloc(1474560).expect("Pointer should be valid")),
    (3473408, allocator.alloc(3473408).expect("Pointer should be valid")),
    (540672, allocator.alloc(540672).expect("Pointer should be valid")),
    (2244608, allocator.alloc(2244608).expect("Pointer should be valid")),
    (1212416, allocator.alloc(1212416).expect("Pointer should be valid")),
    (1392640, allocator.alloc(1392640).expect("Pointer should be valid")),
    (671744, allocator.alloc(671744).expect("Pointer should be valid")),
    (1196032, allocator.alloc(1196032).expect("Pointer should be valid")),
    (1294336, allocator.alloc(1294336).expect("Pointer should be valid")),
    (1998848, allocator.alloc(1998848).expect("Pointer should be valid")),
    (3489792, allocator.alloc(3489792).expect("Pointer should be valid")),
    (753664, allocator.alloc(753664).expect("Pointer should be valid")),
    (1949696, allocator.alloc(1949696).expect("Pointer should be valid")),
    (933888, allocator.alloc(933888).expect("Pointer should be valid")),
    (3309568, allocator.alloc(3309568).expect("Pointer should be valid")),
    (1425408, allocator.alloc(1425408).expect("Pointer should be valid")),
    (950272, allocator.alloc(950272).expect("Pointer should be valid")),
    (3489792, allocator.alloc(3489792).expect("Pointer should be valid")),
    (770048, allocator.alloc(770048).expect("Pointer should be valid")),
    (2048000, allocator.alloc(2048000).expect("Pointer should be valid")),
    (3899392, allocator.alloc(3899392).expect("Pointer should be valid")),
    (1916928, allocator.alloc(1916928).expect("Pointer should be valid")),
    (655360, allocator.alloc(655360).expect("Pointer should be valid")),
    (1097728, allocator.alloc(1097728).expect("Pointer should be valid")),
    (2605056, allocator.alloc(2605056).expect("Pointer should be valid")),
    (4145152, allocator.alloc(4145152).expect("Pointer should be valid")),
    (3817472, allocator.alloc(3817472).expect("Pointer should be valid")),
    (1294336, allocator.alloc(1294336).expect("Pointer should be valid")),
    (2080768, allocator.alloc(2080768).expect("Pointer should be valid")),
    (4128768, allocator.alloc(4128768).expect("Pointer should be valid")),
    (3751936, allocator.alloc(3751936).expect("Pointer should be valid")),
    (2752512, allocator.alloc(2752512).expect("Pointer should be valid")),
    (1097728, allocator.alloc(1097728).expect("Pointer should be valid")),
    (2490368, allocator.alloc(2490368).expect("Pointer should be valid")),
    (3260416, allocator.alloc(3260416).expect("Pointer should be valid")),
    (3096576, allocator.alloc(3096576).expect("Pointer should be valid")),
    (3801088, allocator.alloc(3801088).expect("Pointer should be valid")),
    (1130496, allocator.alloc(1130496).expect("Pointer should be valid")),
    (2768896, allocator.alloc(2768896).expect("Pointer should be valid")),
    (1245184, allocator.alloc(1245184).expect("Pointer should be valid")),
    (1638400, allocator.alloc(1638400).expect("Pointer should be valid")),
    (2670592, allocator.alloc(2670592).expect("Pointer should be valid")),
    (2506752, allocator.alloc(2506752).expect("Pointer should be valid")),
    (3375104, allocator.alloc(3375104).expect("Pointer should be valid")),
    (1867776, allocator.alloc(1867776).expect("Pointer should be valid")),
    (327680, allocator.alloc(327680).expect("Pointer should be valid")),
    (606208, allocator.alloc(606208).expect("Pointer should be valid")),
    (2867200, allocator.alloc(2867200).expect("Pointer should be valid")),
    (2031616, allocator.alloc(2031616).expect("Pointer should be valid")),
    (3735552, allocator.alloc(3735552).expect("Pointer should be valid")),
    (2686976, allocator.alloc(2686976).expect("Pointer should be valid")),
    (2637824, allocator.alloc(2637824).expect("Pointer should be valid")),
    (2326528, allocator.alloc(2326528).expect("Pointer should be valid")),
    (1867776, allocator.alloc(1867776).expect("Pointer should be valid")),
    (360448, allocator.alloc(360448).expect("Pointer should be valid")),
    (786432, allocator.alloc(786432).expect("Pointer should be valid")),
    (81920, allocator.alloc(81920).expect("Pointer should be valid")),
    (2850816, allocator.alloc(2850816).expect("Pointer should be valid")),
    (1851392, allocator.alloc(1851392).expect("Pointer should be valid")),
    (245760, allocator.alloc(245760).expect("Pointer should be valid")),
    (2031616, allocator.alloc(2031616).expect("Pointer should be valid")),
    (3670016, allocator.alloc(3670016).expect("Pointer should be valid")),
    (2129920, allocator.alloc(2129920).expect("Pointer should be valid")),
    (2457600, allocator.alloc(2457600).expect("Pointer should be valid")),
    (835584, allocator.alloc(835584).expect("Pointer should be valid")),
    (409600, allocator.alloc(409600).expect("Pointer should be valid")),
    (3276800, allocator.alloc(3276800).expect("Pointer should be valid")),
    (1064960, allocator.alloc(1064960).expect("Pointer should be valid")),
    (2293760, allocator.alloc(2293760).expect("Pointer should be valid")),
    (1638400, allocator.alloc(1638400).expect("Pointer should be valid")),
    (2621440, allocator.alloc(2621440).expect("Pointer should be valid")),
    (2097152, allocator.alloc(2097152).expect("Pointer should be valid")),
    (2129920, allocator.alloc(2129920).expect("Pointer should be valid")),
    (2408448, allocator.alloc(2408448).expect("Pointer should be valid")),
    (409600, allocator.alloc(409600).expect("Pointer should be valid")),
    (3391488, allocator.alloc(3391488).expect("Pointer should be valid")),
    (1998848, allocator.alloc(1998848).expect("Pointer should be valid")),
    (3407872, allocator.alloc(3407872).expect("Pointer should be valid")),
    (49152, allocator.alloc(49152).expect("Pointer should be valid")),
    (2539520, allocator.alloc(2539520).expect("Pointer should be valid")),
    (3604480, allocator.alloc(3604480).expect("Pointer should be valid")),
    (3735552, allocator.alloc(3735552).expect("Pointer should be valid")),
    (2654208, allocator.alloc(2654208).expect("Pointer should be valid")),
    (2375680, allocator.alloc(2375680).expect("Pointer should be valid")),
    (245760, allocator.alloc(245760).expect("Pointer should be valid")),
    (1998848, allocator.alloc(1998848).expect("Pointer should be valid")),
    (3522560, allocator.alloc(3522560).expect("Pointer should be valid")),
    (1015808, allocator.alloc(1015808).expect("Pointer should be valid")),
    (3981312, allocator.alloc(3981312).expect("Pointer should be valid")),
    (458752, allocator.alloc(458752).expect("Pointer should be valid")),
    (3670016, allocator.alloc(3670016).expect("Pointer should be valid")),
    (2129920, allocator.alloc(2129920).expect("Pointer should be valid")),
    (2392064, allocator.alloc(2392064).expect("Pointer should be valid")),
    (278528, allocator.alloc(278528).expect("Pointer should be valid")),
    (163840, allocator.alloc(163840).expect("Pointer should be valid")),
    (1425408, allocator.alloc(1425408).expect("Pointer should be valid")),
    (983040, allocator.alloc(983040).expect("Pointer should be valid")),
    (3751936, allocator.alloc(3751936).expect("Pointer should be valid")),
    (2785280, allocator.alloc(2785280).expect("Pointer should be valid")),
    (1343488, allocator.alloc(1343488).expect("Pointer should be valid")),
    (360448, allocator.alloc(360448).expect("Pointer should be valid")),
    (851968, allocator.alloc(851968).expect("Pointer should be valid")),
    (524288, allocator.alloc(524288).expect("Pointer should be valid")),
    (2113536, allocator.alloc(2113536).expect("Pointer should be valid")),
    (2277376, allocator.alloc(2277376).expect("Pointer should be valid")),
    (1507328, allocator.alloc(1507328).expect("Pointer should be valid")),
    (3751936, allocator.alloc(3751936).expect("Pointer should be valid")),
    (2801664, allocator.alloc(2801664).expect("Pointer should be valid")),
    (1490944, allocator.alloc(1490944).expect("Pointer should be valid")),
    (3571712, allocator.alloc(3571712).expect("Pointer should be valid")),
    (3473408, allocator.alloc(3473408).expect("Pointer should be valid")),
    (557056, allocator.alloc(557056).expect("Pointer should be valid")),
    (2375680, allocator.alloc(2375680).expect("Pointer should be valid")),
    (147456, allocator.alloc(147456).expect("Pointer should be valid")),
    (1228800, allocator.alloc(1228800).expect("Pointer should be valid")),
    (1523712, allocator.alloc(1523712).expect("Pointer should be valid")),
    (3801088, allocator.alloc(3801088).expect("Pointer should be valid")),
    (1114112, allocator.alloc(1114112).expect("Pointer should be valid")),
    (2637824, allocator.alloc(2637824).expect("Pointer should be valid")),
    (2310144, allocator.alloc(2310144).expect("Pointer should be valid")),
    (1720320, allocator.alloc(1720320).expect("Pointer should be valid")),
    (1294336, allocator.alloc(1294336).expect("Pointer should be valid")),
    (1966080, allocator.alloc(1966080).expect("Pointer should be valid")),
    (3178496, allocator.alloc(3178496).expect("Pointer should be valid")),
    (2408448, allocator.alloc(2408448).expect("Pointer should be valid")),
    (491520, allocator.alloc(491520).expect("Pointer should be valid")),
    (3997696, allocator.alloc(3997696).expect("Pointer should be valid")),
    (540672, allocator.alloc(540672).expect("Pointer should be valid")),
    (2326528, allocator.alloc(2326528).expect("Pointer should be valid")),
    (1933312, allocator.alloc(1933312).expect("Pointer should be valid")),
    (868352, allocator.alloc(868352).expect("Pointer should be valid")),
    (655360, allocator.alloc(655360).expect("Pointer should be valid")),
    (1064960, allocator.alloc(1064960).expect("Pointer should be valid")),
    (2277376, allocator.alloc(2277376).expect("Pointer should be valid")),
    (1556480, allocator.alloc(1556480).expect("Pointer should be valid")),
    (4161536, allocator.alloc(4161536).expect("Pointer should be valid")),
    (4014080, allocator.alloc(4014080).expect("Pointer should be valid")),
    (753664, allocator.alloc(753664).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (49152, allocator.alloc(49152).expect("Pointer should be valid")),
    (2572288, allocator.alloc(2572288).expect("Pointer should be valid")),
    (3915776, allocator.alloc(3915776).expect("Pointer should be valid")),
    (2031616, allocator.alloc(2031616).expect("Pointer should be valid")),
    (3768320, allocator.alloc(3768320).expect("Pointer should be valid")),
    (2981888, allocator.alloc(2981888).expect("Pointer should be valid")),
    (868352, allocator.alloc(868352).expect("Pointer should be valid")),
    (704512, allocator.alloc(704512).expect("Pointer should be valid")),
    (1441792, allocator.alloc(1441792).expect("Pointer should be valid")),
    (3178496, allocator.alloc(3178496).expect("Pointer should be valid")),
    (2408448, allocator.alloc(2408448).expect("Pointer should be valid")),
    (507904, allocator.alloc(507904).expect("Pointer should be valid")),
    (4079616, allocator.alloc(4079616).expect("Pointer should be valid")),
    (3375104, allocator.alloc(3375104).expect("Pointer should be valid")),
    (1884160, allocator.alloc(1884160).expect("Pointer should be valid")),
    (491520, allocator.alloc(491520).expect("Pointer should be valid")),
    (4046848, allocator.alloc(4046848).expect("Pointer should be valid")),
    (966656, allocator.alloc(966656).expect("Pointer should be valid")),
    (3538944, allocator.alloc(3538944).expect("Pointer should be valid")),
    (3194880, allocator.alloc(3194880).expect("Pointer should be valid")),
    (2523136, allocator.alloc(2523136).expect("Pointer should be valid")),
    (3440640, allocator.alloc(3440640).expect("Pointer should be valid")),
    (344064, allocator.alloc(344064).expect("Pointer should be valid")),
    (720896, allocator.alloc(720896).expect("Pointer should be valid")),
    (1638400, allocator.alloc(1638400).expect("Pointer should be valid")),
    (2736128, allocator.alloc(2736128).expect("Pointer should be valid")),
    (3129344, allocator.alloc(3129344).expect("Pointer should be valid")),
    (4079616, allocator.alloc(4079616).expect("Pointer should be valid")),
    (3375104, allocator.alloc(3375104).expect("Pointer should be valid")),
    (1933312, allocator.alloc(1933312).expect("Pointer should be valid")),
    (802816, allocator.alloc(802816).expect("Pointer should be valid")),
    (163840, allocator.alloc(163840).expect("Pointer should be valid")),
    (1376256, allocator.alloc(1376256).expect("Pointer should be valid")),
    (622592, allocator.alloc(622592).expect("Pointer should be valid")),
    (2883584, allocator.alloc(2883584).expect("Pointer should be valid")),
    (81920, allocator.alloc(81920).expect("Pointer should be valid")),
    (2801664, allocator.alloc(2801664).expect("Pointer should be valid")),
    (1441792, allocator.alloc(1441792).expect("Pointer should be valid")),
    (3244032, allocator.alloc(3244032).expect("Pointer should be valid")),
    (2981888, allocator.alloc(2981888).expect("Pointer should be valid")),
    (819200, allocator.alloc(819200).expect("Pointer should be valid")),
    (344064, allocator.alloc(344064).expect("Pointer should be valid")),
    (671744, allocator.alloc(671744).expect("Pointer should be valid")),
    (1196032, allocator.alloc(1196032).expect("Pointer should be valid")),
    (1245184, allocator.alloc(1245184).expect("Pointer should be valid")),
    (1572864, allocator.alloc(1572864).expect("Pointer should be valid")),
    (2097152, allocator.alloc(2097152).expect("Pointer should be valid")),
    (2162688, allocator.alloc(2162688).expect("Pointer should be valid")),
    (2719744, allocator.alloc(2719744).expect("Pointer should be valid")),
    (2932736, allocator.alloc(2932736).expect("Pointer should be valid")),
    (507904, allocator.alloc(507904).expect("Pointer should be valid")),
    (4112384, allocator.alloc(4112384).expect("Pointer should be valid")),
    (3620864, allocator.alloc(3620864).expect("Pointer should be valid")),
    (3801088, allocator.alloc(3801088).expect("Pointer should be valid")),
    (1163264, allocator.alloc(1163264).expect("Pointer should be valid")),
    (3047424, allocator.alloc(3047424).expect("Pointer should be valid")),
    (3424256, allocator.alloc(3424256).expect("Pointer should be valid")),
    (229376, allocator.alloc(229376).expect("Pointer should be valid")),
    (1933312, allocator.alloc(1933312).expect("Pointer should be valid")),
    (901120, allocator.alloc(901120).expect("Pointer should be valid")),
    (1032192, allocator.alloc(1032192).expect("Pointer should be valid")),
    (4079616, allocator.alloc(4079616).expect("Pointer should be valid")),
    (3293184, allocator.alloc(3293184).expect("Pointer should be valid")),
    (1294336, allocator.alloc(1294336).expect("Pointer should be valid")),
    (2080768, allocator.alloc(2080768).expect("Pointer should be valid")),
    (4128768, allocator.alloc(4128768).expect("Pointer should be valid")),
    (3702784, allocator.alloc(3702784).expect("Pointer should be valid")),
    (2392064, allocator.alloc(2392064).expect("Pointer should be valid")),
    (262144, allocator.alloc(262144).expect("Pointer should be valid")),
    (114688, allocator.alloc(114688).expect("Pointer should be valid")),
    (3047424, allocator.alloc(3047424).expect("Pointer should be valid")),
    (3473408, allocator.alloc(3473408).expect("Pointer should be valid")),
    (622592, allocator.alloc(622592).expect("Pointer should be valid")),
    (2965504, allocator.alloc(2965504).expect("Pointer should be valid")),
    (753664, allocator.alloc(753664).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (81920, allocator.alloc(81920).expect("Pointer should be valid")),
    (2785280, allocator.alloc(2785280).expect("Pointer should be valid")),
    (1359872, allocator.alloc(1359872).expect("Pointer should be valid")),
    (507904, allocator.alloc(507904).expect("Pointer should be valid")),
    (4063232, allocator.alloc(4063232).expect("Pointer should be valid")),
    (3211264, allocator.alloc(3211264).expect("Pointer should be valid")),
    (2736128, allocator.alloc(2736128).expect("Pointer should be valid")),
    (3129344, allocator.alloc(3129344).expect("Pointer should be valid")),
    (4128768, allocator.alloc(4128768).expect("Pointer should be valid")),
    (3735552, allocator.alloc(3735552).expect("Pointer should be valid")),
    (2686976, allocator.alloc(2686976).expect("Pointer should be valid")),
    (2637824, allocator.alloc(2637824).expect("Pointer should be valid")),
    (2244608, allocator.alloc(2244608).expect("Pointer should be valid")),
    (1212416, allocator.alloc(1212416).expect("Pointer should be valid")),
    (1310720, allocator.alloc(1310720).expect("Pointer should be valid")),
    (81920, allocator.alloc(81920).expect("Pointer should be valid")),
    (2752512, allocator.alloc(2752512).expect("Pointer should be valid")),
    (1163264, allocator.alloc(1163264).expect("Pointer should be valid")),
    (3014656, allocator.alloc(3014656).expect("Pointer should be valid")),
    (3194880, allocator.alloc(3194880).expect("Pointer should be valid")),
    (2523136, allocator.alloc(2523136).expect("Pointer should be valid")),
    (3424256, allocator.alloc(3424256).expect("Pointer should be valid")),
    (212992, allocator.alloc(212992).expect("Pointer should be valid")),
    (1736704, allocator.alloc(1736704).expect("Pointer should be valid")),
    (1376256, allocator.alloc(1376256).expect("Pointer should be valid")),
    (622592, allocator.alloc(622592).expect("Pointer should be valid")),
    (2883584, allocator.alloc(2883584).expect("Pointer should be valid")),
    (2162688, allocator.alloc(2162688).expect("Pointer should be valid")),
    (2686976, allocator.alloc(2686976).expect("Pointer should be valid")),
    (2621440, allocator.alloc(2621440).expect("Pointer should be valid")),
    (2097152, allocator.alloc(2097152).expect("Pointer should be valid")),
    (2146304, allocator.alloc(2146304).expect("Pointer should be valid")),
    (2605056, allocator.alloc(2605056).expect("Pointer should be valid")),
    (4145152, allocator.alloc(4145152).expect("Pointer should be valid")),
    (3833856, allocator.alloc(3833856).expect("Pointer should be valid")),
    (1327104, allocator.alloc(1327104).expect("Pointer should be valid")),
    (229376, allocator.alloc(229376).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (98304, allocator.alloc(98304).expect("Pointer should be valid")),
    (2916352, allocator.alloc(2916352).expect("Pointer should be valid")),
    (344064, allocator.alloc(344064).expect("Pointer should be valid")),
    (704512, allocator.alloc(704512).expect("Pointer should be valid")),
    (1523712, allocator.alloc(1523712).expect("Pointer should be valid")),
    (3883008, allocator.alloc(3883008).expect("Pointer should be valid")),
    (1703936, allocator.alloc(1703936).expect("Pointer should be valid")),
    (1163264, allocator.alloc(1163264).expect("Pointer should be valid")),
    (3047424, allocator.alloc(3047424).expect("Pointer should be valid")),
    (3440640, allocator.alloc(3440640).expect("Pointer should be valid")),
    (262144, allocator.alloc(262144).expect("Pointer should be valid")),
    (2129920, allocator.alloc(2129920).expect("Pointer should be valid")),
    (2392064, allocator.alloc(2392064).expect("Pointer should be valid")),
    (294912, allocator.alloc(294912).expect("Pointer should be valid")),
    (327680, allocator.alloc(327680).expect("Pointer should be valid")),
    (606208, allocator.alloc(606208).expect("Pointer should be valid")),
    (2834432, allocator.alloc(2834432).expect("Pointer should be valid")),
    (1736704, allocator.alloc(1736704).expect("Pointer should be valid")),
    (1359872, allocator.alloc(1359872).expect("Pointer should be valid")),
    (425984, allocator.alloc(425984).expect("Pointer should be valid")),
    (3522560, allocator.alloc(3522560).expect("Pointer should be valid")),
    (933888, allocator.alloc(933888).expect("Pointer should be valid")),
    (3293184, allocator.alloc(3293184).expect("Pointer should be valid")),
    (1277952, allocator.alloc(1277952).expect("Pointer should be valid")),
    (1916928, allocator.alloc(1916928).expect("Pointer should be valid")),
    (737280, allocator.alloc(737280).expect("Pointer should be valid")),
    (1802240, allocator.alloc(1802240).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (81920, allocator.alloc(81920).expect("Pointer should be valid")),
    (2768896, allocator.alloc(2768896).expect("Pointer should be valid")),
    (1212416, allocator.alloc(1212416).expect("Pointer should be valid")),
    (1343488, allocator.alloc(1343488).expect("Pointer should be valid")),
    (278528, allocator.alloc(278528).expect("Pointer should be valid")),
    (147456, allocator.alloc(147456).expect("Pointer should be valid")),
    (1196032, allocator.alloc(1196032).expect("Pointer should be valid")),
    (1179648, allocator.alloc(1179648).expect("Pointer should be valid")),
    (1064960, allocator.alloc(1064960).expect("Pointer should be valid")),
    (2310144, allocator.alloc(2310144).expect("Pointer should be valid")),
    (1785856, allocator.alloc(1785856).expect("Pointer should be valid")),
    (1736704, allocator.alloc(1736704).expect("Pointer should be valid")),
    (1392640, allocator.alloc(1392640).expect("Pointer should be valid")),
    (655360, allocator.alloc(655360).expect("Pointer should be valid")),
    (1163264, allocator.alloc(1163264).expect("Pointer should be valid")),
    (3112960, allocator.alloc(3112960).expect("Pointer should be valid")),
    (3981312, allocator.alloc(3981312).expect("Pointer should be valid")),
    (409600, allocator.alloc(409600).expect("Pointer should be valid")),
    (3342336, allocator.alloc(3342336).expect("Pointer should be valid")),
    (1622016, allocator.alloc(1622016).expect("Pointer should be valid")),
    (2523136, allocator.alloc(2523136).expect("Pointer should be valid")),
    (3424256, allocator.alloc(3424256).expect("Pointer should be valid")),
    (245760, allocator.alloc(245760).expect("Pointer should be valid")),
    (2048000, allocator.alloc(2048000).expect("Pointer should be valid")),
    (3817472, allocator.alloc(3817472).expect("Pointer should be valid")),
    (1228800, allocator.alloc(1228800).expect("Pointer should be valid")),
    (1556480, allocator.alloc(1556480).expect("Pointer should be valid")),
    (4177920, allocator.alloc(4177920).expect("Pointer should be valid")),
    (4161536, allocator.alloc(4161536).expect("Pointer should be valid")),
    (3981312, allocator.alloc(3981312).expect("Pointer should be valid")),
    (475136, allocator.alloc(475136).expect("Pointer should be valid")),
    (3883008, allocator.alloc(3883008).expect("Pointer should be valid")),
    (1785856, allocator.alloc(1785856).expect("Pointer should be valid")),
    (1720320, allocator.alloc(1720320).expect("Pointer should be valid")),
    (1277952, allocator.alloc(1277952).expect("Pointer should be valid")),
    (1884160, allocator.alloc(1884160).expect("Pointer should be valid")),
    (442368, allocator.alloc(442368).expect("Pointer should be valid")),
    (3571712, allocator.alloc(3571712).expect("Pointer should be valid")),
    (3522560, allocator.alloc(3522560).expect("Pointer should be valid")),
    (983040, allocator.alloc(983040).expect("Pointer should be valid")),
    (3702784, allocator.alloc(3702784).expect("Pointer should be valid")),
    (2441216, allocator.alloc(2441216).expect("Pointer should be valid")),
    (704512, allocator.alloc(704512).expect("Pointer should be valid")),
    (1507328, allocator.alloc(1507328).expect("Pointer should be valid")),
    (3751936, allocator.alloc(3751936).expect("Pointer should be valid")),
    (2867200, allocator.alloc(2867200).expect("Pointer should be valid")),
    (2064384, allocator.alloc(2064384).expect("Pointer should be valid")),
    (3997696, allocator.alloc(3997696).expect("Pointer should be valid")),
    (638976, allocator.alloc(638976).expect("Pointer should be valid")),
    (3063808, allocator.alloc(3063808).expect("Pointer should be valid")),
    (3571712, allocator.alloc(3571712).expect("Pointer should be valid")),
    (3489792, allocator.alloc(3489792).expect("Pointer should be valid")),
    (737280, allocator.alloc(737280).expect("Pointer should be valid")),
    (1785856, allocator.alloc(1785856).expect("Pointer should be valid")),
    (1802240, allocator.alloc(1802240).expect("Pointer should be valid")),
    (1884160, allocator.alloc(1884160).expect("Pointer should be valid")),
    (425984, allocator.alloc(425984).expect("Pointer should be valid")),
    (3506176, allocator.alloc(3506176).expect("Pointer should be valid")),
    (868352, allocator.alloc(868352).expect("Pointer should be valid")),
    (655360, allocator.alloc(655360).expect("Pointer should be valid")),
    (1114112, allocator.alloc(1114112).expect("Pointer should be valid")),
    (2637824, allocator.alloc(2637824).expect("Pointer should be valid")),
    (2244608, allocator.alloc(2244608).expect("Pointer should be valid")),
    (1277952, allocator.alloc(1277952).expect("Pointer should be valid")),
    (1900544, allocator.alloc(1900544).expect("Pointer should be valid")),
    (540672, allocator.alloc(540672).expect("Pointer should be valid")),
    (2342912, allocator.alloc(2342912).expect("Pointer should be valid")),
    (2015232, allocator.alloc(2015232).expect("Pointer should be valid")),
    (3538944, allocator.alloc(3538944).expect("Pointer should be valid")),
    (3145728, allocator.alloc(3145728).expect("Pointer should be valid")),
    (2195456, allocator.alloc(2195456).expect("Pointer should be valid")),
    (2965504, allocator.alloc(2965504).expect("Pointer should be valid")),
    (671744, allocator.alloc(671744).expect("Pointer should be valid")),
    (1261568, allocator.alloc(1261568).expect("Pointer should be valid")),
    (1769472, allocator.alloc(1769472).expect("Pointer should be valid")),
    (1622016, allocator.alloc(1622016).expect("Pointer should be valid")),
    (2523136, allocator.alloc(2523136).expect("Pointer should be valid")),
    (3489792, allocator.alloc(3489792).expect("Pointer should be valid")),
    (770048, allocator.alloc(770048).expect("Pointer should be valid")),
    (1982464, allocator.alloc(1982464).expect("Pointer should be valid")),
    (3375104, allocator.alloc(3375104).expect("Pointer should be valid")),
    (1835008, allocator.alloc(1835008).expect("Pointer should be valid")),
    (49152, allocator.alloc(49152).expect("Pointer should be valid")),
    (2588672, allocator.alloc(2588672).expect("Pointer should be valid")),
    (3981312, allocator.alloc(3981312).expect("Pointer should be valid")),
    (393216, allocator.alloc(393216).expect("Pointer should be valid")),
    (3194880, allocator.alloc(3194880).expect("Pointer should be valid")),
    (2555904, allocator.alloc(2555904).expect("Pointer should be valid")),
    (3768320, allocator.alloc(3768320).expect("Pointer should be valid")),
    (2981888, allocator.alloc(2981888).expect("Pointer should be valid")),
    (819200, allocator.alloc(819200).expect("Pointer should be valid")),
    (376832, allocator.alloc(376832).expect("Pointer should be valid")),
    (933888, allocator.alloc(933888).expect("Pointer should be valid")),
    (3391488, allocator.alloc(3391488).expect("Pointer should be valid")),
    (2048000, allocator.alloc(2048000).expect("Pointer should be valid")),
    (3866624, allocator.alloc(3866624).expect("Pointer should be valid")),
    (1605632, allocator.alloc(1605632).expect("Pointer should be valid")),
    (2441216, allocator.alloc(2441216).expect("Pointer should be valid")),
    (671744, allocator.alloc(671744).expect("Pointer should be valid")),
    (1245184, allocator.alloc(1245184).expect("Pointer should be valid")),
    (1671168, allocator.alloc(1671168).expect("Pointer should be valid")),
    (2932736, allocator.alloc(2932736).expect("Pointer should be valid")),
    (442368, allocator.alloc(442368).expect("Pointer should be valid")),
    (3620864, allocator.alloc(3620864).expect("Pointer should be valid")),
    (3850240, allocator.alloc(3850240).expect("Pointer should be valid")),
    (1474560, allocator.alloc(1474560).expect("Pointer should be valid")),
    (3506176, allocator.alloc(3506176).expect("Pointer should be valid")),
    (819200, allocator.alloc(819200).expect("Pointer should be valid")),
    (327680, allocator.alloc(327680).expect("Pointer should be valid")),
    (638976, allocator.alloc(638976).expect("Pointer should be valid")),
    (3129344, allocator.alloc(3129344).expect("Pointer should be valid")),
    (4079616, allocator.alloc(4079616).expect("Pointer should be valid")),
    (3276800, allocator.alloc(3276800).expect("Pointer should be valid")),
    (1163264, allocator.alloc(1163264).expect("Pointer should be valid")),
    (3080192, allocator.alloc(3080192).expect("Pointer should be valid")),
    (3751936, allocator.alloc(3751936).expect("Pointer should be valid")),
    (2818048, allocator.alloc(2818048).expect("Pointer should be valid")),
    (1589248, allocator.alloc(1589248).expect("Pointer should be valid")),
    (2244608, allocator.alloc(2244608).expect("Pointer should be valid")),
    (1294336, allocator.alloc(1294336).expect("Pointer should be valid")),
    (2031616, allocator.alloc(2031616).expect("Pointer should be valid")),
    (3784704, allocator.alloc(3784704).expect("Pointer should be valid")),
    (3047424, allocator.alloc(3047424).expect("Pointer should be valid")),
    (3506176, allocator.alloc(3506176).expect("Pointer should be valid")),
    (868352, allocator.alloc(868352).expect("Pointer should be valid")),
    (753664, allocator.alloc(753664).expect("Pointer should be valid")),
    (1916928, allocator.alloc(1916928).expect("Pointer should be valid")),
    (770048, allocator.alloc(770048).expect("Pointer should be valid")),
    (1982464, allocator.alloc(1982464).expect("Pointer should be valid")),
    (3276800, allocator.alloc(3276800).expect("Pointer should be valid")),
    (1048576, allocator.alloc(1048576).expect("Pointer should be valid")),
    (2162688, allocator.alloc(2162688).expect("Pointer should be valid")),
    (2736128, allocator.alloc(2736128).expect("Pointer should be valid")),
    (3112960, allocator.alloc(3112960).expect("Pointer should be valid")),
    (3932160, allocator.alloc(3932160).expect("Pointer should be valid")),
    (65536, allocator.alloc(65536).expect("Pointer should be valid")),
    (2637824, allocator.alloc(2637824).expect("Pointer should be valid")),
    (2293760, allocator.alloc(2293760).expect("Pointer should be valid")),
    (1622016, allocator.alloc(1622016).expect("Pointer should be valid")),
    (2605056, allocator.alloc(2605056).expect("Pointer should be valid")),
    (4128768, allocator.alloc(4128768).expect("Pointer should be valid")),
    (3735552, allocator.alloc(3735552).expect("Pointer should be valid")),
    (2621440, allocator.alloc(2621440).expect("Pointer should be valid")),
    (2179072, allocator.alloc(2179072).expect("Pointer should be valid")),
    (2867200, allocator.alloc(2867200).expect("Pointer should be valid")),
    (1966080, allocator.alloc(1966080).expect("Pointer should be valid")),
    (3194880, allocator.alloc(3194880).expect("Pointer should be valid")),
    (2506752, allocator.alloc(2506752).expect("Pointer should be valid")),
    (3342336, allocator.alloc(3342336).expect("Pointer should be valid")),
    (1687552, allocator.alloc(1687552).expect("Pointer should be valid")),
    (3112960, allocator.alloc(3112960).expect("Pointer should be valid")),
    (3964928, allocator.alloc(3964928).expect("Pointer should be valid")),
    (262144, allocator.alloc(262144).expect("Pointer should be valid")),
    (49152, allocator.alloc(49152).expect("Pointer should be valid")),
    (2539520, allocator.alloc(2539520).expect("Pointer should be valid")),
    (3571712, allocator.alloc(3571712).expect("Pointer should be valid")),
    (3407872, allocator.alloc(3407872).expect("Pointer should be valid")),
    (49152, allocator.alloc(49152).expect("Pointer should be valid")),
    (2555904, allocator.alloc(2555904).expect("Pointer should be valid")),
    (3702784, allocator.alloc(3702784).expect("Pointer should be valid")),
    (2424832, allocator.alloc(2424832).expect("Pointer should be valid")),
    (589824, allocator.alloc(589824).expect("Pointer should be valid")),
    (2621440, allocator.alloc(2621440).expect("Pointer should be valid")),
    (2129920, allocator.alloc(2129920).expect("Pointer should be valid")),
    (2375680, allocator.alloc(2375680).expect("Pointer should be valid")),
    (196608, allocator.alloc(196608).expect("Pointer should be valid")),
    (1671168, allocator.alloc(1671168).expect("Pointer should be valid")),
    (2916352, allocator.alloc(2916352).expect("Pointer should be valid")),
    (376832, allocator.alloc(376832).expect("Pointer should be valid")),
    (983040, allocator.alloc(983040).expect("Pointer should be valid")),
    (3719168, allocator.alloc(3719168).expect("Pointer should be valid")),
    (2539520, allocator.alloc(2539520).expect("Pointer should be valid")),
    (3555328, allocator.alloc(3555328).expect("Pointer should be valid")),
  ];

  // dbg!(&allocator);

  for (size, ptr) in allocations {
    if size == 3555328 {
      dbg!(&allocator);
      println!("{size} {}", ptr as usize / 4096 / 32);
    }
    let free_result = allocator.free(ptr);
    free_result.expect("Should free");
  }

  dbg!(&allocator);
}

#[test]
pub fn fuzz_test_random_allocations() {
  let allocator = BlockRecordAllocator::<4096, u64, _>::new(_1GB).expect("Should be fine!");

  let records_before_allocation = unsafe { allocator.0.as_ref().unwrap().record_snapshot() };

  let mut allocations = Vec::with_capacity(1000);
  let random_state = std::collections::hash_map::RandomState::new();
  let mut rn = random_state.build_hasher().finish();

  let r = NSReporter::new("alloc");
  loop {
    let old = rn;
    let val = 1024 * (rn & 0xFF0);

    if val > 0 {
      match allocator.alloc(val as usize) {
        Some(ptr) => allocations.push((ptr, val)),
        None => {
          break;
        }
      }
    }
    rn = (rn.rotate_left(3) ^ (0x123456F8456)) + 0x400;

    if rn == 0 || rn == old {
      break;
    }
  }
  r.report();

  if false {
    for (_, val) in &allocations {
      println!(r###"({val}, allocator.alloc({val}).expect("Pointer should be valid")),"###);
    }
  }
  //dbg!(&allocator);

  println!("len: {}", allocations.len());

  let r = NSReporter::new("free");
  for (ptr, _) in allocations {
    allocator.free(ptr).expect("Should free");
  }
  r.report();

  let records_after_free = unsafe { allocator.0.as_ref().unwrap().record_snapshot() };

  assert_eq!(records_before_allocation, records_after_free);

  //dbg!(&allocator);
}

#[test]
fn large_non_power_of_2_allocations() {
  let ptr = unsafe { std::alloc::alloc(Layout::from_size_align(208896, 64).unwrap()) };
  let _ = BlockRecordAllocator::<128, u32, _, true>::new_managed(ptr, 208896 as usize).unwrap();

  let ptr = unsafe { std::alloc::alloc(Layout::from_size_align(41763456, 64).unwrap()) };
  let _ = BlockRecordAllocator::<128, u32, _, true>::new_managed(ptr, 41763456 as usize).unwrap();
}

#[test]
fn large_non_power_of_2_allocation() {
  let ptr = unsafe { std::alloc::alloc(Layout::from_size_align(41763456, 64).unwrap()) };
  let allocator =
    BlockRecordAllocator::<128, u32, _, true>::new_managed(ptr, 41763456 as usize).unwrap();

  allocator.free(allocator.alloc(41000000 - 4194304).unwrap()).expect("failed to free");

  dbg!(allocator);
}
