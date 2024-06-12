use super::job::Job;
use core::{
  sync::atomic::{AtomicU32, Ordering::*},
  time::Duration,
};

pub type JobBuffer = *mut (usize, Job);

fn create_queue_layout(job_pool_size: usize) -> std::alloc::Layout {
  let layout_job_buffer =
    std::alloc::Layout::array::<Job>(job_pool_size).expect("Could not build JOB array");

  let layout_usize = std::alloc::Layout::new::<usize>();

  let (main_layout, _) =
    layout_usize.extend(layout_job_buffer).expect("Could not build Combined array");
  main_layout
}

pub fn create_queue(job_pool_size: usize) -> JobBuffer {
  unsafe {
    debug_assert!(job_pool_size <= u16::MAX as usize, "JOB_POOL_SIZE must be less than 65546");

    let main_layout = create_queue_layout(job_pool_size);

    let ptr = std::alloc::alloc(main_layout);
    let job_ptr: JobBuffer = std::mem::transmute(ptr);

    (&mut *job_ptr).0 = job_pool_size;

    for (i, job) in get_job_slice(job_ptr).iter_mut().enumerate() {
      std::mem::forget(std::mem::replace(job, Job::default()));

      job.id = i as u16;
      job.next = if i < (job_pool_size - 1) { (i + 1) as u16 } else { u16::MAX };
    }

    job_ptr
  }
}

pub fn free_queue(queue: JobBuffer) {
  unsafe {
    let slice = get_job_slice(queue);
    let len = slice.len();

    for job in slice {
      let _ = Box::from_raw(job.fence);
      let _ = job.task.take();
    }

    let main_layout = create_queue_layout(len);

    std::alloc::dealloc(queue as *mut _, main_layout);
  }
}

unsafe fn get_job_slice<'a>(queue: JobBuffer) -> &'a mut [Job] {
  let job_pool_size = (&mut *queue).0;
  std::slice::from_raw_parts_mut(get_job_queue_ptr(queue), job_pool_size)
}

pub(crate) fn get_job_queue_ptr(queue: *mut (usize, Job)) -> *mut Job {
  &mut (unsafe { &mut *queue }).1
}

pub(crate) fn create_queues(
  job_pool_size: usize,
  free_name: &'static str,
  job_name: &'static str,
) -> (JobBuffer, Box<RawQueueAtomic>, Box<RawQueueAtomic>, MTLLFIFOQueue16<Job>, MTLLFIFOQueue16<Job>)
{
  let jobs = create_queue(job_pool_size);

  let mut free_queue_ptr = Box::new(RawQueueAtomic::new(0, job_pool_size as u16 - 1));
  let mut job_queue_ptr = Box::new(RawQueueAtomic::new_empty());
  let free_queue = MTLLFIFOQueue16 {
    nodes: get_job_queue_ptr(jobs),
    list: &mut *free_queue_ptr,
    #[cfg(debug_assertions)]
    name: free_name,
  };
  let job_queue = MTLLFIFOQueue16 {
    nodes: get_job_queue_ptr(jobs),
    list: &mut *job_queue_ptr,
    #[cfg(debug_assertions)]
    name: job_name,
  };
  (jobs, free_queue_ptr, job_queue_ptr, free_queue, job_queue)
}

// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------

/// This trait is required by MTLLFIFOQueue16 to properly link nodes in a
/// contiguous region of memory. This uses 16bit identifiers for node indices,
/// and as such only supports queue sizes of 2^16-1. The 0xFFFF identifier is
/// reserved for the null indice.
pub trait MTLLNode16 {
  /// Get the id of the node. The id should be have 1-to-1 relationship with the
  /// position of the node within its buffer.
  fn get_id(&mut self) -> u16;
  /// Returns the node id of the next link.
  fn get_next(&mut self) -> u16;
  /// Creates a link between this node and another.
  fn set_next(&mut self, next: u16);
  /// Removes the link between this node and the next.
  fn __clear_next(&mut self) {
    self.set_next(u16::MAX);
  }
}

/// Store head and tail info for a RawQueue
///
/// Tries to maintain an exclusive cache line.
#[derive(Debug)]
#[cfg_attr(target_arch = "x86_64", repr(align(64)))]
pub struct RawQueueAtomic {
  info:    AtomicU32,
  #[allow(unused)]
  #[cfg(debug_assertions)]
  counter: AtomicU32,
}

impl RawQueueAtomic {
  pub fn new_empty() -> Self {
    RawQueueAtomic {
      info: AtomicU32::new(u32::MAX),
      #[cfg(debug_assertions)]
      counter: AtomicU32::new(0),
    }
  }

  pub fn new(head: u16, tail: u16) -> Self {
    RawQueueAtomic {
      info: AtomicU32::new((tail as u32) | ((head as u32) << 16)),
      #[cfg(debug_assertions)]
      counter: AtomicU32::new((tail - head + 1) as u32),
    }
  }
}

/// Multi-threaded Linked-List FIFO  Queue
#[derive(Debug)]
pub struct MTLLFIFOQueue16<Node: MTLLNode16> {
  pub(crate) list:  *mut RawQueueAtomic,
  pub(crate) nodes: *mut Node,
  #[cfg(debug_assertions)]
  pub(crate) name:  &'static str,
}

unsafe impl<Node: MTLLNode16> Send for MTLLFIFOQueue16<Node> {}
unsafe impl<Node: MTLLNode16> Sync for MTLLFIFOQueue16<Node> {}

impl<Node: MTLLNode16> Copy for MTLLFIFOQueue16<Node> {}
impl<Node: MTLLNode16> Clone for MTLLFIFOQueue16<Node> {
  fn clone(&self) -> Self {
    MTLLFIFOQueue16 {
      list: self.list,
      nodes: self.nodes,
      #[cfg(debug_assertions)]
      name: self.name,
    }
  }
}

impl<Node: MTLLNode16> Default for MTLLFIFOQueue16<Node> {
  fn default() -> Self {
    Self {
      list: std::ptr::null_mut(),
      nodes: std::ptr::null_mut(),
      #[cfg(debug_assertions)]
      name: "",
    }
  }
}

const QUEUE_WRITE_LOCKED: u32 = 0xFFFF_0000;
const QUEUE_EMPTY: u32 = 0xFFFF_FFFF;
const QUEUE_ELEMENT_MASK: u32 = 0xFFFF;

impl<Node: MTLLNode16> MTLLFIFOQueue16<Node> {
  pub fn new(list: *mut RawQueueAtomic, nodes: *mut Node, name: &'static str) -> Self {
    Self {
      list,
      nodes,
      #[cfg(debug_assertions)]
      name,
    }
  }

  pub fn push_back(&self, node: *mut Node) {
    unsafe {
      let list = &mut *self.list;

      (*node).__clear_next();

      let id: u16 = (*node).get_id();

      debug_assert!(id != QUEUE_ELEMENT_MASK as u16, "The 0xFFFF id is reserved!");

      loop {
        // Load tail. Ensure it is not null. If it is null
        // Then head should also be null.

        let list_fields = list.info.load(Acquire);

        if list_fields == QUEUE_WRITE_LOCKED {
          // The list is locked. Wait and try again
          std::thread::sleep(Duration::from_nanos(100));
          std::hint::spin_loop();
          continue;
        }

        let mut head = ((list_fields >> 16) & QUEUE_ELEMENT_MASK) as u16;
        let mut tail = ((list_fields) & QUEUE_ELEMENT_MASK) as u16;

        if tail == u16::MAX {
          debug_assert_eq!(tail, head);
          head = id;
          tail = id;

          match list.info.compare_exchange_weak(
            list_fields,
            ((head as u32) << 16) | tail as u32,
            SeqCst,
            Relaxed,
          ) {
            Ok(_) => {
              #[cfg(debug_assertions)]
              {
                //let count = list.counter.fetch_add(1, Relaxed) as isize + 1;
                //rum_log!("[{}] {}: push_back - {count} left",
                // Thread::get_id(), self.name)
              }
              return;
            }
            Err(_) => {
              // Someone got to this queue first. Try again.
              std::hint::spin_loop();
              continue;
            }
          }
        } else {
          let tail_node = self.nodes.wrapping_add(tail as usize);
          tail = id;
          // Lock the list to prevent broadcast of intermediate results.
          match list.info.compare_exchange_weak(list_fields, QUEUE_WRITE_LOCKED, Relaxed, Relaxed) {
            Ok(_) => {
              (*tail_node).set_next(id);
              match list.info.compare_exchange(
                QUEUE_WRITE_LOCKED,
                ((head as u32) << 16) | tail as u32,
                Release,
                Relaxed,
              ) {
                Err(_) => {
                  panic!("Queue has been corrupted");
                }
                Ok(_) => {
                  #[cfg(debug_assertions)]
                  {
                    // let count = list.counter.fetch_add(1, Relaxed) as isize +
                    // 1; rum_log!("[{}] {}: push_back -
                    // {count} left", Thread::get_id(), self.name)
                  }
                  return;
                }
              }
            }
            Err(_) => {
              // Someone got to this queue first. Try again.
              std::hint::spin_loop();
              continue;
            }
          }
        }
      }
    }
  }

  pub fn pop_front(&self) -> Option<*mut Node> {
    unsafe {
      let list = &mut *self.list;
      loop {
        match list.info.load(SeqCst) {
          QUEUE_WRITE_LOCKED => {
            // The list is locked. Wait and try again
            std::thread::sleep(Duration::from_nanos(100));
            std::hint::spin_loop();
            continue;
          }
          QUEUE_EMPTY => {
            // List is empty
            #[cfg(debug_assertions)]
            {
              //rum_log!("[{}] {}: empty list", Thread::get_id(), self.name)
            }
            return None;
          }
          list_fields => {
            let mut head = ((list_fields >> 16) & QUEUE_ELEMENT_MASK) as u16;
            let tail = ((list_fields) & QUEUE_ELEMENT_MASK) as u16;

            let head_node = self.nodes.wrapping_add(head as usize);

            if head == tail {
              match list.info.compare_exchange_weak(list_fields, QUEUE_EMPTY, Release, Relaxed) {
                Err(_) => {
                  std::hint::spin_loop();
                  continue;
                }
                Ok(_) => {
                  #[cfg(debug_assertions)]
                  {
                    // let count = list.counter.fetch_sub(1, Relaxed) as isize -
                    // 1; rum_log!("[{}] {}: pop_front -
                    // {count} left", Thread::get_id(), self.name)
                  }
                  return Some(&mut *head_node);
                }
              }
            } else {
              head = (*head_node).get_next();

              match list.info.compare_exchange_weak(
                list_fields,
                ((head as u32) << 16) | tail as u32,
                Release,
                Relaxed,
              ) {
                Err(_) => {
                  std::hint::spin_loop();
                  continue;
                }
                Ok(_) => {
                  #[cfg(debug_assertions)]
                  {
                    // let count = list.counter.fetch_sub(1, Relaxed) as isize -
                    // 1; rum_log!("[{}] {}: pop_front -
                    // {count} left", Thread::get_id(), self.name)
                  }
                  return Some(&mut *head_node);
                }
              }
            }
          }
        }
      }
    }
  }
}
