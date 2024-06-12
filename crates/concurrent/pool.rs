use std::{
  sync::mpsc::{Receiver, Sender},
  thread::JoinHandle,
  time::Duration,
};

use rum_logger::dbg_println;

use crate::{containers::get_job_queue_ptr, error::RumResult, BroadcastIterator, ThreadId};

use super::{
  containers::{create_queues, free_queue, JobBuffer, MTLLFIFOQueue16, RawQueueAtomic},
  job::{Job, RumFuture, Task},
  specialists::{SpecializationTable, ThreadSpecializationTable},
  sync::Fence,
  thread::{Thread, ThreadDo, ThreadSays},
  ThreadHost,
};

fn create_worker(
  job_pool_size: usize,
  global_free_queue: MTLLFIFOQueue16<Job>,
  global_job_queue: MTLLFIFOQueue16<Job>,
  thread_comm: Sender<ThreadSays>,
  specialization_lut: *mut dyn SpecializationTable,
  id: usize,
) -> (JobBuffer, Box<RawQueueAtomic>, Box<RawQueueAtomic>, Thread) {
  let (local_jobs, local_free_queue_ptr, local_job_queue_ptr, local_free_queue, local_job_queue) =
    create_queues(job_pool_size, "Local Free", "Local Job");

  let worker = Thread::new(
    global_free_queue,
    global_job_queue,
    local_free_queue,
    local_job_queue,
    thread_comm,
    id,
    specialization_lut,
  );

  (local_jobs, local_free_queue_ptr, local_job_queue_ptr, worker)
}

pub(super) struct ThreadRef {
  handle:     JoinHandle<()>,
  job_buffer: JobBuffer,
  free_box:   Box<RawQueueAtomic>,
  job_box:    Box<RawQueueAtomic>,
  free_queue: MTLLFIFOQueue16<Job>,
  job_queue:  MTLLFIFOQueue16<Job>,
}

#[allow(unused)]
pub struct AppThreadPool {
  pub(super) num_of_threads:        usize,
  pub(super) c_signal:              Receiver<ThreadSays>,
  pub(super) job_store:             JobBuffer,
  pub(super) global_free_queue_ptr: Box<RawQueueAtomic>,
  pub(super) global_job_queue_ptr:  Box<RawQueueAtomic>,
  pub(super) threads:               Vec<ThreadRef>,
  pub(super) local_thread:          (Thread, JobBuffer, Box<RawQueueAtomic>, Box<RawQueueAtomic>),
  pub(super) threads_started:       usize,
  pub(super) specialty_lut:         Box<dyn SpecializationTable>,
}

impl ThreadHost for AppThreadPool {
  fn broadcast_message<'a>(&'a self) -> BroadcastIterator<'a> {
    BroadcastIterator {
      threads: self
        .threads
        .iter()
        .map(|ThreadRef { free_queue, job_queue, .. }| (free_queue, job_queue))
        .collect(),
      index:   0,
    }
  }

  fn num_of_threads(&self) -> usize {
    self.num_of_threads
  }

  fn __add_global_task_base(&self, task: Task, fence: bool) -> Option<Fence> {
    self.local_thread.0.__add_global_task_base(task, fence)
  }

  fn __add_specialized_task_base<Specialization: Into<usize>>(
    &self,
    specialization: Specialization,
    task: Task,
  ) {
    self.local_thread.0.__add_specialized_task_base(specialization, task)
  }

  fn add_task<T: FnOnce(&Thread) + Send + Sync + 'static>(&self, task: T) {
    self.local_thread.0.add_task(task)
  }

  fn add_task_specialized<S: Into<usize>, T: FnOnce(&Thread) + Send + Sync + 'static>(
    &self,
    specialization: S,
    task: T,
  ) {
    self.local_thread.0.add_task_specialized(specialization, task)
  }

  fn add_fenced_task<T: FnOnce(&Thread) + Send + Sync + 'static>(&self, task: T) -> Fence {
    self.local_thread.0.add_fenced_task(task)
  }

  fn add_async_task<T: RumFuture>(&self, task: T) {
    self.local_thread.0.add_async_task(task)
  }

  fn add_boxed_lz_task<Data: Sized, T: (FnOnce(&Thread) -> Data) + Send + Sync + 'static>(
    &self,
    create_fence: bool,
    task: T,
  ) -> (Option<Fence>, Box<super::LandingZone<Data>>) {
    self.local_thread.0.add_boxed_lz_task(create_fence, task)
  }

  unsafe fn add_lz_async_task<
    Data: Sized,
    F: std::future::Future<Output = Data> + Send + 'static,
  >(
    &self,
    signal: &mut super::LandingZone<Data>,
    create_fence: bool,
    task: F,
  ) -> Option<Fence> {
    self.local_thread.0.add_lz_async_task(signal, create_fence, task)
  }

  fn add_boxed_lz_async_task<
    Data: Sized,
    F: std::future::Future<Output = Data> + Send + 'static,
  >(
    &self,
    create_fence: bool,
    task: F,
  ) -> (Option<Fence>, Box<super::LandingZone<Data>>) {
    self.local_thread.0.add_boxed_lz_async_task(create_fence, task)
  }

  unsafe fn add_lz_task<Data: Sized, T: (FnOnce(&Thread) -> Data) + Send + Sync + 'static>(
    &self,
    signal: &mut super::LandingZone<Data>,
    create_fence: bool,
    task: T,
  ) -> Option<Fence> {
    self.local_thread.0.add_lz_task(signal, create_fence, task)
  }
}

impl AppThreadPool {
  pub fn new(
    num_of_threads: usize,
    job_buffer_size: usize,
    num_of_specialists: usize,
    num_of_specializations: usize,
  ) -> RumResult<Self> {
    debug_assert!(job_buffer_size <= u16::MAX as usize, "JOB_POOL_SIZE must be less than 65546");

    let (c_sender, receiver) = std::sync::mpsc::channel();

    let size = match std::thread::available_parallelism() {
      Ok(parallism) => num_of_threads.max(1).min(parallism.into()),
      Err(_) => return Err(crate::error::ConcurrentError::NoParallelismAvailable),
    };

    let (
      global_jobs,
      global_free_queue_ptr,
      global_job_queue_ptr,
      global_free_queue,
      global_job_queue,
    ) = create_queues(job_buffer_size, "Global Free", "Global Job");
    let mut specialty_lut =
      Box::new(ThreadSpecializationTable::new(num_of_specialists, num_of_specializations));

    let specialty_lut_ptr = specialty_lut.as_mut() as *mut _;

    let workers = (0..size)
      .into_iter()
      .map(|id| {
        let (job_buffer, mut free_ptr, mut job_ptr, mut worker) = create_worker(
          job_buffer_size,
          global_free_queue.clone(),
          global_job_queue.clone(),
          c_sender.clone(),
          specialty_lut_ptr,
          id,
        );

        let handle = std::thread::Builder::new()
          .name(format!("rum_thread_{}", id))
          .stack_size(10 * 1024 * 1024)
          .spawn(move || {
            worker.run();
          })
          .unwrap();

        let free_queue = MTLLFIFOQueue16::new(free_ptr.as_mut(), get_job_queue_ptr(job_buffer), "");
        let job_queue = MTLLFIFOQueue16::new(job_ptr.as_mut(), get_job_queue_ptr(job_buffer), "");

        ThreadRef {
          handle,
          job_buffer,
          free_box: free_ptr,
          job_box: job_ptr,
          free_queue,
          job_queue,
        }
      })
      .collect();

    let (local_jobs, local_free_queue, local_jobs_queue, worker) = create_worker(
      job_buffer_size,
      global_free_queue.clone(),
      global_job_queue.clone(),
      c_sender.clone(),
      specialty_lut_ptr,
      ThreadId::MAIN_THREAD_ID,
    );

    Ok(Self {
      num_of_threads,
      job_store: global_jobs,
      threads: workers,
      local_thread: (worker, local_jobs, local_free_queue, local_jobs_queue),
      global_job_queue_ptr,
      global_free_queue_ptr,
      c_signal: receiver,
      threads_started: 0,
      specialty_lut,
    })
  }

  /// Blocks the main thread until all threads complete their startup process.
  ///
  /// During this time the main thread may participate in processing any
  /// queued tasks.
  pub fn await_startup(&mut self) {
    while self.threads_started < self.thread_count() {
      dbg_println!("Awaiting thread startup");
      self.monitor();
      std::thread::sleep(Duration::from_micros(100));
    }
  }

  /// Gives threads a chance to gracefully exit after receiving a halt signal
  pub fn await_graceful_exit(mut self, timeout_secs: u64) {
    let time_now = std::time::Instant::now();

    for ThreadRef {
      handle,
      job_buffer: queue,
      free_box: free_ptr,
      job_box: job_ptr,
      free_queue,
      job_queue,
      ..
    } in self.threads.iter_mut()
    {
      if let Some(job) = free_queue.pop_front() {
        unsafe { (*job).task = Task::Command(ThreadDo::HaltQueued) };
        job_queue.push_back(job);
      } else {
        panic!("No more free jobs --------------------------")
      }

      if let Some(job) = self.local_thread.0.global_free_queue.pop_front() {
        unsafe { (*job).task = Task::Command(super::ThreadDo::Halt) };
        self.local_thread.0.global_job_queue.push_back(job);
      } else {
        panic!("No more free jobs --------------------------")
      }
    }

    while self.threads_started > 0
      && ((std::time::Instant::now() - time_now).as_secs() < timeout_secs)
    {
      self.monitor()
    }

    drop(self);
  }

  pub fn monitor(&mut self) {
    // Runs jobs locally until threads are able to spin up.
    if self.threads_started < 1 {
      loop {
        if let Ok(message) = self.c_signal.recv_timeout(Duration::from_micros(1)) {
          match message {
            ThreadSays::Initialized(thread_id) => {
              dbg_println!("Thread {thread_id} Started");
              self.threads_started += 1;
              break;
            }
            _ => {}
          }
        }
        self.local_thread.0.local_run();
      }
    }

    while let Ok(message) = self.c_signal.recv_timeout(Duration::from_micros(1)) {
      match message {
        ThreadSays::Initialized(thread_id) => {
          dbg_println!("Thread {thread_id} Started");
          self.threads_started += 1;
        }
        ThreadSays::Parked(thread_id) => {
          dbg_println!("Thread {thread_id} Parked");
        }
        ThreadSays::Halted(thread_id) => {
          dbg_println!("Thread {thread_id} Halted");
          self.threads_started -= 1;
        }
        _ => {}
      }
    }
  }

  pub fn thread_count(&self) -> usize {
    self.threads.len()
  }
}

impl Drop for AppThreadPool {
  fn drop(&mut self) {
    unsafe {
      for _ in self
        .threads
        .drain(..)
        .map(|ThreadRef { handle, job_buffer, free_box, job_box, free_queue, job_queue }| {
          if let Some(job) = free_queue.pop_front() {
            (*job).task = Task::Command(ThreadDo::Halt);
            job_queue.push_back(job);
          }

          (handle, job_buffer, free_box, job_box)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(join, job_store, ..)| {
          match join.join() {
            Err(_err) => {
              dbg_println!("Error occured while freeing thread resources")
            }
            Ok(_) => {}
          }
          free_queue(job_store);
        })
      {}

      free_queue(self.job_store);
    }
  }
}
