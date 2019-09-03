#![feature(ptr_internals)]
#![feature(allocator_api, alloc_layout_extra)]

use std::{mem, ptr};
use std::ptr::{Unique, NonNull};
use std::alloc::{Global, Layout, Alloc, handle_alloc_error};
use std::cmp::PartialEq;
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;


/// -----------------------------
/// base vec type for composition
/// -----------------------------
#[derive(Debug)]
struct RawVec<T> {
  ptr: Unique<T>,
  cap: usize,
}

impl<T> RawVec<T> {
  ///
  /// init a zero capacity raw vec, no alloc take place.
  ///
  fn new() -> Self {
    RawVec {
      ptr: Unique::empty(),
      cap: 0,
    }
  }

  fn from(s: &[T]) -> Self {
    unsafe {
      let layout = Layout::array::<T>(s.len()).unwrap();
      let ptr = Global.alloc(layout);

      // handle alloc fail
      if ptr.is_err() {
        handle_alloc_error(layout);
      }

      let ptr = ptr.unwrap();

      ptr::copy_nonoverlapping(s.as_ptr(), ptr.as_ptr().cast(), s.len());

      RawVec {
        ptr: Unique::from(ptr.cast()),
        cap: s.len(),
      }
    }
  }

  fn with_capacity(cap: usize) -> Self {
    unsafe {
      let layout = Layout::array::<T>(cap).unwrap();

      let ptr = Global.alloc(layout);

      // handle alloc fail
      if ptr.is_err() {
        handle_alloc_error(layout);
      }

      RawVec {
        ptr: Unique::from(ptr.unwrap().cast()),
        cap,
      }
    }
  }

  #[inline]
  fn offset(&self, i: usize) -> *const T {
    assert!(i < self.cap, "out of bound");
    unsafe { self.ptr.as_ptr().offset(i as isize) }
  }

  #[inline]
  fn offset_mut(&self, i: usize) -> *mut T {
    unsafe { self.ptr.as_ptr().offset(i as isize) }
  }

  #[inline]
  fn get_ref(&self, i: usize) -> &T {
    assert!(i < self.cap, "out of bound");
    unsafe { self.ptr.as_ptr().offset(i as isize).as_ref().unwrap() }
  }

  #[inline]
  fn get_mut(&self, i: usize) -> &mut T {
    assert!(i < self.cap, "out of bound");
    unsafe { self.ptr.as_ptr().offset(i as isize).as_mut().unwrap() }
  }

  #[inline]
  fn read(&self, i: usize) -> T {
    assert!(i < self.cap, "out of bound");
    unsafe { self.ptr.as_ptr().offset(i as isize).read() }
  }

  #[inline]
  fn write(&self, i: usize, elem: T) {
    assert!(i < self.cap, "out of bound");
    unsafe { self.ptr.as_ptr().offset(i as isize).write(elem); }
  }

  fn grow_cap(&mut self, factor: f32) {
    unsafe {
      let (new_cap, ptr) = if self.cap == 0 {
        (1, Global.alloc(Layout::array::<T>(1).unwrap()))
      } else {
        let new_cap = (self.cap as f32 * factor) as usize;
        let ptr = Global.realloc(NonNull::new_unchecked(self.ptr.as_ptr()).cast(),
                                 Layout::array::<T>(self.cap).unwrap(),
                                 new_cap * elem_size).unwrap();
        (new_cap, Ok(ptr))
      };

      if ptr.is_err() {
        handle_alloc_error(Layout::array::<T>(new_cap).unwrap())
      }

      let ptr = ptr.unwrap();

      self.ptr = Unique::new_unchecked(ptr.as_ptr() as *mut _);
      self.cap = new_cap;
    }
  }
}

impl<T> Drop for RawVec<T> {
  fn drop(&mut self) {
    if self.cap != 0 && elem_size != 0 {
      unsafe {
        let nn_ptr: NonNull<T> = self.ptr.into();
        Global.dealloc(nn_ptr.cast(),Layout::array::<T>(self.cap).unwrap());
      }
    }
  }
}

#[derive(Debug)]
pub struct HybridVec<T> {
  data_buffer: RawVec<T>,
  index_buffer: RawVec<u16>,
  len: usize,
}

impl<T> HybridVec<T> {
  fn double_cap(&mut self) {
    self.data_buffer.grow_cap(2f32);
    self.index_buffer.grow_cap(2f32);
  }

  fn grow_if_required(&mut self) {
    if self.len == self.data_buffer.cap {
      self.double_cap();
      // set new indexes to the free memory region
      for i in self.len..self.index_buffer.cap {
        self.index_buffer.write(i, i as u16);
      }
    }
  }

  pub fn new() -> Self {
    HybridVec { data_buffer: RawVec::new(), index_buffer: RawVec::new(), len: 0 }
  }

  pub fn from(array: &[T]) -> Self {
    let data_buffer = RawVec::from(array);
    // index buffer initialized to incr int sequence
    let index_buffer = RawVec::with_capacity(array.len());

    for i in 0..array.len() {
      index_buffer.write(i, i as u16);
    }

    HybridVec {
      data_buffer,
      index_buffer,
      len: array.len(),
    }
  }

  pub fn with_capacity(n: usize) -> Self {
    // uninitialized buffer
    let data_buffer = RawVec::with_capacity(n);
    // index buffer initialized to incr int sequence
    let index_buffer = RawVec::with_capacity(n);

    for i in 0..n {
      index_buffer.write(i, i as u16);
    }

    HybridVec {
      data_buffer,
      index_buffer,
      len: 0,
    }
  }

  #[inline]
  pub fn size(&self) -> usize { self.len }

  #[inline]
  pub fn empty(&self) -> bool { self.len == 0 }

  pub fn push(&mut self, elem: T) {
    self.grow_if_required();

    let new_elem_index = self.index_buffer.read(self.len);
    // write blindly elem in new element location
    self.data_buffer.write(new_elem_index as usize, elem);

    self.len += 1;
  }

  pub fn insert(&mut self, pos: usize, elem: T) {
    assert!(pos < self.data_buffer.cap, "out of bound");
    self.grow_if_required();

    unsafe {
      // shift right indexes buffer from pos
      ptr::copy(self.index_buffer.offset(pos),
                self.index_buffer.offset_mut(pos + 1),
                self.len - pos);

      // make index pos point to last element in data buffer
      self.index_buffer.write(pos, self.len as u16);
      self.data_buffer.write(self.len, elem);
    }

    self.len += 1;
  }

  pub fn erase(&mut self, pos: usize) -> Option<T> {
    if self.len == 0 || pos >= self.len {
      None
    } else {
      unsafe {
        // true position of the erased element in memory
        let erased_elem_index = self.index_buffer.read(pos);

        // shift left indexes buffer from pos
        ptr::copy(self.index_buffer.offset(pos + 1),
                  self.index_buffer.offset_mut(pos),
                  self.len - pos - 1);

        self.len -= 1;
        // make last index point to erased element in data buffer
        self.index_buffer.write(self.len, erased_elem_index);
        self.index_buffer.write(pos, self.len as u16);

        Some(self.data_buffer.read(erased_elem_index as usize))
      }
    }
  }

  pub fn clear(&mut self) {
    self.len = 0;
    for i in 0..self.index_buffer.cap {
      self.index_buffer.write(i, i as u16);
    }
  }

  pub fn get(&self, i: usize) -> Option<&T> {
    match i {
      i if i < self.len => {
        let elem_index = self.index_buffer.read(i);
        Some(self.data_buffer.get_ref(elem_index as usize))
      }
      _ => {
        None
      }
    }
  }

  pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
    match i {
      i if i < self.len => {
        let elem_index = self.index_buffer.read(i);
        Some(self.data_buffer.get_mut(elem_index as usize))
      }
      _ => {
        None
      }
    }
  }

  pub fn iter(&self) -> HIter<T> {
    unsafe {
      HIter {
        data: self.data_buffer.ptr.as_ptr(),
        start_index: self.index_buffer.ptr.as_ptr(),
        end_index: self.index_buffer.ptr.as_ptr().offset(self.len as isize),
        _marker: PhantomData,
      }
    }
  }
}

pub struct HIter<'a, T> {
  data: *const T,
  start_index: *const u16,
  end_index: *const u16,

  _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for HIter<'a, T> {
  type Item = &'a T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.start_index == self.end_index {
      None
    } else {
      unsafe {
        let item = self.data.offset(self.start_index.read() as isize);
        self.start_index = self.start_index.offset(1);
        item.as_ref()
      }
    }
  }
}

impl<'a, T> DoubleEndedIterator for HIter<'a, T> {
  fn next_back(&mut self) -> Option<Self::Item> {
    if self.start_index == self.end_index {
      None
    } else {
      unsafe {
        self.end_index = self.end_index.offset(-1);
        self.data.offset(self.end_index.read() as isize).as_ref()
      }
    }
  }
}

impl<T> PartialEq for HybridVec<T> where T: PartialEq {
  fn eq(&self, other: &Self) -> bool {
    if self.size() != other.size() {
      return false;
    }

    let mut zipped_iter = self.iter().zip(other.iter());

    loop {
      match zipped_iter.next() {
        None => { break; }
        Some((lhs, rhs)) => {
          if lhs != rhs {
            return false;
          }
        }
      }
    }

    true
  }
}

impl<T> Clone for HybridVec<T> {
  fn clone(&self) -> Self {
    let cap = self.data_buffer.cap;
    let mut hv_copy = Self::with_capacity(cap);
    unsafe {
      ptr::copy_nonoverlapping(self.data_buffer.ptr.as_ptr(),
                               hv_copy.data_buffer.ptr.as_ptr(), cap);
      ptr::copy_nonoverlapping(self.index_buffer.ptr.as_ptr(),
                               hv_copy.index_buffer.ptr.as_ptr(), cap);
    }
    hv_copy.len = self.len;
    hv_copy
  }

  fn clone_from(&mut self, source: &Self) {
    let src_cap = source.data_buffer.cap;

    unsafe {
      if self.data_buffer.cap < src_cap {
        let new_data_ptr = Global.realloc(
          NonNull::<T>::new_unchecked(self.data_buffer.ptr.as_ptr()).cast(),
          Layout::array::<T>(self.data_buffer.cap).unwrap(),
          src_cap * elem_size);

        let new_index_ptr = Global.realloc(
          NonNull::<u16>::new_unchecked(self.index_buffer.ptr.as_ptr()).cast(),
          Layout::array::<i16>(self.index_buffer.cap).unwrap(),
          src_cap * u16size);

        if new_data_ptr.is_err() {
          handle_alloc_error(Layout::array::<T>(src_cap).unwrap());
        }

        if new_index_ptr.is_err() {
          handle_alloc_error(Layout::array::<i16>(src_cap).unwrap());
        }

        self.data_buffer.ptr = Unique::from(new_data_ptr);
        self.index_buffer.ptr = Unique::from(new_index_ptr);
        self.data_buffer.cap = src_cap;
        self.index_buffer.cap = src_cap;

        ptr::copy_nonoverlapping(source.data_buffer.ptr.as_ptr(),
                                 self.data_buffer.ptr.as_ptr(), source.len);

        ptr::copy_nonoverlapping(source.index_buffer.ptr.as_ptr(),
                                 self.index_buffer.ptr.as_ptr(), source.len);

        self.len = source.len;
      }
    }
  }
}

impl<T> Index<usize> for HybridVec<T> {
  type Output = T;

  fn index(&self, index: usize) -> &Self::Output {
    assert!(index < self.len, "out of bound");
    let elem_index = self.index_buffer.read(index) as usize;
    self.data_buffer.get_ref(elem_index)
  }
}

impl<T> IndexMut<usize> for HybridVec<T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.len, "out of bound");
    let elem_index = self.index_buffer.read(index) as usize;
    self.data_buffer.get_mut(elem_index)
  }
}

#[macro_export]
macro_rules! hvec {
  ($($x:expr), *) => {
    {
      let len = [$($x), *].len();
      let mut hv = HybridVec::with_capacity(len);
      $(hv.push($x);)*
      hv
    }
  };
  ($elem:expr; $n:expr) => {
    [$elem; n].to_vec();
  };
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::{SeedableRng, Rng};
  use rand::rngs::SmallRng;

  #[test]
  fn create() {
    let empty_hv = HybridVec::<i32>::new();
    assert!(empty_hv.empty());
  }

  #[test]
  fn create_from() {
    let hv = HybridVec::from(&[0; 3]);
    assert_eq!(hv.size(), 3);
    assert_eq!(hv[0], 0);
    assert_eq!(hv[1], 0);
    assert_eq!(hv[2], 0);

    let hv = HybridVec::from(&[42; 3]);
    assert_eq!(hv[0], 42);
    assert_eq!(hv[1], 42);
    assert_eq!(hv[2], 42);
  }

  #[test]
  fn macro_create() {
    let hv = hvec![1, 2, 3];
    assert_eq!(hv.size(), 3);
  }

  #[test]
  fn iter() {
    let hv = hvec![1, 2, 3, 4];
    let mut iter = hv.iter();
    assert_eq!(hv.size(), 4);
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), Some(&4));
    assert_eq!(iter.next(), None);
  }

  #[test]
  fn rev_iter() {
    let hv = hvec![1, 2, 3, 4];
    let mut iter = hv.iter();

    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next_back(), Some(&4));
    assert_eq!(iter.next_back(), Some(&3));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), None);
  }

  #[test]
  fn get() {
    let hv = hvec![1, 2, 3];
    assert_eq!(hv.size(), 3);
    assert_eq!(hv.get(0), Some(&1));
    assert_eq!(hv.get(1), Some(&2));
    assert_eq!(hv.get(2), Some(&3));
    assert_eq!(hv.get(8723), None);
  }

  #[test]
  fn index() {
    let hv = hvec![1, 2, 3];
    assert_eq!(hv.size(), 3);

    assert_eq!(hv[0], 1);
    assert_eq!(hv[1], 2);
    assert_eq!(hv[2], 3);
  }

  #[test]
  fn push_with_capacity() {
    let mut hv = HybridVec::<i32>::with_capacity(2);
    hv.push(42);
    hv.push(3);
    assert_eq!(hv.size(), 2);
    assert_eq!(hv, hvec!(42, 3));
  }

  #[test]
  fn insert() {
    let mut hv = hvec![1, 2, 3];
    hv.insert(0, 0x2a);
    hv.insert(4, 0x2a);
    assert_eq!(hv.size(), 5);
    assert_eq!(hv, hvec![42, 1, 2, 3, 42]);
  }

  #[test]
  fn remove() {
    let mut hv = HybridVec::<i32>::new();
    hv.push(42);
    hv.push(3);
    let erased = hv.erase(0).unwrap();
    assert_eq!(erased, 42);

    assert_eq!(hv.size(), 1);
    assert_eq!(hv[0], 3);

    hv.erase(0);

    assert!(hv.empty());
  }

  #[test]
  fn clone() {
    let hv = hvec![1, 2, 3, 4];

    let hv_clone = hv.clone();

    assert_eq!(hv, hv_clone);
  }

  #[test]
  fn clone_from() {
    let hv = hvec![1, 2, 3, 4];

    let mut hv_clone = HybridVec::<i32>::new();
    hv_clone.clone_from(&hv);

    assert_eq!(hv, hv_clone);
  }

  #[test]
  fn toy() {
    let mut rng = SmallRng::seed_from_u64(32);

    for _ in 0..5 {
      println!("{}", rng.gen_range(0, 4));
    }
  }
}
