#![feature(ptr_internals)]
#![feature(allocator_api)]

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
  pub fn new() -> Self {
    assert_ne!(mem::size_of::<T>(),
               0,
               "we are not ready to handle Zero size types");
    RawVec {
      ptr: Unique::empty(),
      cap: 0,
    }
  }

  fn with_capacity(cap: usize) -> Self {
    unsafe {
      let ptr = Global
        .alloc(Layout::from_size_align_unchecked(cap * mem::size_of::<T>(),
                                                 mem::align_of::<T>())).unwrap();
      RawVec {
        ptr: Unique::new_unchecked(ptr.as_ptr() as *mut _),
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
      let align = mem::align_of::<T>();
      let elem_size = mem::size_of::<T>();

      let (new_cap, ptr) = if self.cap == 0 {
        (1, Global.alloc(Layout::from_size_align_unchecked(elem_size, align)))
      } else {
        let new_cap = (self.cap as f32 * factor) as usize;
        let ptr = Global.realloc(NonNull::new_unchecked(self.ptr.as_ptr()).cast(),
                                 Layout::from_size_align_unchecked(self.cap * elem_size, align),
                                 new_cap * elem_size).unwrap();
        (new_cap, Ok(ptr))
      };

      if ptr.is_err() {
        handle_alloc_error(Layout::from_size_align_unchecked(elem_size * new_cap, align))
      }

      let ptr = ptr.unwrap();

      self.ptr = Unique::new_unchecked(ptr.as_ptr() as *mut _);
      self.cap = new_cap;
    }
  }
}

impl<T> Drop for RawVec<T> {
  fn drop(&mut self) {
    let elem_size = mem::size_of::<T>();
    let align = mem::align_of::<T>();

    if self.cap != 0 && elem_size != 0 {
      unsafe {
        let nn_ptr: NonNull<T> = self.ptr.into();
        Global.dealloc(nn_ptr.cast(),
                       Layout::from_size_align_unchecked(self.cap * elem_size, align));
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
    assert!(pos <= self.len, "out of bound");
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
    if self.len == 0 {
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
        _marker: PhantomData
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

  fn ne(&self, other: &Self) -> bool {
    !self.eq(other)
  }
}

impl<T> Clone for HybridVec<T> {
  fn clone(&self) -> Self {
    let hv_copy = Self::with_capacity(self.len);
    unsafe {
      let cap = self.data_buffer.cap;
      ptr::copy_nonoverlapping(self.data_buffer.ptr.as_ptr(),
                               hv_copy.data_buffer.ptr.as_ptr(), cap);
      ptr::copy_nonoverlapping(self.index_buffer.ptr.as_ptr(),
                               hv_copy.index_buffer.ptr.as_ptr(), cap);
    }
    hv_copy
  }

  fn clone_from(&mut self, source: &Self) {
    let source_cap = source.data_buffer.cap;
    let elem_size = mem::size_of::<T>();
    let align = mem::align_of::<T>();
    let u16size = mem::size_of::<u16>();
    let u16align = mem::align_of::<u16>();

    unsafe {
      if self.data_buffer.cap < source_cap {
        let new_data_ptr = Global.realloc(
          NonNull::<T>::new_unchecked(self.data_buffer.ptr.as_ptr()).cast(),
          Layout::from_size_align_unchecked(self.data_buffer.cap * elem_size, align),
          source_cap * elem_size);

        let new_index_ptr = Global.realloc(
          NonNull::<u16>::new_unchecked(self.index_buffer.ptr.as_ptr()).cast(),
          Layout::from_size_align_unchecked(self.index_buffer.cap * u16size, u16align),
          source_cap * u16size);

        if new_data_ptr.is_err() {
          handle_alloc_error(Layout::from_size_align_unchecked(source_cap * elem_size, align));
        }

        if new_index_ptr.is_err() {
          handle_alloc_error(Layout::from_size_align_unchecked(source_cap * u16size, u16align));
        }

        self.data_buffer.ptr = Unique::new_unchecked(new_data_ptr.unwrap().as_ptr().cast());
        self.index_buffer.ptr = Unique::new_unchecked(new_index_ptr.unwrap().as_ptr().cast());
        self.data_buffer.cap = source_cap;
        self.index_buffer.cap = source_cap;

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
    let elem_index = self.index_buffer.read(index) as usize;
    self.data_buffer.get_ref(elem_index)
  }
}

impl<T> IndexMut<usize> for HybridVec<T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
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
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn create() {
    let empty_hv = HybridVec::<i32>::new();
    assert!(empty_hv.empty());
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
  fn toy() {
    let ve = vec![1, 2, 3];
    let mut it = ve.iter().rev();
    let elem = it.next();
  }
}
