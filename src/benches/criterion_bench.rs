use std::iter;
use std::cell::UnsafeCell;
use hybrid_vec::HybridVec;
use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;

#[macro_use]
extern crate criterion;

use criterion::{Criterion, BenchmarkId, BatchSize};
use std::borrow::Borrow;

trait Insertable<T> {
  fn insert(&mut self, pos: usize, elem: T);

  fn clear(&mut self);
}

impl<T> Insertable<T> for HybridVec<T> {
  fn insert(&mut self, pos: usize, elem: T) {
    self.insert(pos, elem);
  }

  fn clear(&mut self) { self.clear(); }
}

impl<T> Insertable<T> for Vec<T> {
  fn insert(&mut self, pos: usize, elem: T) {
    self.insert(pos, elem);
  }

  fn clear(&mut self) { self.clear(); }
}

#[derive(Default)]
struct NonSmallData {
  _chunk: [f64; 32],
}

fn setup_routine<'a, T, I: Insertable<T>>(size: usize, insertable: &'a UnsafeCell<I>) -> impl FnMut() -> Vec<usize> + 'a {
  move || {
    let mut rng = SmallRng::seed_from_u64(32);
    unsafe { (*insertable.get()).clear(); }
    // generate random indexes
    let mut k = 0;
    iter::repeat(()).take(size).map(|()| {
      k += 1;
      rng.gen_range(0, k)
    }).collect()
  }
}

fn bench_routine<'a, T: Default, I: Insertable<T>>(insertable: &'a UnsafeCell<I>) -> impl FnMut(Vec<usize>) + 'a {
  move |indexes| {
    for i in indexes {
      unsafe { (*insertable.get()).insert(i, T::default()); }
    }
  }
}

fn criterion_benchmark(c: &mut Criterion) {
  let size = 200;
  let hvector = UnsafeCell::new(HybridVec::<NonSmallData>::with_capacity(size));
  let vector = UnsafeCell::new(Vec::<NonSmallData>::with_capacity(size));

  c.bench_with_input(BenchmarkId::new("hybrid_vec bench with 200 size", size),
                     &size, |b, &input_size_boxed| {
      b.iter_batched(setup_routine(input_size_boxed, &hvector),
                     bench_routine(&hvector),
                     BatchSize::NumIterations(input_size_boxed as u64));
    });

  c.bench_with_input(BenchmarkId::new("vector bench with 200 size", size),
                     &size, |b, &input_size_boxed| {
      b.iter_batched(setup_routine(size, &vector),
                     bench_routine(&vector),
                     BatchSize::NumIterations(input_size_boxed as u64));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
