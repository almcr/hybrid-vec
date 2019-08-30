use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;
use std::time::{Instant, Duration};
use hybrid_vec::HybridVec;
use std::iter;

trait Insertable<T> {
  fn insert(&mut self, pos: usize, elem: T);

  fn clear(&mut self);
}

impl<T> Insertable<T> for HybridVec<T> {
  fn insert(&mut self, pos: usize, elem: T) {
    self.insert(pos, elem);
  }

  fn clear(&mut self) {
    self.clear();
  }
}

impl<T> Insertable<T> for Vec<T> {
  fn insert(&mut self, pos: usize, elem: T) {
    self.insert(pos, elem);
  }

  fn clear(&mut self) {
    self.clear();
  }
}

fn insert_bench<T: Default, I: Insertable<T>>(hv: &mut I, len: usize, iter_n: usize) {
  let mut rng = SmallRng::seed_from_u64(32);
  // generate random indexes
  let mut c = 0;
  let random_indexes: Vec<usize> = iter::repeat(()).take(len).map(|()| {
    c += 1;
    rng.gen_range(0, c)
  }).collect();

  let mut elapsed = Duration::default();

  for _ in 0..iter_n {
    let timing = Instant::now();
    for i in 0..len {
      hv.insert(random_indexes[i], T::default());
    }
    elapsed += timing.elapsed();
    hv.clear();
  }

  println!("{} iterations done, {:?}", iter_n, elapsed);
}

#[derive(Default)]
struct NonSmallData {
  _chunk: [f64; 32],
}

fn main() {
  let mut vec_of_nonsmall_data = Vec::<NonSmallData>::with_capacity(300);
  let mut hv_of_nonsmall_data = HybridVec::<NonSmallData>::with_capacity(300);

  insert_bench(&mut vec_of_nonsmall_data, 300, 10000);
  insert_bench(&mut hv_of_nonsmall_data, 300, 10000);
}