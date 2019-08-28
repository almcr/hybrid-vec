use hybrid_vec::HybridVec;
use std::time::Instant;

fn insert() {

  let mut hv = HybridVec::<i32>::with_capacity(10);

  let time = Instant::now();

  for i in 0..10 {
    hv.insert(0, i);
  }

  println!("{:?}", time.elapsed());
}

fn insert_vec() {
  let mut hv = Vec::<i32>::with_capacity(10);

  let before = Instant::now();

  for i in 0..10 {
    hv.insert(0, i);
  }

  println!("{:?}", before.elapsed());
}

fn main() {
  insert();
  insert_vec();
}