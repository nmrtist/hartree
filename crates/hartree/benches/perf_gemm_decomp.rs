use std::time::{Duration, Instant};

use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use hartree::linalg::{Mat, gemm};
use hartree::tensor::{Tensor, tensordot};

fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn med_min(reps: usize, mut f: impl FnMut()) -> (Duration, Duration) {
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        ts.push(t0.elapsed());
    }
    ts.sort();
    (ts[ts.len() / 2], ts[0])
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn decompose(label: &str, m: usize, k: usize, n: usize, reps: usize) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);

    let am = Mat::from_fn(m, k, |i, j| a[i * k + j]);
    let bm = Mat::from_fn(k, n, |i, j| b[i * n + j]);
    let mut cm = Mat::zeros(m, n);
    let (l0, _) = med_min(reps, || {
        matmul(
            cm.as_mut(),
            Accum::Replace,
            am.as_ref(),
            bm.as_ref(),
            1.0,
            Par::Seq,
        );
        std::hint::black_box(&cm);
    });
    let threads = rayon::current_num_threads();
    let par = if threads > 1 {
        Par::rayon(threads)
    } else {
        Par::Seq
    };
    let (l0p, _) = med_min(reps, || {
        matmul(
            cm.as_mut(),
            Accum::Replace,
            am.as_ref(),
            bm.as_ref(),
            1.0,
            par,
        );
        std::hint::black_box(&cm);
    });

    let (l1, _) = med_min(reps, || {
        std::hint::black_box(gemm(&a, m, k, &b, n));
    });

    println!(
        "{label:<28} m={m:>5} k={k:>5} n={n:>5} | L0(faer-seq) {:>8.3}  L0p(faer-par×{threads}) {:>8.3}  L1(gemm+copies) {:>8.3} ms",
        ms(l0),
        ms(l0p),
        ms(l1)
    );
    println!(
        "{:<28} copy overhead (L1−L0)/L1 = {:>5.1}%   parallel headroom L0/L0p = {:>5.2}×",
        "",
        100.0 * (ms(l1) - ms(l0)) / ms(l1).max(1e-9),
        ms(l0) / ms(l0p).max(1e-9),
    );
}

fn ladder_tensordot(label: &str, o: usize, v: usize, reps: usize) {
    let wvvvv = Tensor::new(vec![v, v, v, v], fill(v * v * v * v, 3));
    let tau = Tensor::new(vec![o, o, v, v], fill(o * o * v * v, 4));
    let (l2, _) = med_min(reps, || {
        let r = tensordot(&wvvvv, &[2, 3], &tau, &[2, 3]); // [v,v,o,o]
        let r = r.permute(&[2, 3, 0, 1]); // -> [o,o,v,v]  (ijab)
        std::hint::black_box(r);
    });

    let (m, k, n) = (v * v, v * v, o * o);
    let am = Mat::from_fn(m, k, |i, j| ((i * 7 + j) as f64).sin());
    let bm = Mat::from_fn(k, n, |i, j| ((i * 3 + j) as f64).cos());
    let mut cm = Mat::zeros(m, n);
    let (l0, _) = med_min(reps, || {
        matmul(
            cm.as_mut(),
            Accum::Replace,
            am.as_ref(),
            bm.as_ref(),
            1.0,
            Par::Seq,
        );
        std::hint::black_box(&cm);
    });
    println!(
        "{label:<28} o={o} v={v}  Wvvvv·τ | L2(tensordot full) {:>8.3} ms  vs L0(matmul floor [{m}×{k}]·[{k}×{n}]) {:>8.3} ms  → wrapping overhead {:>5.1}%",
        ms(l2),
        ms(l0),
        100.0 * (ms(l2) - ms(l0)) / ms(l2).max(1e-9),
    );
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("== GEMM/tensordot cost decomposition (rayon threads = {threads}) ==");
    println!("-- CCSD particle-particle ladder GEMM shape [v²,v²]·[v²,o²] --");
    decompose("water/cc-pvdz ladder", 19 * 19, 19 * 19, 4 * 4, 30);
    decompose("ethylene/cc-pvdz ladder", 40 * 40, 40 * 40, 6 * 6, 10);
    println!("-- representative AO→MO transform quarter GEMMs (ethylene, n=48) --");
    decompose("transform q1 [v,n]·[n,n³]", 40, 48, 48 * 48 * 48, 3);
    println!("-- full tensordot path vs matmul floor --");
    ladder_tensordot("water/cc-pvdz", 4, 19, 30);
    ladder_tensordot("ethylene/cc-pvdz", 6, 40, 10);
    println!("== done ==");
}
