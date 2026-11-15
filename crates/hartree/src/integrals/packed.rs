//! In-core ERIs stored once per symmetry-unique integral, and their J/K contraction.

use rayon::prelude::*;

/// The `nao⁴/8` symmetry-unique ERIs as the lower triangle over AO pairs, in the
/// shell-pair-blocked pair order of [`integral::PackedEriBuilder`].
pub(crate) struct PackedEri {
    n: usize,
    pairs: Vec<(usize, usize)>,
    /// `row_span[t]`: one past the largest AO index appearing in pairs `0..=t`.
    row_span: Vec<usize>,
    data: Vec<f64>,
}

/// Number of row chunks a contraction is split into, fixed so that results are
/// reproducible across runs and thread counts.
const CONTRACTION_CHUNKS: usize = 64;

struct Accumulator {
    coulomb: Vec<Vec<f64>>,
    exchange: Vec<Vec<f64>>,
    exchange_transposed: Vec<Vec<f64>>,
    square: Vec<f64>,
}

impl PackedEri {
    pub(crate) fn build(basis: &integral::Basis, tau: f64) -> Self {
        let builder = basis.packed_eri_builder();
        let mut data = vec![0.0; builder.output_len()];
        builder.partition(&mut data).into_par_iter().for_each_init(
            integral::PackedEriWorkspace::new,
            |workspace, mut task| {
                builder.fill(workspace, &mut task, tau);
            },
        );
        let pairs = builder.ao_pairs().to_vec();
        let row_span = pairs
            .iter()
            .scan(0, |span, &(p, _)| {
                *span = (*span).max(p + 1);
                Some(*span)
            })
            .collect();
        Self {
            n: basis.nao(),
            pairs,
            row_span,
            data,
        }
    }

    /// `J[D]` and (when `want_k`) `K[D]` for each row-major `nao²` density.
    /// Densities need not be symmetric.
    pub(crate) fn contract(
        &self,
        densities: &[Vec<f64>],
        want_k: bool,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.n;
        let m = self.pairs.len();
        let nd = densities.len();
        let folded: Vec<Vec<f64>> = densities
            .iter()
            .map(|d| {
                self.pairs
                    .iter()
                    .map(|&(p, q)| {
                        if p == q {
                            d[p * n + p]
                        } else {
                            d[p * n + q] + d[q * n + p]
                        }
                    })
                    .collect()
            })
            .collect();
        // K = A[D] + A[Dᵀ]ᵀ over the stored half; a symmetric D needs A only once.
        let transposed: Vec<Option<Vec<f64>>> = densities
            .iter()
            .map(|d| (want_k && !is_symmetric(d, n)).then(|| transpose(d, n)))
            .collect();

        // Rows are summed in fixed chunks and the chunks combined in order, so the
        // result does not depend on how rayon schedules the work. Chunk bounds
        // follow √c because a row's cost grows with its index.
        let chunks = m.clamp(1, CONTRACTION_CHUNKS);
        let bound = |c: usize| (m as f64 * (c as f64 / chunks as f64).sqrt()).round() as usize;
        let partials: Vec<Accumulator> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let mut acc = Accumulator {
                    coulomb: vec![vec![0.0; m]; nd],
                    exchange: vec![vec![0.0; if want_k { n * n } else { 0 }]; nd],
                    exchange_transposed: transposed
                        .iter()
                        .map(|t| vec![0.0; if t.is_some() { n * n } else { 0 }])
                        .collect(),
                    square: vec![0.0; if want_k { n * n } else { 0 }],
                };
                for t in bound(c)..bound(c + 1) {
                    let row = &self.data[t * (t + 1) / 2..][..t + 1];
                    for (jp, dp) in acc.coulomb.iter_mut().zip(&folded) {
                        jp[t] += dot(&row[..t], &dp[..t]) + row[t] * dp[t];
                        let weight = dp[t];
                        for (slot, g) in jp[..t].iter_mut().zip(row) {
                            *slot += g * weight;
                        }
                    }
                    if want_k {
                        self.exchange_row(t, row, densities, &transposed, &mut acc);
                    }
                }
                acc
            })
            .collect();
        let mut partials = partials.into_iter();
        let mut total = partials.next().expect("at least one chunk");
        for part in partials {
            for (x, y) in total.coulomb.iter_mut().zip(&part.coulomb) {
                add_assign(x, y);
            }
            for (x, y) in total.exchange.iter_mut().zip(&part.exchange) {
                add_assign(x, y);
            }
            for (x, y) in total
                .exchange_transposed
                .iter_mut()
                .zip(&part.exchange_transposed)
            {
                add_assign(x, y);
            }
        }

        let coulomb = total
            .coulomb
            .iter()
            .map(|jp| {
                let mut j = vec![0.0; n * n];
                for (&(p, q), &v) in self.pairs.iter().zip(jp) {
                    j[p * n + q] = v;
                    j[q * n + p] = v;
                }
                j
            })
            .collect();
        let exchange = if want_k {
            total
                .exchange
                .iter()
                .zip(&total.exchange_transposed)
                .zip(&transposed)
                .map(|((a, at), t)| {
                    let other = if t.is_some() { at } else { a };
                    let mut k = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            k[i * n + j] = a[i * n + j] + other[j * n + i];
                        }
                    }
                    k
                })
                .collect()
        } else {
            vec![Vec::new(); nd]
        };
        (coulomb, exchange)
    }

    fn exchange_row(
        &self,
        t: usize,
        row: &[f64],
        densities: &[Vec<f64>],
        transposed: &[Option<Vec<f64>>],
        acc: &mut Accumulator,
    ) {
        let n = self.n;
        let (p, q) = self.pairs[t];
        let span = self.row_span[t];
        // Rows of one chunk arrive in increasing order, so every entry of `square`
        // left over from the previous row is overwritten below.
        let g = &mut acc.square;
        for (&(r, s), &v) in self.pairs.iter().zip(&row[..t]) {
            g[r * n + s] = v;
            g[s * n + r] = v;
        }
        // The pair-diagonal integral belongs to both halves of K = A + Aᵀ.
        g[p * n + q] = 0.5 * row[t];
        g[q * n + p] = 0.5 * row[t];

        let apply = |d: &[f64], a: &mut [f64]| {
            for r in 0..span {
                let g_row = &g[r * n..][..span];
                a[p * n + r] += dot(g_row, &d[q * n..][..span]);
                if p != q {
                    a[q * n + r] += dot(g_row, &d[p * n..][..span]);
                }
            }
        };
        for (di, d) in densities.iter().enumerate() {
            apply(d, &mut acc.exchange[di]);
            if let Some(dt) = &transposed[di] {
                apply(dt, &mut acc.exchange_transposed[di]);
            }
        }
    }

    /// The dense row-major `nao⁴` tensor.
    pub(crate) fn expand(&self) -> Vec<f64> {
        let n = self.n;
        let mut index = vec![0usize; n * n];
        for (t, &(p, q)) in self.pairs.iter().enumerate() {
            index[p * n + q] = t;
            index[q * n + p] = t;
        }
        let mut out = vec![0.0; n * n * n * n];
        out.par_chunks_mut(n * n)
            .zip(index.par_iter())
            .for_each(|(slab, &t)| {
                for (slot, &u) in slab.iter_mut().zip(&index) {
                    let (hi, lo) = if t >= u { (t, u) } else { (u, t) };
                    *slot = self.data[hi * (hi + 1) / 2 + lo];
                }
            });
        out
    }
}

/// Dot product over eight independent partial sums: a plain `sum()` is a single
/// dependency chain the compiler may not reorder, which blocks vectorization.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0; 8];
    let (a_chunks, b_chunks) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f64 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in a_chunks.zip(b_chunks) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    lanes.iter().sum::<f64>() + tail
}

fn add_assign(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Symmetric to rounding: SCF densities come out of matrix products whose two
/// triangles differ in the last bits.
fn is_symmetric(d: &[f64], n: usize) -> bool {
    let scale = d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = 1e-14 * scale;
    (0..n).all(|i| (0..i).all(|j| (d[i * n + j] - d[j * n + i]).abs() <= tol))
}

fn transpose(d: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = d[i * n + j];
        }
    }
    t
}
