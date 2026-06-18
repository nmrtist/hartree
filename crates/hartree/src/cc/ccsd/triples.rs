use crate::integrals::InCoreEri;
use crate::scf::ScfResult;
use rayon::prelude::*;

use super::spin_adapted::{CcsdState, rccsd_spin_adapted_state};
use super::{CcsdOptions, CcsdTResult};

pub fn rccsd_t_spin_adapted<P: InCoreEri>(
    provider: &P,
    scf: &ScfResult,
    n_frozen: usize,
    options: &CcsdOptions,
) -> CcsdTResult {
    let state = rccsd_spin_adapted_state(provider, scf, n_frozen, options);
    let triples_energy = triples_energy(&state);
    CcsdTResult {
        ccsd: state.result,
        triples_energy,
        total_energy: state.result.total_energy + triples_energy,
    }
}

fn triples_energy(st: &CcsdState) -> f64 {
    let (o, v) = (st.o, st.v);
    let t1 = st.t1.as_slice(); // [o, v]
    let t2 = st.t2.as_slice(); // [o, o, v, v]
    let ovvv = st.ovvv.data(); // (ia|bc)  [o, v, v, v]
    let ovoo = st.ovoo.data(); // (ia|jk)  [o, v, o, o]
    let ovov = st.ovov.data(); // (ia|jb)  [o, v, o, v]

    let mut eris_vvov = vec![0.0; v * v * o * v];
    let mut eris_vooo = vec![0.0; v * o * o * o];
    let mut eris_vvoo = vec![0.0; v * v * o * o];
    let mut t2t = vec![0.0; v * v * o * o];
    let mut t1t = vec![0.0; v * o];
    for a in 0..v {
        for i in 0..o {
            t1t[a * o + i] = t1[i * v + a];
        }
    }
    for a in 0..v {
        for b in 0..v {
            for i in 0..o {
                for f in 0..v {
                    eris_vvov[((a * v + b) * o + i) * v + f] = ovvv[((i * v + a) * v + f) * v + b];
                }
                for j in 0..o {
                    eris_vvoo[((a * v + b) * o + i) * o + j] = ovov[((i * v + a) * o + j) * v + b];
                    t2t[((a * v + b) * o + i) * o + j] = t2[((i * o + j) * v + a) * v + b];
                }
            }
        }
    }
    for a in 0..v {
        for i in 0..o {
            for j in 0..o {
                for m in 0..o {
                    eris_vooo[((a * o + i) * o + j) * o + m] = ovoo[((i * v + a) * o + j) * o + m];
                }
            }
        }
    }

    let blocks = Blocks {
        o,
        v,
        eo: &st.eo,
        ev: &st.ev,
        eris_vvov: &eris_vvov,
        eris_vooo: &eris_vooo,
        eris_vvoo: &eris_vvoo,
        t2t: &t2t,
        t1t: &t1t,
    };

    let triples: Vec<(usize, usize, usize)> = (0..v)
        .flat_map(|a| (0..=a).flat_map(move |b| (0..=b).map(move |c| (a, b, c))))
        .collect();

    let et: f64 = triples
        .par_iter()
        .map(|&(a, b, c)| triple_contrib(&blocks, a, b, c))
        .sum();
    2.0 * et
}

struct Blocks<'a> {
    o: usize,
    v: usize,
    eo: &'a [f64],
    ev: &'a [f64],
    eris_vvov: &'a [f64], // [v,v,o,v]
    eris_vooo: &'a [f64], // [v,o,o,o]
    eris_vvoo: &'a [f64], // [v,v,o,o]
    t2t: &'a [f64],       // [v,v,o,o]
    t1t: &'a [f64],       // [v,o]
}

fn get_w(b: &Blocks, a: usize, bb: usize, c: usize) -> Vec<f64> {
    let (o, v) = (b.o, b.v);
    let mut w = vec![0.0; o * o * o];
    for i in 0..o {
        for j in 0..o {
            for k in 0..o {
                let mut s = 0.0;
                for f in 0..v {
                    s += b.eris_vvov[((a * v + bb) * o + i) * v + f]
                        * b.t2t[((c * v + f) * o + k) * o + j];
                }
                for m in 0..o {
                    s -= b.eris_vooo[((a * o + i) * o + j) * o + m]
                        * b.t2t[((bb * v + c) * o + m) * o + k];
                }
                w[(i * o + j) * o + k] = s;
            }
        }
    }
    w
}

fn get_v(b: &Blocks, a: usize, bb: usize, c: usize) -> Vec<f64> {
    let o = b.o;
    let mut vv = vec![0.0; o * o * o];
    for i in 0..o {
        for j in 0..o {
            let g = b.eris_vvoo[((a * b.v + bb) * o + i) * o + j];
            for k in 0..o {
                vv[(i * o + j) * o + k] = g * b.t1t[c * o + k];
            }
        }
    }
    vv
}

fn r3(w: &[f64], o: usize) -> Vec<f64> {
    let g = |i: usize, j: usize, k: usize| w[(i * o + j) * o + k];
    let mut r = vec![0.0; o * o * o];
    for i in 0..o {
        for j in 0..o {
            for k in 0..o {
                r[(i * o + j) * o + k] = 4.0 * g(i, j, k) + g(k, i, j) + g(j, k, i)
                    - 2.0 * g(k, j, i)
                    - 2.0 * g(i, k, j)
                    - 2.0 * g(j, i, k);
            }
        }
    }
    r
}

fn dot_perm(w: &[f64], z: &[f64], o: usize, sel: [usize; 3]) -> f64 {
    let mut s = 0.0;
    for v0 in 0..o {
        for v1 in 0..o {
            for v2 in 0..o {
                let c = [v0, v1, v2];
                let wi = (c[sel[0]] * o + c[sel[1]]) * o + c[sel[2]];
                s += w[wi] * z[(v0 * o + v1) * o + v2];
            }
        }
    }
    s
}

fn triple_contrib(blk: &Blocks, a: usize, b: usize, c: usize) -> f64 {
    let o = blk.o;

    let wabc = get_w(blk, a, b, c);
    let wacb = get_w(blk, a, c, b);
    let wbac = get_w(blk, b, a, c);
    let wbca = get_w(blk, b, c, a);
    let wcab = get_w(blk, c, a, b);
    let wcba = get_w(blk, c, b, a);
    let vabc = get_v(blk, a, b, c);
    let vacb = get_v(blk, a, c, b);
    let vbac = get_v(blk, b, a, c);
    let vbca = get_v(blk, b, c, a);
    let vcab = get_v(blk, c, a, b);
    let vcba = get_v(blk, c, b, a);

    let degen = if a == c {
        6.0
    } else if a == b || b == c {
        2.0
    } else {
        1.0
    };
    let evsum = blk.ev[a] + blk.ev[b] + blk.ev[c];
    let mut d3 = vec![0.0; o * o * o];
    for i in 0..o {
        for j in 0..o {
            for k in 0..o {
                d3[(i * o + j) * o + k] = (blk.eo[i] + blk.eo[j] + blk.eo[k] - evsum) * degen;
            }
        }
    }

    let z = |w: &[f64], vv: &[f64]| -> Vec<f64> {
        let mut comb = w.to_vec();
        for (cval, &vval) in comb.iter_mut().zip(vv) {
            *cval += 0.5 * vval;
        }
        let mut r = r3(&comb, o);
        for (rv, &d) in r.iter_mut().zip(&d3) {
            *rv /= d;
        }
        r
    };
    let zabc = z(&wabc, &vabc);
    let zacb = z(&wacb, &vacb);
    let zbac = z(&wbac, &vbac);
    let zbca = z(&wbca, &vbca);
    let zcab = z(&wcab, &vcab);
    let zcba = z(&wcba, &vcba);

    let dp = |w: &[f64], zt: &[f64], sel: [usize; 3]| dot_perm(w, zt, o, sel);

    let mut et = 0.0;
    et += dp(&wabc, &zabc, [0, 1, 2])
        + dp(&wacb, &zabc, [0, 2, 1])
        + dp(&wbac, &zabc, [1, 0, 2])
        + dp(&wbca, &zabc, [1, 2, 0])
        + dp(&wcab, &zabc, [2, 0, 1])
        + dp(&wcba, &zabc, [2, 1, 0]);
    et += dp(&wacb, &zacb, [0, 1, 2])
        + dp(&wabc, &zacb, [0, 2, 1])
        + dp(&wcab, &zacb, [1, 0, 2])
        + dp(&wcba, &zacb, [1, 2, 0])
        + dp(&wbac, &zacb, [2, 0, 1])
        + dp(&wbca, &zacb, [2, 1, 0]);
    et += dp(&wbac, &zbac, [0, 1, 2])
        + dp(&wbca, &zbac, [0, 2, 1])
        + dp(&wabc, &zbac, [1, 0, 2])
        + dp(&wacb, &zbac, [1, 2, 0])
        + dp(&wcba, &zbac, [2, 0, 1])
        + dp(&wcab, &zbac, [2, 1, 0]);
    et += dp(&wbca, &zbca, [0, 1, 2])
        + dp(&wbac, &zbca, [0, 2, 1])
        + dp(&wcba, &zbca, [1, 0, 2])
        + dp(&wcab, &zbca, [1, 2, 0])
        + dp(&wabc, &zbca, [2, 0, 1])
        + dp(&wacb, &zbca, [2, 1, 0]);
    et += dp(&wcab, &zcab, [0, 1, 2])
        + dp(&wcba, &zcab, [0, 2, 1])
        + dp(&wacb, &zcab, [1, 0, 2])
        + dp(&wabc, &zcab, [1, 2, 0])
        + dp(&wbca, &zcab, [2, 0, 1])
        + dp(&wbac, &zcab, [2, 1, 0]);
    et += dp(&wcba, &zcba, [0, 1, 2])
        + dp(&wcab, &zcba, [0, 2, 1])
        + dp(&wbca, &zcba, [1, 0, 2])
        + dp(&wbac, &zcba, [1, 2, 0])
        + dp(&wacb, &zcba, [2, 0, 1])
        + dp(&wabc, &zcba, [2, 1, 0]);
    et
}
