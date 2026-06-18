use crate::integrals::InCoreEri;
use crate::scf::ScfResult;

use super::diis::AmplitudeDiis;
use super::{CcsdOptions, CcsdResult};
use crate::cc::transform::{column_block, transform_block};

pub fn rccsd_spin_orbital<P: InCoreEri>(
    provider: &P,
    scf: &ScfResult,
    n_frozen: usize,
    options: &CcsdOptions,
) -> CcsdResult {
    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let n_occ = scf.n_alpha; // doubly occupied (RHF)
    assert!(n_frozen <= n_occ, "more frozen orbitals than occupied");

    let nmo = m - n_frozen; // active spatial orbitals
    let n_occ_act = n_occ - n_frozen; // active occupied spatial
    let eps = &scf.orbital_energies_alpha;
    let c = &scf.mo_coeff_alpha;

    let c_act = column_block(c, n, m, n_frozen, nmo);
    let mo = transform_block(provider.ao_eri(), n, [&c_act, &c_act, &c_act, &c_act]).into_data();

    let nso = 2 * nmo;
    let no = 2 * n_occ_act; // occupied spin orbitals (low indices)
    let nv = nso - no; // virtual spin orbitals
    let eri = spin_orbital_eri(&mo, nmo);
    let fs: Vec<f64> = (0..nso).map(|p| eps[n_frozen + p / 2]).collect();

    let g = |p: usize, q: usize, r: usize, s: usize| eri[((p * nso + q) * nso + r) * nso + s];

    let d_ia = |i: usize, a: usize| fs[i] - fs[no + a];
    let d_ijab = |i: usize, j: usize, a: usize, b: usize| fs[i] + fs[j] - fs[no + a] - fs[no + b];

    let t1_idx = |i: usize, a: usize| i * nv + a;
    let t2_idx = |i: usize, j: usize, a: usize, b: usize| ((i * no + j) * nv + a) * nv + b;

    let mut t1 = vec![0.0; no * nv];
    let mut t2 = vec![0.0; no * no * nv * nv];

    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    t2[t2_idx(i, j, a, b)] = g(i, j, no + a, no + b) / d_ijab(i, j, a, b);
                }
            }
        }
    }

    let energy = |t1: &[f64], t2: &[f64]| -> f64 {
        let mut e = 0.0;
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let ijab = g(i, j, no + a, no + b);
                        e += 0.25 * ijab * t2[t2_idx(i, j, a, b)]
                            + 0.5 * ijab * t1[t1_idx(i, a)] * t1[t1_idx(j, b)];
                    }
                }
            }
        }
        e
    };

    let mp2_correlation = energy(&t1, &t2);

    let mut diis = AmplitudeDiis::new(options.diis_dim);
    let mut e_prev = mp2_correlation;
    let mut correlation_energy = mp2_correlation;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=options.max_iter {
        iterations = iter;

        let mut tau = vec![0.0; t2.len()];
        let mut taut = vec![0.0; t2.len()];
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let cross = t1[t1_idx(i, a)] * t1[t1_idx(j, b)]
                            - t1[t1_idx(i, b)] * t1[t1_idx(j, a)];
                        let d = t2[t2_idx(i, j, a, b)];
                        tau[t2_idx(i, j, a, b)] = d + cross;
                        taut[t2_idx(i, j, a, b)] = d + 0.5 * cross;
                    }
                }
            }
        }

        let mut fae = vec![0.0; nv * nv];
        for a in 0..nv {
            for e in 0..nv {
                let mut v = 0.0;
                for mm in 0..no {
                    for f in 0..nv {
                        v += t1[t1_idx(mm, f)] * g(mm, no + a, no + f, no + e);
                    }
                }
                for mm in 0..no {
                    for nn in 0..no {
                        for f in 0..nv {
                            v -= 0.5 * taut[t2_idx(mm, nn, a, f)] * g(mm, nn, no + e, no + f);
                        }
                    }
                }
                fae[a * nv + e] = v;
            }
        }

        let mut fmi = vec![0.0; no * no];
        for mm in 0..no {
            for i in 0..no {
                let mut v = 0.0;
                for nn in 0..no {
                    for e in 0..nv {
                        v += t1[t1_idx(nn, e)] * g(mm, nn, i, no + e);
                    }
                }
                for nn in 0..no {
                    for e in 0..nv {
                        for f in 0..nv {
                            v += 0.5 * taut[t2_idx(i, nn, e, f)] * g(mm, nn, no + e, no + f);
                        }
                    }
                }
                fmi[mm * no + i] = v;
            }
        }

        let mut fme = vec![0.0; no * nv];
        for mm in 0..no {
            for e in 0..nv {
                let mut v = 0.0;
                for nn in 0..no {
                    for f in 0..nv {
                        v += t1[t1_idx(nn, f)] * g(mm, nn, no + e, no + f);
                    }
                }
                fme[mm * nv + e] = v;
            }
        }

        let w_mnij_idx = |m: usize, n: usize, i: usize, j: usize| ((m * no + n) * no + i) * no + j;
        let mut w_mnij = vec![0.0; no * no * no * no];
        for mm in 0..no {
            for nn in 0..no {
                for i in 0..no {
                    for j in 0..no {
                        let mut v = g(mm, nn, i, j);
                        for e in 0..nv {
                            v += t1[t1_idx(j, e)] * g(mm, nn, i, no + e)
                                - t1[t1_idx(i, e)] * g(mm, nn, j, no + e);
                        }
                        for e in 0..nv {
                            for f in 0..nv {
                                v += 0.25 * tau[t2_idx(i, j, e, f)] * g(mm, nn, no + e, no + f);
                            }
                        }
                        w_mnij[w_mnij_idx(mm, nn, i, j)] = v;
                    }
                }
            }
        }

        let w_abef_idx = |a: usize, b: usize, e: usize, f: usize| ((a * nv + b) * nv + e) * nv + f;
        let mut w_abef = vec![0.0; nv * nv * nv * nv];
        for a in 0..nv {
            for b in 0..nv {
                for e in 0..nv {
                    for f in 0..nv {
                        let mut v = g(no + a, no + b, no + e, no + f);
                        for mm in 0..no {
                            v -= t1[t1_idx(mm, b)] * g(no + a, mm, no + e, no + f)
                                - t1[t1_idx(mm, a)] * g(no + b, mm, no + e, no + f);
                        }
                        for mm in 0..no {
                            for nn in 0..no {
                                v += 0.25 * tau[t2_idx(mm, nn, a, b)] * g(mm, nn, no + e, no + f);
                            }
                        }
                        w_abef[w_abef_idx(a, b, e, f)] = v;
                    }
                }
            }
        }

        let w_mbej_idx = |m: usize, b: usize, e: usize, j: usize| ((m * nv + b) * nv + e) * no + j;
        let mut w_mbej = vec![0.0; no * nv * nv * no];
        for mm in 0..no {
            for b in 0..nv {
                for e in 0..nv {
                    for j in 0..no {
                        let mut v = g(mm, no + b, no + e, j);
                        for f in 0..nv {
                            v += t1[t1_idx(j, f)] * g(mm, no + b, no + e, no + f);
                        }
                        for nn in 0..no {
                            v -= t1[t1_idx(nn, b)] * g(mm, nn, no + e, j);
                        }
                        for nn in 0..no {
                            for f in 0..nv {
                                v -= (0.5 * t2[t2_idx(j, nn, f, b)]
                                    + t1[t1_idx(j, f)] * t1[t1_idx(nn, b)])
                                    * g(mm, nn, no + e, no + f);
                            }
                        }
                        w_mbej[w_mbej_idx(mm, b, e, j)] = v;
                    }
                }
            }
        }

        let mut t1_new = vec![0.0; t1.len()];
        for i in 0..no {
            for a in 0..nv {
                let mut v = 0.0; // f_ia = 0
                for e in 0..nv {
                    v += t1[t1_idx(i, e)] * fae[a * nv + e];
                }
                for mm in 0..no {
                    v -= t1[t1_idx(mm, a)] * fmi[mm * no + i];
                }
                for mm in 0..no {
                    for e in 0..nv {
                        v += t2[t2_idx(i, mm, a, e)] * fme[mm * nv + e];
                    }
                }
                for nn in 0..no {
                    for f in 0..nv {
                        v -= t1[t1_idx(nn, f)] * g(nn, no + a, i, no + f);
                    }
                }
                for mm in 0..no {
                    for e in 0..nv {
                        for f in 0..nv {
                            v -= 0.5 * t2[t2_idx(i, mm, e, f)] * g(mm, no + a, no + e, no + f);
                        }
                    }
                }
                for mm in 0..no {
                    for nn in 0..no {
                        for e in 0..nv {
                            v -= 0.5 * t2[t2_idx(mm, nn, a, e)] * g(nn, mm, no + e, i);
                        }
                    }
                }
                t1_new[t1_idx(i, a)] = v / d_ia(i, a);
            }
        }

        let mut t2_new = vec![0.0; t2.len()];
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let mut v = g(i, j, no + a, no + b);

                        let ab_term = |a: usize, b: usize| {
                            let mut s = 0.0;
                            for e in 0..nv {
                                let mut fbe = fae[b * nv + e];
                                for mm in 0..no {
                                    fbe -= 0.5 * t1[t1_idx(mm, b)] * fme[mm * nv + e];
                                }
                                s += t2[t2_idx(i, j, a, e)] * fbe;
                            }
                            s
                        };
                        v += ab_term(a, b) - ab_term(b, a);

                        let ij_term = |i: usize, j: usize| {
                            let mut s = 0.0;
                            for mm in 0..no {
                                let mut fmj = fmi[mm * no + j];
                                for e in 0..nv {
                                    fmj += 0.5 * t1[t1_idx(j, e)] * fme[mm * nv + e];
                                }
                                s += t2[t2_idx(i, mm, a, b)] * fmj;
                            }
                            s
                        };
                        v -= ij_term(i, j) - ij_term(j, i);

                        for mm in 0..no {
                            for nn in 0..no {
                                v += 0.5
                                    * tau[t2_idx(mm, nn, a, b)]
                                    * w_mnij[w_mnij_idx(mm, nn, i, j)];
                            }
                        }
                        for e in 0..nv {
                            for f in 0..nv {
                                v += 0.5 * tau[t2_idx(i, j, e, f)] * w_abef[w_abef_idx(a, b, e, f)];
                            }
                        }

                        let ring = |i: usize, j: usize, a: usize, b: usize| {
                            let mut s = 0.0;
                            for mm in 0..no {
                                for e in 0..nv {
                                    s += t2[t2_idx(i, mm, a, e)] * w_mbej[w_mbej_idx(mm, b, e, j)]
                                        - t1[t1_idx(i, e)]
                                            * t1[t1_idx(mm, a)]
                                            * g(mm, no + b, no + e, j);
                                }
                            }
                            s
                        };
                        v += ring(i, j, a, b) - ring(j, i, a, b) - ring(i, j, b, a)
                            + ring(j, i, b, a);

                        for e in 0..nv {
                            v += t1[t1_idx(i, e)] * g(no + a, no + b, no + e, j)
                                - t1[t1_idx(j, e)] * g(no + a, no + b, no + e, i);
                        }
                        for mm in 0..no {
                            v -= t1[t1_idx(mm, a)] * g(mm, no + b, i, j)
                                - t1[t1_idx(mm, b)] * g(mm, no + a, i, j);
                        }

                        t2_new[t2_idx(i, j, a, b)] = v / d_ijab(i, j, a, b);
                    }
                }
            }
        }

        let mut amplitude: Vec<f64> = t1_new.iter().chain(t2_new.iter()).copied().collect();
        let error: Vec<f64> = t1_new
            .iter()
            .zip(&t1)
            .chain(t2_new.iter().zip(&t2))
            .map(|(new, old)| new - old)
            .collect();
        let rms = (error.iter().map(|x| x * x).sum::<f64>() / error.len() as f64).sqrt();

        diis.push(amplitude.clone(), error);
        amplitude = diis.extrapolate();
        let n_t1 = t1.len();
        t1.copy_from_slice(&amplitude[..n_t1]);
        t2.copy_from_slice(&amplitude[n_t1..]);

        correlation_energy = energy(&t1, &t2);
        let de = correlation_energy - e_prev;
        if de.abs() < options.energy_tol && rms < options.amplitude_tol {
            converged = true;
            break;
        }
        e_prev = correlation_energy;
    }

    CcsdResult {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        scf_energy: scf.energy,
        mp2_correlation,
        converged,
        iterations,
        n_frozen,
        t1_diagnostic: super::t1_diagnostic_from(&t1, no),
    }
}

fn spin_orbital_eri(mo: &[f64], nmo: usize) -> Vec<f64> {
    let nso = 2 * nmo;
    let sp = |p: usize, q: usize, r: usize, s: usize| mo[((p * nmo + q) * nmo + r) * nmo + s];
    let mut out = vec![0.0; nso * nso * nso * nso];
    for p in 0..nso {
        for q in 0..nso {
            for r in 0..nso {
                for s in 0..nso {
                    let direct = if p % 2 == r % 2 && q % 2 == s % 2 {
                        sp(p / 2, r / 2, q / 2, s / 2)
                    } else {
                        0.0
                    };
                    let exchange = if p % 2 == s % 2 && q % 2 == r % 2 {
                        sp(p / 2, s / 2, q / 2, r / 2)
                    } else {
                        0.0
                    };
                    out[((p * nso + q) * nso + r) * nso + s] = direct - exchange;
                }
            }
        }
    }
    out
}
