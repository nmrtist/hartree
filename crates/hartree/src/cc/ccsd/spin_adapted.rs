use crate::integrals::InCoreEri;
use crate::scf::ScfResult;
use crate::tensor::{Tensor, tensordot};

use super::diis::AmplitudeDiis;
use super::{CcsdOptions, CcsdResult};
use crate::cc::transform::{column_block, transform_block};

pub(crate) struct CcsdState {
    pub(crate) result: CcsdResult,
    pub(crate) o: usize,
    pub(crate) v: usize,
    pub(crate) t1: Vec<f64>, // [o, v]
    pub(crate) t2: Vec<f64>, // [o, o, v, v]
    pub(crate) eo: Vec<f64>, // active-occupied orbital energies
    pub(crate) ev: Vec<f64>, // virtual orbital energies
    pub(crate) ovvv: Tensor, // (ia|bc)  [o, v, v, v]
    pub(crate) ovoo: Tensor, // (ia|jk)  [o, v, o, o]
    pub(crate) ovov: Tensor, // (ia|jb)  [o, v, o, v]
}

fn einsum(a: &Tensor, al: &str, b: &Tensor, bl: &str, ol: &str) -> Tensor {
    let ac: Vec<char> = al.chars().collect();
    let bc: Vec<char> = bl.chars().collect();
    let oc: Vec<char> = ol.chars().collect();

    let mut a_axes = Vec::new();
    let mut b_axes = Vec::new();
    for (i, ch) in ac.iter().enumerate() {
        if bc.contains(ch) && !oc.contains(ch) {
            a_axes.push(i);
            b_axes.push(bc.iter().position(|x| x == ch).unwrap());
        }
    }

    let res = tensordot(a, &a_axes, b, &b_axes);

    let mut out_labels: Vec<char> = ac
        .iter()
        .enumerate()
        .filter(|(i, _)| !a_axes.contains(i))
        .map(|(_, &c)| c)
        .collect();
    out_labels.extend(
        bc.iter()
            .enumerate()
            .filter(|(i, _)| !b_axes.contains(i))
            .map(|(_, &c)| c),
    );

    let perm: Vec<usize> = oc
        .iter()
        .map(|c| out_labels.iter().position(|x| x == c).unwrap())
        .collect();
    res.permute(&perm)
}

fn axpy(dst: &mut [f64], alpha: f64, src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += alpha * s;
    }
}

fn add_sym(dst: &mut [f64], alpha: f64, tmp: &Tensor) {
    axpy(dst, alpha, tmp.data());
    let p = tmp.permute(&[1, 0, 3, 2]);
    axpy(dst, alpha, p.data());
}

fn ccsd_energy(ovov: &Tensor, o: usize, v: usize, t1: &[f64], t2: &[f64]) -> f64 {
    let t1t = Tensor::new(vec![o, v], t1.to_vec());
    let mut d = t2.to_vec();
    axpy(&mut d, 1.0, einsum(&t1t, "ia", &t1t, "jb", "ijab").data());
    let tau = Tensor::new(vec![o, o, v, v], d);
    let e1 = einsum(ovov, "iajb", &tau, "ijab", "").data()[0];
    let e2 = einsum(ovov, "ibja", &tau, "ijab", "").data()[0];
    2.0 * e1 - e2
}

pub fn rccsd_spin_adapted<P: InCoreEri>(
    provider: &P,
    scf: &ScfResult,
    n_frozen: usize,
    options: &CcsdOptions,
) -> CcsdResult {
    rccsd_spin_adapted_state(provider, scf, n_frozen, options).result
}

pub(crate) fn rccsd_spin_adapted_state<P: InCoreEri>(
    provider: &P,
    scf: &ScfResult,
    n_frozen: usize,
    options: &CcsdOptions,
) -> CcsdState {
    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let n_occ = scf.n_alpha; // doubly occupied (RHF)
    assert!(n_frozen <= n_occ, "more frozen orbitals than occupied");

    let o = n_occ - n_frozen; // active occupied (spatial)
    let v = m - n_occ; // virtual (spatial)
    let eps = &scf.orbital_energies_alpha;
    let c = &scf.mo_coeff_alpha;

    let co = column_block(c, n, m, n_frozen, o);
    let cv = column_block(c, n, m, n_occ, v);

    let ao = provider.ao_eri();
    let oooo = transform_block(ao, n, [&co, &co, &co, &co]); // (ij|kl)
    let ovoo = transform_block(ao, n, [&co, &cv, &co, &co]); // (ia|jk)
    let oovv = transform_block(ao, n, [&co, &co, &cv, &cv]); // (ij|ab)
    let ovov = transform_block(ao, n, [&co, &cv, &co, &cv]); // (ia|jb)
    let ovvv = transform_block(ao, n, [&co, &cv, &cv, &cv]); // (ia|bc)
    let vvvv = transform_block(ao, n, [&cv, &cv, &cv, &cv]); // (ab|cd)
    let ovvo = ovov.permute(&[0, 1, 3, 2]); // [o,v,v,o]

    let lovov = {
        let swapped = ovov.permute(&[0, 3, 2, 1]); // (kd|lc)
        let mut data: Vec<f64> = ovov.data().iter().map(|x| 2.0 * x).collect();
        axpy(&mut data, -1.0, swapped.data());
        Tensor::new(vec![o, v, o, v], data)
    };

    let eo: Vec<f64> = (0..o).map(|i| eps[n_frozen + i]).collect();
    let ev: Vec<f64> = (0..v).map(|a| eps[n_occ + a]).collect();

    let d_ia = |i: usize, a: usize| eo[i] - ev[a];
    let d_ijab = |i: usize, j: usize, a: usize, b: usize| eo[i] + eo[j] - ev[a] - ev[b];

    let mut t1 = vec![0.0; o * v];
    let mut t2 = vec![0.0; o * o * v * v];

    {
        let g = ovov.data();
        let ovov_i = |i: usize, a: usize, j: usize, b: usize| g[((i * v + a) * o + j) * v + b];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        t2[((i * o + j) * v + a) * v + b] = ovov_i(i, a, j, b) / d_ijab(i, j, a, b);
                    }
                }
            }
        }
    }

    let mp2_correlation = ccsd_energy(&ovov, o, v, &t1, &t2);

    let mut diis = AmplitudeDiis::new(options.diis_dim);
    let mut e_prev = mp2_correlation;
    let mut correlation_energy = mp2_correlation;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=options.max_iter {
        iterations = iter;

        let t1t = Tensor::new(vec![o, v], t1.clone());
        let t2t = Tensor::new(vec![o, o, v, v], t2.clone());

        let tau = {
            let mut d = t2.clone();
            axpy(&mut d, 1.0, einsum(&t1t, "ia", &t1t, "jb", "ijab").data());
            Tensor::new(vec![o, o, v, v], d)
        };
        let pt = einsum(&t1t, "id", &t1t, "la", "ilda");

        let f_oo = einsum(&lovov, "kcld", &tau, "ilcd", "ki");
        let f_vv = {
            let f = einsum(&lovov, "kcld", &tau, "klad", "ac");
            Tensor::new(vec![v, v], f.data().iter().map(|x| -x).collect())
        };
        let f_ov = einsum(&lovov, "kcld", &t1t, "ld", "kc");

        let l_oo = {
            let mut d = f_oo.data().to_vec();
            axpy(&mut d, 2.0, einsum(&ovoo, "lcki", &t1t, "lc", "ki").data());
            axpy(&mut d, -1.0, einsum(&ovoo, "kcli", &t1t, "lc", "ki").data());
            Tensor::new(vec![o, o], d)
        };
        let l_vv = {
            let mut d = f_vv.data().to_vec();
            axpy(&mut d, 2.0, einsum(&ovvv, "kdac", &t1t, "kd", "ac").data());
            axpy(&mut d, -1.0, einsum(&ovvv, "kcad", &t1t, "kd", "ac").data());
            Tensor::new(vec![v, v], d)
        };

        let woooo = {
            let mut d = vec![0.0; o * o * o * o];
            axpy(
                &mut d,
                1.0,
                einsum(&ovoo, "lcki", &t1t, "jc", "klij").data(),
            );
            axpy(
                &mut d,
                1.0,
                einsum(&ovoo, "kclj", &t1t, "ic", "klij").data(),
            );
            axpy(
                &mut d,
                1.0,
                einsum(&ovov, "kcld", &tau, "ijcd", "klij").data(),
            );
            axpy(&mut d, 1.0, oooo.permute(&[0, 2, 1, 3]).data()); // (ki|lj)
            Tensor::new(vec![o, o, o, o], d)
        };

        let wvvvv = {
            let mut d = vec![0.0; v * v * v * v];
            axpy(
                &mut d,
                -1.0,
                einsum(&ovvv, "kdac", &t1t, "kb", "abcd").data(),
            );
            axpy(
                &mut d,
                -1.0,
                einsum(&ovvv, "kcbd", &t1t, "ka", "abcd").data(),
            );
            axpy(&mut d, 1.0, vvvv.permute(&[0, 2, 1, 3]).data()); // (ac|bd)
            Tensor::new(vec![v, v, v, v], d)
        };

        let wvoov = {
            let mut d = vec![0.0; v * o * o * v];
            axpy(
                &mut d,
                1.0,
                einsum(&ovvv, "kcad", &t1t, "id", "akic").data(),
            );
            axpy(
                &mut d,
                -1.0,
                einsum(&ovoo, "kcli", &t1t, "la", "akic").data(),
            );
            axpy(&mut d, 1.0, ovvo.permute(&[2, 0, 3, 1]).data()); // (kc|ai)
            axpy(
                &mut d,
                -0.5,
                einsum(&ovov, "ldkc", &t2t, "ilda", "akic").data(),
            );
            axpy(
                &mut d,
                -0.5,
                einsum(&ovov, "lckd", &t2t, "ilad", "akic").data(),
            );
            axpy(
                &mut d,
                -1.0,
                einsum(&ovov, "ldkc", &pt, "ilda", "akic").data(),
            );
            axpy(
                &mut d,
                1.0,
                einsum(&ovov, "ldkc", &t2t, "ilad", "akic").data(),
            );
            Tensor::new(vec![v, o, o, v], d)
        };

        let wvovo = {
            let mut d = vec![0.0; v * o * v * o];
            axpy(
                &mut d,
                1.0,
                einsum(&ovvv, "kdac", &t1t, "id", "akci").data(),
            );
            axpy(
                &mut d,
                -1.0,
                einsum(&ovoo, "lcki", &t1t, "la", "akci").data(),
            );
            axpy(&mut d, 1.0, oovv.permute(&[2, 0, 3, 1]).data()); // (ki|ac)
            axpy(
                &mut d,
                -0.5,
                einsum(&ovov, "lckd", &t2t, "ilda", "akci").data(),
            );
            axpy(
                &mut d,
                -1.0,
                einsum(&ovov, "lckd", &pt, "ilda", "akci").data(),
            );
            Tensor::new(vec![v, o, v, o], d)
        };

        let mut t1_new = vec![0.0; o * v];
        axpy(
            &mut t1_new,
            1.0,
            einsum(&l_vv, "ac", &t1t, "ic", "ia").data(),
        );
        axpy(
            &mut t1_new,
            -1.0,
            einsum(&l_oo, "ki", &t1t, "ka", "ia").data(),
        );
        axpy(
            &mut t1_new,
            2.0,
            einsum(&f_ov, "kc", &t2t, "kica", "ia").data(),
        );
        axpy(
            &mut t1_new,
            -1.0,
            einsum(&f_ov, "kc", &t2t, "ikca", "ia").data(),
        );
        {
            let w_ik = einsum(&t1t, "ic", &f_ov, "kc", "ik");
            axpy(
                &mut t1_new,
                1.0,
                einsum(&w_ik, "ik", &t1t, "ka", "ia").data(),
            );
        }
        axpy(
            &mut t1_new,
            2.0,
            einsum(&ovvo, "kcai", &t1t, "kc", "ia").data(),
        );
        axpy(
            &mut t1_new,
            -1.0,
            einsum(&oovv, "kiac", &t1t, "kc", "ia").data(),
        );
        axpy(
            &mut t1_new,
            2.0,
            einsum(&ovvv, "kdac", &t2t, "ikcd", "ia").data(),
        );
        axpy(
            &mut t1_new,
            -1.0,
            einsum(&ovvv, "kcad", &t2t, "ikcd", "ia").data(),
        );
        axpy(
            &mut t1_new,
            -2.0,
            einsum(&ovoo, "lcki", &t2t, "klac", "ia").data(),
        );
        axpy(
            &mut t1_new,
            1.0,
            einsum(&ovoo, "kcli", &t2t, "klac", "ia").data(),
        );
        for i in 0..o {
            for a in 0..v {
                t1_new[i * v + a] /= d_ia(i, a);
            }
        }

        let mut t2_new = vec![0.0; o * o * v * v];

        {
            let mut tmp2 = vec![0.0; v * v * o * v];
            axpy(
                &mut tmp2,
                -1.0,
                einsum(&oovv, "kibc", &t1t, "ka", "abic").data(),
            );
            axpy(&mut tmp2, 1.0, ovvv.permute(&[1, 3, 0, 2]).data()); // (ia|cb)
            let tmp2 = Tensor::new(vec![v, v, o, v], tmp2);
            let tmp = einsum(&tmp2, "abic", &t1t, "jc", "ijab");
            add_sym(&mut t2_new, 1.0, &tmp);
        }
        {
            let mut tmp2 = vec![0.0; v * o * o * o];
            axpy(
                &mut tmp2,
                1.0,
                einsum(&ovvo, "kcai", &t1t, "jc", "akij").data(),
            );
            axpy(&mut tmp2, 1.0, ovoo.permute(&[1, 3, 0, 2]).data()); // (ia|jk)
            let tmp2 = Tensor::new(vec![v, o, o, o], tmp2);
            let tmp = einsum(&tmp2, "akij", &t1t, "kb", "ijab");
            add_sym(&mut t2_new, -1.0, &tmp);
        }

        axpy(&mut t2_new, 1.0, ovov.permute(&[0, 2, 1, 3]).data()); // (ia|jb)
        axpy(
            &mut t2_new,
            1.0,
            einsum(&woooo, "klij", &tau, "klab", "ijab").data(),
        );
        axpy(
            &mut t2_new,
            1.0,
            einsum(&wvvvv, "abcd", &tau, "ijcd", "ijab").data(),
        );

        add_sym(&mut t2_new, 1.0, &einsum(&l_vv, "ac", &t2t, "ijcb", "ijab"));
        add_sym(
            &mut t2_new,
            -1.0,
            &einsum(&l_oo, "ki", &t2t, "kjab", "ijab"),
        );

        {
            let mut tmp = einsum(&wvoov, "akic", &t2t, "kjcb", "ijab").data().to_vec();
            for x in tmp.iter_mut() {
                *x *= 2.0;
            }
            axpy(
                &mut tmp,
                -1.0,
                einsum(&wvovo, "akci", &t2t, "kjcb", "ijab").data(),
            );
            add_sym(&mut t2_new, 1.0, &Tensor::new(vec![o, o, v, v], tmp));
        }
        add_sym(
            &mut t2_new,
            -1.0,
            &einsum(&wvoov, "akic", &t2t, "kjbc", "ijab"),
        );
        add_sym(
            &mut t2_new,
            -1.0,
            &einsum(&wvovo, "bkci", &t2t, "kjac", "ijab"),
        );

        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        t2_new[((i * o + j) * v + a) * v + b] /= d_ijab(i, j, a, b);
                    }
                }
            }
        }

        let amplitude: Vec<f64> = t1_new.iter().chain(t2_new.iter()).copied().collect();
        let error: Vec<f64> = t1_new
            .iter()
            .zip(&t1)
            .chain(t2_new.iter().zip(&t2))
            .map(|(new, old)| new - old)
            .collect();
        let rms = (error.iter().map(|x| x * x).sum::<f64>() / error.len() as f64).sqrt();

        diis.push(amplitude, error);
        let extrapolated = diis.extrapolate();
        let n_t1 = t1.len();
        t1.copy_from_slice(&extrapolated[..n_t1]);
        t2.copy_from_slice(&extrapolated[n_t1..]);

        correlation_energy = ccsd_energy(&ovov, o, v, &t1, &t2);
        let de = correlation_energy - e_prev;
        if de.abs() < options.energy_tol && rms < options.amplitude_tol {
            converged = true;
            break;
        }
        e_prev = correlation_energy;
    }

    let result = CcsdResult {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        scf_energy: scf.energy,
        mp2_correlation,
        converged,
        iterations,
        n_frozen,
        t1_diagnostic: super::t1_diagnostic_from(&t1, o),
    };
    CcsdState {
        result,
        o,
        v,
        eo: eo.clone(),
        ev: ev.clone(),
        t1,
        t2,
        ovvv,
        ovoo,
        ovov,
    }
}
