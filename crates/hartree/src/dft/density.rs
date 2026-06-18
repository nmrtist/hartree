use crate::dft::ao::AoBatch;

#[derive(Debug, Clone)]
pub struct BatchDensity {
    pub rho: Vec<f64>,
    pub grad: Vec<[f64; 3]>,
    pub tau: Vec<f64>,
}

pub fn batch_density(ao: &AoBatch, d: &[f64], with_grad: bool) -> BatchDensity {
    batch_density_tau(ao, d, with_grad, false)
}

pub fn batch_density_tau(ao: &AoBatch, d: &[f64], with_grad: bool, with_tau: bool) -> BatchDensity {
    let npts = ao.npts;
    let nao = ao.nao;
    assert_eq!(d.len(), nao * nao, "density must be nao×nao");
    debug_assert!(
        !(with_grad || with_tau) || ao.with_grad,
        "gradients requested but AO batch has none"
    );

    let t = crate::linalg::gemm(&ao.phi, npts, nao, d, nao);

    let mut rho = vec![0.0; npts];
    let mut grad = if with_grad {
        vec![[0.0; 3]; npts]
    } else {
        Vec::new()
    };
    let mut tau = if with_tau {
        vec![0.0; npts]
    } else {
        Vec::new()
    };
    if with_tau {
        for dk in &ao.dphi {
            let tk = crate::linalg::gemm(dk, npts, nao, d, nao);
            for p in 0..npts {
                let trow = &tk[p * nao..p * nao + nao];
                let drow = &dk[p * nao..p * nao + nao];
                let mut s = 0.0;
                for mu in 0..nao {
                    s += trow[mu] * drow[mu];
                }
                tau[p] += 0.5 * s;
            }
        }
    }

    for p in 0..npts {
        let trow = &t[p * nao..p * nao + nao];
        let prow = &ao.phi[p * nao..p * nao + nao];
        let mut r = 0.0;
        for mu in 0..nao {
            r += trow[mu] * prow[mu];
        }
        rho[p] = r;

        if with_grad {
            let mut g = [0.0; 3];
            for (k, gk) in g.iter_mut().enumerate() {
                let dk = &ao.dphi[k][p * nao..p * nao + nao];
                let mut s = 0.0;
                for mu in 0..nao {
                    s += trow[mu] * dk[mu];
                }
                *gk = 2.0 * s;
            }
            grad[p] = g;
        }
    }

    BatchDensity { rho, grad, tau }
}

pub fn sigma_dot(g_sigma: &[[f64; 3]], g_tau: &[[f64; 3]]) -> Vec<f64> {
    assert_eq!(g_sigma.len(), g_tau.len());
    g_sigma
        .iter()
        .zip(g_tau)
        .map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        .collect()
}

pub fn pack_rho_polarized(rho_a: &[f64], rho_b: &[f64]) -> Vec<f64> {
    assert_eq!(rho_a.len(), rho_b.len());
    let mut out = vec![0.0; 2 * rho_a.len()];
    for p in 0..rho_a.len() {
        out[2 * p] = rho_a[p];
        out[2 * p + 1] = rho_b[p];
    }
    out
}

pub fn pack_tau_polarized(tau_a: &[f64], tau_b: &[f64]) -> Vec<f64> {
    pack_rho_polarized(tau_a, tau_b)
}

pub fn pack_sigma_polarized(grad_a: &[[f64; 3]], grad_b: &[[f64; 3]]) -> Vec<f64> {
    assert_eq!(grad_a.len(), grad_b.len());
    let np = grad_a.len();
    let mut out = vec![0.0; 3 * np];
    for p in 0..np {
        let a = grad_a[p];
        let b = grad_b[p];
        out[3 * p] = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; // σ_αα
        out[3 * p + 1] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; // σ_αβ
        out[3 * p + 2] = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]; // σ_ββ
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::ShellData;
    use crate::dft::ao::eval_ao_batch;

    fn s_shell(center: [f64; 3], a: f64) -> ShellData {
        ShellData {
            l: 0,
            center,
            exponents: vec![a],
            coefficients: vec![1.0],
            spherical: false,
        }
    }

    #[test]
    fn single_orbital_density_and_gradient() {
        let shells = vec![s_shell([0.0, 0.0, 0.0], 0.7)];
        let pts = vec![[0.3, -0.2, 0.5], [1.0, 0.0, 0.0]];
        let ao = eval_ao_batch(&shells, 1, &pts, true);
        let d = vec![2.0]; // 1×1 density, 2 electrons in the orbital
        let bd = batch_density_tau(&ao, &d, true, true);

        for p in 0..pts.len() {
            let phi = ao.phi[p];
            assert!((bd.rho[p] - 2.0 * phi * phi).abs() < 1e-12);
            let mut g2 = 0.0;
            for k in 0..3 {
                let dphi = ao.dphi[k][p];
                assert!((bd.grad[p][k] - 4.0 * phi * dphi).abs() < 1e-12);
                g2 += dphi * dphi;
            }
            assert!((bd.tau[p] - g2).abs() < 1e-12, "τ != |∇φ|² at point {p}");
            if bd.rho[p] > 1e-12 {
                let vw = (bd.grad[p][0].powi(2) + bd.grad[p][1].powi(2) + bd.grad[p][2].powi(2))
                    / (8.0 * bd.rho[p]);
                assert!((bd.tau[p] - vw).abs() < 1e-12, "vW not saturated at {p}");
            }
        }
    }

    #[test]
    fn restricted_equals_polarized_sum() {
        let shells = vec![s_shell([0.0, 0.0, 0.0], 0.9), s_shell([0.0, 0.0, 1.4], 1.1)];
        let pts = vec![[0.2, 0.1, 0.7], [0.0, 0.0, 0.7]];
        let ao = eval_ao_batch(&shells, 2, &pts, true);
        let d_tot = vec![1.3, 0.4, 0.4, 0.9];
        let d_half: Vec<f64> = d_tot.iter().map(|x| x * 0.5).collect();

        let tot = batch_density(&ao, &d_tot, true);
        let spin = batch_density(&ao, &d_half, true);
        for p in 0..pts.len() {
            assert!((tot.rho[p] - 2.0 * spin.rho[p]).abs() < 1e-12);
            for k in 0..3 {
                assert!((tot.grad[p][k] - 2.0 * spin.grad[p][k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn polarized_packing_layout() {
        let rho_a = [1.0, 2.0];
        let rho_b = [3.0, 4.0];
        assert_eq!(pack_rho_polarized(&rho_a, &rho_b), vec![1.0, 3.0, 2.0, 4.0]);

        let ga = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let gb = [[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(
            pack_sigma_polarized(&ga, &gb),
            vec![1.0, 0.0, 1.0, 4.0, 0.0, 9.0]
        );
    }
}
