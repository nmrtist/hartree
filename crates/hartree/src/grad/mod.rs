//! Analytic nuclear gradients for RHF/UHF and Kohn-Sham references.

use crate::core::Molecule;
use crate::integrals::{GradientError, GradientProvider, IntegralProvider};
use crate::linalg::{mat_from_row_major, mat_to_row_major};
use crate::scf::{RangeSeparation, XcContributor};

pub type Gradient = Vec<[f64; 3]>;

pub fn hf_gradient<P>(
    provider: &P,
    molecule: &Molecule,
    density_alpha: &[f64],
    density_beta: &[f64],
) -> Result<Gradient, GradientError>
where
    P: IntegralProvider + GradientProvider,
{
    gradient_core(
        provider,
        molecule,
        density_alpha,
        density_beta,
        1.0,
        None,
        None,
    )
}

pub fn ks_gradient<P>(
    provider: &P,
    molecule: &Molecule,
    xc: &dyn XcContributor,
    density_alpha: &[f64],
    density_beta: &[f64],
    restricted: bool,
) -> Result<Gradient, GradientError>
where
    P: IntegralProvider + GradientProvider,
{
    let n = provider.n_basis();
    let xc_grad = xc
        .gradient(density_alpha, density_beta, restricted)
        .ok_or_else(|| {
            GradientError::Backend(
                "XC contributor exposes no analytic gradient; use finite differences".to_string(),
            )
        })?;
    let contrib = xc.eval(density_alpha, density_beta, n, restricted);
    let mut grad = gradient_core(
        provider,
        molecule,
        density_alpha,
        density_beta,
        xc.exx_fraction(),
        Some((&contrib.vxc_alpha, &contrib.vxc_beta)),
        xc.range_separation(),
    )?;
    assert_eq!(xc_grad.len(), grad.len(), "XC gradient atom count");
    for (g, x) in grad.iter_mut().zip(&xc_grad) {
        for k in 0..3 {
            g[k] += x[k];
        }
    }
    Ok(grad)
}

fn gradient_core<P>(
    provider: &P,
    molecule: &Molecule,
    density_alpha: &[f64],
    density_beta: &[f64],
    c_x: f64,
    vxc: Option<(&[f64], &[f64])>,
    rs: Option<RangeSeparation>,
) -> Result<Gradient, GradientError>
where
    P: IntegralProvider + GradientProvider,
{
    let n = provider.n_basis();
    let nn = n * n;
    let natom = molecule.len();
    assert_eq!(density_alpha.len(), nn, "density_alpha must be n²");
    assert_eq!(density_beta.len(), nn, "density_beta must be n²");

    let mut p = vec![0.0; nn];
    for i in 0..nn {
        p[i] = density_alpha[i] + density_beta[i];
    }

    let hcore = mat_to_row_major(&provider.core_hamiltonian());
    let jk = provider.build_jk(&[
        mat_from_row_major(n, density_alpha),
        mat_from_row_major(n, density_beta),
    ]);
    let ja = mat_to_row_major(&jk.coulomb[0]);
    let jb = mat_to_row_major(&jk.coulomb[1]);
    let mut ka = mat_to_row_major(&jk.exchange[0]);
    let mut kb = mat_to_row_major(&jk.exchange[1]);
    let c_x_fock = match &rs {
        Some(rs) => {
            debug_assert!(
                (c_x - rs.alpha).abs() < 1e-12,
                "RS hybrid: exx_fraction must equal α"
            );
            let klr = provider
                .build_k_erf(
                    &[
                        mat_from_row_major(n, density_alpha),
                        mat_from_row_major(n, density_beta),
                    ],
                    rs.omega,
                )
                .ok_or_else(|| {
                    GradientError::Backend(
                        "range-separated gradient: the integral backend has no erf-attenuated \
                         exchange (build_k_erf declined)"
                            .to_string(),
                    )
                })?;
            let klr_a = mat_to_row_major(&klr[0]);
            let klr_b = mat_to_row_major(&klr[1]);
            for i in 0..nn {
                ka[i] = rs.alpha * ka[i] + rs.beta * klr_a[i];
                kb[i] = rs.alpha * kb[i] + rs.beta * klr_b[i];
            }
            1.0
        }
        None => c_x,
    };
    let mut fa = vec![0.0; nn];
    let mut fb = vec![0.0; nn];
    for i in 0..nn {
        let j_total = ja[i] + jb[i];
        fa[i] = hcore[i] + j_total - c_x_fock * ka[i];
        fb[i] = hcore[i] + j_total - c_x_fock * kb[i];
    }
    if let Some((va, vb)) = vxc {
        assert_eq!(va.len(), nn, "vxc_alpha must be n²");
        assert_eq!(vb.len(), nn, "vxc_beta must be n²");
        for i in 0..nn {
            fa[i] += va[i];
            fb[i] += vb[i];
        }
    }

    let wa = triple_product(density_alpha, &fa, density_alpha, n);
    let wb = triple_product(density_beta, &fb, density_beta, n);
    let mut w = vec![0.0; nn];
    for i in 0..nn {
        w[i] = wa[i] + wb[i];
    }

    let dt = provider.kinetic_gradient()?;
    let dv = provider.nuclear_gradient()?;
    let ds = provider.overlap_gradient()?;
    assert_eq!(
        dt.natom(),
        natom,
        "integral-derivative atom count {} disagrees with molecule {natom}",
        dt.natom()
    );

    let mut grad: Gradient = vec![[0.0; 3]; natom];
    for (atom, g_atom) in grad.iter_mut().enumerate() {
        for (axis, g) in g_atom.iter_mut().enumerate() {
            let bt = dt.block(atom, axis);
            let bv = dv.block(atom, axis);
            let bs = ds.block(atom, axis);
            let mut acc = 0.0;
            for i in 0..nn {
                acc += p[i] * (bt[i] + bv[i]) - w[i] * bs[i];
            }
            *g = acc;
        }
    }

    if let Some(g_ecp) = provider.ecp_gradient_contract(&p)? {
        assert_eq!(
            g_ecp.len(),
            natom,
            "ECP gradient atom count {} disagrees with molecule {natom}",
            g_ecp.len()
        );
        for (atom, g_atom) in grad.iter_mut().enumerate() {
            for axis in 0..3 {
                g_atom[axis] += g_ecp[atom][axis];
            }
        }
    }

    let gamma = two_particle_density(&p, density_alpha, density_beta, n, c_x);
    let g2e = provider.eri_gradient_contract(&gamma)?;
    for (atom, g_atom) in grad.iter_mut().enumerate() {
        for axis in 0..3 {
            g_atom[axis] += g2e[atom][axis];
        }
    }

    if let Some(rs) = &rs {
        let gamma_lr = exchange_density(density_alpha, density_beta, n, rs.beta);
        let g_lr = provider
            .eri_gradient_contract_erf(&gamma_lr, rs.omega)?
            .ok_or_else(|| {
                GradientError::Backend(
                    "range-separated gradient: the integral backend has no erf-attenuated \
                     ERI derivative (eri_gradient_contract_erf declined)"
                        .to_string(),
                )
            })?;
        for (atom, g_atom) in grad.iter_mut().enumerate() {
            for axis in 0..3 {
                g_atom[axis] += g_lr[atom][axis];
            }
        }
    }

    let zeff = provider.effective_nuclear_charges().unwrap_or_else(|| {
        molecule
            .atoms
            .iter()
            .map(|a| a.element.z() as f64)
            .collect()
    });
    assert_eq!(zeff.len(), natom, "effective-charge count");
    let vnn = nuclear_repulsion_gradient(molecule, &zeff);
    for (atom, g_atom) in grad.iter_mut().enumerate() {
        for axis in 0..3 {
            g_atom[axis] += vnn[atom][axis];
        }
    }

    Ok(grad)
}

fn two_particle_density(p: &[f64], da: &[f64], db: &[f64], n: usize, c_x: f64) -> Vec<f64> {
    let mut gamma = vec![0.0; n * n * n * n];
    for pi in 0..n {
        for q in 0..n {
            let pq = p[pi * n + q];
            for r in 0..n {
                let da_pr = da[pi * n + r];
                let db_pr = db[pi * n + r];
                let base = ((pi * n + q) * n + r) * n;
                for s in 0..n {
                    gamma[base + s] = 0.5 * pq * p[r * n + s]
                        - c_x * (0.5 * da_pr * da[q * n + s] + 0.5 * db_pr * db[q * n + s]);
                }
            }
        }
    }
    gamma
}

fn exchange_density(da: &[f64], db: &[f64], n: usize, c: f64) -> Vec<f64> {
    let mut gamma = vec![0.0; n * n * n * n];
    for pi in 0..n {
        for q in 0..n {
            for r in 0..n {
                let da_pr = da[pi * n + r];
                let db_pr = db[pi * n + r];
                let base = ((pi * n + q) * n + r) * n;
                for s in 0..n {
                    gamma[base + s] =
                        -c * (0.5 * da_pr * da[q * n + s] + 0.5 * db_pr * db[q * n + s]);
                }
            }
        }
    }
    gamma
}

fn nuclear_repulsion_gradient(mol: &Molecule, charges: &[f64]) -> Gradient {
    let natom = mol.len();
    let mut g: Gradient = vec![[0.0; 3]; natom];
    for (a, g_a) in g.iter_mut().enumerate() {
        let za = charges[a];
        let ra = mol.atoms[a].position;
        for (b, other) in mol.atoms.iter().enumerate() {
            if a == b {
                continue;
            }
            let zb = charges[b];
            let rb = other.position;
            let d = [ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r3 = r2 * r2.sqrt();
            let f = za * zb / r3;
            for k in 0..3 {
                g_a[k] -= f * d[k];
            }
        }
    }
    g
}

fn triple_product(a: &[f64], m: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let am = matmul(a, m, n);
    matmul(&am, b, n)
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            let brow = k * n;
            let crow = i * n;
            for j in 0..n {
                c[crow + j] += a_ik * b[brow + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    #[test]
    fn vnn_gradient_diatomic() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.4]),
            ],
            0,
            1,
        );
        let g = nuclear_repulsion_gradient(&mol, &[1.0, 1.0]);
        let expected = 1.0 / (1.4 * 1.4);
        assert!((g[0][2] - expected).abs() < 1e-12, "g0z = {}", g[0][2]);
        assert!((g[1][2] + expected).abs() < 1e-12, "g1z = {}", g[1][2]);
        for (axis, (a, b)) in g[0].iter().zip(g[1].iter()).enumerate() {
            let s = a + b;
            assert!(s.abs() < 1e-12, "Σg axis {axis} = {s}");
        }
    }
}
