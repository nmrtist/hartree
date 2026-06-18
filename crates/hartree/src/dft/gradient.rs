use crate::basis::ShellData;
use crate::core::Molecule;

use crate::dft::ao::{self, AoBatch, hess_index};
use crate::dft::density::{
    batch_density_tau, pack_rho_polarized, pack_sigma_polarized, pack_tau_polarized,
};
use crate::dft::error::{DftError, Result};
use crate::dft::grid::BeckePartition;
use crate::dft::xc::GridXc;
use xcx::XcInput;

pub(crate) fn ao_atom_map(shells: &[ShellData], mol: &Molecule) -> Result<Vec<usize>> {
    let mut map = Vec::new();
    for s in shells {
        let atom = mol
            .atoms
            .iter()
            .position(|a| a.position == s.center)
            .ok_or(DftError::ShellAtomMismatch)?;
        let nf = ao::n_func(s.l as usize, s.spherical);
        map.extend(std::iter::repeat_n(atom, nf));
    }
    Ok(map)
}

impl GridXc {
    pub fn xc_gradient(
        &self,
        d_alpha: &[f64],
        d_beta: &[f64],
        restricted: bool,
    ) -> Result<Vec<[f64; 3]>> {
        let nao = self.nao();
        let natom = self.natom;
        let ao_atom = &self.ao_atom;
        let partition = &self.partition;
        let grid = self.grid();
        let weights = &grid.weights;
        let atom_of_point = &grid.atom_of_point;
        let needs_sigma = self.needs_sigma();
        let needs_tau = self.needs_tau();
        debug_assert!(!needs_tau || needs_sigma, "τ-meta-GGA without σ");

        let d_tot: Vec<f64>;
        let (da, db): (&[f64], &[f64]) = if restricted {
            d_tot = d_alpha.iter().zip(d_beta).map(|(a, b)| a + b).collect();
            (&d_tot, &[])
        } else {
            (d_alpha, d_beta)
        };

        ao::par_blocks_fold_full(
            self.shells(),
            nao,
            &grid.points,
            true,
            needs_sigma,
            || vec![[0.0; 3]; natom],
            |mut acc, batch, start| {
                let np = batch.npts;
                let w = &weights[start..start + np];

                if restricted {
                    let bd = batch_density_tau(batch, da, needs_sigma, needs_tau);
                    let res = self.eval_xc_unpol(np, &bd.rho, &bd.grad, &bd.tau);
                    let gvec = needs_sigma.then(|| {
                        (0..np)
                            .map(|p| {
                                let g = bd.grad[p];
                                let c = 2.0 * res.vsigma[p];
                                [c * g[0], c * g[1], c * g[2]]
                            })
                            .collect::<Vec<_>>()
                    });
                    orbital_term(
                        batch,
                        w,
                        &atom_of_point[start..start + np],
                        da,
                        &res.vrho,
                        1,
                        0,
                        gvec.as_deref(),
                        needs_tau.then_some(res.vtau.as_slice()),
                        ao_atom,
                        &mut acc,
                    );
                    weight_term(
                        partition,
                        &grid.points[start..start + np],
                        &atom_of_point[start..start + np],
                        w,
                        &bd.rho,
                        &res.exc,
                        &mut acc,
                    );
                } else {
                    let bd_a = batch_density_tau(batch, da, needs_sigma, needs_tau);
                    let bd_b = batch_density_tau(batch, db, needs_sigma, needs_tau);
                    let rho = pack_rho_polarized(&bd_a.rho, &bd_b.rho);
                    let res = if needs_sigma {
                        let sigma = pack_sigma_polarized(&bd_a.grad, &bd_b.grad);
                        let input = XcInput::gga(&rho, &sigma);
                        if needs_tau {
                            let tau = pack_tau_polarized(&bd_a.tau, &bd_b.tau);
                            self.func_pol()
                                .eval(np, &input.with_tau(&tau))
                                .expect("xcx polarized meta-GGA eval")
                        } else {
                            self.func_pol()
                                .eval(np, &input)
                                .expect("xcx polarized GGA eval")
                        }
                    } else {
                        self.func_pol()
                            .eval(np, &XcInput::lda(&rho))
                            .expect("xcx polarized LDA eval")
                    };
                    let make_g = |same: &[[f64; 3]], other: &[[f64; 3]], s_idx: usize| {
                        (0..np)
                            .map(|p| {
                                let (vs, vx) = (res.vsigma[3 * p + s_idx], res.vsigma[3 * p + 1]);
                                let (gs, go) = (same[p], other[p]);
                                [
                                    2.0 * vs * gs[0] + vx * go[0],
                                    2.0 * vs * gs[1] + vx * go[1],
                                    2.0 * vs * gs[2] + vx * go[2],
                                ]
                            })
                            .collect::<Vec<_>>()
                    };
                    let g_a = needs_sigma.then(|| make_g(&bd_a.grad, &bd_b.grad, 0));
                    let g_b = needs_sigma.then(|| make_g(&bd_b.grad, &bd_a.grad, 2));
                    let par = &atom_of_point[start..start + np];
                    let vtau = needs_tau.then_some(res.vtau.as_slice());
                    orbital_term(
                        batch,
                        w,
                        par,
                        da,
                        &res.vrho,
                        2,
                        0,
                        g_a.as_deref(),
                        vtau,
                        ao_atom,
                        &mut acc,
                    );
                    orbital_term(
                        batch,
                        w,
                        par,
                        db,
                        &res.vrho,
                        2,
                        1,
                        g_b.as_deref(),
                        vtau,
                        ao_atom,
                        &mut acc,
                    );
                    let n_tot: Vec<f64> =
                        bd_a.rho.iter().zip(&bd_b.rho).map(|(a, b)| a + b).collect();
                    weight_term(
                        partition,
                        &grid.points[start..start + np],
                        &atom_of_point[start..start + np],
                        w,
                        &n_tot,
                        &res.exc,
                        &mut acc,
                    );
                }
                acc
            },
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    for k in 0..3 {
                        x[k] += y[k];
                    }
                }
                a
            },
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn orbital_term(
    batch: &AoBatch,
    w: &[f64],
    parents: &[usize],
    d: &[f64],
    vrho: &[f64],
    stride: usize,
    offset: usize,
    gvec: Option<&[[f64; 3]]>,
    vtau: Option<&[f64]>,
    ao_atom: &[usize],
    acc: &mut [[f64; 3]],
) {
    let np = batch.npts;
    let nao = batch.nao;
    debug_assert!(gvec.is_none() || batch.with_hess);
    debug_assert!(
        vtau.is_none() || gvec.is_some(),
        "the vtau term rides the GGA second-derivative path"
    );

    let t = crate::linalg::gemm(&batch.phi, np, nao, d, nao);
    let tk: Option<[Vec<f64>; 3]> =
        gvec.map(|_| std::array::from_fn(|k| crate::linalg::gemm(&batch.dphi[k], np, nao, d, nao)));

    for p in 0..np {
        let wp = w[p];
        let vr = vrho[stride * p + offset];
        let row = p * nao;
        let trow = &t[row..row + nao];
        let dx = &batch.dphi[0][row..row + nao];
        let dy = &batch.dphi[1][row..row + nao];
        let dz = &batch.dphi[2][row..row + nao];

        let mut point_motion = [0.0; 3];
        match gvec {
            None => {
                for mu in 0..nao {
                    let c = 2.0 * wp * vr * trow[mu];
                    let g = &mut acc[ao_atom[mu]];
                    g[0] -= c * dx[mu];
                    g[1] -= c * dy[mu];
                    g[2] -= c * dz[mu];
                    point_motion[0] += c * dx[mu];
                    point_motion[1] += c * dy[mu];
                    point_motion[2] += c * dz[mu];
                }
            }
            Some(gv) => {
                let gp = gv[p];
                let vt = vtau.map(|v| v[stride * p + offset]);
                let tks = tk.as_ref().unwrap();
                let h: [&[f64]; 6] = std::array::from_fn(|i| &batch.hess[i][row..row + nao]);
                for mu in 0..nao {
                    let tkmu = [tks[0][row + mu], tks[1][row + mu], tks[2][row + mu]];
                    let u = gp[0] * tkmu[0] + gp[1] * tkmu[1] + gp[2] * tkmu[2];
                    let c1 = vr * trow[mu] + u;
                    let tmu = trow[mu];
                    let g = &mut acc[ao_atom[mu]];
                    for x in 0..3 {
                        let dphi_x = match x {
                            0 => dx[mu],
                            1 => dy[mu],
                            _ => dz[mu],
                        };
                        let hg = gp[0] * h[hess_index(x, 0)][mu]
                            + gp[1] * h[hess_index(x, 1)][mu]
                            + gp[2] * h[hess_index(x, 2)][mu];
                        let mut v = 2.0 * wp * (c1 * dphi_x + tmu * hg);
                        if let Some(vt) = vt {
                            let ht = tkmu[0] * h[hess_index(x, 0)][mu]
                                + tkmu[1] * h[hess_index(x, 1)][mu]
                                + tkmu[2] * h[hess_index(x, 2)][mu];
                            v += wp * vt * ht;
                        }
                        g[x] -= v;
                        point_motion[x] += v;
                    }
                }
            }
        }
        let g = &mut acc[parents[p]];
        for k in 0..3 {
            g[k] += point_motion[k];
        }
    }
}

fn weight_term(
    partition: &BeckePartition,
    points: &[[f64; 3]],
    parents: &[usize],
    w: &[f64],
    n_tot: &[f64],
    exc: &[f64],
    acc: &mut [[f64; 3]],
) {
    let natom = acc.len();
    if natom == 1 {
        return; // single-atom partition is constant 1
    }
    let mut dp = vec![[0.0; 3]; natom];
    for p in 0..points.len() {
        let e = n_tot[p] * exc[p];
        if e == 0.0 {
            continue;
        }
        let p_cell = partition.weight_derivatives(points[p], parents[p], &mut dp);
        let w0 = w[p] / p_cell;
        for (g, d) in acc.iter_mut().zip(&dp) {
            for k in 0..3 {
                g[k] += w0 * e * d[k];
            }
        }
    }
}
