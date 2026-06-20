//! FIRE relaxation of the band: the inertial optimizer (Bitzek *et al.*, Phys. Rev.
//! Lett. 97, 170201 (2006)) and the iteration loop that drives the interior images
//! toward the minimum-energy path while the endpoints stay fixed.
//!
//! FIRE is Hessian-free and robust for the indefinite, many-coordinate NEB force —
//! it integrates a damped equation of motion, growing the time step while descending
//! and resetting velocity the instant the power `F·v` goes negative. The per-image
//! true gradients are evaluated **sequentially** on the single [`Surface`]: one band
//! relaxation step costs `n_images` gradient evaluations.

use super::band;
use super::{NebError, NebOptions, NebStatus};
use crate::opt::OptStep;
use crate::opt::Surface;
use crate::opt::ts::numerics::{dot, gradient, norm};
use crate::opt::ts::{Flow, Progress};

/// The relaxed band and the data the orchestrator packages into a `NebResult`.
pub(super) struct Relaxed {
    pub(super) images: Vec<Vec<[f64; 3]>>,
    pub(super) energies: Vec<f64>,
    pub(super) climbing_image: usize,
    pub(super) peak_tangent: Vec<[f64; 3]>,
    pub(super) status: NebStatus,
    pub(super) iterations: usize,
    pub(super) history: Vec<OptStep>,
}

/// Relax the full band (`images[0]` and `images[last]` are the fixed endpoints) onto
/// the minimum-energy path with climbing-image NEB + FIRE.
pub(super) fn relax<S: Surface>(
    surface: &mut S,
    mut images: Vec<Vec<[f64; 3]>>,
    options: &NebOptions,
    progress: Option<&dyn Progress>,
) -> Result<Relaxed, NebError> {
    let n_total = images.len();
    let n_images = n_total - 2;
    let ndof = 3 * images[0].len();

    // Endpoints never move, so their energies are constant — evaluate them once.
    let mut energies = vec![0.0f64; n_total];
    energies[0] = surface.energy(&images[0])?;
    energies[n_total - 1] = surface.energy(&images[n_total - 1])?;
    // Interior gradients (endpoints carry empty placeholders that are never read).
    let mut grads: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_total];

    let mut fire = Fire::new(n_images * ndof, options);
    // Climbing from a cold band is unstable: only invert the peak force once the band
    // force is modest, or after a few warm-up iterations (whichever comes first).
    let climb_threshold = (10.0 * options.gtol).max(0.05);
    let mut ci_latched = false;

    let mut history = Vec::new();
    let mut prev_interior: Option<Vec<f64>> = None;
    let mut status = NebStatus::NotConverged;
    let mut iterations = 0;
    let mut hei = 1; // highest-energy interior image (full-band index)

    for iter in 1..=options.max_iter {
        iterations = iter;

        // Per-image energies and gradients, evaluated one image at a time.
        for i in 1..=n_images {
            energies[i] = surface.energy(&images[i])?;
            grads[i] = gradient(surface, &images[i], options.fd_step)?;
        }
        hei = highest_energy_image(&energies, n_images);

        if options.climbing && !ci_latched {
            let (_f, plain_max, _r) =
                band::neb_forces(&images, &energies, &grads, options.spring_k, None);
            if iter >= options.climb_after || plain_max < climb_threshold {
                ci_latched = true;
            }
        }
        let ci_index = if options.climbing && ci_latched {
            Some(hei)
        } else {
            None
        };

        let (force, max_f, rms_f) =
            band::neb_forces(&images, &energies, &grads, options.spring_k, ci_index);

        let interior = concat_interior(&images, n_images, ndof);
        let (max_disp, rms_disp) = match &prev_interior {
            Some(p) => disp_norms(&interior, p),
            None => (0.0, 0.0),
        };
        let step = OptStep {
            iteration: iter,
            energy: energies[hei],
            max_force: max_f,
            rms_force: rms_f,
            max_disp,
            rms_disp,
        };
        history.push(step);
        if let Some(obs) = progress {
            if obs.step(&step) == Flow::Stop {
                status = NebStatus::StoppedEarly;
                break;
            }
        }

        // Converge only once the climbing image (if requested) is active, so the peak
        // has actually been driven to the barrier top rather than left mid-band.
        if max_f < options.gtol && (!options.climbing || ci_latched) {
            status = NebStatus::Converged;
            break;
        }
        if iter == options.max_iter {
            status = NebStatus::NotConverged;
            break;
        }

        // Keep the pre-step geometry so the *next* iteration's history measures the
        // actual step: after `scatter_interior`, re-`concat`-ing the band reproduces
        // the post-step geometry exactly, so storing `x` here would compare a geometry
        // to itself and report a flat (zero) displacement trace.
        let mut x = interior.clone();
        fire.step(&mut x, &force);
        scatter_interior(&mut images, &x, n_images, ndof);
        prev_interior = Some(interior);
    }

    // The reaction-coordinate tangent at the peak, surfaced as the refiner's seed.
    let peak_tangent_flat = band::improved_tangent(
        &images[hei - 1],
        &images[hei],
        &images[hei + 1],
        energies[hei - 1],
        energies[hei],
        energies[hei + 1],
    );
    Ok(Relaxed {
        peak_tangent: band::reshape(&peak_tangent_flat),
        images,
        energies,
        climbing_image: hei,
        status,
        iterations,
        history,
    })
}

fn highest_energy_image(energies: &[f64], n_images: usize) -> usize {
    (1..=n_images)
        .max_by(|&a, &b| energies[a].total_cmp(&energies[b]))
        .unwrap_or(1)
}

fn concat_interior(images: &[Vec<[f64; 3]>], n_images: usize, ndof: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_images * ndof);
    for image in images.iter().take(n_images + 1).skip(1) {
        for p in image {
            out.extend_from_slice(p);
        }
    }
    out
}

fn scatter_interior(images: &mut [Vec<[f64; 3]>], x: &[f64], n_images: usize, ndof: usize) {
    for (idx, image) in images.iter_mut().enumerate().take(n_images + 1).skip(1) {
        let base = (idx - 1) * ndof;
        for (a, p) in image.iter_mut().enumerate() {
            p[0] = x[base + 3 * a];
            p[1] = x[base + 3 * a + 1];
            p[2] = x[base + 3 * a + 2];
        }
    }
}

fn disp_norms(cur: &[f64], prev: &[f64]) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut sumsq = 0.0;
    for (a, b) in cur.iter().zip(prev) {
        let d = (a - b).abs();
        max = max.max(d);
        sumsq += d * d;
    }
    (max, (sumsq / cur.len().max(1) as f64).sqrt())
}

/// The Fast Inertial Relaxation Engine over a flat coordinate vector (unit mass).
struct Fire {
    v: Vec<f64>,
    dt: f64,
    alpha: f64,
    nsteps: usize,
    dt_max: f64,
    n_min: usize,
    f_inc: f64,
    f_dec: f64,
    alpha_start: f64,
    f_alpha: f64,
    max_step: f64,
}

impl Fire {
    fn new(dim: usize, o: &NebOptions) -> Self {
        Self {
            v: vec![0.0; dim],
            dt: o.fire_dt,
            alpha: o.fire_alpha_start,
            nsteps: 0,
            dt_max: o.fire_dt_max,
            n_min: o.fire_n_min,
            f_inc: o.fire_f_inc,
            f_dec: o.fire_f_dec,
            alpha_start: o.fire_alpha_start,
            f_alpha: o.fire_f_alpha,
            max_step: o.fire_max_step,
        }
    }

    /// One FIRE update: advance `x` along `force` (the direction to move toward —
    /// already a force, not a gradient). Mixes velocity toward the force, grows the
    /// time step while descending, and zeroes velocity when the power goes negative.
    fn step(&mut self, x: &mut [f64], force: &[f64]) {
        let power = dot(force, &self.v);
        if power > 0.0 {
            let vnorm = norm(&self.v);
            let fnorm = norm(force).max(1e-30);
            for (vk, &fk) in self.v.iter_mut().zip(force) {
                *vk = (1.0 - self.alpha) * *vk + self.alpha * vnorm * fk / fnorm;
            }
            if self.nsteps > self.n_min {
                self.dt = (self.dt * self.f_inc).min(self.dt_max);
                self.alpha *= self.f_alpha;
            }
            self.nsteps += 1;
        } else {
            for vk in self.v.iter_mut() {
                *vk = 0.0;
            }
            self.dt *= self.f_dec;
            self.alpha = self.alpha_start;
            self.nsteps = 0;
        }

        // Semi-implicit Euler: accelerate, then displace.
        for (vk, &fk) in self.v.iter_mut().zip(force) {
            *vk += self.dt * fk;
        }
        let mut dr: Vec<f64> = self.v.iter().map(|&vk| self.dt * vk).collect();
        // Cap the largest single-coordinate move (preserving direction) so an early,
        // large-force step cannot launch an image into a non-convergent region.
        let max_abs = dr.iter().fold(0.0f64, |m, &c| m.max(c.abs()));
        if max_abs > self.max_step && max_abs > 0.0 {
            let s = self.max_step / max_abs;
            for d in dr.iter_mut() {
                *d *= s;
            }
        }
        for (xk, &drk) in x.iter_mut().zip(&dr) {
            *xk += drk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIRE drives a simple isotropic quadratic (force `−x`) to its minimum at the
    /// origin from a displaced start — the optimizer's basic descent property.
    #[test]
    fn fire_descends_a_quadratic() {
        let opts = NebOptions::default();
        let mut fire = Fire::new(2, &opts);
        let mut x = vec![1.0, -0.7];
        for _ in 0..1000 {
            let force = vec![-x[0], -x[1]];
            fire.step(&mut x, &force);
        }
        assert!(
            x[0].abs() < 1e-3 && x[1].abs() < 1e-3,
            "FIRE did not converge: x={x:?}"
        );
    }

    /// A single coordinate-component move is capped at `fire_max_step`, regardless of
    /// how large the force is (direction preserved).
    #[test]
    fn fire_caps_the_step_size() {
        let opts = NebOptions {
            fire_max_step: 0.05,
            ..Default::default()
        };
        let mut fire = Fire::new(1, &opts);
        let mut x = vec![0.0];
        fire.step(&mut x, &[1000.0]);
        assert!(x[0].abs() <= 0.05 + 1e-12, "step not capped: {}", x[0]);
    }
}
