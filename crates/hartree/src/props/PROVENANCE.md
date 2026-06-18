# Properties (`props`) — algorithm provenance

## Single-Point Hessian (SPH) — `src/sph.rs`

Method reference: S. Spicher and S. Grimme, "Single-Point Hessian Calculations
for Improved Vibrational Frequencies and Rigid-Rotor-Harmonic-Oscillator
Thermostatistics", *J. Chem. Theory Comput.* 2021, **17**, 1701–1714,
DOI 10.1021/acs.jctc.0c01306.

Projection algorithm reference (the parameter-free variant hartree implements):
W. H. Miller, N. C. Handy and J. E. Adams, "Reaction path Hamiltonian for
polyatomic molecules", *J. Chem. Phys.* 1980, **72**, 99–112 — projection of the
(normalized, mass-weighted) gradient direction out of the Cartesian Hessian
alongside the Eckart translation/rotation vectors.

**No numeric parameters are vendored.** hartree's SPH is parameter-free: it adds
the mass-weighted gradient direction to the Eckart projector. This intentionally
**differs from xtb `--bhess`**, which adds an RMSD-based Gaussian biasing
potential with method-specific `kpush`/`alpha` strengths and computes the
Hessian of the biased surface. hartree does not vendor those bias parameters and
makes no claim to reproduce xtb `--bhess` numbers; it implements the documented
gradient-projection treatment, which shares the intent (meaningful frequencies
from a Hessian at a non-stationary geometry) and the key property that at a true
minimum (g ≈ 0) it reduces exactly to the ordinary harmonic analysis. The
`SPH_GRADIENT_NORM_FLOOR = 1e-3` is an engineering threshold (mass-weighted
gradient 2-norm) separating a converged-optimization residual from a chemically
displaced gradient, documented in the source; it is not a published parameter.

The existing RRHO/mRRHO machinery (`src/thermo.rs`) is unchanged; its
provenance (Grimme, *Chem. Eur. J.* 2012, 18, 9955 for the quasi-RRHO entropy
interpolation) is documented inline in that module.
