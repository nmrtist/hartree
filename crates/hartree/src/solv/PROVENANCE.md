# Parameter provenance — `solv` module

## C-PCM (electrostatics) — `src/surface.rs`, `src/lib.rs`
- SWIG cavity discretization, modified-Bondi radii × 1.2, York–Karplus ξ
  constants, switching function: PySCF 2.13.1 `pyscf/solvent/pcm.py`
  (Apache-2.0) and the cited primary literature (Lange & Herbert,
  J. Chem. Phys. 133, 244111 (2010); York & Karplus, J. Chem. Phys. 122,
  194110 (2005)). See module docs for the per-constant citations.

## SMD (universal solvation model) — `src/smd.rs`
Reference: Marenich, Cramer, Truhlar, *J. Phys. Chem. B* **113**, 6378 (2009)
("SMD"). All numeric tables below were transcribed from, and cross-checked
against, the authoritative Minnesota `MNSOL` Fortran implementation
(`gpu4pyscf/lib/solvent/mnsol.F`, redistributed by the PySCF/GPU4PySCF
projects), which encodes the published SMD parameterization, and the
GPU4PySCF Python port `gpu4pyscf/solvent/smd.py` (tag v0.6.17).

### Intrinsic Coulomb radii (Table 3 of the paper) — `smd_coulomb_radius`
H 1.20, C 1.85, N 1.89, F 1.73, Si 2.47, P 2.12, S 2.49, Cl 2.38 Å.
Oxygen is solvent-dependent (hydrogen-bond-acidity rule, eq. 16):
`r_O = 1.52` for α ≥ 0.43 (e.g. water), else `1.52 + 1.8·(0.43 − α)`.
Source: MNSOL `smd_radii` / `mnsol.F`. Elements without an SMD value fall
back to Bondi radii (H–Ar), matching MNSOL's `VDWRAD` over that range.

### Atomic surface-tension coefficients σ̃ — `tensions`
- Aqueous set (Table 4): MNSOL `SMD_CDS_AQ` (`SIGMA_DATA`, `HSIGMA_DATA`).
- Nonaqueous set σ̃ = σ̃ⁿ·n + σ̃ᵅ·α + σ̃ᵝ·β (Tables 5–6): MNSOL `SMD_CDS_NAQ`
  (`SIGMA_N`, `SIGMA_A`, `SIGMA_B`, `HSIGMA_N`).
- Molecular surface tension σ[M] = 0.35·γ − 4.19·φ² − 6.68·ψ² (+0·β²)
  (Table 6): MNSOL `SIGMA_MOL` = [0.35, 0.00, −4.19, −6.68].

### COT switching functions T(R; r̄, ΔR) — `cot`, `atomic_surface_tensions`
Functional forms and the (r̄, ΔR) distance parameters (Table 7) transcribed
from MNSOL `SMXCDS`: H–C/H–O (1.55, 0.30); C–C (1.84, 0.30); C–N inner-sum
r̄_XC row of `RKKVAL`; N–C (1.84, 0.30) with the inner C–X factor and the
^1.3 / ²  exponents; N–C(3) triple-bond (1.225, 0.065); O–C (1.33, 0.10);
O–N (1.50, 0.30); O–O (1.80, 0.30); O–P (2.10, 0.30). Only the bond terms
used by SMD are implemented (the SM6-only and unused terms in MNSOL are
omitted).

### SASA — `sasa`, `cds_energy`
Solvent (probe) radius 0.4 Å added to Bondi/Mantina vdW radii (MNSOL
`SOLVRD = 0.4`, `VDWRAD`). Areas computed on switching-smoothed Lebedev
spheres (default 590 points/sphere) — the same SWIG discretization as the
C-PCM cavity, so SASA is exact (4π(r+r_s)²) for an isolated sphere and a
smooth function of geometry.

### Solvent descriptors `[n, α, β, γ, ε, φ, ψ]` — `SMD_SOLVENTS`
Minnesota Solvent Descriptor Database (Winget, Dolney, Giesen, Cramer,
Truhlar; `mnsddb.pdf`), values as redistributed in PySCF
`pyscf/solvent/smd.py`. The 20 bundled solvents are a subset of that
database (water + common organics); the unused `n25` column is dropped.

## ALPB / GBSA (generalized-Born implicit solvation) — `src/gbsa.rs`, `src/gbsa_params.rs`
Model references: Ehlert, Stahn, Spicher, Grimme, *J. Chem. Theory Comput.*
**17**, 4250 (2021) (ALPB); the GFN-xTB GBSA generalized-Born/SASA model; the
Born-radii integrator (GBOBC) and the smooth SASA of Im, Lee & Brooks,
*J. Comput. Chem.* **24**, 1691 (2003); the P16 kernel of Lange & Herbert,
*J. Chem. Theory Comput.* **8**, 1999 (2012).

All formulae and **every numeric parameter** were transcribed from the
open-source `xtb` program (Grimme group, github.com/grimme-lab/xtb, LGPL-3.0):
- Energy/Born/SASA/kernel code: `src/solv/{gbsa,born,kernel,sasa,model,input}.f90`.
- Solvent parameter sets: `include/param_{alpb,gbsa}_<solvent>.fh`, the
  **GFN2-xTB** (`gfn2_*`) blocks. `src/gbsa_params.rs` is auto-generated from
  these files (38 sets: 25 ALPB + 13 GBSA solvents); the scalars
  (`epsv, smass, rhos, c1, rprobe, gshift, soset, alpha`) and the three
  per-element arrays (`gamscale, sx, tmp`, Z = 1..=94) are copied verbatim.
- D3 van-der-Waals radii (`vanDerWaalsRadD3`): `src/param/vdwradd3.f90`.
- Fixed constants: ALPB α = 0.571412; P16 ζ = 1.028; GBOBC (α,β,γ) =
  (1.0, 0.8, 4.85); dielectric smoothing w = 0.3 Å; surface-tension unit
  `gamscale·4π·1e-5`; H-bond strength `−tmp²·kcaltoau`.

**Parameterization caveat (provenance, not invention).** xtb's ALPB/GBSA
parameters were fit for the GFN2-xTB Hamiltonian's atomic (Mulliken/EEQ)
charges. hartree is ab initio, so the model is shipped **clearly labeled with its
GFN2 provenance** and applied as a **post-SCF** correction on the converged SCF
Mulliken charges — the same caveat ORCA documents for its ALPB-style models. No
ab-initio-specific parameter set exists; per the task directive we ship the
machinery + the published xtb GFN2 set rather than inventing parameters. The
SASA quadrature uses hartree's 194-point Lebedev grid (xtb's default is 230;
hartree does not ship that order — a negligible, documented difference).

## COSMO file export (`.cosmo`) — `src/cosmo.rs`
Format: the TURBOMOLE/COSMOtherm `.cosmo` layout as written by xtb's
`writeCosmoFile` (`src/solv/cosmo.f90`, LGPL-3.0) and read by openCOSMO-RS
parsers: the `$info / $cosmo / $cosmo_data / $coord_rad / $screening_charge /
$cosmo_energy / $segment_information` blocks. Written from a C-PCM run in the
ideal-conductor limit ε = ∞ (COSMO-RS convention, `fepsi = ½`). Documented
deviations: segments come from hartree's SWIG Lebedev surface (not a TURBOMOLE
segmented cavity); segment potentials are in atomic units; the COSMO-data-free
`$coord_car` block is omitted.

## SMD standard state
SMD ΔG_solv is reported at 298 K with the fixed-concentration convention
(1 mol/L ideal gas → 1 mol/L ideal solution); no `RT·ln(24.46)` 1 atm→1 M
term is added (per the SMD paper's convention).
