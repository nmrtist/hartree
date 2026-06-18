# External interfaces (`ext`) — format & algorithm provenance

This module wraps external programs and implements a fallback conformer
generator. Every input/output format and every algorithm is sourced below.
No format or parameter is guessed; the subprocess parsers are tested only
against fixtures captured from the documented formats.

## xtb interface (`src/xtb.rs`)

Target program: **xtb** (Grimme group), https://github.com/grimme-lab/xtb,
documentation https://xtb-docs.readthedocs.io. Fixtures correspond to the
xtb 6.x output formats (commit-stable since 6.4; the fixtures are labelled
xtb 6.6.1).

- **CLI flags.** Method selection `--gfn 2` (GFN2-xTB) / `--gfnff` (GFN-FF);
  net charge `--chrg <int>`; number of *unpaired* electrons `--uhf <N>` where
  `N = multiplicity − 1`; ALPB implicit solvation `--alpb <solvent>`; energy+
  gradient `--grad`; optimization `--opt`; structured output `--json`.
  Source: xtb-docs "Command Line Options" and `xtb --help`.
- **TURBOMOLE `gradient` file** (written by `--grad`). Layout:
  a `$grad` line; a header line
  ` cycle = <n>    SCF energy = <Eh>   |dE/dxyz| = <norm>`;
  then N coordinate lines (`x y z element`, coordinates in **bohr**, element
  symbol lowercase); then N gradient lines (`gx gy gz`, hartree/bohr, Fortran
  `D`/`E` exponent); a closing `$end`. The TURBOMOLE gradient format is the
  long-standing convention shared by TURBOMOLE and xtb. Parser:
  `parse_turbomole_gradient`. The parser accepts both `D` and `E` exponents.
- **`xtbopt.xyz`** (written by `--opt`): a standard XYZ whose comment (second)
  line reads ` energy: <Eh> gnorm: <Eh/a0> xtb: <version>`. Parser:
  `parse_xtbopt_xyz` (reads the geometry via hartree's XYZ parser and the energy
  from the comment).
- **`xtbout.json`** (written by `--json`): JSON object with the key
  `"total energy"` (hartree) among others. Parser: `parse_json_energy`.
- **stdout total energy**: the boxed summary line
  `| TOTAL ENERGY    <Eh> Eh   |`. Parser: `parse_energy_stdout` (fallback when
  no `xtbout.json` is present).
- **Binary detection**: `HARTREE_XTB_PATH` (if it points at an existing file)
  else `xtb` on `PATH`. Absence → `ExtError::BinaryNotFound` with install
  guidance.

## CREST interface (`src/crest.rs`)

Target program: **CREST**, https://github.com/crest-lab/crest, documentation
https://crest-lab.github.io/crest-docs/. Fixture labelled CREST 2.12.

- **CLI flags.** `crest <input.xyz> --gfn2|--gfnff --chrg <int> --uhf <N>
  [--alpb <solvent>] [-T <threads>]`.
- **`crest_conformers.xyz`** (and `crest_rotamers.xyz`, `crest_best.xyz`):
  multi-frame XYZ; each frame's comment line is the conformer total energy in
  **hartree**, written in ascending energy order. Parser:
  `parse_crest_ensemble` → `Ensemble` with relative energies and Boltzmann
  weights.
- Binary detection: `HARTREE_CREST_PATH` or `PATH`.

## Kabsch RMSD (`src/kabsch.rs`)

W. Kabsch, *Acta Crystallogr. A* **32**, 922 (1976); **34**, 827 (1978). The
optimal rotation is obtained from the polar decomposition of the 3×3
cross-covariance matrix, computed via the eigendecomposition of `HᵀH` (analytic
3×3 Jacobi), with the standard `det(R) < 0` reflection correction. No external
SVD/linear-algebra dependency.

## Fallback conformer generator (`src/confgen.rs`)

A deterministic torsion-driving generator — **not** a CREST replacement.

- **Connectivity**: covalent-radius overlap, `r_ij < 1.3·(r_cov,i + r_cov,j)`,
  using hartree's Cordero (2008) covalent radii (the hartree connectivity
  convention, matching `hartree_grad`/optimizer bond perception).
- **Rotatable bonds**: acyclic, non-terminal, heavy-atom (Z > 1) single bonds.
  All detected bonds are treated as single (no bond-order perception — a
  documented limitation: conjugated systems are over-rotated, rings are not
  puckered).
- **Torsion grid**: staggered, default 3 positions/bond (0°/120°/240°, the
  sp³–sp³ staggered minima). The full grid is the Cartesian product
  `g^n_bonds`, capped at `ConfGenOptions::max_candidates` (default 2000) by
  reducing the driven-bond count (lowest-index bonds first).
- **Candidate geometry**: rigid Rodrigues rotation of the smaller fragment
  about each bond axis. No relaxation.
- **Clash filter**: reject if any non-bonded pair is closer than
  `clash_factor·(r_vdw,i + r_vdw,j)` (default factor 0.6). Van-der-Waals radii:
  A. Bondi, *J. Phys. Chem.* **68**, 441 (1964), with the conventional 1.20 Å
  for H; light main-group elements tabulated, heavier → 2.0 Å fallback.
- **Ranking/dedup**: caller-supplied single-point energy (kept out of this
  crate to avoid a dependency cycle); deduplicate by Kabsch RMSD
  (`rmsd_threshold_bohr`, default 0.1 bohr) together with an energy window
  (`energy_window_hartree`, default 1e-4 Eh).

Limits: rigid rotamers only (no relaxation), no ring/pyramidal-inversion
sampling, no bond-order perception. For quantitative work, re-optimize the
unique conformers.
