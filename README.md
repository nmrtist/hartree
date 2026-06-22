# hartree

`hartree` is a quantum chemistry and solid-state physics software package implemented in Rust, with a shared library API and CLI.

- No C or Fortran dependencies in the hartree workspace.
- Molecular methods and periodic methods live in the same release surface.
- Validation is based on committed reference data, so build and test do not require external programs.

Core dependencies:
[`integral`](https://crates.io/crates/integral) for molecular and periodic integrals, [`xcx`](https://crates.io/crates/xcx) for exchange-correlation functionals, and [`faer`](https://crates.io/crates/faer) for dense linear algebra.

## Fundamental Capabilities

### Molecular

| Area | Notes |
|---|---|
| **SCF** | RHF / UHF / ROHF, DIIS, level shift, GWH guess, in-core / direct / RI-JK backends |
| **Post-HF** | MP2, CCSD, CCSD(T) for closed-shell RHF references |
| **DFT** | RKS / UKS with LDA, GGA, tau-meta-GGA, and global hybrids |
| **Gradients & optimization** | Analytic RHF/UHF and KS gradients, redundant-internal-coordinate optimization, finite-difference fallback where needed |
| **Transition states & reaction paths** | P-RFO and dimer saddle search in mass-weighted or redundant-internal coordinates, single-geometry / two-endpoint (IDPP and climbing-image NEB) / distinguished-coordinate-scan guessing, harmonic saddle verification, and IRC tracing (DVV / GS2 / EulerPC) |
| **Properties** | dipole, Mulliken/Lowdin charges, Mayer bond orders, T1 and HOMO-LUMO diagnostics |
| **Frequencies** | RHF numerical Hessian, harmonic frequencies, RRHO thermochemistry |
| **Dispersion & composites** | D3(BJ), D4, gCP, SRB, and `r2scan-3c`, `b3lyp-3c`, `b97-3c`, `pbeh-3c` |
| **Solvation** | C-PCM, SMD, ALPB, GBSA, and `.cosmo` export for downstream COSMO-RS workflows |
| **Special workflows** | ghost atoms, counterpoise correction, Fermi smearing, FOD, X2C, COSX |

### Periodic

Periodic solid-state DFT is available through the `periodic` module, the top-level `PeriodicJob` / `run_periodic` APIs, and the CLI periodic path.

Current periodic support includes:

- GPW with GTH pseudopotentials and GTH basis sets
- spin-restricted fixed-occupation insulating and semiconducting systems
- LDA grid XC (`pade` or `lda`)
- Gamma and Monkhorst-Pack k-point meshes
- total energy, analytic forces, analytic stress, bands, and DOS

## Validation

hartree is validated against committed oracle data from ORCA, PySCF, dftd4, mctc-gcp, and CP2K where applicable. The reference tables under [`tests/ref/`](tests/ref) and the integration test suite are the release record.

Recommended validation commands:

```sh
cargo build --release
cargo test --workspace --release
cargo test --workspace --release -- --ignored
```

Recent stable Rust is required (`edition = 2024`, MSRV `1.87`).

## Quick Start

### Molecular CLI

```sh
# RHF single point
hartree water.xyz --basis cc-pvdz --method rhf

# Open-shell UHF
hartree oh.xyz --basis cc-pvdz --method uhf --multiplicity 2

# Closed-shell CCSD
hartree water.xyz --basis cc-pvdz --method ccsd

# Kohn-Sham DFT
hartree water.xyz --basis cc-pvdz --method pbe --grid 3

# Optimization followed by properties and frequencies
hartree water.xyz --basis cc-pvdz --method rhf --opt
hartree water.xyz --basis cc-pvdz --method rhf --properties --freq --symmetry-number 2

# Transition-state search from a single guess, confirmed by an IRC trace
hartree ts_guess.xyz --basis def2-svp --method rhf --ts --ts-irc

# Double-ended search from reactant and product endpoints via a climbing-image NEB band
hartree reactant.xyz --basis def2-svp --method rhf --ts --ts-product product.xyz --ts-neb

# Dispersion, RI-JK, and solvent
hartree water.xyz --basis def2-svp --method b3lyp-d3 --ri --solvent water

# SMD
hartree water.xyz --basis def2-svp --method pbe --smd water

# ALPB or GBSA single points
hartree water.xyz --basis def2-svp --method pbe --alpb water
hartree water.xyz --basis def2-svp --method pbe --gbsa water

# 3c composite
hartree water.xyz --method r2scan-3c
```

### Periodic CLI

Bulk Si example (`si8.xyz`, coordinates and lattice in angstrom):

```text
8
Si conventional cubic diamond; Lattice="5.4306975 0 0 0 5.4306975 0 0 0 5.4306975"
Si 0.000000000 0.000000000 0.000000000
Si 0.000000000 2.715348750 2.715348750
Si 2.715348750 0.000000000 2.715348750
Si 2.715348750 2.715348750 0.000000000
Si 1.357674375 1.357674375 1.357674375
Si 1.357674375 4.073023125 4.073023125
Si 4.073023125 1.357674375 4.073023125
Si 4.073023125 4.073023125 1.357674375
```

```sh
hartree si8.xyz --cell file --basis DZVP-GTH-PADE --xc pade --kpoints gamma --cutoff 300 --forces
```

The validated CP2K reference for this 8-atom Gamma-point setup is
`E_total ~= -31.297820 Ha`; hartree tracks the same setup to grid / SCF tolerance.

Run `hartree --help` for the full CLI surface.

## Library Surface

The `hartree` crate is the main entry point for library users.

- `Job` / `JobResult` drive molecular workflows.
- `PeriodicJob` / `run_periodic` drive periodic GPW workflows.
- Public submodules expose the major subsystems: `core`, `basis`, `integrals`, `scf`, `grad`, `cc`, `dft`, `disp`, `solv`, `periodic`, and others.

## Current Boundaries

- **Post-HF** is closed-shell, RHF-reference only; no open-shell MP2/CCSD/CCSD(T), and no post-HF on Kohn-Sham orbitals.
- **Analytic gradients** cover RHF/UHF and KS LDA/GGA/tau-meta-GGA/global hybrids; post-HF, ROHF, and solvated optimizations fall back where documented.
- **Frequencies** are RHF-only and use a numerical Hessian of the analytic gradient.
- **Relativistic scope** is all-electron through Kr; selected heavier elements are available with def2-ECP support and the documented guardrails.
- **DFT scope** is LDA/GGA/tau-meta-GGA/global-hybrid; no range-separated hybrids, VV10 optimization/frequencies, or ROKS.
- **RI-JK** is energy-only; no RI-MP2 or DF gradients.
- **Solvation** has no analytic solvation gradients; solvated `--freq` is rejected, and `.cosmo` export is for downstream COSMO-RS workflows rather than a built-in COSMO-RS solver.
- **Periodic DFT** does not yet cover metals, smearing, spin polarization, periodic exact exchange/hybrids, GGA grid-gradient XC, or high-level cell optimization workflows.

## Basis Sets And Data

Molecular orbital and auxiliary basis data are vendored from the [Basis Set Exchange](https://www.basissetexchange.org) as native JSON and compiled into the workspace. The shipped families cover minimal, Pople, correlation-consistent, and def2 basis sets, including the auxiliary `def2-universal-JKFIT` set and the GTH periodic basis / pseudopotential data.

Retrieval details and provenance for the molecular basis library are documented
in [`crates/hartree/src/basis/data/basis/PROVENANCE.md`](crates/hartree/src/basis/data/basis/PROVENANCE.md).

## Library Layout

The `hartree` library is organized into modules, with `hartree-cli` providing the
command-line driver:

| Module | Responsibility |
|---|---|
| `core` | molecule, periodic table, units, constants |
| `linalg` / `tensor` | dense linear algebra and contraction kernels |
| `basis` | molecular basis and GTH basis/pseudopotential loading |
| `integrals` | molecular and periodic integral access seam |
| `scf` | RHF/UHF/ROHF and method-agnostic SCF driver |
| `dft` | molecular KS DFT |
| `periodic` | periodic GPW DFT |
| `cc` | MP2, CCSD, CCSD(T) |
| `grad` / `opt` / `props` | gradients, optimization, transition-state search / IRC, properties, frequencies |
| `disp` / `solv` | dispersion, composite corrections, and solvent models |
| `ext` | external-program interfaces (xtb, CREST) and conformer generation |

## License

Dual-licensed under either [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
