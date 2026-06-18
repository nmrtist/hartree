# Vendored basis-set data — provenance

These JSON files are verbatim responses from the **Basis Set Exchange** (BSE) REST API,
embedded into `hartree` at compile time via `include_str!`.

- **Source:** https://www.basissetexchange.org
- **API endpoint:** `GET /api/basis/{name}/format/json/?elements=1,2,…,N`
- **Elements:** H–Ar (Z = 1–18) for the Pople/Dunning/STO sets; **H–Kr (Z = 1–36,
  all-electron)** for the entire def2 family (no ECPs are needed below Kr)
- **Retrieved:** 2026-06-07 (sto-3g, 6-31g, cc-pvdz, def2-svp); 2026-06-08 (6-311G family;
  cc-pvtz, def2-tzvp, aug-cc-pvtz; cc-pvqz, def2-qzvp); 2026-06-11 (def2-tzvpp,
  def2-qzvpp, def2-tzvpd, def2-svpd; def2-universal-jkfit; def2-mtzvpp);
  2026-06-12 (def2-tzvppd, and the re-download of every def2 file with
  `elements=1,…,36` — the H–Ar `electron_shells` data of the refreshed files is
  identical to the previously vendored responses, verified field-by-field before
  replacement); 2026-06-12 (def2-svp-c, def2-tzvp-c — the MP2-fit /C auxiliary
  sets, BSE names `def2-SVP-RIFIT` / `def2-TZVP-RIFIT`, `elements=1,…,36`)
- **Format:** native BSE JSON (numeric values are strings, full precision preserved)

| File | Basis set | BSE name / version | Convention used by hartree |
|------|-----------|--------------------|--------------------------|
| `sto-3g.json`         | STO-3G        | STO-3G / 1        | Cartesian |
| `6-31g.json`          | 6-31G         | 6-31G / 1         | Cartesian |
| `6-311g.json`         | 6-311G        | 6-311G / 0        | spherical (5d) |
| `6-311g(d,p).json`    | 6-311G(d,p)   | 6-311G(d,p) / 0   | spherical (5d) |
| `6-311+g(d,p).json`   | 6-311+G(d,p)  | 6-311+G** / 0     | spherical (5d) |
| `6-311++g(d,p).json`  | 6-311++G(d,p) | 6-311++G** / 0    | spherical (5d) |
| `cc-pvdz.json`        | cc-pVDZ       | cc-pVDZ / 1       | spherical (pure) |
| `cc-pvtz.json`        | cc-pVTZ       | cc-pVTZ / 1       | spherical (pure, f) |
| `cc-pvqz.json`        | cc-pVQZ       | cc-pVQZ / 1       | spherical (pure, g) |
| `def2-svp.json`       | def2-SVP      | def2-SVP / 1      | spherical (pure) |
| `def2-tzvp.json`      | def2-TZVP     | def2-TZVP / 1     | spherical (pure, f) |
| `def2-qzvp.json`      | def2-QZVP     | def2-QZVP / 1     | spherical (pure, g) |
| `aug-cc-pvtz.json`    | aug-cc-pVTZ   | aug-cc-pVTZ / 1   | spherical (pure, f, diffuse) |
| `def2-tzvpp.json`     | def2-TZVPP    | def2-TZVPP / 1    | spherical (pure, f) |
| `def2-qzvpp.json`     | def2-QZVPP    | def2-QZVPP / 1    | spherical (pure, g) |
| `def2-tzvpd.json`     | def2-TZVPD    | def2-TZVPD / 1    | spherical (pure, f, diffuse) |
| `def2-tzvppd.json`    | def2-TZVPPD   | def2-TZVPPD / 1   | spherical (pure, f, diffuse) |
| `def2-svpd.json`      | def2-SVPD     | def2-SVPD / 1     | spherical (pure, diffuse) |
| `def2-mtzvpp.json`    | def2-mTZVPP   | def2-mTZVPP / 1   | spherical (pure, max l = d) |
| `def2-msvp.json`      | def2-mSVP     | (not on BSE — Psi4 distribution; see below) | spherical (pure) |
| `def2-mtzvp.json`     | def2-mTZVP (mTZVP) | (not on BSE — Psi4 distribution; see below) | spherical (pure, max l = d for H–Ar) |
| `def2-universal-jkfit.json` | def2-universal-JKFIT (**auxiliary**) | def2-universal-JKFIT / 1 | spherical (pure, g) |
| `def2-svp-c.json`     | def2-SVP/C (**auxiliary**, MP2-fit)  | def2-SVP-RIFIT / 1  | spherical (pure) |
| `def2-tzvp-c.json`    | def2-TZVP/C (**auxiliary**, MP2-fit) | def2-TZVP-RIFIT / 1 | spherical (pure) |

**Auxiliary sets.** `def2-universal-jkfit` is a density-fitting (RI-JK) expansion set, not
an orbital basis: it loads only through `hartree_basis::BasisSet::load_aux` and is rejected
as `--basis`. It is the universal Coulomb/exchange fitting set for the def2 family
(Weigend 2008), acceptable for the Pople/Dunning orbital sets at the HF/DFT level too.
Max angular momentum is g (l = 4) for every H–Ar element. The S-quadrature gate does not
apply (aux functions are not orbitals and never enter the DFT grid).

`def2-svp-c.json` / `def2-tzvp-c.json` are the def2 **MP2-fit (/C, RI-C)** correlation
fitting sets (Weigend, Köhn, Hättig, *J. Chem. Phys.* 116, 3175 (2002); Hättig, *Phys.
Chem. Chem. Phys.* 7, 59 (2005); Hellweg, Hättig, Höfener, Klopper, *Theor. Chem. Acc.*
117, 587 (2007) for Rb–Rn analogues), vendored verbatim from BSE under the names
`def2-SVP-RIFIT` and `def2-TZVP-RIFIT` and exposed by hartree as `def2-svp/c` /
`def2-tzvp/c` through `load_aux`. They expand occupied×virtual orbital products for
RI-MP2 (and double-hybrid PT2) and are **not** interchangeable with the JK fitting set;
hartree pairs each with its like-named orbital basis and refuses to substitute the jkfit
set silently.

The spherical/Cartesian choice is hartree's, applied at build time
(`hartree_basis::BasisSet::spherical`, via `bse::default_spherical`). Minimal and
double-zeta Pople/STO sets (STO-3G, 6-31G*) are Cartesian (6d); the **6-311G family is
spherical (5d)** — Pople's 1980 original definition and the Gaussian/ORCA default — as
are the correlation-consistent (cc-*) and Karlsruhe (def2-*) families. The triple-zeta
sets (`cc-pvtz`, `def2-tzvp`, `aug-cc-pvtz`) carry f functions (l = 3) and the
quadruple-zeta sets (`cc-pvqz`, `def2-qzvp`) carry g functions (l = 4); all H–Ar are
defined for each. `aug-cc-pvtz` adds a diffuse function per angular momentum. The
Karlsruhe energy bases `def2-tzvpp` (f) and `def2-qzvpp` (g) and the diffuse-augmented
property bases `def2-tzvpd` (f, diffuse) and `def2-svpd` (diffuse) define all H–Ar with
no exceptions.

**def2-mTZVPP.** The *modified* TZVPP defined for the r²SCAN-3c composite method
(Grimme, Hansen, Ehlert, Mewes, J. Chem. Phys. 154, 064103 (2021)): a TZVPP-derived set
with reduced polarization (max l = d for every H–Ar element — no f functions). It is a
regular user-selectable orbital basis; r²SCAN-3c (`--method r2scan-3c`) implies it.

**def2-mSVP.** The *modified* SVP defined for the PBEh-3c composite method (Grimme,
Brandenburg, Bannwarth, Hansen, J. Chem. Phys. 143, 054107 (2015) / TURBOMOLE), also
used by HSE-3c and B3LYP-3c. It is **not hosted on BSE**; the data (`def2-msvp.json`,
retrieved 2026-06-12) was transcribed mechanically from the authoritative **Psi4
distribution** file `psi4/share/psi4/basis/def2-msvp.gbs` (github.com/psi4/psi4,
master), which encodes the published composition: H = def2-SVP s set with exponents
scaled by 1.2² = 1.44 (scale factor 1.2 on the Slater exponent) and the p polarization
removed; He–Be = def2-SVP; B–Ne = the old Ahlrichs DZ s core (one extra
single-primitive 1s relative to def2-SVP) with the def2-SVP (B, Ne) / 6-31G* (C–F)
polarization; Na–Kr = def2-SV(P). Only format conversion (Gaussian94 → BSE JSON, H–Kr)
was performed; every exponent and contraction coefficient is verbatim. No value was
fitted or guessed. It is a regular user-selectable orbital basis; the `b3lyp-3c`
composite implies it.

**def2-mTZVP (mTZVP).** The *modified* TZVP defined for the B97-3c composite method
(Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, 064104 (2018), Sec. II B:
"mTZVP", shipped with Turbomole/ORCA). It is **not hosted on BSE**; the data
(`def2-mtzvp.json`, retrieved 2026-06-12) was transcribed mechanically from the
authoritative **Psi4 distribution** file `psi4/share/psi4/basis/def2-mtzvp.gbs`
(github.com/psi4/psi4, master; header: "exported from ORCA; converted to Gaussian fmt
by MolSSI BSE tool"), elements H–Kr (the file continues to Rn with ECPs, which hartree
does not vendor — the Z > 36 range is outside the all-electron H–Kr convention of the
modified sets). Composition relative to def2-TZVP per the paper: hydrogen reduced to a
3s set with refitted exponents and no polarization; heavy-atom polarization reduced
(max l = d for H–Ar; no f functions). The spherical basis-function counts match the
reference implementation `mctc-gcp`'s `BASdef2mtzvp` table (H = 3, C = 19, S = 22, …).
Only format conversion (Gaussian94 → BSE JSON) was performed; every exponent and
contraction coefficient is verbatim. No value was fitted or guessed. It is a regular
user-selectable orbital basis (alias `mtzvp`); the `b97-3c` composite implies it.

**6-311G family naming.** The diffuse-and-polarized members are exposed under their
`(d,p)` Pople synonyms (`6-311+g(d,p)`, `6-311++g(d,p)`); the vendored JSON is BSE's
equivalent double-star set (`6-311+G**` = `6-311+G(d,p)`, `6-311++G**` =
`6-311++G(d,p)` — `**` ⇔ d on heavy atoms + p on H ⇔ `(d,p)`). `6-311++G(d,p)` does
**not** define He (BSE omits it); all other H–Ar elements are present.

**def2 element range (H–Kr, plus an ECP heavy-element subset).** Every def2 file
(orbital sets, `def2-mtzvpp`, and `def2-universal-jkfit`) carries elements 1–36. The
def2 sets are all-electron through Kr. Beyond Kr, def2 is defined with the def2-ECP
effective core potentials: hartree vendors a representative heavy-element subset for
`def2-svp` and `def2-tzvp` (see the ECP section below). For Z = 19–36 the orbital sets
stay within l ≤ 4 (g); `def2-universal-jkfit` reaches l = 6 (i) on the transition
metals, which the integral engine's `MAX_L = 6` supports (aux functions never need
gradients, where the limit is `MAX_L − 1`).

**ma-def2 sets (no JSON files).** `ma-def2-SVP` and `ma-def2-TZVP` are **not** on the
Basis Set Exchange; hartree derives them from the vendored parents at load time by
Truhlar's exact published prescription (Zheng, Xu, Truhlar, *Theor. Chem. Acc.* 128,
295 (2011)): one diffuse s and one diffuse p single-primitive function on every atom
except hydrogen, with exponent equal to one third of the smallest parent exponent of
the same angular momentum. No exponent is fitted or guessed.

## def2-ECP and the heavy-element (Z > 36) def2 subset

- **Files** (retrieved **2026-06-12**, verbatim BSE REST responses):
  - `def2-svp.heavy.json` — `GET /api/basis/def2-svp/format/json/?elements=47,50,53,79`
  - `def2-tzvp.heavy.json` — `GET /api/basis/def2-tzvp/format/json/?elements=47,50,53,79`
  - `../ecp/def2-ecp.json` — `GET /api/basis/def2-ecp/format/json/?elements=47,50,53,79`
    (BSE name **def2-ECP / 1**, "Data from Turbomole 7.3")
- **Elements:** Ag (47), Sn (50), I (53), Au (79) — a representative subset
  (4d transition metal, 5p main group ×2, 5d row-6 metal); extending to the
  full Rb–Rn range is a data-only re-download with a wider `elements=` list.
- **Layout choice.** The heavy elements live in *separate* `.heavy.json` files
  merged at load time, so the established all-electron H–Kr files stay
  byte-identical (no light-element drift risk) and extension is mechanical.
- **ECPs:** Ag/Sn/I use 28-core potentials (ECP28: MWB for Ag — Andrae,
  Häußermann, Dolg, Stoll, Preuß, Theor. Chim. Acta 77, 123 (1990) — MDF for
  Sn/I), Au the 60-core ECP60MDF; max angular momentum of the orbital sets on
  these elements is f (l = 3) and the ECP local channel is f, both inside the
  integral engine's validated l ≤ 4 range.
- **Conventions (verified by hand against the published Ag ECP28MWB table):**
  the BSE block with the highest angular momentum is the local channel `U_L`;
  lower channels are the already-subtracted differences `U_l − U_L` (the
  `∓U_L` primitives appear verbatim in each row); a BSE `r_exponents` value
  `k` means an `r^{k−2}` prefactor — identical to the integral engine's
  `EcpPrimitive::n` convention, so values pass through unchanged (the whole
  def2-ECP tabulation is pure Gaussians, `n = 2`). See `src/ecp.rs`.
- **Spin–orbit terms are ignored** (scalar-relativistic ECPs only); the BSE
  def2-ECP data for these elements contains none, and any non-`scalar_ecp`
  block is rejected at parse time rather than mis-summed.
- The `def2-svp.heavy.json` / `def2-tzvp.heavy.json` responses also embed the
  same ECP blocks per element; hartree reads ECP parameters only from
  `def2-ecp.json` (verified field-identical to the embedded copies at
  download time) and ignores the embedded ones.

## Refreshing or extending

Re-download (PowerShell), e.g. to widen the element range:

```powershell
$els = (1..36) -join ','
Invoke-WebRequest "https://www.basissetexchange.org/api/basis/cc-pvdz/format/json/?elements=$els" `
  -OutFile data/basis/cc-pvdz.json
```

Always cite the literature references inside each JSON (the `references` field)
when reporting results obtained with these basis sets.
