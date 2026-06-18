use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::integral::Basis as IntegralBasis;
use hartree::integrals::{ConventionalProvider, InCoreEri};

fn atom(sym: &str, position: [f64; 3]) -> Atom {
    Atom::new(Element::from_symbol(sym).unwrap(), position)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            atom("O", [0.0, -0.143225816552, 0.0]),
            atom("H", [1.638036840407, 1.136548822547, 0.0]),
            atom("H", [-1.638036840407, 1.136548822547, 0.0]),
        ],
        0,
        1,
    )
}

fn h2() -> Molecule {
    Molecule::new(
        vec![atom("H", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 1.4])],
        0,
        1,
    )
}

fn integral_basis(mol: &Molecule, basis: &str) -> IntegralBasis {
    BasisSet::load(basis)
        .unwrap()
        .build(mol)
        .unwrap()
        .into_integral()
}

fn ao_to_shell(basis: &IntegralBasis) -> Vec<usize> {
    let mut map = Vec::new();
    for (s, shell) in basis.shells().iter().enumerate() {
        for _ in 0..shell.n_func() {
            map.push(s);
        }
    }
    map
}

fn canon(a: usize, b: usize) -> (usize, usize) {
    if a >= b { (a, b) } else { (b, a) }
}

fn pair_ge(p: (usize, usize), q: (usize, usize)) -> bool {
    p.0 > q.0 || (p.0 == q.0 && p.1 >= q.1)
}

fn assert_matches_reference(name: &str, mol: &Molecule, basis: &str) {
    let ox = integral_basis(mol, basis);

    let shell_of = ao_to_shell(&ox);
    let nao = ox.nao();
    let reference = ox.eri();

    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ox, charges);
    let candidate = provider.ao_eri();

    assert_eq!(
        candidate.len(),
        reference.len(),
        "{name}/{basis}: length mismatch"
    );
    assert_eq!(reference.len(), nao.pow(4), "{name}/{basis}: not nao⁴");

    let peak = reference.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let floor = 1e-3 * peak;
    let mut worst_sig = 0.0_f64;
    let mut worst_abs = 0.0_f64;

    for mu in 0..nao {
        for nu in 0..nao {
            let bp = canon(shell_of[mu], shell_of[nu]);
            for la in 0..nao {
                for sg in 0..nao {
                    let idx = ((mu * nao + nu) * nao + la) * nao + sg;
                    let (r, c) = (reference[idx], candidate[idx]);
                    let dv = (r - c).abs();
                    worst_abs = worst_abs.max(dv);
                    if r.abs() >= floor {
                        worst_sig = worst_sig.max(dv / r.abs());
                    }
                    let kp = canon(shell_of[la], shell_of[sg]);
                    if pair_ge(bp, kp) {
                        assert_eq!(
                            r.to_bits(),
                            c.to_bits(),
                            "{name}/{basis}: non-swapped element ({mu}{nu}|{la}{sg}) \
                             must be bit-identical: ref={r:e} cand={c:e}"
                        );
                    }
                }
            }
        }
    }

    assert!(
        worst_sig < 1e-11,
        "{name}/{basis}: worst significant-element relative diff {worst_sig:e} exceeds 1e-11"
    );
    assert!(
        worst_abs < 1e-11 * peak.max(1.0) + 1e-12,
        "{name}/{basis}: worst absolute diff {worst_abs:e} exceeds floor (peak {peak:e})"
    );
}

#[test]
fn parallel_eri_matches_serial() {
    assert_matches_reference("h2", &h2(), "sto-3g");
    assert_matches_reference("h2", &h2(), "6-31g");
    assert_matches_reference("water", &water(), "sto-3g");
    assert_matches_reference("water", &water(), "6-31g");
}

#[test]
#[ignore]
fn parallel_eri_matches_serial_ccpvdz() {
    assert_matches_reference("water", &water(), "cc-pvdz");
}
