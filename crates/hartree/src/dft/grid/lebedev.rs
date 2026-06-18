#![allow(clippy::excessive_precision)]

pub const ORDERS: [usize; 13] = [6, 14, 26, 38, 50, 86, 110, 146, 170, 194, 302, 434, 590];

#[derive(Debug, Clone)]
pub struct LebedevGrid {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
}

impl LebedevGrid {
    pub fn new(npts: usize) -> Option<Self> {
        let table = generators(npts)?;
        let mut points = Vec::with_capacity(npts);
        let mut weights = Vec::with_capacity(npts);
        for &entry in table {
            gen_oh(entry, &mut points, &mut weights);
        }
        debug_assert_eq!(points.len(), npts, "Lebedev {npts}: wrong point count");
        Some(Self { points, weights })
    }
}

pub fn degree(npts: usize) -> Option<usize> {
    Some(match npts {
        6 => 3,
        14 => 5,
        26 => 7,
        38 => 9,
        50 => 11,
        86 => 15,
        110 => 17,
        146 => 19,
        170 => 21,
        194 => 23,
        302 => 29,
        434 => 35,
        590 => 41,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct Gen {
    code: u8,
    a: f64,
    b: f64,
    v: f64,
}

const fn g(code: u8, a: f64, b: f64, v: f64) -> Gen {
    Gen { code, a, b, v }
}

fn gen_oh(entry: Gen, points: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let Gen { code, a, b, v } = entry;
    let mut push = |p: [f64; 3]| {
        points.push(p);
        weights.push(v);
    };
    match code {
        0 => {
            let a = 1.0;
            push([a, 0.0, 0.0]);
            push([-a, 0.0, 0.0]);
            push([0.0, a, 0.0]);
            push([0.0, -a, 0.0]);
            push([0.0, 0.0, a]);
            push([0.0, 0.0, -a]);
        }
        1 => {
            let a = (0.5_f64).sqrt();
            push([0.0, a, a]);
            push([0.0, -a, a]);
            push([0.0, a, -a]);
            push([0.0, -a, -a]);
            push([a, 0.0, a]);
            push([-a, 0.0, a]);
            push([a, 0.0, -a]);
            push([-a, 0.0, -a]);
            push([a, a, 0.0]);
            push([-a, a, 0.0]);
            push([a, -a, 0.0]);
            push([-a, -a, 0.0]);
        }
        2 => {
            let a = (1.0_f64 / 3.0).sqrt();
            push([a, a, a]);
            push([-a, a, a]);
            push([a, -a, a]);
            push([-a, -a, a]);
            push([a, a, -a]);
            push([-a, a, -a]);
            push([a, -a, -a]);
            push([-a, -a, -a]);
        }
        3 => {
            let b = (1.0 - 2.0 * a * a).sqrt();
            push([a, a, b]);
            push([-a, a, b]);
            push([a, -a, b]);
            push([-a, -a, b]);
            push([a, a, -b]);
            push([-a, a, -b]);
            push([a, -a, -b]);
            push([-a, -a, -b]);
            push([a, b, a]);
            push([-a, b, a]);
            push([a, -b, a]);
            push([-a, -b, a]);
            push([a, b, -a]);
            push([-a, b, -a]);
            push([a, -b, -a]);
            push([-a, -b, -a]);
            push([b, a, a]);
            push([-b, a, a]);
            push([b, -a, a]);
            push([-b, -a, a]);
            push([b, a, -a]);
            push([-b, a, -a]);
            push([b, -a, -a]);
            push([-b, -a, -a]);
        }
        4 => {
            let b = (1.0 - a * a).sqrt();
            push([a, b, 0.0]);
            push([-a, b, 0.0]);
            push([a, -b, 0.0]);
            push([-a, -b, 0.0]);
            push([b, a, 0.0]);
            push([-b, a, 0.0]);
            push([b, -a, 0.0]);
            push([-b, -a, 0.0]);
            push([a, 0.0, b]);
            push([-a, 0.0, b]);
            push([a, 0.0, -b]);
            push([-a, 0.0, -b]);
            push([b, 0.0, a]);
            push([-b, 0.0, a]);
            push([b, 0.0, -a]);
            push([-b, 0.0, -a]);
            push([0.0, a, b]);
            push([0.0, -a, b]);
            push([0.0, a, -b]);
            push([0.0, -a, -b]);
            push([0.0, b, a]);
            push([0.0, -b, a]);
            push([0.0, b, -a]);
            push([0.0, -b, -a]);
        }
        5 => {
            let c = (1.0 - a * a - b * b).sqrt();
            push([a, b, c]);
            push([-a, b, c]);
            push([a, -b, c]);
            push([-a, -b, c]);
            push([a, b, -c]);
            push([-a, b, -c]);
            push([a, -b, -c]);
            push([-a, -b, -c]);
            push([a, c, b]);
            push([-a, c, b]);
            push([a, -c, b]);
            push([-a, -c, b]);
            push([a, c, -b]);
            push([-a, c, -b]);
            push([a, -c, -b]);
            push([-a, -c, -b]);
            push([b, a, c]);
            push([-b, a, c]);
            push([b, -a, c]);
            push([-b, -a, c]);
            push([b, a, -c]);
            push([-b, a, -c]);
            push([b, -a, -c]);
            push([-b, -a, -c]);
            push([b, c, a]);
            push([-b, c, a]);
            push([b, -c, a]);
            push([-b, -c, a]);
            push([b, c, -a]);
            push([-b, c, -a]);
            push([b, -c, -a]);
            push([-b, -c, -a]);
            push([c, a, b]);
            push([-c, a, b]);
            push([c, -a, b]);
            push([-c, -a, b]);
            push([c, a, -b]);
            push([-c, a, -b]);
            push([c, -a, -b]);
            push([-c, -a, -b]);
            push([c, b, a]);
            push([-c, b, a]);
            push([c, -b, a]);
            push([-c, -b, a]);
            push([c, b, -a]);
            push([-c, b, -a]);
            push([c, -b, -a]);
            push([-c, -b, -a]);
        }
        _ => unreachable!("invalid Lebedev gen_oh code {code}"),
    }
}

fn generators(npts: usize) -> Option<&'static [Gen]> {
    Some(match npts {
        6 => LD0006,
        14 => LD0014,
        26 => LD0026,
        38 => LD0038,
        50 => LD0050,
        86 => LD0086,
        110 => LD0110,
        146 => LD0146,
        170 => LD0170,
        194 => LD0194,
        302 => LD0302,
        434 => LD0434,
        590 => LD0590,
        _ => return None,
    })
}

#[rustfmt::skip]
const LD0006: &[Gen] = &[
    g(0, 0.0, 0.0, 0.1666666666666667e0),
];

#[rustfmt::skip]
const LD0014: &[Gen] = &[
    g(0, 0.0, 0.0, 0.6666666666666667e-1),
    g(2, 0.0, 0.0, 0.7500000000000000e-1),
];

#[rustfmt::skip]
const LD0026: &[Gen] = &[
    g(0, 0.0, 0.0, 0.4761904761904762e-1),
    g(1, 0.0, 0.0, 0.3809523809523810e-1),
    g(2, 0.0, 0.0, 0.3214285714285714e-1),
];

#[rustfmt::skip]
const LD0038: &[Gen] = &[
    g(0, 0.0, 0.0, 0.9523809523809524e-2),
    g(2, 0.0, 0.0, 0.3214285714285714e-1),
    g(4, 0.4597008433809831e0, 0.0, 0.2857142857142857e-1),
];

#[rustfmt::skip]
const LD0050: &[Gen] = &[
    g(0, 0.0, 0.0, 0.1269841269841270e-1),
    g(1, 0.0, 0.0, 0.2257495590828924e-1),
    g(2, 0.0, 0.0, 0.2109375000000000e-1),
    g(3, 0.3015113445777636e0, 0.0, 0.2017333553791887e-1),
];

#[rustfmt::skip]
const LD0086: &[Gen] = &[
    g(0, 0.0, 0.0, 0.1154401154401154e-1),
    g(2, 0.0, 0.0, 0.1194390908585628e-1),
    g(3, 0.3696028464541502e0, 0.0, 0.1111055571060340e-1),
    g(3, 0.6943540066026664e0, 0.0, 0.1187650129453714e-1),
    g(4, 0.3742430390903412e0, 0.0, 0.1181230374690448e-1),
];

#[rustfmt::skip]
const LD0110: &[Gen] = &[
    g(0, 0.0, 0.0, 0.3828270494937162e-2),
    g(2, 0.0, 0.0, 0.9793737512487512e-2),
    g(3, 0.1851156353447362e0, 0.0, 0.8211737283191111e-2),
    g(3, 0.6904210483822922e0, 0.0, 0.9942814891178103e-2),
    g(3, 0.3956894730559419e0, 0.0, 0.9595471336070963e-2),
    g(4, 0.4783690288121502e0, 0.0, 0.9694996361663028e-2),
];

#[rustfmt::skip]
const LD0146: &[Gen] = &[
    g(0, 0.0, 0.0, 0.5996313688621381e-3),
    g(1, 0.0, 0.0, 0.7372999718620756e-2),
    g(2, 0.0, 0.0, 0.7210515360144488e-2),
    g(3, 0.6764410400114264e0, 0.0, 0.7116355493117555e-2),
    g(3, 0.4174961227965453e0, 0.0, 0.6753829486314477e-2),
    g(3, 0.1574676672039082e0, 0.0, 0.7574394159054034e-2),
    g(5, 0.1403553811713183e0, 0.4493328323269557e0, 0.6991087353303262e-2),
];

#[rustfmt::skip]
const LD0170: &[Gen] = &[
    g(0, 0.0, 0.0, 0.5544842902037365e-2),
    g(1, 0.0, 0.0, 0.6071332770670752e-2),
    g(2, 0.0, 0.0, 0.6383674773515093e-2),
    g(3, 0.2551252621114134e0, 0.0, 0.5183387587747790e-2),
    g(3, 0.6743601460362766e0, 0.0, 0.6317929009813725e-2),
    g(3, 0.4318910696719410e0, 0.0, 0.6201670006589077e-2),
    g(4, 0.2613931360335988e0, 0.0, 0.5477143385137348e-2),
    g(5, 0.4990453161796037e0, 0.1446630744325115e0, 0.5968383987681156e-2),
];

#[rustfmt::skip]
const LD0194: &[Gen] = &[
    g(0, 0.0, 0.0, 0.1782340447244611e-2),
    g(1, 0.0, 0.0, 0.5716905949977102e-2),
    g(2, 0.0, 0.0, 0.5573383178848738e-2),
    g(3, 0.6712973442695226e0, 0.0, 0.5608704082587997e-2),
    g(3, 0.2892465627575439e0, 0.0, 0.5158237711805383e-2),
    g(3, 0.4446933178717437e0, 0.0, 0.5518771467273614e-2),
    g(3, 0.1299335447650067e0, 0.0, 0.4106777028169394e-2),
    g(4, 0.3457702197611283e0, 0.0, 0.5051846064614808e-2),
    g(5, 0.1590417105383530e0, 0.8360360154824589e0, 0.5530248916233094e-2),
];

#[rustfmt::skip]
const LD0302: &[Gen] = &[
    g(0, 0.0, 0.0, 0.8545911725128148e-3),
    g(2, 0.0, 0.0, 0.3599119285025571e-2),
    g(3, 0.3515640345570105e0, 0.0, 0.3449788424305883e-2),
    g(3, 0.6566329410219612e0, 0.0, 0.3604822601419882e-2),
    g(3, 0.4729054132581005e0, 0.0, 0.3576729661743367e-2),
    g(3, 0.9618308522614784e-1, 0.0, 0.2352101413689164e-2),
    g(3, 0.2219645236294178e0, 0.0, 0.3108953122413675e-2),
    g(3, 0.7011766416089545e0, 0.0, 0.3650045807677255e-2),
    g(4, 0.2644152887060663e0, 0.0, 0.2982344963171804e-2),
    g(4, 0.5718955891878961e0, 0.0, 0.3600820932216460e-2),
    g(5, 0.2510034751770465e0, 0.8000727494073952e0, 0.3571540554273387e-2),
    g(5, 0.1233548532583327e0, 0.4127724083168531e0, 0.3392312205006170e-2),
];

#[rustfmt::skip]
const LD0434: &[Gen] = &[
    g(0, 0.0, 0.0, 0.5265897968224436e-3),
    g(1, 0.0, 0.0, 0.2548219972002607e-2),
    g(2, 0.0, 0.0, 0.2512317418927307e-2),
    g(3, 0.6909346307509111e0, 0.0, 0.2530403801186355e-2),
    g(3, 0.1774836054609158e0, 0.0, 0.2014279020918528e-2),
    g(3, 0.4914342637784746e0, 0.0, 0.2501725168402936e-2),
    g(3, 0.6456664707424256e0, 0.0, 0.2513267174597564e-2),
    g(3, 0.2861289010307638e0, 0.0, 0.2302694782227416e-2),
    g(3, 0.7568084367178018e-1, 0.0, 0.1462495621594614e-2),
    g(3, 0.3927259763368002e0, 0.0, 0.2445373437312980e-2),
    g(4, 0.8818132877794288e0, 0.0, 0.2417442375638981e-2),
    g(4, 0.9776428111182649e0, 0.0, 0.1910951282179532e-2),
    g(5, 0.2054823696403044e0, 0.8689460322872412e0, 0.2416930044324775e-2),
    g(5, 0.5905157048925271e0, 0.7999278543857286e0, 0.2512236854563495e-2),
    g(5, 0.5550152361076807e0, 0.7717462626915901e0, 0.2496644054553086e-2),
    g(5, 0.9371809858553722e0, 0.3344363145343455e0, 0.2236607760437849e-2),
];

#[rustfmt::skip]
const LD0590: &[Gen] = &[
    g(0, 0.0, 0.0, 0.3095121295306187e-3),
    g(2, 0.0, 0.0, 0.1852379698597489e-2),
    g(3, 0.7040954938227469e0, 0.0, 0.1871790639277744e-2),
    g(3, 0.6807744066455243e0, 0.0, 0.1858812585438317e-2),
    g(3, 0.6372546939258752e0, 0.0, 0.1852028828296213e-2),
    g(3, 0.5044419707800358e0, 0.0, 0.1846715956151242e-2),
    g(3, 0.4215761784010967e0, 0.0, 0.1818471778162769e-2),
    g(3, 0.3317920736472123e0, 0.0, 0.1749564657281154e-2),
    g(3, 0.2384736701421887e0, 0.0, 0.1617210647254411e-2),
    g(3, 0.1459036449157763e0, 0.0, 0.1384737234851692e-2),
    g(3, 0.6095034115507196e-1, 0.0, 0.9764331165051050e-3),
    g(4, 0.6116843442009876e0, 0.0, 0.1857161196774078e-2),
    g(4, 0.3964755348199858e0, 0.0, 0.1705153996395864e-2),
    g(4, 0.1724782009907724e0, 0.0, 0.1300321685886048e-2),
    g(5, 0.5610263808622060e0, 0.3518280927733519e0, 0.1842866472905286e-2),
    g(5, 0.4742392842551980e0, 0.2634716655937950e0, 0.1802658934377451e-2),
    g(5, 0.5984126497885380e0, 0.1816640840360209e0, 0.1849830560443660e-2),
    g(5, 0.3791035407695563e0, 0.1720795225656878e0, 0.1713904507106709e-2),
    g(5, 0.2778673190586244e0, 0.8213021581932511e-1, 0.1555213603396808e-2),
    g(5, 0.5033564271075117e0, 0.8999205842074875e-1, 0.1802239128008525e-2),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn double_factorial_odd(n: i64) -> f64 {
        let mut r = 1.0;
        let mut k = n;
        while k > 1 {
            r *= k as f64;
            k -= 2;
        }
        r
    }

    fn sphere_average(a: u32, b: u32, c: u32) -> f64 {
        if a % 2 == 1 || b % 2 == 1 || c % 2 == 1 {
            return 0.0;
        }
        let (a, b, c) = (a as i64, b as i64, c as i64);
        double_factorial_odd(a - 1) * double_factorial_odd(b - 1) * double_factorial_odd(c - 1)
            / double_factorial_odd(a + b + c + 1)
    }

    #[test]
    fn point_counts_and_weight_sums() {
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            assert_eq!(grid.points.len(), npts, "npts={npts}: point count");
            assert_eq!(grid.weights.len(), npts, "npts={npts}: weight count");
            let sum: f64 = grid.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "npts={npts}: Σw = {sum}");
            assert!(
                grid.weights.iter().all(|&w| w > 0.0),
                "npts={npts}: expected all-positive weights"
            );
        }
        assert!(LebedevGrid::new(7).is_none());
        assert!(LebedevGrid::new(74).is_none());
        assert!(degree(74).is_none());
    }

    #[test]
    fn points_are_unit_vectors() {
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            for p in &grid.points {
                let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                assert!((r2 - 1.0).abs() < 1e-13, "npts={npts}: |r|² = {r2}");
            }
        }
    }

    #[test]
    fn exact_for_all_monomials_up_to_degree() {
        let mut global_max = 0.0_f64;
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            let deg = degree(npts).unwrap();
            let n = grid.points.len();
            let stride = deg + 1;

            let mut px = vec![0.0; n * stride];
            let mut py = vec![0.0; n * stride];
            let mut pz = vec![0.0; n * stride];
            for (i, p) in grid.points.iter().enumerate() {
                let (mut ex, mut ey, mut ez) = (1.0, 1.0, 1.0);
                for k in 0..stride {
                    px[i * stride + k] = ex;
                    py[i * stride + k] = ey;
                    pz[i * stride + k] = ez;
                    ex *= p[0];
                    ey *= p[1];
                    ez *= p[2];
                }
            }

            let mut max_err = 0.0_f64;
            let (mut worst, mut worst_q, mut worst_a) = ((0, 0, 0), 0.0, 0.0);
            for a in 0..=deg {
                for b in 0..=(deg - a) {
                    for c in 0..=(deg - a - b) {
                        let analytic = sphere_average(a as u32, b as u32, c as u32);
                        let mut quad = 0.0;
                        for i in 0..n {
                            quad += grid.weights[i]
                                * px[i * stride + a]
                                * py[i * stride + b]
                                * pz[i * stride + c];
                        }
                        let err = (quad - analytic).abs();
                        if err > max_err {
                            max_err = err;
                            worst = (a, b, c);
                            worst_q = quad;
                            worst_a = analytic;
                        }
                    }
                }
            }
            assert!(
                max_err < 1e-12,
                "npts={npts} deg={deg}: max monomial error {max_err:e} at \
                 x^{}y^{}z^{} (quad={worst_q}, exact={worst_a})",
                worst.0,
                worst.1,
                worst.2
            );
            global_max = global_max.max(max_err);
        }
        println!("Lebedev max monomial error over all shipped orders: {global_max:e}");
    }
}
