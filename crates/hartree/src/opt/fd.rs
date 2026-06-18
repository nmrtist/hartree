use crate::opt::{OptError, Surface};

pub fn central_difference<S: Surface>(
    surface: &mut S,
    positions: &[[f64; 3]],
    step: f64,
) -> Result<Vec<[f64; 3]>, OptError> {
    let natom = positions.len();
    let mut g = vec![[0.0; 3]; natom];
    for (atom, g_atom) in g.iter_mut().enumerate() {
        for (axis, slot) in g_atom.iter_mut().enumerate() {
            let mut plus = positions.to_vec();
            plus[atom][axis] += step;
            let mut minus = positions.to_vec();
            minus[atom][axis] -= step;
            let e_plus = surface.energy(&plus)?;
            let e_minus = surface.energy(&minus)?;
            *slot = (e_plus - e_minus) / (2.0 * step);
        }
    }
    Ok(g)
}
