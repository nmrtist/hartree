//! The opt-in stalled-Hessian refresh ([`TsOptions::stall_refresh`]): a soft-surface
//! aid that refreshes the maintained (Bofill) Hessian from finite differences once
//! the projected force has stalled for several steps. Disabled by default, so the
//! historical climb is byte-for-byte unchanged — the guarantee these tests pin down.

use super::*;
use crate::opt::ts::{TsOptions, TsStatus, find_transition_state};

/// Inert on a well-behaved climb: a quadratic saddle converges with a monotonically
/// decreasing force, so the stall counter resets every step and the refresh never
/// fires. Enabling it (`stall_refresh = 5`) must therefore produce the identical
/// iteration count and the bitwise-identical converged geometry as the default
/// (`stall_refresh = 0`) — the byte-identical-default guarantee for the new knob.
#[test]
fn stall_refresh_is_inert_on_well_behaved_climb() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.06 * basis[0][i] + 0.04 * basis[1][i];
        }
    }

    let run = |stall_refresh: usize| {
        let mut surf = Quadratic {
            x0: x0.clone(),
            h: h.clone(),
        };
        let mut opts = TsOptions::default();
        opts.stall_refresh = stall_refresh;
        find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap()
    };

    let baseline = run(0);
    let enabled = run(5);
    assert_eq!(baseline.status, TsStatus::Converged);
    assert_eq!(enabled.status, TsStatus::Converged);
    assert_eq!(baseline.iterations, enabled.iterations);
    assert_eq!(baseline.positions, enabled.positions);
}

/// Backward compatibility: a `TsOptions` serialized before `stall_refresh` existed
/// (no such key) deserializes with the field defaulted to `0` (the aid disabled).
#[test]
fn options_round_trip_defaults_new_stall_refresh_field() {
    let opts = TsOptions::default();
    assert_eq!(opts.stall_refresh, 0);
    let json = serde_json::to_string(&opts).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("stall_refresh");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.stall_refresh, 0);
}
