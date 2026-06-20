//! Parsing and validation for the `--ts-*` transition-state CLI flags.
//!
//! Pure and side-effect-free so it is unit-testable without touching argv or the
//! solver: [`apply_ts_option`] mutates a borrowed [`TsOptions`] (built via
//! `TsOptions::default()`, the documented way to construct the `#[non_exhaustive]`
//! struct) from one flag/value pair, and [`parse_ts_algo`] decodes `--ts-algo`.
//! Error strings mirror the existing CLI style ("--flag must be ...").

use hartree::opt::ts::{IrcMethod, TsAlgorithm, TsOptions, VerifyHessian};

/// The value-taking `--ts-*` flags, so the argv loop in `main.rs` can recognise
/// them in one membership test before routing through [`apply_ts_option`].
/// `--ts-irc` is a bare boolean and is handled directly in `main.rs`.
pub const TS_VALUE_FLAGS: &[&str] = &[
    "--ts-max-iter",
    "--ts-follow",
    "--ts-recalc-hessian",
    "--ts-trust",
    "--ts-fd-step",
    "--ts-neg-tol",
    "--ts-algo",
    "--ts-irc-method",
    "--ts-irc-step",
    "--ts-irc-max-steps",
    "--ts-verify-hessian",
];

/// Parse the `--ts-algo` value. Accepts "prfo" and "dimer" (case-insensitive).
pub fn parse_ts_algo(s: &str) -> Result<TsAlgorithm, String> {
    match s.to_ascii_lowercase().as_str() {
        "prfo" => Ok(TsAlgorithm::Prfo),
        "dimer" => Ok(TsAlgorithm::Dimer),
        _ => Err(format!("--ts-algo must be one of prfo, dimer (got {s:?})")),
    }
}

/// Parse the `--ts-irc-method` value. Accepts "dvv", "gs2", and "eulerpc"
/// (case-insensitive).
pub fn parse_irc_method(s: &str) -> Result<IrcMethod, String> {
    match s.to_ascii_lowercase().as_str() {
        "dvv" => Ok(IrcMethod::Dvv),
        "gs2" => Ok(IrcMethod::GonzalezSchlegel),
        "eulerpc" => Ok(IrcMethod::EulerPc),
        _ => Err(format!(
            "--ts-irc-method must be one of dvv, gs2, eulerpc (got {s:?})"
        )),
    }
}

/// Parse the `--ts-verify-hessian` value. Accepts "strict", "maintained", and "auto"
/// (case-insensitive).
pub fn parse_verify_hessian(s: &str) -> Result<VerifyHessian, String> {
    match s.to_ascii_lowercase().as_str() {
        "strict" => Ok(VerifyHessian::Strict),
        "maintained" => Ok(VerifyHessian::Maintained),
        "auto" => Ok(VerifyHessian::Auto),
        _ => Err(format!(
            "--ts-verify-hessian must be one of strict, maintained, auto (got {s:?})"
        )),
    }
}

/// Apply one value-taking `--ts-*` flag to `opts`, validating the value. Returns
/// an error string (mirroring the existing CLI error style) on bad input.
pub fn apply_ts_option(opts: &mut TsOptions, flag: &str, value: &str) -> Result<(), String> {
    match flag {
        "--ts-max-iter" => opts.max_iter = parse_usize(flag, value)?, // >= 1
        "--ts-follow" => opts.follow_mode = parse_usize_zero_ok(flag, value)?, // >= 0
        "--ts-recalc-hessian" => opts.recalc_hessian = parse_usize_zero_ok(flag, value)?, // 0 = Bofill only
        "--ts-trust" => opts.trust_radius = parse_pos_f64(flag, value, "bohr")?,          // > 0
        "--ts-fd-step" => opts.fd_step = parse_pos_f64(flag, value, "bohr")?,             // > 0
        "--ts-neg-tol" => opts.negative_mode_tol = parse_pos_f64(flag, value, "a.u.")?,   // > 0
        "--ts-algo" => opts.algorithm = parse_ts_algo(value)?,
        "--ts-irc-method" => opts.irc_method = parse_irc_method(value)?,
        "--ts-irc-step" => opts.irc_step = parse_pos_f64(flag, value, "√amu·bohr")?, // > 0
        "--ts-irc-max-steps" => opts.irc_max_steps = parse_usize(flag, value)?,      // >= 1
        "--ts-verify-hessian" => opts.verify_hessian = parse_verify_hessian(value)?,
        other => return Err(format!("internal error: unhandled ts flag {other}")),
    }
    Ok(())
}

/// Parse a `usize` that must be at least 1 (e.g. an iteration count).
fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    let n: usize = value
        .parse()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if n == 0 {
        return Err(format!("{flag} must be a positive integer (>= 1)"));
    }
    Ok(n)
}

/// Parse a `usize` that may be 0 (e.g. a mode index / recompute cadence).
fn parse_usize_zero_ok(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} must be a non-negative integer"))
}

/// Parse a strictly positive, finite `f64`. `unit` names the value's units for
/// the error message (e.g. "bohr", "a.u.").
fn parse_pos_f64(flag: &str, value: &str, unit: &str) -> Result<f64, String> {
    let x: f64 = value
        .parse()
        .map_err(|_| format!("{flag} must be a positive number ({unit})"))?;
    if !x.is_finite() || x <= 0.0 {
        return Err(format!("{flag} must be a positive number ({unit})"));
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ts_algo_accepts_known_values() {
        assert_eq!(parse_ts_algo("prfo").unwrap(), TsAlgorithm::Prfo);
        assert_eq!(parse_ts_algo("dimer").unwrap(), TsAlgorithm::Dimer);
        // Case-insensitive.
        assert_eq!(parse_ts_algo("PRFO").unwrap(), TsAlgorithm::Prfo);
        assert_eq!(parse_ts_algo("Dimer").unwrap(), TsAlgorithm::Dimer);
    }

    #[test]
    fn parse_ts_algo_rejects_unknown_values() {
        assert!(parse_ts_algo("neb").is_err());
        assert!(parse_ts_algo("bogus").is_err());
        assert!(parse_ts_algo("").is_err());
    }

    #[test]
    fn parse_irc_method_accepts_known_values() {
        assert_eq!(parse_irc_method("dvv").unwrap(), IrcMethod::Dvv);
        assert_eq!(
            parse_irc_method("gs2").unwrap(),
            IrcMethod::GonzalezSchlegel
        );
        assert_eq!(parse_irc_method("eulerpc").unwrap(), IrcMethod::EulerPc);
        // Case-insensitive.
        assert_eq!(parse_irc_method("DVV").unwrap(), IrcMethod::Dvv);
        assert_eq!(parse_irc_method("EulerPC").unwrap(), IrcMethod::EulerPc);
    }

    #[test]
    fn parse_irc_method_rejects_unknown_values() {
        assert!(parse_irc_method("gonzalez").is_err());
        assert!(parse_irc_method("bogus").is_err());
        assert!(parse_irc_method("").is_err());
    }

    #[test]
    fn apply_sets_each_field_from_valid_value() {
        let mut o = TsOptions::default();
        apply_ts_option(&mut o, "--ts-max-iter", "42").unwrap();
        assert_eq!(o.max_iter, 42);

        apply_ts_option(&mut o, "--ts-follow", "2").unwrap();
        assert_eq!(o.follow_mode, 2);

        apply_ts_option(&mut o, "--ts-recalc-hessian", "5").unwrap();
        assert_eq!(o.recalc_hessian, 5);

        apply_ts_option(&mut o, "--ts-trust", "0.15").unwrap();
        assert_eq!(o.trust_radius, 0.15);

        apply_ts_option(&mut o, "--ts-fd-step", "1e-3").unwrap();
        assert_eq!(o.fd_step, 1e-3);

        apply_ts_option(&mut o, "--ts-neg-tol", "2e-4").unwrap();
        assert_eq!(o.negative_mode_tol, 2e-4);

        apply_ts_option(&mut o, "--ts-algo", "dimer").unwrap();
        assert_eq!(o.algorithm, TsAlgorithm::Dimer);

        apply_ts_option(&mut o, "--ts-irc-method", "gs2").unwrap();
        assert_eq!(o.irc_method, IrcMethod::GonzalezSchlegel);

        apply_ts_option(&mut o, "--ts-irc-step", "0.05").unwrap();
        assert_eq!(o.irc_step, 0.05);

        apply_ts_option(&mut o, "--ts-irc-max-steps", "250").unwrap();
        assert_eq!(o.irc_max_steps, 250);

        apply_ts_option(&mut o, "--ts-verify-hessian", "auto").unwrap();
        assert_eq!(o.verify_hessian, VerifyHessian::Auto);
    }

    #[test]
    fn parse_verify_hessian_accepts_known_values() {
        assert_eq!(
            parse_verify_hessian("strict").unwrap(),
            VerifyHessian::Strict
        );
        assert_eq!(
            parse_verify_hessian("maintained").unwrap(),
            VerifyHessian::Maintained
        );
        assert_eq!(parse_verify_hessian("auto").unwrap(), VerifyHessian::Auto);
        // Case-insensitive.
        assert_eq!(parse_verify_hessian("Auto").unwrap(), VerifyHessian::Auto);
        assert!(parse_verify_hessian("bogus").is_err());
        assert!(parse_verify_hessian("").is_err());
    }

    #[test]
    fn apply_accepts_zero_for_zero_ok_fields() {
        let mut o = TsOptions::default();
        apply_ts_option(&mut o, "--ts-follow", "0").unwrap();
        assert_eq!(o.follow_mode, 0);
        apply_ts_option(&mut o, "--ts-recalc-hessian", "0").unwrap();
        assert_eq!(o.recalc_hessian, 0);
    }

    #[test]
    fn apply_rejects_bad_values() {
        let mut o = TsOptions::default();
        assert!(apply_ts_option(&mut o, "--ts-trust", "0").is_err());
        assert!(apply_ts_option(&mut o, "--ts-trust", "-1").is_err());
        assert!(apply_ts_option(&mut o, "--ts-fd-step", "nan").is_err());
        assert!(apply_ts_option(&mut o, "--ts-fd-step", "-1").is_err());
        assert!(apply_ts_option(&mut o, "--ts-max-iter", "0").is_err());
        assert!(apply_ts_option(&mut o, "--ts-neg-tol", "0").is_err());
        assert!(apply_ts_option(&mut o, "--ts-algo", "bogus").is_err());
        assert!(apply_ts_option(&mut o, "--ts-irc-method", "bogus").is_err());
        assert!(apply_ts_option(&mut o, "--ts-irc-step", "0").is_err());
        assert!(apply_ts_option(&mut o, "--ts-irc-step", "-1").is_err());
        assert!(apply_ts_option(&mut o, "--ts-irc-max-steps", "0").is_err());
        // A non-numeric integer is rejected too.
        assert!(apply_ts_option(&mut o, "--ts-max-iter", "abc").is_err());
        assert!(apply_ts_option(&mut o, "--ts-follow", "-1").is_err());
    }

    #[test]
    fn rejected_values_do_not_mutate_defaults() {
        // A failed parse must leave the option untouched (validation happens
        // before assignment).
        let mut o = TsOptions::default();
        let before = o.trust_radius;
        let _ = apply_ts_option(&mut o, "--ts-trust", "-1");
        assert_eq!(o.trust_radius, before);
    }

    #[test]
    fn value_flag_table_is_exhaustive() {
        // Every value flag routes through `apply_ts_option` without hitting the
        // "internal error" arm.
        let mut o = TsOptions::default();
        for &flag in TS_VALUE_FLAGS {
            // Flags taking an enum keyword need a valid keyword; the rest accept "1".
            let value = match flag {
                "--ts-algo" => "prfo",
                "--ts-irc-method" => "dvv",
                "--ts-verify-hessian" => "strict",
                _ => "1",
            };
            apply_ts_option(&mut o, flag, value)
                .unwrap_or_else(|e| panic!("flag {flag} should parse value {value:?}: {e}"));
        }
    }
}
