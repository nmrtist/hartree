# External performance comparisons

Keep approximate and exact comparisons separate:

- Compare Hartree `--ri` with ORCA RI-J using the same auxiliary basis and a
  matched integration-grid point count.
- Compare Hartree exact/direct with ORCA RI disabled.

For every run record the total energy, SCF iteration count, grid-point count,
stage timings, total wall time, thread count, and peak RSS. Use one warm-up and
at least five measured repetitions at 1, 2, 4, and 8 threads, and compare
medians. Do not compare a Hartree RI run with an exact ORCA run as if they were
the same numerical method.
