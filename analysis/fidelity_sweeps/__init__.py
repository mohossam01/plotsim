"""M103 fidelity sweeps — empirical characterization harness.

Module is package-shaped (rather than two flat scripts) so the smoke test in
tests/test_fidelity_smoke.py can import sweep_runner.run_sweep and the small
strategy helpers without shelling out. Result CSVs and the report itself live
alongside this package under analysis/.
"""
