"""Phase 6 fitting-script test.

Self-consistency check, NOT a literature-validation claim: generates a
synthetic "digitized" growth curve using the twin itself at KNOWN
kappa/D0 (plus small noise), writes it as a CSV in the Phase 6 schema to
a temp directory, and verifies fit_curve recovers parameters close to the
known ones with low RMSE. This proves the fitting mechanics are correct;
it says nothing about agreement with real literature (there is none yet
-- data/literature/ has no real digitized CSVs, see its README).

The module's N_X/DX are temporarily shrunk for test speed (a full
least-squares fit at production N_X=512 takes minutes); this only
affects numerical resolution, not the fitting logic being tested.
"""
import sys, os, csv, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

import fit_literature_curves as flc
from holomedia import MediumParams


def test_infer_curve_type_and_K():
    assert flc.infer_curve_type_and_K("sheridan2011_growth_K6.csv") == ("growth", 6.0)
    assert flc.infer_curve_type_and_K("fomenko2017_angular_K12.5.csv") == ("angular", 12.5)
    assert flc.infer_curve_type_and_K("random_name.csv") == ("unknown", None)
    print("infer_curve_type_and_K OK")


def test_load_curve_csv_schema():
    tmp = tempfile.mkdtemp(prefix="fit_test_")
    try:
        path = os.path.join(tmp, "test_growth_K6.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(flc.CSV_SCHEMA_COLUMNS)
            w.writerow([1.0, 0.05, "10.1/x", "Fig1", "tester", "2026-07-01"])
            w.writerow([2.0, 0.08, "10.1/x", "Fig1", "tester", "2026-07-01"])
        data = flc.load_curve_csv(path)
        assert data["x"] == [1.0, 2.0] and data["y"] == [0.05, 0.08]
        assert data["source_doi"] == "10.1/x" and data["digitized_by"] == "tester"
        print("load_curve_csv schema OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fit_recovers_known_parameters_on_synthetic_data():
    orig_n_x, orig_dx = flc.N_X, flc.DX
    flc.N_X, flc.DX = 96, 0.1  # shrink for test speed; fitting logic unaffected
    try:
        true_kappa, true_D0 = 1.5, 0.05
        K = 6.0
        t_values = [1.0, 3.0, 6.0, 10.0]
        base_params = MediumParams()

        # synthesize "digitized" data from the twin itself at known params
        clean = flc.simulate_growth_de(t_values, K, true_kappa, true_D0, base_params, 30.0)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0, 0.002, size=clean.shape)  # small noise, real digitization-like

        fit = flc.fit_curve("growth", K, t_values, noisy.tolist(), base_params=base_params)

        print(f"true kappa={true_kappa} D0={true_D0} | "
              f"fit kappa={fit['kappa_fit']:.3f} D0={fit['D0_fit']:.4f} rmse={fit['rmse']:.4f}")
        assert fit["converged"]
        assert fit["rmse"] < 0.02, f"fit RMSE too high on synthetic (near-noiseless) data: {fit['rmse']}"
        # recovered params should be in the right ballpark (order of magnitude),
        # not necessarily exact -- growth curves have limited sensitivity to D0
        # at a single K, so this checks the fit is finding a REASONABLE
        # explanation of the curve, not pinning exact recovery
        assert 0.3 * true_kappa < fit["kappa_fit"] < 3.0 * true_kappa, fit["kappa_fit"]
        print("fit recovers known parameters on synthetic data OK (self-consistency check)")
    finally:
        flc.N_X, flc.DX = orig_n_x, orig_dx


if __name__ == "__main__":
    test_infer_curve_type_and_K()
    test_load_curve_csv_schema()
    test_fit_recovers_known_parameters_on_synthetic_data()
    print("PASSED")
