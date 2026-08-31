"""Figure tests: verify make_all.py produces valid PDFs in both the
no-data (placeholder) and real-data cases, using a temp output dir and
temp results dir so this never touches the real figures/paper/ or
results/ trees.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "figures"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

torch.set_default_dtype(torch.float64)

import make_all as fig_mod
import run_manifest as rm
from manifest import _job, DEFAULT_MEDIUM


def _is_valid_pdf(path):
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def test_placeholder_figures_are_valid_pdfs_when_no_data():
    """Every make_F* function returns True iff it rendered real content,
    False iff it emitted a placeholder -- checked directly via the return
    value rather than sniffing the PDF's text content, which matplotlib
    compresses (FlateDecode), making a raw byte-search for placeholder
    text unreliable (see the commit fixing an earlier, broken version of
    this check that used a byte-size heuristic and silently never
    exercised the real-data path for months)."""
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out_")
    tmp_results = tempfile.mkdtemp(prefix="fig_test_results_")
    try:
        fig_mod.OUT_DIR = tmp_out
        rm.set_results_root(tmp_results)
        fns_and_names = [
            (fig_mod.make_F4_headline_gain_vs_K, "F4_headline_gain_vs_K.pdf"),
            (fig_mod.make_F6_cliff_shift, "F6_cliff_shift.pdf"),
            (fig_mod.make_F7_physics_ablation, "F7_physics_ablation.pdf"),
            (fig_mod.make_F8_sensitivity_band, "F8_sensitivity_band.pdf"),
            (fig_mod.make_F3b_regime_map, "F3b_regime_map.pdf"),
        ]
        for fn, name in fns_and_names:
            is_real = fn()
            assert is_real is False, f"{fn.__name__} should report no real content (no data present)"
            p = os.path.join(tmp_out, name)
            assert os.path.exists(p), f"{name} not written"
            assert _is_valid_pdf(p), f"{name} is not a valid PDF"
        print("placeholder figures OK (valid PDFs, correctly report no real content)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


def test_F1_and_F9_real_data_figures_render():
    """F1 has no data dependency; F9a/F9b/F9c/F3a/F2 use real files already
    committed in the repo -- render against the ACTUAL repo state (not a
    temp dir) since that's the real data these functions are contracted
    to use. F2 joined this list once data/literature/*.csv and
    results_literature_fit.json became real, committed data -- like F3a,
    make_F2_twin_validation() resolves its input via a HERE-relative path
    in figures/make_all.py, not rm.set_results_root(), so it always reads
    the real repo files regardless of the OTHER test's temp-dir isolation
    (checked: a prior version of this test still expected F2 to report
    no-real-content in that isolated case, which stopped being true the
    moment real literature data was committed -- see git log)."""
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out2_")
    try:
        fig_mod.OUT_DIR = tmp_out
        fns_and_names = [
            (fig_mod.make_F1_pipeline_schematic, "F1_pipeline_schematic.pdf"),
            (fig_mod.make_F9a_gradient_ablation, "F9a_gradient_ablation.pdf"),
            (fig_mod.make_F3a_rcwa_validity_envelope, "F3a_rcwa_validity_envelope.pdf"),
            (fig_mod.make_F9b_mesh_convergence, "F9b_mesh_convergence.pdf"),
            (fig_mod.make_F9c_wavelength_detuning, "F9c_wavelength_detuning.pdf"),
            (fig_mod.make_F2_twin_validation, "F2_twin_validation.pdf"),
        ]
        for fn, name in fns_and_names:
            is_real = fn()
            assert is_real is True, f"{fn.__name__} should report real content given committed data"
            p = os.path.join(tmp_out, name)
            assert os.path.exists(p) and _is_valid_pdf(p), name
        print("F1 + F9/F3a/F2 real-data figures OK (valid PDFs, real content confirmed)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_F4_renders_real_content_when_M1_data_present():
    """With real (tiny) M1 data present, F4 must render the actual curve
    path (_render_F4), not fall back to the placeholder."""
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out3_")
    tmp_results = tempfile.mkdtemp(prefix="fig_test_results3_")
    try:
        fig_mod.OUT_DIR = tmp_out
        rm.set_results_root(tmp_results)
        device = rm.get_device()
        commit = rm.git_commit_hash()
        n_x = 48
        dx = 51.2 / n_x
        for K, period_px in [(2.0, 16), (6.0, 5)]:
            for method_id in ["BSGD", "MIL"]:
                for seed in [0, 1]:
                    config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=3,
                                 converge_tol=None, contrast_cap=4.0, dose_budget=1.0,
                                 medium=DEFAULT_MEDIUM, target=dict(kind="bars", period_px=period_px),
                                 K_nominal=K)
                    job = _job("M1", method_id, seed, config)
                    result = rm.run_job(job, device, commit)
                    path = rm.result_path(job["experiment_id"], job["method_id"],
                                          job["config_hash"], job["seed"])
                    rm.atomic_write_json(path, result)

        is_real = fig_mod.make_F4_headline_gain_vs_K()
        assert is_real is True, "F4 should report real content given real M1 data"
        p = os.path.join(tmp_out, "F4_headline_gain_vs_K.pdf")
        assert os.path.exists(p) and _is_valid_pdf(p)
        print(f"F4 with real M1 data: {os.path.getsize(p)} bytes, real curve rendered")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


if __name__ == "__main__":
    test_placeholder_figures_are_valid_pdfs_when_no_data()
    test_F1_and_F9_real_data_figures_render()
    test_F4_renders_real_content_when_M1_data_present()
    print("PASSED")
