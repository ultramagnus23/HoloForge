"""Phase 5 figure tests: verify make_all.py produces valid PDFs in both
the no-data (placeholder) and real-data cases, using a temp output dir
and temp results dir so this never touches the real figures/paper/ or
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
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out_")
    tmp_results = tempfile.mkdtemp(prefix="fig_test_results_")
    try:
        fig_mod.OUT_DIR = tmp_out
        rm.set_results_root(tmp_results)
        fig_mod.make_F4_headline_gain_vs_K()
        fig_mod.make_F6_sigma_probe()
        fig_mod.make_F7_twin_validation()
        for name in ["F4_headline_gain_vs_K.pdf", "F6_sigma_probe.pdf", "F7_twin_validation.pdf"]:
            p = os.path.join(tmp_out, name)
            assert os.path.exists(p), f"{name} not written"
            assert _is_valid_pdf(p), f"{name} is not a valid PDF"
        print("placeholder figures OK (valid PDFs, correctly emitted when no data)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


def test_F1_and_F8_real_data_figures_render():
    """F1 has no data dependency; F8a/F8b/F8c/F8d/F8e use real files already
    committed in the repo -- render against the ACTUAL repo state (not a
    temp dir) since that's the real data this function is contracted to use."""
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out2_")
    try:
        fig_mod.OUT_DIR = tmp_out
        fig_mod.make_F1_pipeline_schematic()
        fig_mod.make_F8a_gradient_ablation()
        fig_mod.make_F8b_rcwa()
        fig_mod.make_F8c_mesh_convergence()
        fig_mod.make_F8d_wavelength_detuning()
        fig_mod.make_F8e_shrinkage_prelim()
        names = ["F1_pipeline_schematic.pdf", "F8a_gradient_ablation.pdf",
                "F8b_rcwa.pdf", "F8c_mesh_convergence.pdf",
                "F8d_wavelength_detuning.pdf", "F8e_shrinkage_prelim.pdf"]
        for name in names:
            p = os.path.join(tmp_out, name)
            assert os.path.exists(p) and _is_valid_pdf(p), name
            assert os.path.getsize(p) > 2000, f"{name} suspiciously small (placeholder fallback?)"
        print("F1 + F8 real-data figures OK (valid, non-trivial PDFs)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_F4_renders_real_content_when_E1_data_present():
    """With real (tiny) E1 data present, F4 must render the actual curve
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
            for method_id in ["M2", "M4"]:
                for seed in [0, 1]:
                    config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=3,
                                 converge_tol=None, contrast_cap=4.0, dose_budget=1.0,
                                 medium=DEFAULT_MEDIUM, target=dict(kind="bars", period_px=period_px),
                                 K_nominal=K)
                    job = _job("E1", method_id, seed, config)
                    result = rm.run_job(job, device, commit)
                    path = rm.result_path(job["experiment_id"], job["method_id"],
                                          job["config_hash"], job["seed"])
                    rm.atomic_write_json(path, result)

        fig_mod.make_F4_headline_gain_vs_K()
        p = os.path.join(tmp_out, "F4_headline_gain_vs_K.pdf")
        assert os.path.exists(p) and _is_valid_pdf(p)
        # placeholder PDFs are small (short text-only figure); a real
        # rendered curve+legend+fill_between is reliably larger
        size = os.path.getsize(p)
        print(f"F4 with real E1 data: {size} bytes (placeholder is typically <15000)")
        assert size > 15000, f"F4 suspiciously small ({size}b) -- may have hit the placeholder path"
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


if __name__ == "__main__":
    test_placeholder_figures_are_valid_pdfs_when_no_data()
    test_F1_and_F8_real_data_figures_render()
    test_F4_renders_real_content_when_E1_data_present()
    print("PASSED")
