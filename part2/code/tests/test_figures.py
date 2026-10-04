"""Figure tests: every figure function in figures/make_all.py returns True
and writes a valid PDF when its data exist, and returns False with a
placeholder PDF when they do not. Output goes to a temporary directory, so the
real figures/paper/ tree is never touched.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "figures"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

torch.set_default_dtype(torch.float64)

import make_all as fig_mod
import run_manifest as rm
from manifest import _job, DEFAULT_MEDIUM

RESULTS_TREE_FIGS = [fig_mod.make_fig1_paired_gain, fig_mod.make_fig3_slant_mismatch,
                     fig_mod.make_figS2_ablation, fig_mod.make_figS3_factorial]
JSON_FIGS = [fig_mod.make_fig2_reconstructions, fig_mod.make_fig4_twin_validation,
             fig_mod.make_figS1_exposure_profiles, fig_mod.make_figS4_rcwa_envelope,
             fig_mod.make_figS5_twin_fits_held_kbleach]


def _is_valid_pdf(path):
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def _new_pdfs(before, out_dir):
    return [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f not in before]


def test_every_figure_function_is_registered():
    assert set(RESULTS_TREE_FIGS + JSON_FIGS) == set(fig_mod.ALL_FIGURES)


def test_results_tree_figures_emit_placeholders_without_data():
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out_")
    tmp_results = tempfile.mkdtemp(prefix="fig_test_results_")
    old_out, old_root = fig_mod.OUT_DIR, rm.RESULTS_ROOT
    try:
        fig_mod.OUT_DIR = tmp_out
        rm.set_results_root(tmp_results)
        for fn in RESULTS_TREE_FIGS:
            before = set(os.listdir(tmp_out))
            assert fn() is False, f"{fn.__name__} claimed real content with no data"
            written = _new_pdfs(before, tmp_out)
            assert len(written) == 1 and _is_valid_pdf(written[0]), fn.__name__
    finally:
        fig_mod.OUT_DIR = old_out
        rm.set_results_root(old_root)
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


def test_committed_json_figures_render():
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out2_")
    old_out = fig_mod.OUT_DIR
    try:
        fig_mod.OUT_DIR = tmp_out
        for fn in JSON_FIGS:
            before = set(os.listdir(tmp_out))
            assert fn() is True, f"{fn.__name__} should render from committed data"
            written = _new_pdfs(before, tmp_out)
            assert len(written) == 1 and _is_valid_pdf(written[0]), fn.__name__
    finally:
        fig_mod.OUT_DIR = old_out
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_fig1_renders_real_content_from_tiny_m1_data():
    """With a tiny but complete M1 data set, Fig. 1 must render the real
    curves rather than the placeholder."""
    tmp_out = tempfile.mkdtemp(prefix="fig_test_out3_")
    tmp_results = tempfile.mkdtemp(prefix="fig_test_results3_")
    old_out, old_root = fig_mod.OUT_DIR, rm.RESULTS_ROOT
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
                                  medium=DEFAULT_MEDIUM,
                                  target=dict(kind="bars", period_px=period_px),
                                  K_nominal=K)
                    job = _job("M1", method_id, seed, config)
                    result = rm.run_job(job, device, commit)
                    path = rm.result_path(job["experiment_id"], job["method_id"],
                                          job["config_hash"], job["seed"])
                    rm.atomic_write_json(path, result)
        assert fig_mod.make_fig1_paired_gain() is True
        p = os.path.join(tmp_out, "Fig1_paired_gain.pdf")
        assert os.path.exists(p) and _is_valid_pdf(p)
    finally:
        fig_mod.OUT_DIR = old_out
        rm.set_results_root(old_root)
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


if __name__ == "__main__":
    test_every_figure_function_is_registered()
    test_results_tree_figures_emit_placeholders_without_data()
    test_committed_json_figures_render()
    test_fig1_renders_real_content_from_tiny_m1_data()
    print("PASSED")
