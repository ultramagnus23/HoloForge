"""Phase 7.1/7.3 paper-build infrastructure tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import make_numbers_tex as mnt
from check_consistency import find_violations
from check_refs import find_pending_entries


def test_make_numbers_tex_pending_when_no_data():
    tex = mnt.build_macros({})
    assert r"\newcommand{\SeedCount}{\textcolor{red}{[PENDING]}}" in tex
    assert tex.count("[PENDING]") > 0
    print("make_numbers_tex all-PENDING-on-empty-input OK")


def test_make_numbers_tex_real_values_when_data_present():
    fake_paper_numbers = dict(
        n_result_files=8,
        per_config={"E1/abc": {"M4": {"n_seeds": 3}, "M2": {"n_seeds": 3}}},
        e1_headroom_closure=[
            dict(budget=2.0, measured_contrast_C=1.9, predicted_Kc_from_measured_C=4.1,
                observed_Kstar_interp=4.2, observed_Kstar_ci=4.5,
                gain_curve=[(1.0, 2.0, 1.5, 2.5), (5.0, -0.5, -1.0, 0.0)]),
            dict(budget=4.0, status="no_data"),
            dict(budget=8.0, status="no_data"),
        ],
    )
    tex = mnt.build_macros(fake_paper_numbers)
    assert r"\newcommand{\SeedCount}{3}" in tex
    assert r"\newcommand{\KcPredTwoX}{4.10}" in tex
    assert r"\newcommand{\KstarTwoXInterp}{4.20}" in tex
    assert r"\newcommand{\KcPredFourX}{\textcolor{red}{[PENDING]}}" in tex
    print("make_numbers_tex real-values-when-present OK")


def test_check_consistency_catches_hardcoded_numbers():
    bad = "We ran 5 seeds and a mean gain of 1.45 dB was observed at K = 12 rad/ um."
    v = find_violations(bad)
    assert len(v) >= 3, v
    good = r"We ran \SeedCount seeds and a mean gain of \MeanGainTwoX dB was observed."
    assert find_violations(good) == []
    print("check_consistency catches/passes correctly:", v)


def test_check_refs_finds_todo_verify_entries():
    bib = """
% TODO-VERIFY: title not confirmed
@article{foo2020bar,
  title = {something},
  year = {2020},
}

@article{clean2021baz,
  title = {a verified thing},
  year = {2021},
}
"""
    pending = find_pending_entries(bib)
    assert pending == ["foo2020bar"], pending
    print("check_refs finds exactly the flagged entry OK:", pending)


if __name__ == "__main__":
    test_make_numbers_tex_pending_when_no_data()
    test_make_numbers_tex_real_values_when_data_present()
    test_check_consistency_catches_hardcoded_numbers()
    test_check_refs_finds_todo_verify_entries()
    print("PASSED")
