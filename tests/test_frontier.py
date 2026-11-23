import numpy as np
import pytest

from blindspot_eval.monitors.replay import ScorePoint, ScoreTimeline
from blindspot_eval.results.frontier import (
    carrier_frontiers,
    case_score,
    partial_auc_log,
    tpr_at_fpr,
    verdict_flip_rate,
)
from blindspot_eval.results.plots import plot_arm, results_table
from blindspot_eval.sanitize import LabelRecord


def tl(tid, points, arm="arm"):
    return ScoreTimeline(trace_id=tid, arm=arm, points=[ScorePoint(seq=s, score=v) for s, v in points])


def test_partial_auc_bounds():
    pos = np.array([0.9] * 50)
    neg = np.array([0.1] * 1000)
    assert partial_auc_log(pos, neg) == pytest.approx(1.0)
    same = np.array([0.5] * 1000)
    assert partial_auc_log(np.array([0.5] * 50), same) == pytest.approx(0.0)


def test_tpr_at_fpr_is_conservative_step():
    neg = np.linspace(0, 1, 1000)  # 10% FPR needs threshold >= 0.9
    pos = np.array([0.95, 0.85])
    curve = tpr_at_fpr(pos, neg, grid=np.array([0.1]))
    assert curve[0] == pytest.approx(0.5)


def test_flags_after_release_do_not_count():
    late = tl("p", [(1, 0.1), (5, 0.9)])
    assert case_score(late, LabelRecord(trace_id="p", positive=True, verified=True, R_seq=3)) == 0.1
    assert case_score(late, LabelRecord(trace_id="p", positive=True, verified=True, R_seq=5)) == 0.9
    assert case_score(late, LabelRecord(trace_id="p", positive=False)) == 0.9


def _dataset(rng, n_payloads=40, n_controls=25):
    timelines, labels = [], []
    for k in range(n_payloads):
        pid = f"p{k}"
        for carrier, mean in (("C0", 0.9), ("C3", 0.6)):
            tid = f"{pid}-{carrier}"
            timelines.append(tl(tid, [(0, float(np.clip(rng.normal(mean, 0.1), 0, 1)))]))
            labels.append(LabelRecord(trace_id=tid, positive=True, verified=True, R_seq=0, payload_id=pid, carrier=carrier))
            for j in range(n_controls):
                cid = f"{tid}-ctl{j}"
                timelines.append(tl(cid, [(0, float(np.clip(rng.normal(0.3, 0.12), 0, 1)))]))
                labels.append(LabelRecord(trace_id=cid, positive=False, payload_id=pid, carrier=carrier))
    unver = "p0-C3-unverified"
    timelines.append(tl(unver, [(0, 0.99)]))
    labels.append(LabelRecord(trace_id=unver, positive=True, verified=False, payload_id="p0", carrier="C3"))
    return timelines, labels


def test_carrier_gap_is_positive_and_bootstrapped(tmp_path):
    timelines, labels = _dataset(np.random.default_rng(0))
    res = carrier_frontiers(timelines, labels, n_boot=200, seed=1)
    c0, c3 = res.carriers
    assert (c0.carrier, c3.carrier) == ("C0", "C3")
    assert c3.n_pos_unverified == 1 and c3.n_pos == 40
    assert c3.gap_vs_baseline > 0.2
    assert c3.gap_ci[0] > 0
    assert c0.pauc_ci[0] <= c0.pauc <= c0.pauc_ci[1]
    assert len(c3.tpr_lo) == len(res.fpr_grid)
    png = plot_arm(res, tmp_path / "arm.png")
    assert png.stat().st_size > 10_000
    table = results_table([res])
    assert "| arm | C3 |" in table


def test_verdict_flip_rate():
    run1 = [tl("a", [(0, 0.9)]), tl("b", [(0, 0.2)])]
    run2 = [tl("a", [(0, 0.4)]), tl("b", [(0, 0.2)])]
    assert verdict_flip_rate([run1, run2], threshold=0.5) == 0.5
