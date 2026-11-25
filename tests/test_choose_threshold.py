"""Tests for src/evaluation/choose_threshold.py (synthetic scores, runs in < 2 s)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "choose_threshold.py"


def synth(n_ran, n_ben, seed, ben_hi=0.0):
    r = np.random.default_rng(seed)
    ran = np.clip(r.normal(0.85, 0.15, (n_ran, 4)), 0, 1)
    ben = np.clip(r.beta(0.5, 12, (n_ben, 4)), 0, 1)
    if ben_hi:                                           # a few benign runs that look malicious
        ben[: int(ben_hi * n_ben), 1] = 0.97
    d = pd.DataFrame(np.vstack([ran, ben]), columns=["prob_10", "prob_25", "prob_50", "prob_75"])
    d["true_label"] = ["Ransomware"] * n_ran + ["Benign"] * n_ben
    d["family"] = "x"; d["status"] = "ok"
    return d


def run(tmp_path, val, test=None, extra=()):
    val.to_csv(tmp_path / "val.csv", index=False)
    cmd = [sys.executable, "-I", str(SCRIPT), "--val", str(tmp_path / "val.csv"), "--out", str(tmp_path / "cfg.json"), *extra]
    if test is not None:
        test.to_csv(tmp_path / "test.csv", index=False)
        cmd += ["--test", str(tmp_path / "test.csv")]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_respects_fpr_cap_and_writes_config(tmp_path):
    p = run(tmp_path, synth(100, 120, 0, ben_hi=0.03), synth(150, 180, 1))
    assert p.returncode == 0, p.stderr
    cfg = json.loads((tmp_path / "cfg.json").read_text())
    assert 0 < cfg["threshold"] < 1
    assert cfg["val"]["fpr"] <= 0.05
    assert cfg["val"]["f1"] > 0.9
    assert (tmp_path / "threshold_curve.png").exists()


def test_fails_cleanly_when_cap_unreachable(tmp_path):
    val = synth(50, 100, 2)
    val.loc[val.true_label == "Benign", "prob_25"] = 1.0   # 100 % of benign runs saturate -> no threshold < 1 can help
    p = run(tmp_path, val)
    assert p.returncode != 0 and "No threshold reaches FPR" in (p.stdout + p.stderr)
    assert not (tmp_path / "cfg.json").exists()
