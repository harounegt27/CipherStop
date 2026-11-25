"""
choose_threshold.py  -  CipherStop: choose the alert threshold on VALIDATION data.

Input : one CSV per split, in the format written by simulation/replayer.py
        (columns: family, execution, true_label, prob_10, prob_25, prob_50, prob_75).
        `true_label` may be "Ransomware"/"Benign" or 1/0.
Rule  : same as backend/alerts.py -> an execution raises an alert at the FIRST cut
        (10 -> 25 -> 50 -> 75 %) whose probability is > threshold.
Choice: among thresholds whose validation FPR <= --max-fpr, take the best F1;
        if several thresholds are within 0.005 F1 of the best, take the middle one
        (a threshold in the centre of a plateau is less fragile than its edge).
Output: artifacts/alert_config.json (read by backend/alerts.py) + threshold_curve.png
        + a printed comparison of the chosen threshold with 0.5 and 0.85 on every split.

Usage:
    python src/evaluation/choose_threshold.py --val results_val.csv --test results_test.csv \
        --zeroday results_ryuk.csv --out artifacts/alert_config.json
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CUTS = [10, 25, 50, 75]
PROB_COLS = [f"prob_{c}" for c in CUTS]


def load(path):
    d = pd.read_csv(path)
    if "status" in d.columns:                       # replayer output: keep successful runs only
        d = d[d["status"] == "ok"]
    miss = [c for c in PROB_COLS if c not in d.columns]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}")
    lab = d["true_label"] if "true_label" in d.columns else d["label"]
    y = lab.map(lambda v: 1 if str(v).strip().lower() in ("ransomware", "1", "1.0", "true") else 0).to_numpy()
    return d.reset_index(drop=True), y, d[PROB_COLS].to_numpy(float)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def evaluate(P, y, t):
    """Alert rule of the backend. Returns metrics + detection cut of every run."""
    hit = P > t
    flagged = hit.any(1)
    first = np.where(flagged, hit.argmax(1), -1)
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum()); tn = int((~flagged & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    det = first[(y == 1) & flagged]
    return dict(n=len(y), n_ransom=int(y.sum()), n_benign=int((y == 0).sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, fpr=fpr, fpr_ci=wilson(fp, fp + tn),
                detected_at_10=float((det == 0).mean()) if len(det) else 0.0,
                mean_cut=float(np.mean([CUTS[i] for i in det])) if len(det) else float("nan"))


def choose(P, y, max_fpr, tol=0.005):
    grid = np.unique(np.concatenate([np.round(np.arange(0.01, 1.0, 0.005), 3), np.round(P.ravel(), 4)]))
    grid = grid[(grid > 0) & (grid < 1)]
    rows = [(t, evaluate(P, y, t)) for t in grid]
    ok = [(t, m) for t, m in rows if m["fpr"] <= max_fpr]
    if not ok:
        raise SystemExit(f"No threshold reaches FPR <= {max_fpr:.2%} on validation "
                         f"(lowest FPR = {min(m['fpr'] for _, m in rows):.2%}). Retrain or relax --max-fpr.")
    best = max(m["f1"] for _, m in ok)
    plateau = sorted(t for t, m in ok if m["f1"] >= best - tol)
    return float(plateau[len(plateau) // 2]), rows, (plateau[0], plateau[-1]), best


def fmt(m):
    lo, hi = m["fpr_ci"]
    return (f"F1={m['f1']:.3f} recall={m['recall']:.3f} FPR={m['fpr']:.3f} [{lo:.3f},{hi:.3f}] "
            f"(FP={m['fp']}/{m['n_benign']}, FN={m['fn']}/{m['n_ransom']}, "
            f"flagged at 10%={m['detected_at_10']:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="scores of the VALIDATION executions (threshold is chosen here)")
    ap.add_argument("--test", help="scores of the TEST executions (report only)")
    ap.add_argument("--zeroday", help="scores of the zero-day family (Ryuk) (report only; add benign test runs "
                                      "to this file or pass --test as well to get a meaningful FPR)")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/alert_config.json")
    ap.add_argument("--plot", default=None, help="png path (default: next to --out)")
    a = ap.parse_args()

    dv, yv, Pv = load(a.val)
    if (yv == 0).sum() < 60:
        print(f"WARNING: only {(yv == 0).sum()} benign validation runs -> FPR resolution "
              f"{1 / max((yv == 0).sum(), 1):.1%}; the threshold will be unstable.")
    t, rows, (lo, hi), best = choose(Pv, yv, a.max_fpr)
    mv = evaluate(Pv, yv, t)
    print(f"\nChosen threshold = {t:.3f}  (plateau {lo:.3f}-{hi:.3f}, best val F1 {best:.3f}, FPR cap {a.max_fpr:.0%})")

    report = {}
    splits = [("val", a.val)] + [(n, p) for n, p in (("test", a.test), ("zeroday", a.zeroday)) if p]
    for name, path in splits:
        _, y, P = load(path)
        report[name] = {str(th): evaluate(P, y, th) for th in (t, 0.5, 0.85)}
        print(f"\n[{name}] {path}")
        for th in (t, 0.5, 0.85):
            tag = "chosen" if th == t else "      "
            print(f"  thr={th:.3f} {tag} {fmt(report[name][str(th)])}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfg = dict(threshold=round(t, 4),
               rule="alert at the first cut (10/25/50/75%) whose probability is > threshold",
               selected_on="validation", max_fpr=a.max_fpr, plateau=[round(lo, 4), round(hi, 4)],
               val=dict(f1=mv["f1"], recall=mv["recall"], fpr=mv["fpr"], fpr_ci95=list(mv["fpr_ci"]),
                        n_benign=mv["n_benign"], n_ransom=mv["n_ransom"]),
               created=datetime.now(timezone.utc).isoformat(timespec="seconds"), source_val_file=os.path.basename(a.val))
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nwrote {a.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, [r[1]["recall"] for r in rows], label="recall")
        ax.plot(ts, [r[1]["f1"] for r in rows], label="F1")
        ax.plot(ts, [r[1]["fpr"] for r in rows], label="FPR")
        ax.axhline(a.max_fpr, color="grey", ls=":", lw=1)
        ax.axvline(t, color="red", ls="--", label=f"chosen {t:.2f}")
        ax.set_xlabel("threshold"); ax.set_ylabel("validation metric"); ax.legend(); ax.grid(alpha=.3)
        out_png = a.plot or os.path.join(os.path.dirname(os.path.abspath(a.out)), "threshold_curve.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=150)
        print(f"wrote {out_png}")
    except Exception as e:  # plotting must never block the config
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
