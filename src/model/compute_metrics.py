import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve)

ZERODAY_FAMILY = "Ryuk"
RANSOM = {"Conti", "LockBit", "Darkside", "Ryuk", "REvil", "WannaCry"}
TARGET_F1, TARGET_FPR = 0.80, 0.05


def point_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), accuracy=div(tp + tn, len(y)),
                precision=prec, recall=rec, specificity=div(tn, tn + fp), fpr=div(fp, fp + tn), fnr=div(fn, fn + tp),
                f1=0.0 if np.isnan(f1) else f1, balanced_accuracy=np.nanmean([rec, div(tn, tn + fp)]),
                mcc=matthews_corrcoef(y, pred) if 0 < pred.sum() < len(pred) or (tp + fn) and (tn + fp) else np.nan)


def ranking_metrics(y, s):
    ok = 0 < y.sum() < len(y)
    return dict(auc=roc_auc_score(y, s) if ok else np.nan, pr_auc=average_precision_score(y, s) if ok else np.nan)


def bootstrap(y, s, thr, n, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("f1", "fpr", "recall", "precision", "auc")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if not 0 < y[i].sum() < len(i):
            continue
        m, r = point_metrics(y[i], s[i], thr), ranking_metrics(y[i], s[i])
        for k in ("f1", "fpr", "recall", "precision"):
            acc[k].append(m[k])
        acc["auc"].append(r["auc"])
    out = {}
    for k, v in acc.items():
        v = [x for x in v if not np.isnan(x)]
        out[f"{k}_lo"], out[f"{k}_hi"] = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--out", default="metrics_out")
    ap.add_argument("--thr", action="append", default=[], help="name=value, repeatable (default 0.5=0.5)")
    ap.add_argument("--results", help="results.csv from zeroday_eval_torch.py, for a consistency check")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    thrs = dict(t.split("=") for t in a.thr) if a.thr else {"0.5": "0.5"}
    thrs = {k: float(v) for k, v in thrs.items()}

    R = pd.read_csv(a.per_run)
    R["label"] = R["label"].astype(int)
    cuts = sorted(R.fraction.unique())

    #  full metrics
    rows = []
    for (es, f), g in R.groupby(["eval_set", "fraction"]):
        y, s = g.label.values, g.score.values
        if len(set(y)) < 2:
            continue
        rk = ranking_metrics(y, s)
        for tn_, t in thrs.items():
            rows.append(dict(eval_set=es, cut_pct=f, thr_name=tn_, threshold=t, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                             **point_metrics(y, s, t), **rk, **bootstrap(y, s, t, a.boot)))
    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics_full.csv", index=False)

    if a.results and Path(a.results).exists():           
        ref = pd.read_csv(a.results)
        ref = ref.rename(columns={"fraction": "cut_pct"})
        j = M.merge(ref[["eval_set", "cut_pct", "thr_name", "f1", "fpr", "auc"]], on=["eval_set", "cut_pct", "thr_name"], suffixes=("", "_ref"))
        err = max((j.f1 - j.f1_ref).abs().max(), (j.fpr - j.fpr_ref).abs().max(), (j.auc - j.auc_ref).abs().max())
        print(f"consistency with {a.results}: {len(j)} rows compared, max abs diff = {err:.2e}")
        assert err < 1e-9, "metrics differ from results.csv"

    #  per family
    pf = []
    for (es, f, fam), g in R[R.eval_set.isin(["seen_test", "zeroday"])].groupby(["eval_set", "fraction", "family"]):
        for tn_, t in thrs.items():
            flag = (g.score >= t)
            pf.append(dict(eval_set=es, cut_pct=f, family=fam, kind="ransomware" if fam in RANSOM else "benign", thr_name=tn_,
                           n_runs=len(g), flagged=int(flag.sum()), flag_rate=flag.mean(), median_score=g.score.median()))
    PF = pd.DataFrame(pf).sort_values(["eval_set", "thr_name", "kind", "family", "cut_pct"])
    PF.to_csv(out / "per_family.csv", index=False)

    # detection point (per run)
    base = R[R.eval_set == "seen_test"].copy()
    ry = R[(R.eval_set == "zeroday") & (R.family == ZERODAY_FAMILY)]
    base = pd.concat([base, ry])
    runs = []
    for tn_, t in thrs.items():
        b = base.assign(flag=(base.score >= t))
        for rid, g in b.groupby("rel_dir"):
            g = g.sort_values("fraction")
            hit = g[g.flag]
            first = hit.iloc[0] if len(hit) else None
            runs.append(dict(thr_name=tn_, rel_dir=rid, family=g.family.iloc[0], hw=g.hw.iloc[0], label=int(g.label.iloc[0]),
                             detected=first is not None,
                             detection_cut_pct=first.fraction if first is not None else np.nan,
                             detection_time_s=first.seq_len if first is not None else np.nan,   
                             n_cuts_flagged=int(g.flag.sum())))
    D = pd.DataFrame(runs)
    D.to_csv(out / "detection_point_runs.csv", index=False)

    grp = lambda r: "Ryuk (unseen)" if r.family == ZERODAY_FAMILY else (r.family if r.label else "benign (all) - false alarms")
    D["group"] = D.apply(grp, axis=1)
    rows = []
    for (tn_, gname), g in D.groupby(["thr_name", "group"]):
        d = g[g.detected]
        q = lambda c, p: d[c].quantile(p) if len(d) else np.nan
        rows.append(dict(thr_name=tn_, group=gname, n_runs=len(g), detected=len(d), detected_pct=100 * len(d) / len(g),
                         never_flagged=len(g) - len(d), median_cut_pct=q("detection_cut_pct", .5), q1_cut_pct=q("detection_cut_pct", .25),
                         q3_cut_pct=q("detection_cut_pct", .75), median_time_s=q("detection_time_s", .5),
                         q1_time_s=q("detection_time_s", .25), q3_time_s=q("detection_time_s", .75)))
    DS = pd.DataFrame(rows).sort_values(["thr_name", "group"])
    DS.to_csv(out / "detection_point_summary.csv", index=False)

    #  cumulative (monitor) view
    cum = []
    for es, sub in (("zeroday", pd.concat([R[(R.eval_set == "zeroday")]])), ("seen_test", R[R.eval_set == "seen_test"])):
        for tn_, t in thrs.items():
            P = sub.assign(flag=(sub.score >= t)).pivot_table(index="rel_dir", columns="fraction", values="flag", aggfunc="max")
            lab = sub.drop_duplicates("rel_dir").set_index("rel_dir").label.reindex(P.index).values.astype(int)
            for c in cuts:
                any_flag = P[[x for x in cuts if x <= c]].max(axis=1).astype(int).values
                m = point_metrics(lab, any_flag.astype(float), 0.5)
                cum.append(dict(eval_set=es, thr_name=tn_, up_to_cut_pct=c, recall=m["recall"], fpr=m["fpr"], precision=m["precision"], f1=m["f1"]))
    C = pd.DataFrame(cum)
    C.to_csv(out / "cumulative_detection.csv", index=False)

    #  targets
    T = M[["eval_set", "cut_pct", "thr_name", "f1", "fpr"]].copy()
    T["f1_ok"] = T.f1 > TARGET_F1
    T["fpr_ok"] = T.fpr < TARGET_FPR
    T["both_ok"] = T.f1_ok & T.fpr_ok
    T.to_csv(out / "targets.csv", index=False)

    #  plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(cuts), figsize=(4 * len(cuts), 4), sharey=True)
    for a_, f in zip(np.atleast_1d(ax), cuts):
        for es, ls in (("seen_test", "--"), ("zeroday", "-")):
            g = R[(R.eval_set == es) & (R.fraction == f)]
            fp_, tp_, _ = roc_curve(g.label, g.score)
            a_.plot(fp_, tp_, ls=ls, label=f"{es} (AUC {roc_auc_score(g.label, g.score):.2f})")
        a_.plot([0, 1], [0, 1], c="grey", lw=.6); a_.axvline(TARGET_FPR, c="k", ls=":", lw=.8)
        a_.set_title(f"{f}% prefix"); a_.set_xlabel("FPR"); a_.legend(fontsize=7); a_.grid(alpha=.3)
    np.atleast_1d(ax)[0].set_ylabel("TPR")
    plt.tight_layout(); plt.savefig(out / "roc_curves.png", dpi=150); plt.close()

    t0 = list(thrs)[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for es, ls in (("seen_test", "--"), ("zeroday", "-")):
        g = C[(C.eval_set == es) & (C.thr_name == t0)]
        ax[0].plot(g.up_to_cut_pct, g.recall, marker="o", ls=ls, label=f"{es} recall")
        ax[1].plot(g.up_to_cut_pct, g.fpr, marker="o", ls=ls, label=f"{es} FPR")
    ax[0].set_title(f"Cumulative detection (alarm at any cut <= x), thr={t0}"); ax[1].set_title("Cumulative false-alarm rate")
    ax[1].axhline(TARGET_FPR, c="k", ls=":")
    for a_ in ax: a_.set_xticks(cuts); a_.set_xlabel("Prefix (%)"); a_.grid(alpha=.3); a_.legend()
    plt.tight_layout(); plt.savefig(out / "detection_curve.png", dpi=150); plt.close()

    #  report
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = ["eval_set", "cut_pct", "thr_name", "f1", "f1_lo", "f1_hi", "recall", "precision", "fpr", "fpr_lo", "fpr_hi", "auc", "auc_lo", "auc_hi", "pr_auc", "mcc"]
    print(M[M.thr_name == t0][show].round(3).to_string(index=False))
    print("\nDetection point (thr=%s)\n" % t0, DS[DS.thr_name == t0].round(1).to_string(index=False))
    print("\nCumulative detection (thr=%s)\n" % t0, C[C.thr_name == t0].round(3).to_string(index=False))
    print("\nTargets met (F1>0.80 and FPR<5%):", int(T.both_ok.sum()), "of", len(T), "rows")
    with open(out / "metrics_report.md", "w") as fh:
        fh.write(f"# Metrics (thresholds: {thrs})\n\n## Full metrics\n\n{M[show].round(3).to_markdown(index=False)}\n\n"
                 f"## Detection point\n\n{DS.round(1).to_markdown(index=False)}\n\n## Cumulative detection\n\n{C.round(3).to_markdown(index=False)}\n")


if __name__ == "__main__":
    main()
