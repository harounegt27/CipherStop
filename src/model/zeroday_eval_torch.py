import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

FRACTIONS = [10, 25, 50, 75]
ZERODAY_FAMILY = "Ryuk"
TEST_HW = {"celeron-gen6/ddr4-2133-8g", "i3-gen12/ddr4-3200-16g", "i3-gen12/ddr4-2666-8g"}
META_COLS = ["rel_dir", "family", "top", "cpu", "ram", "hw", "label"]


# data
def seqs_from_merged(root, split, fraction, fill_gaps):
    """Same logic as the sequence-generation notebook: sort by step, features in column order,
    keep the first max(1, int(n*fraction)) windows."""
    p = Path(root) / f"merged_{split}"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    feat = [c for c in d.columns if c not in META_COLS + ["step"]]
    rows = []
    for rid, g in d.groupby("rel_dir"):
        g = g.sort_values("step").set_index("step")[feat]
        if fill_gaps:
            g = g.reindex(range(int(g.index.max()) + 1), fill_value=0)
        x = g.to_numpy(dtype=np.float32)
        rows.append(dict(rel_dir=rid, family=group_first(d, rid, "family"),
                         label=int(group_first(d, rid, "label")),
                         sequence=list(x[:max(1, int(len(x) * fraction))])))
    return pd.DataFrame(rows)


def group_first(d, rid, col):
    return d.loc[d.rel_dir == rid, col].iloc[0]


def load_fraction(args, manifest, frac):
    if args.merged_root:
        parts = []
        for sp in ["train", "val", "test", "zeroday", "variants"]:
            r = seqs_from_merged(args.merged_root, sp, frac / 100, args.fill_gaps)
            if r is not None:
                parts.append(r)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_parquet(Path(args.processed_root) / f"{frac}pct")
    df = df[df["sequence"].map(len) > 0]
    cols = ["rel_dir", "split", "hw", "top"]
    n0 = len(df)
    df = df.drop(columns=[c for c in cols[1:] if c in df.columns]).merge(manifest[cols], on="rel_dir", how="inner")
    if len(df) == 0:
        raise SystemExit(f"{frac}%: no rel_dir of the parquet matches the manifest (run --inspect)")
    if len(df) < n0:
        print(f"[{frac}%] {n0 - len(df)} parquet runs not in manifest were dropped")
    if not (df.family == ZERODAY_FAMILY).any():
        raise SystemExit(f"{frac}%: no Ryuk runs in data/processed -> regenerate the sequences for the zeroday split")
    if not (df.split == "test").any():
        raise SystemExit(f"{frac}%: no test-split runs in the parquet")
    df["seq"] = df["sequence"].map(lambda s: np.stack([np.asarray(r, dtype=np.float32) for r in s]))
    return df.drop(columns=["sequence"]).reset_index(drop=True)


def find_repo_root():
    """Folder that contains src/model/lstm.py, searched upward from the cwd and from this file."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in [start, *start.parents]:
            if (p / "src" / "model" / "lstm.py").exists():
                return p
    return Path.cwd()


def resolve(p, root):
    """Relative paths: as given if they exist from the cwd, otherwise relative to the repo root."""
    if p is None:
        return None
    p = Path(p)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(root / p)


def inspect(args):
    man = pd.read_csv(args.manifest)
    print("manifest:", args.manifest, "| split counts:", man.split.value_counts().to_dict())
    for f in FRACTIONS:
        p = Path(args.processed_root) / f"{f}pct"
        d = pd.read_parquet(p)
        print(f"\n[{f}%] {p} rows={len(d)} columns={list(d.columns)}")
        has_split = "split" in d.columns
        print("   `split` column in parquet:", has_split,
              "-> train.py used the OFFICIAL split" if has_split else
              "-> train.py fell back to a RANDOM 70/15/15 split (Ryuk would be in training!)")
        ov = d.rel_dir.isin(man.rel_dir).mean() if "rel_dir" in d.columns else 0
        print(f"   rel_dir found in manifest: {ov:.0%}")
        if "family" in d.columns:
            print("   Ryuk runs:", int((d.family == ZERODAY_FAMILY).sum()))
        if has_split:
            print("   parquet split counts:", d.split.value_counts().to_dict())
            if "family" in d.columns:
                print("   Ryuk per split:", d[d.family == ZERODAY_FAMILY].split.value_counts().to_dict())
        L = d.sequence.map(len)
        print(f"   seq_len min/mean/max = {L.min()}/{L.mean():.1f}/{L.max()}  | n_features = {len(d.sequence.iloc[0][0])}")
    print("\ncheckpoints:")
    for f in FRACTIONS:
        for c in sorted(glob.glob(os.path.join(args.ckpt_dir, f"lstm_attn_{f}pct_*.pt")), key=os.path.getmtime):
            ck = torch.load(c, map_location="cpu", weights_only=False)
            print(f"  {f}% {Path(c).name}  saved={pd.Timestamp(os.path.getmtime(c), unit='s'):%Y-%m-%d %H:%M} "
                  f"epoch={ck['epoch']} val_auc={ck['val_auc']:.4f} hidden={ck['args']['hidden_dim']}")


#  model / ckpt
def locate_checkpoints(args):
    """One checkpoint per cut. With MLflow: newest FINISHED run of the cut (and verify its split sizes).
    Without: newest .pt in --ckpt-dir for the cut."""
    found, warnings = {}, []
    runs = {}
    if args.tracking_uri:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(args.tracking_uri)
        c = MlflowClient()
        exp = c.get_experiment_by_name(args.train_experiment)
        if exp is None:
            warnings.append(f"MLflow experiment '{args.train_experiment}' not found")
        else:
            for cut in FRACTIONS:
                r = c.search_runs([exp.experiment_id], f"tags.sequence_cut = '{cut}pct' and attributes.status = 'FINISHED'",
                                  order_by=["start_time DESC"], max_results=1)
                if r:
                    runs[cut] = r[0]
    for cut in FRACTIONS:
        pat = f"lstm_attn_{cut}pct_{runs[cut].info.run_id[:8]}.pt" if cut in runs else f"lstm_attn_{cut}pct_*.pt"
        cands = sorted(glob.glob(os.path.join(args.ckpt_dir, pat)), key=os.path.getmtime)
        if not cands:
            raise FileNotFoundError(f"no checkpoint for {cut}% matching {pat} in {args.ckpt_dir}")
        if args.select == "best_val" and len(cands) > 1:
            cands.sort(key=lambda c: torch.load(c, map_location="cpu", weights_only=False)["val_auc"])
        found[cut] = cands[-1]
        if len(cands) > 1:
            print(f"[{cut}%] {len(cands)} checkpoints match -> using {Path(cands[-1]).name} (--select {args.select})")
        if cut in runs:
            p = runs[cut].data.params
            found[f"{cut}_split"] = {k: int(p.get(k, -1)) for k in ("n_train", "n_val", "n_test")}
    return found, warnings


def build_model(ckpt, device):
    from src.model.lstm import RansomwareLSTM
    a = ckpt["args"]
    m = RansomwareLSTM(ckpt["n_features"], a["hidden_dim"], a["num_layers"], a["dropout"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


@torch.no_grad()
def score(model, seqs, mean, std, device, bs=64):
    """Exactly train.py's path: log1p(clip) -> standardise with the TRAIN stats -> pad + true lengths."""
    out, peak = [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        x = [torch.from_numpy(((np.log1p(np.clip(s, 0, None)) - mean) / std).astype(np.float32)) for s in chunk]
        lengths = torch.tensor([len(s) for s in x])
        s, w = model(pad_sequence(x, batch_first=True).to(device), lengths, return_attention=True)
        out += s.cpu().tolist()
        peak += (w.argmax(1).cpu().numpy() / lengths.numpy()).tolist()     
    return np.asarray(out), np.asarray(peak)


# metrics
def metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1=f1_score(y, pred, zero_division=0), recall=tp / max(tp + fn, 1),
                precision=tp / max(tp + fp, 1), fpr=fp / max(fp + tn, 1),
                auc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan)


def boot_ci(y, s, thr, n=1000, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(f1_score(y[i], (s[i] >= thr).astype(int), zero_division=0))
    return tuple(np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan)


def choose_threshold(val, max_fpr=0.05):
    """Chosen on VAL only: centre of the plateau of best mean F1 over cuts, with mean FPR <= max_fpr."""
    sc = {}
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        ms = [metrics(y, s, t) for y, s in val.values()]
        if np.mean([m["fpr"] for m in ms]) <= max_fpr:
            sc[t] = np.mean([m["f1"] for m in ms])
    if not sc:
        return 0.5
    b = max(sc.values())
    return float(np.median([t for t, v in sc.items() if v >= b - 1e-9]))


#  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="default: first existing of data/manifest.csv, manifest.csv, data/processed/manifest.csv")
    ap.add_argument("--processed-root", default="data/processed")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--merged-root", default=None)
    ap.add_argument("--fill-gaps", action="store_true", help="only with --merged-root; default matches the sequence notebook (no fill)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--select", choices=["latest", "best_val"], default="latest", help="which checkpoint when several exist per cut")
    ap.add_argument("--repo-root", default=".", help="folder containing src/model/lstm.py")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--train-experiment", default="cipherstop-lstm-attention")
    ap.add_argument("--experiment", default="cipherstop-zeroday")
    ap.add_argument("--neg-split", default="test", help="benign negatives for zero-day (use 'train' only for a smoke test)")
    ap.add_argument("--out", default="zeroday_results")
    ap.add_argument("--no-mlflow-log", action="store_true")
    args = ap.parse_args()
    root = find_repo_root()
    print("repo root:", root)
    if args.manifest is None:
        cands = ["data/manifest.csv", "manifest.csv", "data/processed/manifest.csv", "data/raw/manifest.csv"]
        args.manifest = next((str(root / c) for c in cands if (root / c).exists()), None)
        if args.manifest is None:
            raise SystemExit(f"manifest.csv not found under {root}; pass --manifest PATH")
    args.manifest = resolve(args.manifest, root)
    args.processed_root = resolve(args.processed_root, root)
    args.merged_root = resolve(args.merged_root, root)
    args.ckpt_dir = resolve(args.ckpt_dir, root)
    args.out = str(root / args.out) if not Path(args.out).is_absolute() else args.out
    args.repo_root = str(root)
    if args.inspect:
        return inspect(args)
    sys.path.insert(0, os.path.abspath(args.repo_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest)
    assert man[(man.family == ZERODAY_FAMILY) & man.split.isin(["train", "val", "test"])].empty, "Ryuk in train/val/test"
    expect = {"n_train": int((man.split == "train").sum()), "n_val": int((man.split == "val").sum()),
              "n_test": int((man.split == "test").sum())}

    ckpts, warns = locate_checkpoints(args)
    invalid = []
    for cut in FRACTIONS:
        got = ckpts.get(f"{cut}_split")
        if got and got != expect:
            invalid.append(f"{cut}%: MLflow n_train/val/test={got} but manifest={expect}")
    for w in warns + invalid:
        print("WARNING:", w)
    if invalid:
        print("!! The models were NOT trained on the manifest split (probably the random 70/15/15 fallback of "
              "train.py because the parquet had no `split` column). Ryuk may be in training: zero-day results INVALID. "
              "Retrain with a `split` column first.")

    rows, per = [], []
    zd_mask_names = ("zeroday", "zeroday_hw", "seen_test", "variants")
    val_cache, scores = {}, {}
    for f in FRACTIONS:
        ck = torch.load(ckpts[f], map_location=device, weights_only=False)
        model, mean, std = build_model(ck, device), ck["scaler_mean"], ck["scaler_std"]
        df = load_fraction(args, man, f)
        assert df.seq.iloc[0].shape[1] == ck["n_features"], "feature count differs from the checkpoint"
        assert not ((df.split.isin(["train", "val", "test"])) & (df.family == ZERODAY_FAMILY)).any()
        df["score"], df["attn_peak"] = score(model, list(df.seq), mean, std, device)
        df["seq_len"] = df.seq.map(len)
        scores[f] = df.drop(columns=["seq"])
        v = df[df.split == "val"]
        val_cache[f] = (v.label.values.astype(int), v.score.values)
        print(f"[{f}%] ckpt={Path(ckpts[f]).name} | runs={len(df)} | val AUC={roc_auc_score(*val_cache[f]):.3f}")

    thr_val = choose_threshold(val_cache)
    thrs = {"0.5": 0.5, "val": thr_val}
    print("thresholds:", thrs)

    for f, d in scores.items():
        neg = d[(d.split == args.neg_split) & (d.label == 0)]
        zd = d[d.family == ZERODAY_FAMILY]
        sets = {"zeroday": pd.concat([zd, neg]),
                "zeroday_hw": pd.concat([zd[zd.hw.isin(TEST_HW)], neg]),
                "seen_test": d[d.split == "test"],
                "variants": d[d.split == "variants"]}
        for name, e in sets.items():
            if e.empty or (name != "variants" and e.label.nunique() < 2):
                continue
            y, s = e.label.values.astype(int), e.score.values
            for tn_, t in thrs.items():
                m = metrics(y, s, t)
                lo, hi = boot_ci(y, s, t) if name != "variants" else (np.nan, np.nan)
                rows.append(dict(fraction=f, eval_set=name, thr_name=tn_, threshold=t, f1_ci_lo=lo, f1_ci_hi=hi, **m))
            per.append(e.assign(fraction=f, eval_set=name, pred_05=(s >= .5).astype(int), pred_val=(s >= thr_val).astype(int))
                       [["fraction", "eval_set", "rel_dir", "family", "hw", "split", "seq_len", "label", "score",
                         "pred_05", "pred_val", "attn_peak"]])

    S, R = pd.DataFrame(rows), pd.concat(per)
    S.to_csv(out / "results.csv", index=False)
    R.to_csv(out / "per_run_scores.csv", index=False)

    z = R[(R.eval_set == "zeroday") & (R.family == ZERODAY_FAMILY)]
    det = []
    for col in ("pred_05", "pred_val"):
        fl = z[z[col] == 1].groupby("rel_dir").fraction.min()
        v = pd.Series({r: fl.get(r, np.nan) for r in z.rel_dir.unique()})
        det.append(dict(threshold=col, n_runs=len(v), never_detected=int(v.isna().sum()), median_cut=v.median(),
                        p25=v.quantile(.25), p75=v.quantile(.75)))
    D = pd.DataFrame(det); D.to_csv(out / "detection_point.csv", index=False)

    ze = R[R.eval_set == "zeroday"]
    ze[(ze.label == 1) & (ze.pred_05 == 0)].to_csv(out / "false_negatives.csv", index=False)
    ze[(ze.label == 0) & (ze.pred_05 == 1)].to_csv(out / "false_positives.csv", index=False)

    piv = S[S.eval_set.isin(["zeroday", "seen_test"])].pivot_table(index=["thr_name", "fraction"], columns="eval_set", values="f1")
    if {"zeroday", "seen_test"} <= set(piv.columns):
        piv["gap_seen_minus_zeroday"] = piv["seen_test"] - piv["zeroday"]
    piv.to_csv(out / "gap_seen_vs_zeroday.csv")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for es, g in S[(S.thr_name == "0.5") & S.eval_set.isin(["zeroday", "seen_test"])].groupby("eval_set"):
        g = g.sort_values("fraction")
        for a_, c in zip(ax[:2], ["f1", "recall"]):
            a_.plot(g.fraction, g[c], marker="o", ls="-" if es == "zeroday" else "--", label=es)
    for a_, t in zip(ax[:2], ["F1", "Recall"]):
        a_.set_xticks(FRACTIONS); a_.set_xlabel("Prefix (%)"); a_.set_ylabel(t); a_.grid(alpha=.3); a_.legend()
    for f in FRACTIONS:
        zz = z[z.fraction == f]
        ax[2].hist(zz.score, bins=20, range=(0, 1), alpha=.4, label=f"{f}%")
    ax[2].axvline(.5, c="k", ls=":"); ax[2].set_title("Ryuk scores"); ax[2].legend()
    plt.tight_layout(); plt.savefig(out / "earliness.png", dpi=150)

    pd.set_option("display.width", 220)
    cols = ["fraction", "eval_set", "thr_name", "f1", "f1_ci_lo", "f1_ci_hi", "recall", "precision", "fpr", "auc", "n_pos", "n_neg"]
    print(S[S.eval_set != "variants"][cols].round(3).to_string(index=False))
    if (S.eval_set == "variants").any():
        print("\nvariants recall\n", S[S.eval_set == "variants"][["fraction", "thr_name", "recall"]].round(3).to_string(index=False))
    print("\ndetection point\n", D.to_string(index=False), "\n\ngap\n", piv.round(3).to_string())

    if args.tracking_uri and not args.no_mlflow_log:
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri); mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"zeroday-{ZERODAY_FAMILY}"):
            mlflow.set_tags({"eval": "zeroday", "heldout_family": ZERODAY_FAMILY,
                             "split_verified": str(not invalid), "scaler_source": "checkpoint"})
            mlflow.log_params({"val_threshold": thr_val, **{f"ckpt_{f}": Path(ckpts[f]).name for f in FRACTIONS}})
            for _, r in S[(S.eval_set == "zeroday") & (S.thr_name == "0.5")].iterrows():
                for k in ("f1", "recall", "precision", "fpr", "auc"):
                    mlflow.log_metric(f"zd_{int(r.fraction)}_{k}", float(r[k]))
            mlflow.log_artifacts(str(out))


if __name__ == "__main__":
    main()