"""CipherStop - training loop with MLflow logging (KAN-27/28/29/30).

Expected data layout (output of the Spark sequence-generation notebook):
    <data-root>/10pct/  25pct/  50pct/  75pct/      (parquet folders)
Each parquet has at least: sequence (array<array<double>>), label (0/1).
If a `split` column (train / val / test) exists it is used; otherwise a
stratified 70/15/15 split is made (with a warning).
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.model.lstm import RansomwareLSTM


# --------------------------------------------------------------------- data
class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def collate(batch):
    """Pad variable-length sequences and return the real lengths."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = pad_sequence(seqs, batch_first=True)  # zero padding
    return x, lengths, torch.tensor(labels, dtype=torch.float32)


def to_array(seq):
    return np.stack([np.asarray(step, dtype=np.float32) for step in seq])


def load_cut(path: Path):
    df = pd.read_parquet(path)
    df = df[df["sequence"].map(len) > 0].reset_index(drop=True)
    if "split" in df.columns:
        parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    else:
        print("WARNING: no `split` column - using a random stratified split. "
              "Prefer the official split to avoid leakage between executions.")
        tr, rest = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        va, te = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)
        parts = {"train": tr, "val": va, "test": te}
    out = {}
    for name, p in parts.items():
        out[name] = ([to_array(s) for s in p["sequence"]], p["label"].astype(int).tolist())
    return out


def preprocess(parts):
    """log1p (counts/sizes are heavy-tailed) + standardise with TRAIN stats."""
    def log(seqs):
        return [np.log1p(np.clip(s, 0, None)) for s in seqs]

    parts = {k: (log(v[0]), v[1]) for k, v in parts.items()}
    allx = np.concatenate(parts["train"][0], axis=0)
    mean, std = allx.mean(0), allx.std(0) + 1e-6
    loaders_data = {}
    for k, (seqs, labels) in parts.items():
        tens = [torch.from_numpy((s - mean) / std) for s in seqs]
        loaders_data[k] = SeqDataset(tens, labels)
    return loaders_data, mean, std


# ------------------------------------------------------------------ metrics
def compute_metrics(y_true, y_prob, thr=0.5):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "auc": auc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, probs, labels = [], [], []
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logit, _ = model.logits(x, lengths)
        losses.append(criterion(logit, y).item() * len(y))
        probs += torch.sigmoid(logit).cpu().tolist()
        labels += y.cpu().tolist()
    return sum(losses) / len(labels), np.array(labels), np.array(probs)


class EarlyStopping:
    """Stop when the monitored metric (higher = better) stops improving."""

    def __init__(self, patience):
        self.patience, self.best, self.bad = patience, -np.inf, 0

    def step(self, value):
        if value > self.best:
            self.best, self.bad = value, 0
            return True  # improved
        self.bad += 1
        return False

    @property
    def stop(self):
        return self.bad >= self.patience


def log_confusion_matrix(y_true, y_prob, name, out_dir):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "ransomware"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
    path = out_dir / f"confusion_matrix_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="confusion_matrix")


# ----------------------------------------------------------------- training
def train_one_cut(cut, args, device):
    data_path = Path(args.data_root) / f"{cut}pct"
    parts = load_cut(data_path)
    datasets, mean, std = preprocess(parts)
    n_features = datasets["train"][0][0].shape[1]

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                        collate_fn=collate)
    train_dl, val_dl, test_dl = mk(datasets["train"], True), mk(datasets["val"], False), mk(datasets["test"], False)

    n_pos = sum(datasets["train"].labels)
    n_neg = len(datasets["train"]) - n_pos
    lens = [len(s) for s in datasets["train"].seqs]
    print(f"[{cut}%] sizes train/val/test = {len(datasets['train'])}/{len(datasets['val'])}/"
          f"{len(datasets['test'])} | train pos/neg = {n_pos}/{n_neg} | "
          f"seq_len min/mean/max = {min(lens)}/{np.mean(lens):.1f}/{max(lens)}")
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = RansomwareLSTM(n_features, args.hidden_dim, args.num_layers, args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"lstm_attn_{cut}pct") as run:
        mlflow.set_tag("sequence_cut", f"{cut}pct")
        mlflow.log_params({
            **{k: v for k, v in vars(args).items() if k not in ("cuts",)},
            "cut": cut, "n_features": n_features, "n_train": len(datasets["train"]),
            "n_val": len(datasets["val"]), "n_test": len(datasets["test"]),
            "pos_weight": round(pos_weight.item(), 3),
        })
        ckpt_path = ckpt_dir / f"lstm_attn_{cut}pct_{run.info.run_id[:8]}.pt"
        stopper = EarlyStopping(args.patience)

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for x, lengths, y in train_dl:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                logit, _ = model.logits(x, lengths)
                loss = criterion(logit, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                tot, n = tot + loss.item() * len(y), n + len(y)

            _, ytr, ptr = evaluate(model, train_dl, criterion, device)
            val_loss, yva, pva = evaluate(model, val_dl, criterion, device)
            tm, vm = compute_metrics(ytr, ptr), compute_metrics(yva, pva)
            mlflow.log_metrics({
                "train_loss": tot / n, "val_loss": val_loss,
                "train_f1": tm["f1"], "train_fpr": tm["fpr"], "train_auc": tm["auc"],
                "val_f1": vm["f1"], "val_fpr": vm["fpr"], "val_auc": vm["auc"],
            }, step=epoch)
            print(f"[{cut}%] epoch {epoch:03d} loss={tot/n:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f} val_fpr={vm['fpr']:.4f}")

            # KAN-29: keep the best checkpoint; KAN-28: early stopping
            monitor = vm["auc"] if not np.isnan(vm["auc"]) else -val_loss
            if stopper.step(monitor):
                torch.save({"state_dict": model.state_dict(), "n_features": n_features,
                            "scaler_mean": mean, "scaler_std": std, "args": vars(args),
                            "epoch": epoch, "val_auc": vm["auc"]}, ckpt_path)
            if stopper.stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # ---- reload the best model, evaluate on test, log artifacts
        best = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"])
        mlflow.log_metric("best_epoch", best["epoch"])
        for split_name, dl in (("val", val_dl), ("test", test_dl)):
            loss, y, p = evaluate(model, dl, criterion, device)
            m = compute_metrics(y, p)
            mlflow.log_metrics({f"final_{split_name}_{k}": v for k, v in m.items()})
            log_confusion_matrix(y, p, f"{cut}pct_{split_name}", ckpt_dir)  # KAN-30
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")     # KAN-29
        # pickle format: no model signature needed (newer MLflow defaults to pt2)
        mlflow.pytorch.log_model(model, artifact_path="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--cuts", type=int, nargs="+", default=[10, 25, 50, 75])
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--experiment", default="cipherstop-lstm-attention")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    print("Tracking URI:", mlflow.get_tracking_uri(), "| device:", device)

    for cut in args.cuts:
        train_one_cut(cut, args, device)


if __name__ == "__main__":
    main()
