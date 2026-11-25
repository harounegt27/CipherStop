"""
simulation/replayer.py
Task 6 — CipherStop trace replayer
Sends test executions to POST /predict and records results.

Usage:
    python simulation/replayer.py
    python simulation/replayer.py --api http://localhost:8000
    python simulation/replayer.py --data data/test_executions --out simulation/results.csv
"""

import os
import csv
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API     = "http://localhost:8000"
DEFAULT_DATA    = "data/test_executions"
DEFAULT_OUT     = "simulation/results.csv"
ALERT_THRESHOLD = 0.85

RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "WannaCry", "Ryuk"}

FILE_PARAMS = ["ata_read", "ata_write", "mem_exec",
               "mem_read", "mem_readwrite", "mem_write"]

FIELDNAMES = [
    "cpu", "ram", "family", "execution", "true_label",
    "prob_10", "prob_25", "prob_50", "prob_75",
    "pred_10", "pred_25", "pred_50", "pred_75",
    "earliest_detection", "detection_score",
    "elapsed_sec", "status", "error"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_earliest_detection(predictions):
    for pct in ["10%", "25%", "50%", "75%"]:
        if predictions[pct]["prediction"] == 1:
            return pct, predictions[pct]["probability"]
    return None, None


def send_execution(api_url, exe_path):
    """Open 6 CSVs, POST to /predict, return (predictions, elapsed_sec)."""
    files = []
    handles = []
    try:
        for param in FILE_PARAMS:
            csv_path = Path(exe_path) / f"{param}.csv"
            fh = open(csv_path, "rb")
            handles.append(fh)
            files.append((param, (f"{param}.csv", fh, "text/csv")))

        t0 = time.time()
        resp = requests.post(f"{api_url}/predict", files=files, timeout=120)
        elapsed = round(time.time() - t0, 2)
        resp.raise_for_status()
        return resp.json()["predictions"], elapsed

    finally:
        for fh in handles:
            fh.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def run(api_url, data_root, output_csv):
    done = set()
    if Path(output_csv).exists():
        prev = pd.read_csv(output_csv)
        done = set(prev[prev['status'] == 'ok']['execution'].tolist())

    data_root = Path(data_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = failed = 0
    
    # Walk cpu/ram/family/execution
    for cpu in sorted(os.listdir(data_root)):
        for ram in sorted(os.listdir(data_root / cpu)):
            for family in sorted(os.listdir(data_root / cpu / ram)):
                family_path = data_root / cpu / ram / family
                true_label  = "Ransomware" if family in RANSOMWARE_FAMILIES else "Benign"

                for execution in sorted(os.listdir(family_path)):
                    exe_path = family_path / execution
                    if execution in done:
                        continue
                    total += 1
                    row = dict.fromkeys(FIELDNAMES, "")
                    row.update(cpu=cpu, ram=ram, family=family,
                               execution=execution, true_label=true_label)

                    try:
                        preds, elapsed = send_execution(api_url, exe_path)
                        earliest, det_score = find_earliest_detection(preds)

                        row.update(
                            prob_10 = preds["10%"]["probability"],
                            prob_25 = preds["25%"]["probability"],
                            prob_50 = preds["50%"]["probability"],
                            prob_75 = preds["75%"]["probability"],
                            pred_10 = preds["10%"]["prediction"],
                            pred_25 = preds["25%"]["prediction"],
                            pred_50 = preds["50%"]["prediction"],
                            pred_75 = preds["75%"]["prediction"],
                            earliest_detection = earliest or "none",
                            detection_score    = det_score or "",
                            elapsed_sec        = elapsed,
                            status             = "ok"
                        )
                        flag = "🚨" if earliest else "✅"
                        print(f"{flag} {family:12} | {cpu} | {execution[:30]:30} "
                              f"→ earliest={earliest or 'none':4}  "
                              f"score={det_score or '':4}  ({elapsed}s)")

                    except Exception as e:
                        row["status"] = "error"
                        row["error"]  = str(e)[:300]
                        failed += 1
                        print(f"💥 {family:12} | {execution[:30]} → {e}")

                    rows.append(row)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Executions total : {total}")
    print(f"Successful       : {total - failed}")
    print(f"Failed           : {failed}")
    print(f"Results saved to : {output_csv}")
    print(f"Timestamp        : {datetime.now().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",  default=DEFAULT_API)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out",  default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.api, args.data, args.out)