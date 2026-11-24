# RQ3 - Zero-day evaluation on an unseen family (Ryuk)

Nov 23, 2026 · @ayett

## Summary

The LSTM+Attention model does not generalize to Ryuk: zero-day F1 peaks at 0.70 (10% cut, FPR 10%) and falls to 0.00-0.19 at the 25%, 50% and 75% cuts, while the same models score F1 0.91-1.00 on seen families. The targets (F1 > 0.80 with FPR < 5%) are not met at any cut.

- **Best case:** 10% cut, F1 0.70 at threshold 0.5 (0.78 at the val-chosen threshold 0.195), but FPR is 0.10 / 0.15.
- **With FPR under 5%:** the best zero-day F1 is 0.19 (75% cut, recall 0.11).
- **From 25% on, Ryuk looks benign to the model:** its median score (0.0005-0.0035) equals the median benign score (0.001-0.003), and AUC drops to 0.65, 0.43 and 0.63.
- **The training was valid:** 770 / 220 / 330 runs in train / val / test, Ryuk only in the zero-day split.

## Protocol

Four frozen LSTM+Attention checkpoints (one per prefix cut: 10, 25, 50, 75%) were trained on the official split without Ryuk, then scored on Ryuk plus held-out benign runs. Nothing was fitted or tuned on Ryuk.

- **Training:** `train.py` on the full dataset with the manifest split (770 train / 220 val / 330 test runs, confirmed from the MLflow run parameters). Ryuk (150 runs: 120 `original` + 30 `extra`) is only in the `zeroday` split.
- **Preprocessing:** the scaler (log1p then z-score on train statistics) is read from each checkpoint, so Ryuk and seen families go through the identical path.
- **Zero-day set:** 150 Ryuk runs (positives) and the 180 benign runs of the test split (negatives). A hardware-matched variant keeps only the 30 Ryuk runs on the three test hardware configurations.
- **Seen-family reference:** the full test split (150 ransomware, 180 benign) with the same checkpoints.
- **Thresholds:** 0.5 (the value used in training reports) and 0.195, chosen on the val split only as the centre of the best-F1 plateau under a 5% mean FPR.
- **Uncertainty:** 95% bootstrap interval on F1, 1,000 resamples over runs.
- **Source data:** `zeroday_results/` from the Kaggle run (`results.csv`, `per_run_scores.csv`, `gap_seen_vs_zeroday.csv`, `detection_point.csv`).

## Results

Seen-family F1 rises from 0.91 to 1.00 with the prefix, while Ryuk F1 is highest at 10% and nearly zero at 25% and 50% (threshold 0.5; 150 Ryuk and 180 benign runs).

| Prefix cut | Seen F1 | Zero-day F1 (95% CI) | Ryuk recall | FPR | Zero-day AUC | Seen AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 10% | 0.913 | 0.703 (0.64-0.77) | 0.607 | 0.100 | 0.888 | 0.980 |
| 25% | 0.969 | 0.026 (0.00-0.06) | 0.013 | 0.006 | 0.649 | 1.000 |
| 50% | 0.993 | 0.000 (0.00-0.00) | 0.000 | 0.000 | 0.433 | 1.000 |
| 75% | 1.000 | 0.193 (0.11-0.27) | 0.107 | 0.000 | 0.629 | 1.000 |

At the val-chosen threshold (0.195) the picture is the same: zero-day F1 is 0.784 / 0.026 / 0.000 / 0.204 against seen F1 0.897 / 0.983 / 0.993 / 1.000. The 10% cut then has Ryuk recall 0.76 but FPR 0.15, above the 5% target; the val threshold did not transfer to the test hardware at that cut.

The hardware-matched set (30 Ryuk runs on test hardware) gives F1 0.61 / 0.00 / 0.00 / 0.13, so the collapse is not a hardware effect. Per-run scores and the earliness plot are in `zeroday_results/`.

## Detection point and additional metrics

Every seen family is flagged by the first (10%) cut, about 19-20 s into the run, while Ryuk is flagged at some cut in only 64% of runs (96 of 150). The detection point is the first prefix cut whose model flags the run; time in seconds equals the number of 1 s windows in that prefix (threshold 0.5).

| Group | Runs | Flagged at any cut | Never flagged | Median detection point |
| --- | --- | --- | --- | --- |
| Conti | 30 | 100% | 0 | 10% cut (20 s) |
| Darkside | 30 | 100% | 0 | 10% cut (19 s) |
| LockBit | 30 | 100% | 0 | 10% cut (19.5 s) |
| REvil | 30 | 100% | 0 | 10% cut (19 s) |
| WannaCry | 30 | 100% | 0 | 10% cut (19 s) |
| Ryuk (unseen) | 150 | 64% (96) | 54 | 10% cut (20 s) |
| Benign (false alarms) | 180 | 10.6% (19) | 161 | 10% cut (19 s) |

- **The metric is capped by the earliest cut.** All seen families sit at the first cut, so it cannot rank earliness; cuts below 10% (for example 2% and 5%) or fixed durations (5, 10, 15 s) are needed.
- **Combining the four models barely helps.** An alarm at any cut up to 75% raises Ryuk recall from 0.607 to 0.640 (5 more runs) and FPR from 0.100 to 0.106.
- **Ryuk's AUC is ranking noise, not partial signal.** At the 25% and 50% cuts Ryuk's median score (0.0017, 0.0005) lies inside the range of the benign families' medians (0.0007-0.0044, 0.0003-0.0015), so the model is confidently benign on it.
- **Targets:** F1 > 0.80 with FPR < 5% holds in 6 of 24 cut / threshold / set combinations, all on the seen-family test set at the 25-75% cuts; none on the zero-day set.

All values (also PR-AUC, MCC, specificity, balanced accuracy and bootstrap intervals) come from `compute_metrics.py` on `per_run_scores.csv` and match `results.csv` to 1e-16.

## Findings

The per-run scores show a confident miss, not a borderline one: from the 25% cut on, Ryuk scores are almost all below 0.01.

- **Seen ransomware is caught, Ryuk is not.** At the 25% cut the same model flags 97% of Conti, 97% of LockBit, 100% of Darkside, 93% of REvil and 87% of WannaCry test runs, but 1.3% of Ryuk runs (2 of 150).
- **Ryuk looks benign to the model.** Median Ryuk score is 0.0017 / 0.0005 / 0.0035 at the 25 / 50 / 75% cuts, against 0.001 / 0.001 / 0.003 for benign runs. An AUC of 0.43 at 50% means Ryuk scores slightly lower than benign.
- **The 10% signal is weak and not confirmed later.** 91 of 150 Ryuk runs are flagged at 10%, but benign runs are flagged too (FPR 0.10: AESCrypt 23%, Idle 20%, Firefox 10%, Office 7%). Of the 91, 78 are never flagged at a later cut; 54 Ryuk runs are never flagged at any cut. The coarse detection point is therefore 10% for 91 runs, undefined for 54.
- **Late detections come from the slowest hardware only.** All 16 runs flagged at the 75% cut are on Celeron configurations (20-40% of runs on each, except `ddr4-1866-8g`), none on i3, i5 or i7.
- **One odd configuration.** `i3-gen12/ddr5-4800-16g` (not in training) has the lowest 10% detection (20%, median score 0.03) and the source of both 25% detections (2 of its 10 runs).
- **Sequence lengths are normal.** Ryuk prefixes have the same length as other families (20.7 vs 18-20 steps at 10%), and the 10% AUC of 0.89 shows the features are read correctly, so a pipeline error is unlikely.

## Interpretation, limits and next steps

The data show that the models detect the five training families, not ransomware in general; the reason for the Ryuk failure is not yet established. Three explanations fit the numbers and none is tested:

1. **Family signatures.** The models learned the sustained encryption pattern of Conti, LockBit, Darkside, REvil and WannaCry, and Ryuk's mid-run activity resembles benign activity.
2. **Front-loaded activity.** Ryuk may act early and then go quiet, which would fit the best result at 10% and the fall after it (the 10% hits are also shared with AESCrypt and Idle start-up activity).
3. **Hardware-dependent pacing.** Slower Celeron machines may stretch Ryuk's activity into the prefix, which would explain why all late detections are on Celeron.

Ryuk is widely reported to be related to Conti, which is in training, so this failure is more striking than a miss on an unrelated family; that relationship is background knowledge and was not checked here.

**Limits of this evidence**

- One held-out family is one data point; it cannot separate a Ryuk-specific outlier from a general lack of generalization.
- A prefix cut is a fraction of the run's time windows, not of the attack's progress, so "early detection" here means early in the recording.
- The detection point is coarse (four cuts, four separate models, no continuous monitoring).
- Seen-family F1 is saturated at 50-75%, so it does not discriminate between models.
- All negatives are the 180 benign test runs, shared by the zero-day and seen-family sets.

