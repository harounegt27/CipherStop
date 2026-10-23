# RanSMAP Ransomware Detection Project — Progress Summary

## 1. Project overview

The project uses the **RanSMAP 2024 ransomware behavioral features dataset** from Kaggle to explore differences between ransomware and benign applications, with the longer-term goal of preparing data for ransomware classification.

The dataset is very large (approximately **160.26 GB**), so the work was carried out in Kaggle notebooks rather than downloading the entire dataset locally.

**Kaggle dataset:** `hiranomanabu/ransmap-2024-ransomware-behavioral-features`

## 2. Dataset classes

### Ransomware families
- Conti
- Darkside
- LockBit
- REvil
- Ryuk
- WannaCry

### Benign applications
- AESCrypt
- Firefox
- Idle
- Office
- SDelete
- Zip

## 3. Work completed so far

### 3.1 Dataset exploration
- Explored the directory structure.
- Collected CSV paths recursively using `rglob("*.csv")`.
- Separated benign and ransomware files using their names.

### 3.2 ATA-read exploratory data analysis (EDA)
- Examined sample files from both classes.
- Checked file shapes and data types.
- Checked missing values and duplicate rows.
- Reviewed descriptive statistics and histograms.

### 3.3 Class-level comparison
Prepared an EDA report summarizing **1,799 ATA-read files**:
- 900 benign files
- 899 ransomware files

The report found no missing values in these ATA-read files. On average, ransomware recordings contained more rows per file than benign recordings.

### 3.4 Initial feature analysis
- Investigated column distributions in sample ATA-read files.
- Found that columns 4 and 5 were constant in the examined data, making them potential candidates for removal.
- This finding should be rechecked across the broader dataset before making a final preprocessing decision.

## 4. ATA-read EDA findings

| Metric | Benign | Ransomware |
|---|---:|---:|
| Number of files | 900 | 899 |
| Average rows per file | 285,255 | 485,166 |
| Average duplicate rows per file | 0.036 | 3.60 |
| Missing values | 0 | 0 |

### Interpretation
- Ransomware files had approximately 1.7 times as many rows per file on average.
- Benign files were almost duplicate-free on average, while ransomware files had more duplicate rows.
- These are exploratory findings, not proof that either feature can independently classify a file reliably.
- The row-count ranges overlap substantially, so row count alone should not be treated as a reliable classifier.

## 5. Behavioral data types

The dataset documentation groups files into five behavioral data types:

| Behavioral type | Benign files | Ransomware files | Progress |
|---|---:|---:|---|
| ATA-read | 900 | 899 | Initial EDA completed |
| ATA-write | 900 | 900 | Full analysis still needed |
| Memory-read | 1,800 | 1,800 | Full analysis still needed |
| Memory-write | 900 | 900 | Full analysis still needed |
| Memory-execution | 900 | 900 | Full analysis still needed |

### ATA-write discrepancy
During file collection, a discrepancy was encountered: **360 ATA-write CSV files were found, versus 720 expected**. This needs to be investigated by checking the directory structure, file naming, path filters, and counting logic before proceeding.

## 6. Planned next steps

1. Investigate and resolve the ATA-write file-count discrepancy.
2. Repeat and standardize the EDA for ATA-write, Memory-read, Memory-write, and Memory-execution.
3. Compare benign and ransomware behavior for each data type.
4. Validate the relevant columns and data assumptions across files rather than relying only on a few samples.
5. Recheck whether columns 4 and 5 are constant throughout the relevant datasets before dropping them.
6. Engineer file-level or window-level features, such as:
   - Read/write event counts per window
   - Inter-arrival time statistics
   - Address range or entropy
   - Block-size distribution
   - Duplicate-row rate
7. Prepare the validated features for ransomware classification.
8. Later, evaluate generalization using the dataset's **Variants** subset and test simultaneous activity using the **Mix** subset.

## 7. Important limitations

- The more complete analysis so far is for ATA-read; the other behavioral types still need systematic analysis.
- Some column interpretations were inferred from observed values and were not confirmed by column labels.
- The report's detailed distribution analysis included one sample file per class, so those sample-level observations should not be generalized to all files without further testing.
- The dataset report noted that the two inspected samples were recorded on different dates; recording conditions may partly explain differences.
- SDelete and AESCrypt are I/O-heavy benign applications and may resemble ransomware, so they deserve attention during feature analysis and evaluation.

## 8. Current milestone

**Most complete milestone:** ATA-read EDA and its report.

**Next major milestone:** Standardize and complete the EDA across all remaining behavioral data types, validate the assumptions, and then begin feature engineering for classification.
