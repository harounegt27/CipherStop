The notebook has three main parts:

1. Data ingestion and preprocessing
2. Feature extraction 
3. sequence generation

## Part 1 — Preparing the dataset

### Step 1: Set up Apache Spark (cells 0–5)

First, we check whether Python, Java and PySpark are available.
Then we create a SparkSession:

### Step 2: Identify our data (cells 6–7)

We use the RanSMAP dataset, containing computer behavior measurements.

The notebook identifies six types of activities, called streams.

| Stream          | Simple meaning                   |
| --------------- | -------------------------------- |
| `ata_read`      | Reading data from storage        |
| `ata_write`     | Writing data to storage          |
| `mem_exec`      | Executing instructions in memory |
| `mem_read`      | Reading memory                   |
| `mem_write`     | Writing memory                   |
| `mem_readwrite` | Reading and writing memory       |

For example, ransomware may perform many write operations while encrypting files.
The notebook scans the folders to find the CSV files associated with each execution.
It also extracts information such as the CPU, RAM, ransomware family and total file size.
This information is saved in:
`manifest_raw.csv`

### Step 3: Create labels and dataset splits (cell 8)

Next, we prepare the labels that a future machine learning model will use.
We assign:

- `label = 1`: ransomware family
- `label = 0`: other behavior


We also organize executions into different subsets.

| Split      | Purpose                              |
| ---------- | ------------------------------------ |
| `train`    | Train the AI model                   |
| `val`      | Validate model performance           |
| `test`     | Evaluate on held-out hardware        |
| `zeroday`  | Evaluate on the excluded Ryuk family |
| `mix`      | Separate evaluation data             |
| `variants` | Separate evaluation data             |
| `extra`    | Additional held-out hardware data    |



### Step 4: Select a small dataset for testing (cell 9)

Instead of processing everything immediately, we create a smaller dataset called:
`subset_smoke.csv` Because processing a large dataset takes time and memory.
This is called a smoke test.

### Step 5: Read and aggregate the CSV files (cell 10)

Initially, our data contains individual events.

For example:

| Time  | Activity | Size |
| ----- | -------- | ---- |
| 0.1 s | Write    | 100  |
| 0.3 s | Write    | 200  |
| 0.8 s | Read     | 150  |
| 1.2 s | Write    | 300  |

Instead of keeping every individual event, we group them into 1-second windows.
For every second and every stream, the notebook calculates features such as:

- Event count (`_n`): how many valid events occurred.
- Size sum (`_size_sum`): total amount of data involved.
- Address range (`_addr_range`): difference between maximum and minimum addresses.
- Entropy mean and max: statistics measuring data randomness, where available.
The notebook saves these aggregated results as Parquet files.

### Step 6: Merge the streams (cells 11–12)

Previously, each stream was processed separately.
Now we combine them.

For example:

| Second | Read count | Write count | Memory execution count |
| ------ | ---------- | ----------- | ---------------------- |
| 0      | 10         | 5           | 2                      |
| 1      | 15         | 8           | 3                      |
| 2      | 20         | 30          | 6                      |
The result is saved to:

`/kaggle/working/interim_smoke/windows`

### Step 7: Validate the processed data (cells 13–15)

Before continuing, we check the quality of the data.
The notebook verifies that:

- The expected executions are present.
- The step indices do not have gaps.
- Selected numerical features are not negative.
- Event-count columns do not contain missing values.

## Part 2 — Feature extraction and sequence generation

### Step 8: Load the prepared data (cells 17–20)

We start another SparkSession and load the processed Parquet data.
We are working with the features calculated during preprocessing.
We also inspect the number of rows, columns and their data types.

### Step 9: Separate metadata from features (cells 21–22)

Not every column should be used as an input feature.
For example:

- `family`: ransomware family name
- `label`: whether the execution is ransomware
- `cpu`, `ram`: hardware information
- `rel_dir`: execution identifier
- `step`: time-window position
These are treated as metadata.

### Step 10: Convert features into vectors (cells 23–25)

Now we arrange the data by execution and time step.


Then we combine the features of each second into an array:

Imagine that one second has three features:

- Read count = 10
- Write count = 5
- Memory execution count = 2

Instead of storing them separately, we represent them as:

`[10, 5, 2]`

This array is called a feature vector.

Each second now has its own feature vector.

### Step 11: Build complete sequences (cells 26–30)

Previously, we had one row per second.
Now we want one sequence per execution.
For example:

Before: separate time windows

| Step 0 | [10, 5, 2]  |
| ------ | ----------- |
| Step 1 | [15, 8, 3]  |
| Step 2 | [20, 30, 6] |

Group and sort by step

After: one sequence

[   [10,  5, 2],   [15,  8, 3],   [20, 30, 6] ]

The notebook uses:


This collects the time steps belonging to the same execution.

than sorts them chronologically.

Finally, the `transform` expression keeps only the feature arrays.

Why is the order important?

Because ransomware behavior can change over time. An AI model analyzing a sequence can potentially learn how an attack develops, not just which activities occurred.

### Step 12: Calculate sequence lengths (cell 31)

For example, a sequence containing 100 windows has `seq_len = 100`.

The notebook calculates the minimum, average and maximum sequence lengths.

This helps us understand how long the executions are.

### Step 13: Generate partial sequences (cells 32–38)

Now we create four versions of each sequence.

The remaining portion of each sequence is excluded from that version.


Simple explanation:

- `seq`: the complete sequence.
- `ratio`: the percentage we want to keep.
- `len(seq)`: counts the time windows.
- `int(n * ratio)`: calculates the number of windows to keep.
- `seq[:cut]`: keeps only the beginning of the sequence.
- `max(1, ...)`: ensures that a nonempty sequence keeps at least one window.

We then create four datasets:

The notebook repeats this operation for 10%, 25%, 50% and 75%.

Why are we doing this?

Because we want to prepare data for early ransomware detection.

Instead of waiting until an execution finishes, we want to investigate whether a future model can recognize malicious behavior after observing only part of it.

For example, could the model recognize ransomware using just the first 10% of the execution?

That is what these datasets will help us investigate later.

Important: these percentages refer to the number of observed time windows, not a guaranteed percentage of the ransomware attack's actual progression.

### Step 14: Save the datasets (cells 39–42)

Finally, the notebook saves the four partial-sequence datasets.

| Output folder     | Content                    |
| ----------------- | -------------------------- |
| `processed/10pct` | First 10% of each sequence |
| `processed/25pct` | First 25%                  |
| `processed/50pct` | First 50%                  |
| `processed/75pct` | First 75%                  |

All four are saved in Parquet format.

The notebook also compares the number of executions across the original and shortened datasets, then lists the output folders.

Because we shortened the sequences rather than removing executions, the number of executions should remain the same.

