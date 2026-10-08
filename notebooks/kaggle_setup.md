# Running CipherStop on Kaggle Notebooks

## Setup
1. Go to https://www.kaggle.com
2. Create a new Notebook
3. Add the RanSMAP dataset:
   - Notebook → Add Data → search "ransmap-2024-ransomware-behavioral-features"
   - Data will be available at: /kaggle/input/ransmap-2024-ransomware-behavioral-features/

## Connect to GitHub
Run these cells at the start of every Kaggle Notebook session:

```python
import subprocess

# Clone the repo
subprocess.run([
    "git", "clone",
    "https://github.com/harounegt27/CipherStop.git"
])

# Install dependencies
subprocess.run([
    "pip", "install", "-r",
    "CipherStop/requirements.txt"
])
```

## Run the pipeline
```python
import os
os.chdir("CipherStop")

# Run DVC pipeline
subprocess.run(["dvc", "repro"])
```

## Push results back
```python
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "exp: run pipeline on Kaggle"])
subprocess.run(["git", "push"])
```