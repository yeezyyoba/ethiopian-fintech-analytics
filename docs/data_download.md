# Data Download Guide

## Datasets Used in This Project

### 1. IEEE-CIS Fraud Detection (Fraud Detection)
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Files needed**: `train_transaction.csv`, `train_identity.csv`
- **Size**: ~450MB combined
- **Place in**: `data/raw/fraud/`

**Steps:**
1. Create a free Kaggle account at kaggle.com
2. Go to the competition page and click "Join Competition"
3. Download via browser or use the Kaggle CLI:
```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/fraud/
```

---

### 2. Give Me Some Credit (Credit Risk)
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
- **Files needed**: `cs-training.csv`
- **Size**: ~25MB
- **Place in**: `data/raw/credit/`

**Steps:**
```bash
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/credit/
```

---

## Setting Up Kaggle CLI

1. Go to kaggle.com → Your Profile → Settings → API → Create New Token
2. Download `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\YOU\.kaggle\kaggle.json` (Windows)
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

---

## Expected Folder Structure After Download

```
data/
├── raw/
│   ├── fraud/
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── credit/
│       └── cs-training.csv
```
