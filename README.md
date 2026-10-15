# SMS Spam Baseline

Baseline SMS spam detection example using TF–IDF features and a logistic regression classifier.

This repository contains a small, auditable script that downloads the UCI SMS Spam Collection dataset (if
not already present), trains a TF–IDF + Logistic Regression baseline, and writes test and cross-validation
metrics and figures to the `results/` directory.

## Files added
- `sms_spam_baseline.py` — main script (download, train, evaluate, save results)
- `requirements.txt` — minimal Python dependency list
- `.gitignore` — sensible Python ignores
- `LICENSE` — MIT license
- `README.md` — this file

## Quickstart

Clone and run:

```bash
git clone https://github.com/kethakabinoth/sms_spam_baseline.git
cd sms_spam_baseline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python sms_spam_baseline.py
```

The script will download the dataset into `data/` if needed and write outputs into `results/`.

## Requirements
See `requirements.txt` for required packages and suggested minimum versions.

## License
This project is licensed under the MIT License — see `LICENSE`.
