# Balanced, Calibrated LightGBM vs Random Forest for Flagging Malicious Entities

Analysts working with darknet data need to prioritise which actors or sites to investigate, and the ranking and probability scores they see matter. The paper "Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet" (2021) proposes machine learning tools for this kind of analysis. This experiment asks whether a class-balanced, calibrated gradient-boosted model can beat a standard random forest at flagging malicious entities, and whether its probabilities are more trustworthy.

**Question:** Can a class-balanced, calibrated LightGBM model flag malicious entities more reliably than a standard random forest on a public cybersecurity dataset, with probability outputs analysts can trust?

**Dataset:** UCI Phishing Websites (id 327), loaded with the `ucimlrepo` package. It contains URL and HTML features with a malicious-vs-benign label. It is a public stand-in for darknet actor data and is not the paper's own dataset.

**Baseline:** scikit-learn `RandomForestClassifier` with default settings.

**Improved model:** LightGBM with balanced class weights, hyperparameters chosen by cross-validation on the training split only, and isotonic calibration of probabilities via cross-validation.

Both models use the same stratified train/test split (fixed seed), the same preprocessing, the same evaluation function, and the same seeds. The main metric is PR-AUC on the held-out test set. Secondary metrics are the Brier score and F1 on the malicious class.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json`. Charts are saved to `figures/`.

## Results (latest run)

On the public UCI Phishing Websites dataset, a standard random-forest classifier reached PR-AUC 0.996 on the held-out test set, while a class-balanced, calibrated LightGBM reached 0.997, a small gain within likely split variability, with F1 essentially unchanged (0.973 vs 0.973).

| Metric | Value |
|---|---|
| baseline_score | 0.996 |
| improved_score | 0.9968 |
| improvement | 0.0007926 |
| baseline_score_std | 0.00009864 |
| improved_score_std | 0 |
| baseline_f1 | 0.9729 |
| improved_f1 | 0.9728 |
| baseline_brier | 0.01995 |
| improved_brier | 0.01709 |
| brier_improvement | 0.00286 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Phishing URLs are a proxy for darknet malicious actors, not the paper's data or task. Single split; differences of small magnitude may be within split variability. F1 uses a fixed 0.5 threshold on each model's probabilities. Hyperparameters were searched once with a fixed seed and a small grid for speed. The paper's own results are not stated, so no comparison to it is made.

## Reference

This project is motivated by the following paper:

> C Chen, C Peersman, M Edwards (2021). *Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet*. … Conference on Big. https://ieeexplore.ieee.org/abstract/document/9671906/

Paper author: Prof. Awais Rashid, University of Bristol (awais.rashid@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
