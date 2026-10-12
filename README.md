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

## Reference

This project is motivated by the following paper:

> C Chen, C Peersman, M Edwards (2021). *Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet*. … Conference on Big. https://ieeexplore.ieee.org/abstract/document/9671906/

Paper author: Prof. Awais Rashid, University of Bristol (awais.rashid@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
