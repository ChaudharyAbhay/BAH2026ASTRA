# ASTRA

## Adversarial Stability-based Transit Recognition Architecture

AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

---

## Overview

ASTRA (Adversarial Stability-based Transit Recognition Architecture) is a physics-guided AI pipeline designed for the automated detection and classification of exoplanet transit signals in noisy TESS light curve data.

Unlike conventional transit search methods that rely on a single detection metric, ASTRA performs multiple independent validation tests before classifying a signal. The system combines transit detection, physics-based validation, confidence decomposition, adversarial robustness testing, and a multi-stage voting framework to improve reliability and interpretability.

The pipeline is designed to distinguish between:

* Planetary Transit Candidates
* Eclipsing Binaries
* Stellar Variability
* Blended Sources
* Instrumental Artifacts
* Noise

---

## Problem Statement

Exoplanet detection through transit photometry requires identifying extremely small periodic brightness variations in stellar light curves.

In real observations, light curves are contaminated by:

* Detector noise
* Stellar variability
* Starspots
* Crowded field blending
* Eclipsing binary systems
* Instrumental systematics

These effects often mimic planetary transit signals, making reliable detection difficult.

ASTRA addresses this challenge through a multi-validation architecture that combines signal processing, physics constraints, statistical validation, and machine learning-inspired classification.

---

## ASTRA Pipeline

```text
TESS Light Curve
        ↓
Data Cleaning
        ↓
Normalization
        ↓
Detrending
        ↓
BLS Transit Detection
        ↓
Physics Feature Extraction
        ↓
Transit Reliability Score
        ↓
Classification Engine
        ↓
Adversarial Stability Testing
        ↓
Multi-Stage Voting
        ↓
Final Classification
```

---

## Core Components

### 1. Light Curve Preprocessing

The raw TESS light curve is cleaned and normalized.

Operations performed:

* NaN removal
* Outlier rejection
* Flux normalization
* Trend removal using Savitzky-Golay filtering

Output:

* Cleaned light curve
* Detrended flux series

---

### 2. Transit Detection

Potential transit signals are detected using:

* Box Least Squares (BLS)

Estimated parameters:

* Orbital Period
* Transit Duration
* Transit Depth
* Transit Epoch
* Signal-to-Noise Ratio

---

### 3. Physics-Guided Validation

ASTRA extracts physically meaningful features:

* Transit depth
* Transit duration
* Transit symmetry
* Number of observed transits
* Secondary eclipse indicators
* Odd-even transit differences

These features are used to reject astrophysical false positives.

---

### 4. Transit Reliability Score

A composite reliability score is calculated:

Transit Reliability Score =

Periodicity Score

* Shape Score
* SNR Score
* Binary Rejection Score
* Noise Robustness Score

This score quantifies how likely a signal is to represent a genuine transit event.

---

### 5. Confidence Decomposition Engine

Rather than providing a single confidence value, ASTRA evaluates:

* Signal Quality
* Periodicity
* Transit Shape
* Binary Rejection
* Noise Robustness

These components are combined into an overall confidence estimate.

---

### 6. Adversarial Stability Testing

ASTRA introduces a novel validation framework.

After a candidate signal is detected:

* Gaussian noise is injected
* Random data points are removed
* Small perturbations are applied

Transit detection is repeated multiple times.

Example:

```text
100 trials
↓
Signal recovered 92 times
↓
Robustness Score = 92%
```

A genuine astrophysical signal should remain detectable under small perturbations.

---

### 7. Multi-Stage Voting Framework

Final classification is determined using multiple independent validators.

Voting Modules:

1. Signal-to-Noise Validator
2. Transit Reliability Validator
3. Binary Rejection Validator
4. Classification Engine
5. Adversarial Stability Validator

Output:

```text
Planet Candidate: 4 votes
Noise: 1 vote

Final Class:
Planetary Transit Candidate
```

---

## Detection Quality Flags

ASTRA assigns a quality grade based on SNR.

| Flag | SNR Range | Interpretation      |
| ---- | --------- | ------------------- |
| A    | ≥ 10      | Excellent Detection |
| B    | 7 – 10    | Strong Candidate    |
| C    | 5 – 7     | Moderate Candidate  |
| D    | 3 – 5     | Weak Candidate      |
| F    | < 3       | Likely Noise        |

---

## Output Products

For each light curve ASTRA generates:

### Classification Report

Contains:

* Final Classification
* Confidence
* Quality Flag
* Voting Results
* Validation Scores
* Physical Parameters

### CSV Report

Machine-readable summary containing:

* Orbital Period
* Transit Duration
* Transit Depth
* SNR
* Reliability Scores
* Robustness Scores

### Visualization

Generated plots:

* Detrended Light Curve
* Phase-Folded Light Curve

---

## Folder Structure

```text
SolutionAlgorithms/
│
├── main.py
├── requirements.txt
│
├── lightcurvefilesector102/
│   ├── *.fits
│
└── astra_outputs/
    ├── *.txt
    ├── *.csv
    ├── *.png
```

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy pandas matplotlib scipy astropy scikit-learn
```

---

## Usage

### Single Light Curve

```bash
python main.py --input lightcurvefilesector102/example.fits
```

### Entire Folder

```bash
python main.py --folder lightcurvefilesector102
```

### Custom Adversarial Trials

```bash
python main.py --folder lightcurvefilesector102 --adversarial 200
```

---

## Output Example

```text
Final Class:
Planetary Transit Candidate

Confidence:
88.6%

Detection Quality Flag:
B

Orbital Period:
3.412 days

Transit Duration:
0.152 days

Transit Depth:
0.72%

SNR:
8.4

Adversarial Robustness:
91%
```

---

## Future Improvements

* Transit Least Squares (TLS) implementation
* Convolutional Neural Networks (CNN)
* Graph Neural Networks
* Multi-sector validation
* Stellar centroid analysis
* TESS catalog cross-matching
* Explainable AI confidence decomposition
* GPU acceleration for large-scale surveys

---

## Authors

Team: [P.D.G. Engineers]
Abhay Singh , Shivam Gayan , Saransh Mishra

Developed for the Bharatiya Antariksh Hackathon 2026 | ISRO

Project Title:

ASTRA — Adversarial Stability-based Transit Recognition Architecture

AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
# BAH2026ASTRA

