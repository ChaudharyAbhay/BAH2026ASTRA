TESS Light Curve Data
        ↓
Data Cleaning and Normalization
        ↓
Noise Removal / Detrending
        ↓
Periodic Dip Detection
        ↓
Transit Feature Extraction
        ↓
Physics-Based Validation
        ↓
AI Classification
        ↓
Adversarial Stability Testing
        ↓
Multi-Stage Voting
        ↓
Final Class + Confidence Score + Transit Parameters


Problem we are solving
Detecting exoplanets from TESS light curves is difficult because planetary transits are extremely small brightness dips and can be hidden by detector noise, stellar variability, blending from nearby stars, and false positives such as eclipsing binaries. Our goal is to build an AI-enabled pipeline that detects periodic dips, classifies their astrophysical origin, estimates transit parameters, and provides confidence scores.


Our Solution
We propose ASTRA: Adversarial Stability-based Transit Recognition Architecture, a multi-validation AI pipeline for detecting exoplanet transit signals from noisy TESS light curves. Instead of depending on a single detection algorithm, ASTRA performs multiple validation tests including periodicity analysis, transit shape verification, signal-to-noise estimation, binary rejection, physics-based feature checking, AI classification, adversarial robustness testing, and final voting-based confidence scoring. This layered validation approach helps distinguish true planetary transits from eclipsing binaries, stellar variability, blending effects, and random noise.


USP
The unique strength of ASTRA is its multi-validation framework. Existing methods often focus on either detecting dips or classifying candidates, but ASTRA repeatedly verifies each candidate through independent tests before making a final decision. A signal is accepted only if it is periodic, physically consistent, statistically significant, AI-classified correctly, resistant to noise perturbations, and supported by the voting system. This makes the pipeline more reliable, explainable, and robust for noisy astronomical light curves in crowded stellar fields.


Features
1. Automatic TESS light curve preprocessing
2. Periodic dip detection using BLS/TLS
3. Transit parameter estimation
4. AI-based classification of events
5. Binary and false-positive rejection
6. Signal-to-noise and confidence scoring
7. Adversarial robustness testing
8. Final multi-stage voting decision
9. Visualization of detected transit events

Output
For each light curve, ASTRA gives:

- Detected class: planet, eclipsing binary, blend, starspot, or noise
- Orbital period
- Transit depth
- Transit duration
- Signal-to-noise ratio
- Robustness score
- Final confidence percentage
- Light curve visualization with marked transit events


To run:
python main.py --folder lightcurvefilesector102 (for folders)
python main.py --input lightcurvefilesector102/your_file.fits (for single files)