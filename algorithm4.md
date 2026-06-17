Input features:
- Period
- Transit depth
- Transit duration
- SNR
- Odd-even transit depth difference
- Secondary eclipse depth
- Transit shape symmetry
- Number of transits detected
- Local noise level
- Stellar contamination/blending indicators if available

Class 1: Planetary Transit
Class 2: Eclipsing Binary
Class 3: Blended Source / False Positive
Class 4: Stellar Variability / Starspot
Class 5: Noise / Instrumental Artifact
Random Forest / XGBoost for tabular extracted features
1D CNN or LSTM for raw/folded light curve shape
Hybrid model = feature classifier + neural network classifier