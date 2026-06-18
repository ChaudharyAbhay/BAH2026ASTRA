Input: Time, Flux, Flux Error

Steps:
1. Remove NaN values and bad-quality data points.
2. Normalize flux around 1.0.
3. Remove long-term stellar/instrumental trends using median filtering or Savitzky-Golay filtering.
4. Detect and remove extreme outliers.
5. Produce cleaned light curve.

Output: Cleaned and normalized light curve.