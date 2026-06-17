After a candidate transit is detected, ASTRA performs adversarial stability testing by slightly perturbing the light curve multiple times. These perturbations include adding Gaussian noise, randomly removing data points, shifting flux values slightly, and changing the sampling window. The transit detection algorithm is then rerun on each modified version. If the same transit period and event are repeatedly recovered, the signal is considered robust.

Input: Candidate transit signal

detected_count = 0
N = 1000

For i in range(N):
    modified_light_curve = perturb(original_light_curve)
    result = detect_transit(modified_light_curve)

    if result period is close to original period:
        detected_count += 1

Robustness Score = detected_count / N × 100

If the transit is recovered in 988 out of 1000 trials:

Robustness Score = 98.8%