import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD TESS FITS LIGHT CURVE
# ============================================================

def load_tess_fits(file_path):
    with fits.open(file_path, memmap=False) as hdul:
        hdul.verify("silentfix")
        data = hdul[1].data
        columns = data.columns.names
        header = hdul[0].header

        tic_id = header.get("TICID", "Unknown")
        sector = header.get("SECTOR", "Unknown")
        tessmag = header.get("TESSMAG", "Unknown")

        time = np.array(data["TIME"], dtype=float)

        if "PDCSAP_FLUX" in columns:
            flux = np.array(data["PDCSAP_FLUX"], dtype=float)
        elif "SAP_FLUX" in columns:
            flux = np.array(data["SAP_FLUX"], dtype=float)
        else:
            raise ValueError("No SAP_FLUX or PDCSAP_FLUX column found.")

        if "PDCSAP_FLUX_ERR" in columns:
            flux_err = np.array(data["PDCSAP_FLUX_ERR"], dtype=float)
        elif "SAP_FLUX_ERR" in columns:
            flux_err = np.array(data["SAP_FLUX_ERR"], dtype=float)
        else:
            flux_err = np.full_like(flux, np.nanstd(flux))

    return time, flux, flux_err, tic_id, sector, tessmag


# ============================================================
# 2. CLEANING AND DETRENDING
# ============================================================

def clean_light_curve(time, flux, flux_err):
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    median_flux = np.nanmedian(flux)
    if median_flux == 0 or not np.isfinite(median_flux):
        raise ValueError("Invalid median flux.")

    flux = flux / median_flux

    mad = median_abs_deviation(flux, nan_policy="omit")
    if mad == 0 or not np.isfinite(mad):
        mad = np.nanstd(flux)

    outlier_mask = np.abs(flux - np.nanmedian(flux)) < 6 * mad

    return time[outlier_mask], flux[outlier_mask], flux_err[outlier_mask]


def detrend_light_curve(time, flux):
    if len(flux) < 50:
        return flux

    window_length = min(401, len(flux) // 2 * 2 - 1)

    if window_length < 7:
        return flux

    if window_length % 2 == 0:
        window_length += 1

    trend = savgol_filter(flux, window_length=window_length, polyorder=2)

    trend[trend == 0] = 1.0

    return flux / trend


# ============================================================
# 3. BLS TRANSIT DETECTION
# ============================================================

def detect_transit_bls(time, flux):
    time_span = np.nanmax(time) - np.nanmin(time)

    min_period = 0.3
    max_period = min(20, time_span / 2)

    if max_period <= min_period:
        max_period = min_period + 0.5

    periods = np.linspace(min_period, max_period, 3000)

    max_duration = min(0.25, min_period * 0.8)
    durations = np.linspace(0.02, max_duration, 20)

    model = BoxLeastSquares(time, flux)
    result = model.power(periods, durations)

    best_index = np.argmax(result.power)

    best_period = float(result.period[best_index])
    best_duration = float(result.duration[best_index])
    best_t0 = float(result.transit_time[best_index])
    best_power = float(result.power[best_index])
    best_depth = float(abs(result.depth[best_index]))

    phase = ((time - best_t0 + 0.5 * best_period) % best_period) / best_period - 0.5

    duration_fraction = best_duration / best_period
    in_transit = np.abs(phase) < duration_fraction / 2
    out_transit = ~in_transit

    noise = np.nanstd(flux[out_transit])
    snr = best_depth / noise if noise > 0 else 0

    return {
        "period": best_period,
        "duration": best_duration,
        "t0": best_t0,
        "depth": best_depth,
        "snr": float(snr),
        "power": best_power,
        "phase": phase,
        "in_transit": in_transit,
        "out_transit": out_transit
    }


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def extract_features(time, flux, detection):
    period = detection["period"]
    duration = detection["duration"]
    depth = detection["depth"]
    snr = detection["snr"]
    phase = detection["phase"]

    in_transit = detection["in_transit"]
    out_transit = detection["out_transit"]

    duration_fraction = duration / period

    transit_flux = flux[in_transit]
    outside_flux = flux[out_transit]

    transit_std = float(np.nanstd(transit_flux)) if len(transit_flux) > 0 else 0
    outside_std = float(np.nanstd(outside_flux)) if len(outside_flux) > 0 else 0

    left = flux[(phase < 0) & (phase > -duration_fraction)]
    right = flux[(phase > 0) & (phase < duration_fraction)]

    if len(left) > 5 and len(right) > 5:
        shape_asymmetry = float(abs(np.nanmedian(left) - np.nanmedian(right)))
    else:
        shape_asymmetry = 1.0

    secondary_mask = np.abs(np.abs(phase) - 0.5) < duration_fraction / 2

    if np.sum(secondary_mask) > 5:
        secondary_depth = float(abs(1 - np.nanmedian(flux[secondary_mask])))
    else:
        secondary_depth = 0.0

    transit_numbers = np.floor((time - detection["t0"]) / period).astype(int)

    odd_depths = []
    even_depths = []

    for n in np.unique(transit_numbers):
        mask = transit_numbers == n

        if np.sum(mask) < 5:
            continue

        local_time = time[mask]
        local_flux = flux[mask]

        local_phase = ((local_time - detection["t0"] + 0.5 * period) % period) / period - 0.5
        local_in = np.abs(local_phase) < duration_fraction / 2

        if np.sum(local_in) > 3:
            local_depth = abs(1 - np.nanmedian(local_flux[local_in]))

            if n % 2 == 0:
                even_depths.append(local_depth)
            else:
                odd_depths.append(local_depth)

    if len(odd_depths) > 0 and len(even_depths) > 0:
        odd_even_difference = float(abs(np.mean(odd_depths) - np.mean(even_depths)))
    else:
        odd_even_difference = 0.0

    number_of_transits = int((np.nanmax(time) - np.nanmin(time)) / period)

    return {
        "period": period,
        "duration": duration,
        "depth": depth,
        "snr": snr,
        "duration_fraction": duration_fraction,
        "transit_std": transit_std,
        "outside_std": outside_std,
        "shape_asymmetry": shape_asymmetry,
        "secondary_depth": secondary_depth,
        "odd_even_difference": odd_even_difference,
        "number_of_transits": number_of_transits,
        "bls_power": detection["power"]
    }


# ============================================================
# 5. QUALITY FLAG
# ============================================================

def get_detection_quality_flag(snr):
    if snr >= 10:
        return "A"
    elif snr >= 7:
        return "B"
    elif snr >= 5:
        return "C"
    elif snr >= 3:
        return "D"
    else:
        return "F"


# ============================================================
# 6. TRANSIT RELIABILITY SCORE
# ============================================================

def score_candidate(features):
    periodicity_score = min(100, features["number_of_transits"] * 25)

    shape_score = 100 * np.exp(-features["shape_asymmetry"] * 500)

    snr_score = min(100, features["snr"] * 10)

    binary_penalty = (
        features["secondary_depth"] * 1000 +
        features["odd_even_difference"] * 1000 +
        max(0, features["depth"] - 0.05) * 1000
    )

    binary_rejection_score = max(0, 100 - binary_penalty)

    noise_score = 100 * np.exp(-features["outside_std"] * 100)

    reliability_score = (
        0.25 * periodicity_score +
        0.20 * shape_score +
        0.20 * snr_score +
        0.20 * binary_rejection_score +
        0.15 * noise_score
    )

    return {
        "periodicity_score": float(periodicity_score),
        "shape_score": float(shape_score),
        "snr_score": float(snr_score),
        "binary_rejection_score": float(binary_rejection_score),
        "noise_score": float(noise_score),
        "transit_reliability_score": float(reliability_score)
    }


# ============================================================
# 7. RULE-BASED AI CLASSIFIER
# ============================================================

def rule_based_classifier(features, scores):
    if features["snr"] < 3:
        return "Noise / Instrumental Artifact", 45.0

    if features["duration_fraction"] > 0.15:
        return "Noise / Stellar Variability", 50.0

    if features["depth"] > 0.08:
        return "Eclipsing Binary", 75.0

    if features["secondary_depth"] > features["depth"] * 0.35:
        return "Eclipsing Binary", 80.0

    if features["odd_even_difference"] > features["depth"] * 0.30:
        return "Eclipsing Binary", 80.0

    if scores["transit_reliability_score"] > 75:
        return "Planetary Transit Candidate", scores["transit_reliability_score"]

    if scores["transit_reliability_score"] > 55:
        return "Possible Transit / Needs Review", scores["transit_reliability_score"]

    return "Noise / Instrumental Artifact", scores["transit_reliability_score"]


# ============================================================
# 8. ADVERSARIAL STABILITY TEST
# ============================================================

def perturb_light_curve(time, flux):
    new_time = time.copy()
    new_flux = flux.copy()

    noise_level = np.nanstd(flux) * np.random.uniform(0.05, 0.20)
    new_flux = new_flux + np.random.normal(0, noise_level, size=len(new_flux))

    keep_fraction = np.random.uniform(0.90, 0.98)
    keep_mask = np.random.random(len(new_flux)) < keep_fraction

    new_time = new_time[keep_mask]
    new_flux = new_flux[keep_mask]

    flux_shift = np.random.normal(0, np.nanstd(flux) * 0.02)
    new_flux = new_flux + flux_shift

    return new_time, new_flux


def adversarial_stability_test(time, flux, original_period, n_trials=100):
    recovered = 0

    for _ in range(n_trials):
        try:
            t_mod, f_mod = perturb_light_curve(time, flux)

            if len(t_mod) < 100:
                continue

            det = detect_transit_bls(t_mod, f_mod)
            new_period = det["period"]

            relative_error = abs(new_period - original_period) / original_period

            if relative_error < 0.05 and det["snr"] >= 5 and det["depth"] < 0.05:
                recovered += 1

        except Exception:
            continue

    robustness_score = recovered / n_trials * 100

    return float(robustness_score), recovered, n_trials


# ============================================================
# 9. MULTI-STAGE VOTING SYSTEM
# ============================================================

def voting_system(features, scores, ai_label, ai_confidence, robustness_score):
    if features["snr"] < 3:
        votes = {
            "Planetary Transit Candidate": 0,
            "Eclipsing Binary": 0,
            "Noise / Artifact": 5,
            "Uncertain": 0
        }

        final_class = "Noise / Artifact"
        final_confidence = min(40, features["snr"] * 15)

        return final_class, float(final_confidence), votes

    if features["snr"] < 5 and robustness_score < 50:
        votes = {
            "Planetary Transit Candidate": 0,
            "Eclipsing Binary": 0,
            "Noise / Artifact": 4,
            "Uncertain": 1
        }

        final_class = "Noise / Artifact"
        final_confidence = min(50, features["snr"] * 12)

        return final_class, float(final_confidence), votes

    if features["duration_fraction"] > 0.15:
        votes = {
            "Planetary Transit Candidate": 0,
            "Eclipsing Binary": 1,
            "Noise / Artifact": 3,
            "Uncertain": 1
        }

        final_class = "Noise / Artifact"
        final_confidence = 35.0

        return final_class, final_confidence, votes

    votes = {
        "Planetary Transit Candidate": 0,
        "Eclipsing Binary": 0,
        "Noise / Artifact": 0,
        "Uncertain": 0
    }

    if features["snr"] >= 7:
        votes["Planetary Transit Candidate"] += 1
    elif features["snr"] < 5:
        votes["Noise / Artifact"] += 1
    else:
        votes["Uncertain"] += 1

    if scores["transit_reliability_score"] >= 70:
        votes["Planetary Transit Candidate"] += 1
    elif scores["transit_reliability_score"] < 45:
        votes["Noise / Artifact"] += 1
    else:
        votes["Uncertain"] += 1

    if scores["binary_rejection_score"] >= 70:
        votes["Planetary Transit Candidate"] += 1
    else:
        votes["Eclipsing Binary"] += 1

    if "Planet" in ai_label or "Transit" in ai_label:
        votes["Planetary Transit Candidate"] += 1
    elif "Binary" in ai_label:
        votes["Eclipsing Binary"] += 1
    elif "Noise" in ai_label or "Artifact" in ai_label:
        votes["Noise / Artifact"] += 1
    else:
        votes["Uncertain"] += 1

    if robustness_score >= 75:
        votes["Planetary Transit Candidate"] += 1
    elif robustness_score < 40:
        votes["Noise / Artifact"] += 1
    else:
        votes["Uncertain"] += 1

    final_class = max(votes, key=votes.get)

    final_confidence = (
        0.30 * scores["transit_reliability_score"] +
        0.25 * ai_confidence +
        0.25 * robustness_score +
        0.20 * scores["snr_score"]
    )

    return final_class, float(final_confidence), votes


# ============================================================
# 10. PLOTTING
# ============================================================

def plot_results(time, flux, detection, final_class, final_confidence, output_path):
    phase = detection["phase"]
    period = detection["period"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].scatter(time, flux, s=3, alpha=0.6)
    axes[0].set_xlabel("Time [BTJD]")
    axes[0].set_ylabel("Normalized Flux")
    axes[0].set_title("Cleaned and Detrended TESS Light Curve")

    axes[1].scatter(phase, flux, s=3, alpha=0.6)
    axes[1].axvline(0, linestyle="--")
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Normalized Flux")
    axes[1].set_title(
        f"Phase Folded Light Curve | Period = {period:.6f} days | "
        f"{final_class} | Confidence = {final_confidence:.2f}%"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 11. TEXT REPORT
# ============================================================

def save_text_report(result, output_txt):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("ASTRA LIGHT CURVE CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. CLASSIFICATION AND VOTING\n")
        f.write("-" * 50 + "\n")
        f.write(f"File: {result['file']}\n")
        f.write(f"Final Class: {result['final_class']}\n")
        f.write(f"Final Confidence: {result['final_confidence_percent']:.2f}%\n")
        f.write(f"Detection Quality Flag: {result['detection_quality_flag']}\n")
        f.write(f"AI Label: {result['ai_label']}\n")
        f.write(f"AI Confidence: {result['ai_confidence_percent']:.2f}%\n")
        f.write(f"Votes: {result['votes']}\n\n")

        f.write("2. DETECTED SIGNAL PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Orbital Period: {result['orbital_period_days']:.6f} days\n")
        f.write(f"Transit Duration: {result['transit_duration_days']:.6f} days\n")
        f.write(f"Transit Duration: {result['transit_duration_hours']:.4f} hours\n")
        f.write(f"Transit Depth: {result['transit_depth_fraction']:.6f}\n")
        f.write(f"Transit Depth: {result['transit_depth_percent']:.4f}%\n")
        f.write(f"SNR: {result['snr']:.2f}\n")
        f.write(f"Number of Transits: {result['number_of_transits']}\n\n")

        f.write("3. VALIDATION TESTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Periodicity Score: {result['periodicity_score']:.2f}\n")
        f.write(f"Shape Score: {result['shape_score']:.2f}\n")
        f.write(f"SNR Score: {result['snr_score']:.2f}\n")
        f.write(f"Binary Rejection Score: {result['binary_rejection_score']:.2f}\n")
        f.write(f"Noise Score: {result['noise_score']:.2f}\n")
        f.write(f"Transit Reliability Score: {result['transit_reliability_score']:.2f}\n")
        f.write(f"Adversarial Robustness: {result['adversarial_robustness_percent']:.2f}%\n")
        f.write(f"Adversarial Recovery: {result['adversarial_recovered']} / {result['adversarial_trials']}\n\n")

        f.write("4. FALSE POSITIVE INDICATORS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Secondary Eclipse Depth: {result['secondary_depth']:.6f}\n")
        f.write(f"Odd-Even Transit Difference: {result['odd_even_difference']:.6f}\n")
        f.write(f"Duration Fraction: {result['duration_fraction']:.6f}\n\n")

        f.write("5. INTERPRETATION\n")
        f.write("-" * 50 + "\n")
        f.write(result["interpretation"] + "\n")


# ============================================================
# 12. INTERPRETATION
# ============================================================

def generate_interpretation(result):
    if result["final_class"] == "Noise / Artifact":
        return (
            "The strongest dip-like pattern detected in this light curve is not strong "
            "enough to be considered a reliable planetary transit. The signal has low "
            "SNR and/or failed robustness validation. The period, depth, and duration "
            "reported above represent the best-fitting BLS signal, not a confirmed planet."
        )

    if result["final_class"] == "Eclipsing Binary":
        return (
            "The detected signal shows features more consistent with an eclipsing binary "
            "or false positive, such as large depth, secondary eclipse evidence, or "
            "odd-even transit differences."
        )

    if result["final_class"] == "Planetary Transit Candidate":
        return (
            "The detected signal passed multiple validation tests and is classified as "
            "a planetary transit candidate. Further scientific validation would require "
            "centroid analysis, multi-sector consistency checks, and comparison with "
            "known catalogs."
        )

    return (
        "The signal is uncertain and requires further validation with more data, "
        "additional sectors, and stronger astrophysical vetting."
    )


# ============================================================
# 13. FULL PIPELINE FOR ONE FILE
# ============================================================

def run_astra_pipeline(file_path, n_adversarial=100, output_dir="astra_outputs"):
    print("\n================ ASTRA PIPELINE STARTED ================\n")

    os.makedirs(output_dir, exist_ok=True)

    print("[1] Loading FITS file...")
    time, flux, flux_err, tic_id, sector, tessmag = load_tess_fits(file_path)

    print("[2] Cleaning light curve...")
    time, flux, flux_err = clean_light_curve(time, flux, flux_err)

    print("[3] Detrending light curve...")
    flux = detrend_light_curve(time, flux)

    print("[4] Detecting periodic dips using BLS...")
    detection = detect_transit_bls(time, flux)

    print("[5] Extracting features...")
    features = extract_features(time, flux, detection)

    print("[6] Computing validation scores...")
    scores = score_candidate(features)

    print("[7] Classifying signal...")
    ai_label, ai_confidence = rule_based_classifier(features, scores)

    print("[8] Running adversarial validation...")
    robustness_score, recovered, trials = adversarial_stability_test(
        time,
        flux,
        features["period"],
        n_trials=n_adversarial
    )

    print("[9] Running voting system...")
    final_class, final_confidence, votes = voting_system(
        features,
        scores,
        ai_label,
        ai_confidence,
        robustness_score
    )

    quality_flag = get_detection_quality_flag(features["snr"])

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    output_plot = os.path.join(output_dir, f"{base_name}_astra_plot.png")
    output_txt = os.path.join(output_dir, f"{base_name}_astra_report.txt")
    output_csv = os.path.join(output_dir, f"{base_name}_astra_report.csv")

    print("[10] Saving plot...")
    plot_results(time, flux, detection, final_class, final_confidence, output_plot)

    result = {
        "file": file_path,
        "final_class": final_class,
        "final_confidence_percent": final_confidence,
        "detection_quality_flag": quality_flag,
        "ai_label": ai_label,
        "ai_confidence_percent": ai_confidence,
        "tic_id": tic_id,
        "sector": sector,
        "tessmag": tessmag,

        "orbital_period_days": features["period"],
        "transit_duration_days": features["duration"],
        "transit_duration_hours": features["duration"] * 24,
        "transit_depth_fraction": features["depth"],
        "transit_depth_percent": features["depth"] * 100,
        "duration_fraction": features["duration_fraction"],
        "snr": features["snr"],
        "number_of_transits": features["number_of_transits"],

        "secondary_depth": features["secondary_depth"],
        "odd_even_difference": features["odd_even_difference"],

        "periodicity_score": scores["periodicity_score"],
        "shape_score": scores["shape_score"],
        "snr_score": scores["snr_score"],
        "binary_rejection_score": scores["binary_rejection_score"],
        "noise_score": scores["noise_score"],
        "transit_reliability_score": scores["transit_reliability_score"],

        "adversarial_robustness_percent": robustness_score,
        "adversarial_recovered": recovered,
        "adversarial_trials": trials,

        "votes": str(votes),
        "plot_file": output_plot
    }

    result["interpretation"] = generate_interpretation(result)

    pd.DataFrame([result]).to_csv(output_csv, index=False)
    save_text_report(result, output_txt)

    print("\n================ ASTRA FINAL RESULT ================\n")
    print(f"File: {file_path}")
    print(f"TIC ID: {tic_id}")
    print(f"Sector: {sector}")  
    print(f"TESS Magnitude: {tessmag}")
    print(f"Final Class: {final_class}")
    print(f"Final Confidence: {final_confidence:.2f}%")
    print(f"Detection Quality Flag: {quality_flag}")
    print(f"AI Label: {ai_label}")
    print(f"AI Confidence: {ai_confidence:.2f}%")
    print(f"Orbital Period: {features['period']:.6f} days")
    print(f"Transit Duration: {features['duration']:.6f} days")
    print(f"Transit Depth: {features['depth']:.6f}")
    print(f"Transit Depth: {features['depth'] * 100:.4f}%")
    print(f"SNR: {features['snr']:.2f}")
    print(f"Adversarial Robustness: {robustness_score:.2f}%")
    print(f"Votes: {votes}")
    print(f"\nSaved Text Report: {output_txt}")
    print(f"Saved CSV Report: {output_csv}")
    print(f"Saved Plot: {output_plot}")

    return result


# ============================================================
# 14. FOLDER MODE
# ============================================================

def run_folder(folder_path, n_adversarial=50, output_dir="astra_outputs"):
    fits_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(".fits") or file.lower().endswith(".fits.gz")
    ]

    if len(fits_files) == 0:
        print("No FITS files found in folder.")
        return

    all_results = []

    for file in fits_files:
        try:
            result = run_astra_pipeline(
                file_path=file,
                n_adversarial=n_adversarial,
                output_dir=output_dir
            )
            all_results.append(result)

        except Exception as e:
            print(f"\nERROR processing file: {file}")
            print(f"Reason: {e}\n")

    if len(all_results) > 0:
        batch_csv = os.path.join(output_dir, "astra_batch_results.csv")
        batch_txt = os.path.join(output_dir, "astra_batch_summary.txt")

        df = pd.DataFrame(all_results)
        df.to_csv(batch_csv, index=False)

        with open(batch_txt, "w", encoding="utf-8") as f:
            f.write("ASTRA BATCH CLASSIFICATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            for result in all_results:
                f.write("CLASSIFICATION AND VOTING\n")
                f.write("-" * 60 + "\n")
                f.write(f"File: {result['file']}\n")
                f.write(f"TIC ID: {result['tic_id']}\n")
                f.write(f"Sector: {result['sector']}\n")
                f.write(f"TESS Magnitude: {result['tessmag']}\n")
                f.write(f"Final Class: {result['final_class']}\n")
                f.write(f"Final Confidence: {result['final_confidence_percent']:.2f}%\n")
                f.write(f"Detection Quality Flag: {result['detection_quality_flag']}\n")
                f.write(f"AI Label: {result['ai_label']}\n")
                f.write(f"Votes: {result['votes']}\n\n")

                f.write("PARAMETERS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Orbital Period: {result['orbital_period_days']:.6f} days\n")
                f.write(f"Transit Duration: {result['transit_duration_days']:.6f} days\n")
                f.write(f"Transit Depth: {result['transit_depth_percent']:.4f}%\n")
                f.write(f"SNR: {result['snr']:.2f}\n")
                f.write(f"Adversarial Robustness: {result['adversarial_robustness_percent']:.2f}%\n")
                f.write("\n" + "=" * 60 + "\n\n")

        print(f"\nBatch CSV saved to: {batch_csv}")
        print(f"Batch text summary saved to: {batch_txt}")


# ============================================================
# 15. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ASTRA Exoplanet Transit Detection Pipeline")

    parser.add_argument("--input", type=str, help="Path to one FITS light curve file")
    parser.add_argument("--folder", type=str, help="Path to folder containing FITS files")
    parser.add_argument("--adversarial", type=int, default=100, help="Number of adversarial trials")
    parser.add_argument("--output", type=str, default="astra_outputs", help="Output folder")

    args = parser.parse_args()

    if args.input:
        run_astra_pipeline(
            file_path=args.input,
            n_adversarial=args.adversarial,
            output_dir=args.output
        )

    elif args.folder:
        run_folder(
            folder_path=args.folder,
            n_adversarial=args.adversarial,
            output_dir=args.output
        )

    else:
        print("Use one of these:")
        print("python main.py --input path/to/file.fits")
        print("python main.py --folder path/to/folder")


if __name__ == "__main__":
    main()