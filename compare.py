#!/usr/bin/env python3
"""
Process paired p/l wav files and compute objective metrics.

Filename pattern: sx_y_zzz.wav
  - sx  = speaker ID, e.g. s1, s2, ...
  - y   = 'p' or 'l'
  - zzz = utterance ID, e.g. bbai2p, sbwe1n, ...

For each *_p_*.wav file:
  - Apply volume (in dB) and pitch shift (in semitones) to create a processed file.
  - Save as <original_basename>_proc.wav in output folder.
  - Find matching *_l_* file (same speaker + utterance ID).
  - Compute:
      * Mel-cepstral distance (MCD)
      * Spectral cross-correlation
      * HASPI (pyclarity)
      * HASQI (pyclarity)
  - Print per-pair metrics and overall averages.

Dependencies:
    pip install numpy scipy librosa soundfile pyclarity

Usage:
Simple example: +3 dB louder and +2 semitones pitch shift
    python compare.py /path/to/wavs 3.0 2.0

With explicit output directory:
    python process_pl_pairs.py /path/to/wavs 0.0 -1.5 -o /path/to/output
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import sys
from pathlib import Path

mel_cepstral_distance_path = Path("D:\Git_BAA\Evaluations_all\mel-cepstral-distance-main\src")
sys.path.append(str(mel_cepstral_distance_path))
from mel_cepstral_distance.api import compare_audio_files

# Optional imports: pyclarity for HASPI/HASQI
try:
    from clarity.evaluator.haspi.haspi import haspi_v2
    from clarity.evaluator.hasqi.hasqi import hasqi_v2
    from clarity.utils.audiogram import Audiogram

    HASPI_HASQI_AVAILABLE = True
except ImportError:
    HASPI_HASQI_AVAILABLE = False


# --------- Data structures --------- #

@dataclass
class Metrics:
    filename_p: str
    filename_l: str
    mcd: float
    spectral_corr: float
    haspi: Optional[float]
    hasqi: Optional[float]


# --------- Audio utilities --------- #

def load_mono(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as mono.

    Args:
        path: Path to wav file.
        target_sr: If not None, resample to this sample rate.

    Returns:
        (audio, sample_rate)
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """
    Save mono audio to a WAV file.
    """
    sf.write(path, audio, sr)


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain in dB: y = x * 10^(gain_db / 20)
    """
    gain_lin = 10.0 ** (gain_db / 20.0)
    return audio * gain_lin


def apply_pitch_shift_semitones(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Pitch-shift audio by the given number of semitones, keeping duration.
    Uses librosa.effects.pitch_shift.
    """
    if semitones == 0.0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


# --------- Metric computation --------- #

def compute_mcd(
    ref: np.ndarray,
    target: np.ndarray,
    sr: int,
    n_mfcc: int = 25,
) -> float:
    """
    Compute Mel Cepstral Distance (MCD) between reference and target.

    Implementation:
      - Compute MFCCs (as a proxy for mel-cepstra) for both signals.
      - Align MFCC sequences with DTW.
      - MCD = (10 / ln(10)) * sqrt(2) * mean over aligned frames of L2 distance.

    Returns:
        MCD in dB (float).
    """
    # Ensure neither is zero-length
    if len(ref) == 0 or len(target) == 0:
        return np.nan

    # MFCCs: shape (n_mfcc, T)
    mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)
    mfcc_target = librosa.feature.mfcc(y=target, sr=sr, n_mfcc=n_mfcc)

    # Transpose to (T, n_mfcc) for DTW
    X = mfcc_ref.T
    Y = mfcc_target.T

    # DTW alignment based on Euclidean distance
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")

    # Extract aligned frames and compute average L2 distance
    dists = []
    for i, j in wp:
        diff = X[i] - Y[j]
        dists.append(np.linalg.norm(diff))

    if not dists:
        return np.nan

    dists = np.asarray(dists, dtype=float)
    factor = 10.0 / np.log(10.0) * np.sqrt(2.0)
    mcd = factor * np.mean(dists)
    return float(mcd)


def compute_spectral_cross_correlation(
    ref: np.ndarray,
    target: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """
    Compute a simple spectral cross-correlation metric between reference and target.

    Implementation:
      - Compute magnitude STFT for each signal.
      - Average magnitude across time to get one spectrum per signal.
      - Compute Pearson correlation coefficient between the two spectra.

    Returns:
        Correlation coefficient in [-1, 1] (float).
    """
    if len(ref) == 0 or len(target) == 0:
        return np.nan

    S_ref = librosa.stft(ref, n_fft=n_fft, hop_length=hop_length)
    S_tgt = librosa.stft(target, n_fft=n_fft, hop_length=hop_length)

    mag_ref = np.mean(np.abs(S_ref), axis=1)
    mag_tgt = np.mean(np.abs(S_tgt), axis=1)

    # Remove any-constant arrays to avoid division by zero
    if np.allclose(mag_ref, mag_ref[0]) or np.allclose(mag_tgt, mag_tgt[0]):
        return np.nan

    # Pearson correlation
    mag_ref_centered = mag_ref - np.mean(mag_ref)
    mag_tgt_centered = mag_tgt - np.mean(mag_tgt)

    numerator = np.sum(mag_ref_centered * mag_tgt_centered)
    denominator = np.sqrt(np.sum(mag_ref_centered ** 2) * np.sum(mag_tgt_centered ** 2))

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def compute_haspi_hasqi(
    ref: np.ndarray,
    ref_sr: int,
    target: np.ndarray,
    target_sr: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute HASPI and HASQI using pyclarity.

    Uses a 'normal hearing' audiogram (0 dB loss at standard frequencies)
    for both metrics.

    Returns:
        (haspi_score or None, hasqi_score or None)

    If pyclarity is not available, returns (None, None).
    """
    if not HASPI_HASQI_AVAILABLE:
        return None, None

    # Ensure non-empty audio
    if len(ref) == 0 or len(target) == 0:
        return None, None

    # Standard frequencies assumed by HASPI/HASQI
    freqs = np.array([250, 500, 1000, 2000, 4000, 6000], dtype=float)
    levels = np.zeros_like(freqs, dtype=float)  # 0 dB HL => normal hearing

    audiogram = Audiogram(levels=levels, frequencies=freqs)

    # HASPI
    try:
        haspi_score, _ = haspi_v2(
            reference=ref,
            reference_sample_rate=float(ref_sr),
            processed=target,
            processed_sample_rate=float(target_sr),
            audiogram=audiogram,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] HASPI computation failed: {e}")
        haspi_score = None

    # HASQI
    try:
        hasqi_combined, _, _, _ = hasqi_v2(
            reference=ref,
            reference_sample_rate=float(ref_sr),
            processed=target,
            processed_sample_rate=float(target_sr),
            audiogram=audiogram,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] HASQI computation failed: {e}")
        hasqi_combined = None

    return haspi_score, hasqi_combined


# --------- File / naming utilities --------- #

FILENAME_RE = re.compile(r"^(?P<speaker>s\d+)_"
                         r"(?P<kind>[pl])_"
                         r"(?P<utt>.+)\.wav$")


def parse_filename(fname: str) -> Optional[Dict[str, str]]:
    """
    Parse filenames like 's1_p_swbxza.wav' into components.

    Returns dict with keys: speaker, kind, utt
    or None if pattern does not match.
    """
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    return m.groupdict()


def find_l_pair_for_p(
    p_path: str,
    input_dir: str,
) -> Optional[str]:
    """
    Given a *_p_*.wav file, find the matching *_l_*.wav file in the same folder.

    Returns:
        Full path to l-file or None if not found.
    """
    base = os.path.basename(p_path)
    info = parse_filename(base)
    if info is None:
        return None

    speaker = info["speaker"]
    utt = info["utt"]

    l_name = f"{speaker}_l_{utt}.wav"
    l_path = os.path.join(input_dir, l_name)

    if os.path.exists(l_path):
        return l_path
    return None


def ensure_output_dir(input_dir: str, out_dir: Optional[str]) -> str:
    """
    Decide on and create the output directory.

    If out_dir is None, create <input_dir>/processed
    """
    if out_dir is None:
        out_dir = os.path.join(input_dir, "processed")

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def make_processed_filename(base_name: str) -> str:
    """
    Given 's1_p_swbxza.wav', produce 's1_p_swbxza_proc.wav'
    """
    if base_name.lower().endswith(".wav"):
        base_name = base_name[:-4]
    return base_name + "_proc.wav"


# --------- Main processing logic --------- #

def process_folder(
    input_dir: str,
    gain_db: float,
    pitch_semitones: float,
    output_dir: Optional[str] = None,
) -> List[Metrics]:
    """
    Process all *_p_*.wav files in input_dir.

    Steps:
      - Transform p-files (gain + pitch) and save to output_dir.
      - Find corresponding l-file.
      - Compute metrics for each p/l pair.

    Returns:
        List[Metrics] with per-pair scores.
    """
    input_dir = os.path.abspath(input_dir)
    out_dir = ensure_output_dir(input_dir, output_dir)

    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Gain (dB)   : {gain_db}")
    print(f"Pitch (st)  : {pitch_semitones}")
    if not HASPI_HASQI_AVAILABLE:
        print("[INFO] pyclarity not installed; HASPI/HASQI will be skipped.")

    metrics_list: List[Metrics] = []

    # Collect all *_p_*.wav
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        if "_p_" not in fname:
            continue

        info = parse_filename(fname)
        if info is None or info["kind"] != "p":
            continue

        p_path = os.path.join(input_dir, fname)
        l_path = find_l_pair_for_p(p_path, input_dir)

        if l_path is None:
            print(f"[WARN] No matching l-file found for: {fname}")
            continue

        # --- Load and transform p audio ---
        p_audio, p_sr = load_mono(p_path, target_sr=None)
        p_audio_proc = apply_gain_db(p_audio, gain_db)
        p_audio_proc = apply_pitch_shift_semitones(p_audio_proc, p_sr, pitch_semitones)

        # Save processed p audio
        out_fname = make_processed_filename(fname)
        out_path = os.path.join(out_dir, out_fname)
        save_wav(out_path, p_audio_proc, p_sr)

        # --- Load l audio (reference) ---
        l_audio, l_sr = load_mono(l_path, target_sr=None)

        # For our own metrics, align in length and sample rate
        # (HASPI/HASQI can handle different sample rates internally)
        min_len = min(len(p_audio_proc), len(l_audio))
        if min_len == 0:
            print(f"[WARN] One of the files is empty: {fname} / {os.path.basename(l_path)}")
            mcd = np.nan
            corr = np.nan
            haspi = None
            hasqi = None
        else:
            p_aligned = p_audio_proc[:min_len]
            l_aligned = l_audio[:min_len]

            # If sample rates differ (unlikely if from same corpus),
            # resample l to match p for MCD / spectral corr.
            if p_sr != l_sr:
                l_aligned = librosa.resample(l_aligned, orig_sr=l_sr, target_sr=p_sr)
                sr_for_local = p_sr
            else:
                sr_for_local = p_sr

            # Compute metrics
            #mcd = compute_mcd(l_aligned, p_aligned, sr_for_local)
            mcd, _ = compare_audio_files(l_aligned, l_sr, p_aligned, p_sr)
            corr = compute_spectral_cross_correlation(l_aligned, p_aligned, sr_for_local)
            haspi, hasqi = compute_haspi_hasqi(l_audio, l_sr, p_audio_proc, p_sr)

        metrics = Metrics(
            filename_p=fname,
            filename_l=os.path.basename(l_path),
            mcd=mcd,
            spectral_corr=corr,
            haspi=haspi,
            hasqi=hasqi,
        )
        metrics_list.append(metrics)

        # Per-pair printout
        print(
            f"\nPair: {metrics.filename_p}  vs  {metrics.filename_l}\n"
            f"  MCD (dB)         : {metrics.mcd:.4f}\n"
            f"  Spectral Corr    : {metrics.spectral_corr:.4f}\n"
            f"  HASPI            : {metrics.haspi if metrics.haspi is not None else 'N/A'}\n"
            f"  HASQI            : {metrics.hasqi if metrics.hasqi is not None else 'N/A'}"
        )

    return metrics_list


def summarize_metrics(metrics_list: List[Metrics]) -> None:
    """
    Compute and print overall averages of all metrics.
    """
    if not metrics_list:
        print("\nNo valid p/l pairs were processed.")
        return

    mcd_vals = np.array([m.mcd for m in metrics_list], dtype=float)
    corr_vals = np.array([m.spectral_corr for m in metrics_list], dtype=float)
    haspi_vals = np.array(
        [m.haspi for m in metrics_list if m.haspi is not None], dtype=float
    )
    hasqi_vals = np.array(
        [m.hasqi for m in metrics_list if m.hasqi is not None], dtype=float
    )

    print("\n================= Overall summary =================")
    print(f"Number of pairs: {len(metrics_list)}")

    if np.isfinite(mcd_vals).any():
        print(f"Mean MCD            : {np.nanmean(mcd_vals):.4f} dB")
    else:
        print("Mean MCD            : N/A")

    if np.isfinite(corr_vals).any():
        print(f"Mean Spectral Corr  : {np.nanmean(corr_vals):.4f}")
    else:
        print("Mean Spectral Corr  : N/A")

    if haspi_vals.size > 0:
        print(f"Mean HASPI          : {np.nanmean(haspi_vals):.4f}")
    else:
        print("Mean HASPI          : N/A")

    if hasqi_vals.size > 0:
        print(f"Mean HASQI          : {np.nanmean(hasqi_vals):.4f}")
    else:
        print("Mean HASQI          : N/A")
    print("===================================================\n")


# --------- CLI --------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform *_p_*.wav files and compute p/l metrics."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input folder containing .wav files.",
    )

    parser.add_argument(
        "gain_db",
        type=float,
        help="Desired volume change in dB (e.g. +3.0 or -6.0).",
    )

    parser.add_argument(
        "pitch_semitones",
        type=float,
        help="Desired pitch shift in semitones (positive = higher, negative = lower).",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional output folder. If omitted, a 'processed/' subfolder is created inside input_dir.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_list = process_folder(
        input_dir=args.input_dir,
        gain_db=args.gain_db,
        pitch_semitones=args.pitch_semitones,
        output_dir=args.output_dir,
    )
    summarize_metrics(metrics_list)


if __name__ == "__main__":
    main()
