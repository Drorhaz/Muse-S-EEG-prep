# prep_output.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import EpochsArray, create_info
from scipy.signal import welch
from scipy.stats import pearsonr
from datetime import timedelta

from prep_config import (
    RAW_CHANNEL_NAMES, CH_NAMES, SFREQ, FREQ_BANDS,
    DEFAULT_CHANNEL_ORDER, MEDIAN_FILTER_SETTINGS,
    DYNAMIC_THRESHOLD_SETTINGS, AMPLITUDE_REJECTION_SETTINGS, ICA_SETTINGS,
    OUTPUT_SETTINGS
)
from prep_function import load_muse_data

def plot_annotated_eeg(raw, output_dir, duration_sec=20):
    os.makedirs(output_dir, exist_ok=True)
    n_samples = int(duration_sec * SFREQ)
    times = raw.times[:n_samples]
    data = raw.get_data()[:, :n_samples] * 1e6  # µV

    labels = list(set(ann['description'] for ann in raw.annotations))
    colors = plt.get_cmap('tab10', len(labels))
    label_colors = {label: colors(i) for i, label in enumerate(labels)}

    for ch_idx, ch_name in enumerate(CH_NAMES):
        plt.figure(figsize=(12, 4))
        plt.plot(times, data[ch_idx], label=f'{ch_name} Base-Filtered', color='black')
        plt.title(f'{ch_name} - Annotated Artifacts (Preview)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.grid(True)

        for ann in raw.annotations:
            onset = ann['onset']
            duration = ann['duration']
            label = ann['description']
            end = onset + duration
            if onset < duration_sec:
                plt.axvspan(onset, min(end, duration_sec), color=label_colors[label], alpha=0.3, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"annotated_{ch_name}.png"))
        plt.close()

def compute_bandpower(data, sfreq, band):
    freqs, psd = welch(data, sfreq, nperseg=sfreq*2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx], axis=1)

def extract_brainwaves(raw, timestamp_list):
    window_sec = OUTPUT_SETTINGS.get('brainwave_window_sec', 0.5)
    sfreq = int(raw.info['sfreq'])
    win_size = int(window_sec * sfreq)
    n_windows = raw.n_times // win_size

    if len(timestamp_list) < n_windows:
        raise ValueError(f"Not enough timestamps ({len(timestamp_list)}) for {n_windows} windows.")

    results = []
    for i in range(n_windows):
        start = i * win_size
        stop = start + win_size
        cropped = raw.copy().crop(tmin=start / sfreq, tmax=(stop - 1) / sfreq)
        band_power = {}

        for band, (l_freq, h_freq) in FREQ_BANDS.items():
            filtered = cropped.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)
            power = np.mean(filtered.get_data() ** 2, axis=1)
            band_name = band.capitalize()
            for ch, p in zip(CH_NAMES, power):
                band_power[f'{band_name}_{ch}'] = p

        band_power['TimeStamp'] = timestamp_list[i]
        results.append(band_power)

    df = pd.DataFrame(results)
    return df

def export_cleaned_data(raw, brainwave_df, output_path):
    brainwave_df.to_csv(output_path, index=False)

def plot_global_brainwaves(df, output_path):
    waves = [band.capitalize() for band in FREQ_BANDS.keys()]
    values = [df.filter(like=f'{band}_').mean().mean() for band in waves]
    plt.figure(figsize=(10, 4))
    plt.bar(waves, values, color='skyblue')
    plt.title('Global Brainwave Power')
    plt.ylabel('Power (µV^2)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def verify_against_reference(computed_df, reference_csv, output_path):
    ref_df = pd.read_csv(reference_csv)
    metric = OUTPUT_SETTINGS.get('verification_metric', 'mse')
    downsample = OUTPUT_SETTINGS.get('downsample_reference', False)
    fill_missing = OUTPUT_SETTINGS.get('fill_missing', 'none')  # options: 'zero', 'interpolate', 'none'

    # Downsample if enabled
    if downsample:
        factor = len(ref_df) // len(computed_df)
        ref_df = ref_df.iloc[::factor].reset_index(drop=True)

    # Merge on timestamp
    merged_cols = list(set(computed_df.columns) & set(ref_df.columns))
    merged_cols = [col for col in merged_cols if col != 'TimeStamp']
    merged = pd.merge(computed_df[['TimeStamp'] + merged_cols], ref_df[['TimeStamp'] + merged_cols], on='TimeStamp', suffixes=('_comp', '_ref'))

    if fill_missing == 'zero':
        merged = merged.fillna(0)
    elif fill_missing == 'interpolate':
        merged = merged.interpolate()

    metrics = []
    for col in merged_cols:
        col_comp = f'{col}_comp'
        col_ref = f'{col}_ref'
        if col_comp not in merged or col_ref not in merged:
            continue

        ref_vals = merged[col_ref].values
        comp_vals = merged[col_comp].values

        if metric == 'mse':
            value = np.mean((ref_vals - comp_vals) ** 2)
        elif metric == 'corr':
            value, _ = pearsonr(ref_vals, comp_vals)
        else:
            raise ValueError(f"Unsupported verification metric: {metric}")

        metrics.append({'channel_band': col, metric: value})

    pd.DataFrame(metrics).to_csv(output_path, index=False)
