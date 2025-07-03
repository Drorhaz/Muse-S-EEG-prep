# prep_output.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
from datetime import timedelta
import mne

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

def extract_frequency_bands(raw, bands=None):
    """
    Extract frequency bands from MNE Raw object.
    Returns a dictionary with band names as keys and filtered Raw objects as values.
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta':  (12, 30), 'gamma': (30, 40)}

    # 1) For each channel, make a new RawArray
    per_channel_raws = {}
    sfreq = raw.info['sfreq']
    for ch in raw.ch_names:
        # grab the 1D data for this channel
        sig = raw.copy().pick_channels([ch]).get_data()[0]  # shape (n_times,)
        data_list   = [sig]
        ch_names    = [ch]
        ch_types    = ['eeg']      # keep the original as EEG
        # for each band, filter and add as a "misc" channel
        for band_name, (l_hz, h_hz) in bands.items():
            filt = mne.filter.filter_data(sig[np.newaxis, :], sfreq, l_hz, h_hz, method='fir', fir_design='firwin')[0]
            data_list.append(filt)
            ch_names.append(f"{ch}_{band_name}")
            ch_types.append('misc')
        
        # stack into array of shape (n_chans=6, n_times)
        data_mat = np.vstack(data_list)

        # create the RawArray
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw_chan = mne.io.RawArray(data_mat, info)

        # copy over all annotations
        raw_chan.set_annotations(raw.annotations)

        per_channel_raws[ch] = raw_chan
    return per_channel_raws


def export_all_data_to_csv(per_channel_raws, csv_path):
    # 1. Grab the common time axis
    times = per_channel_raws[next(iter(per_channel_raws))].times  # in seconds

    # 2. Build a dict of columns
    all_data = {'time_s': times}
    for raw_ch in per_channel_raws.values():
        data = raw_ch.get_data()  # shape (n_ch, n_samp)
        for idx, name in enumerate(raw_ch.ch_names):
            all_data[name] = data[idx]

    # 3. Make DataFrame and save
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined data to {csv_path}")


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
