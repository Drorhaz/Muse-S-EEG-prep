import numpy as np
import pandas as pd
import mne
from scipy import signal
from mne.preprocessing import annotate_amplitude, ICA
from itertools import groupby

from prep_config import (
    RAW_CHANNEL_NAMES, CH_NAMES, SFREQ, CH_TYPES,
    MUSE_POSITIONS, BASE_FILTER_SETTINGS,
    MEDIAN_FILTER_SETTINGS, DYNAMIC_THRESHOLD_SETTINGS,
    AMPLITUDE_REJECTION_SETTINGS
)

# === 1. Load EEG data from CSV ===
def load_muse_data(csv_path):
    # === Load and extract timestamps ===
    df = pd.read_csv(csv_path)
    if not any(x.lower() == 'timestamp' for x in df.columns):
        raise ValueError("Input CSV must contain a 'TimeStamp' column.")
    time_key = [x for x in df.columns if x.lower() == 'timestamp'][0]
    timestamp_list = df[time_key].tolist()
    
    df_eeg = df[RAW_CHANNEL_NAMES]
    eeg_data = df_eeg.values.T * 1e-6  # Convert µV to V
    nan_prec = np.isnan(eeg_data).sum() / df_eeg.shape[0]
    print(f"precentage of NaN values in EEG data: {nan_prec * 100:.2f}%")

    # 1. In pandas, fill all NaNs:
    df_clean = (df_eeg.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill'))
    
    # 2. Remove DC offset (baseline correction) per channel
    #    Now each channel will have mean ≈ 0 counts
    df_centered = df_clean - df_clean.mean(axis=0)
    
    # 3. Apply your scale factor to go from “counts” → volts
    #    e.g. if 1 count = 0.488 µV, then:
    scale_factor = 0.488e-6  # volts per count
    data_volts = df_centered.values * scale_factor  # shape: (n_times, n_channels)
    
    # 4. Build your MNE RawArray with zero-mean, physical-unit data
    info = mne.create_info(RAW_CHANNEL_NAMES, SFREQ, CH_TYPES)
    raw = mne.io.RawArray(data_volts.T, info)
    # montage = mne.channels.make_dig_montage(ch_pos=MUSE_POSITIONS)
    # raw.set_montage(montage)
    return raw, timestamp_list

# === 2. Filtering ===
def base_filtering(raw):
    raw.filter(BASE_FILTER_SETTINGS['highpass_freq'], None, fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.notch_filter(BASE_FILTER_SETTINGS['notch_freq'], fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.filter(None, BASE_FILTER_SETTINGS['lowpass_freq'], fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.set_eeg_reference(ref_channels=BASE_FILTER_SETTINGS['eeg_reference'])

# === 3. Median Filter ===
def median_filter_artifact_removal(raw):
    if not MEDIAN_FILTER_SETTINGS['enabled']:
        return raw

    data = raw.get_data().copy()
    kernel_size = MEDIAN_FILTER_SETTINGS['kernel_size']
    for i in range(data.shape[0]):
        signal_clean = signal.medfilt(data[i], kernel_size=kernel_size)
        artifact_mask = data[i] != signal_clean

        indices = np.where(artifact_mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(g)
            start = group[0][1]
            end = group[-1][1]
            onset = start / raw.info['sfreq']
            duration = (end - start + 1) / raw.info['sfreq']
            raw.annotations.append(onset, duration, MEDIAN_FILTER_SETTINGS['annotation_label'])

        data[i] = signal_clean

    raw_clean = raw.copy()
    raw_clean._data = data
    return raw_clean

# === 4. Dynamic Threshold Cleaning ===
def dynamic_threshold_artifact_removal(raw):
    if not DYNAMIC_THRESHOLD_SETTINGS['enabled']:
        return raw

    data = raw.get_data().copy()
    n_mads = DYNAMIC_THRESHOLD_SETTINGS['n_mads']
    action = DYNAMIC_THRESHOLD_SETTINGS['action']
    label = DYNAMIC_THRESHOLD_SETTINGS['annotation_label']

    for i in range(data.shape[0]):
        median = np.median(data[i])
        mad = np.median(np.abs(data[i] - median))
        threshold = median + n_mads * mad
        mask = np.abs(data[i]) > threshold

        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(g)
            start = group[0][1]
            end = group[-1][1]
            onset = start / raw.info['sfreq']
            duration = (end - start + 1) / raw.info['sfreq']
            raw.annotations.append(onset, duration, label)

        if action == 'replace':
            data[i, mask] = median
        elif action == 'remove':
            data[i, mask] = np.nan
        else:
            raise ValueError(f"Unknown action: {action}")

    raw_clean = raw.copy()
    raw_clean._data = data
    return raw_clean

# === 5. Amplitude-Based Rejection ===
def auto_artifact_rejection(raw):
    if not AMPLITUDE_REJECTION_SETTINGS['enabled']:
        return raw

    annotations, _ = annotate_amplitude(raw, peak=AMPLITUDE_REJECTION_SETTINGS['threshold'])
    for onset, duration, _ in zip(annotations.onset, annotations.duration, annotations.description):
        raw.annotations.append(onset, duration, AMPLITUDE_REJECTION_SETTINGS['annotation_label'])
    return raw

# === 6. ICA ===
def run_ica(raw):
    ica = ICA(n_components=4, method='fastica', random_state=42)
    ica.fit(raw)
    ica.plot_components(inst=raw)
    ica.plot_sources(raw)
    for i in range(ica.n_components_):
        ica.plot_properties(raw, picks=i)
    return ica

# === 7. Annotate ICA artifacts ===
def annotate_ica_artifacts(raw, ica, label='ica_artifact', n_mads=6):
    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info['sfreq']

    for i in ica.exclude:
        component = sources[i]
        median = np.median(component)
        mad = np.median(np.abs(component - median))
        threshold = median + n_mads * mad
        mask = np.abs(component) > threshold

        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(g)
            start = group[0][1]
            end = group[-1][1]
            onset = start / sfreq
            duration = (end - start + 1) / sfreq
            raw.annotations.append(onset, duration, label)

    return raw
