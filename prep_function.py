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
    
    df_eeg = df[RAW_CHANNEL_NAMES]
    eeg_data = df_eeg.values.T * 1e-6  # Convert µV to V
    nan_prec = np.isnan(eeg_data).sum() / df_eeg.shape[0]
    print(f"Percentage of NaN values interpolated in EEG data: {nan_prec * 100:.2f}%")

    # 1. In pandas, fill all NaNs:
    df_clean = (df_eeg.interpolate(method='linear', axis=0).bfill().ffill())
    
    # 2. Remove DC offset (baseline correction) per channel
    #    Now each channel will have mean ≈ 0 counts
    df_centered = df_clean - df_clean.mean(axis=0)
    
    # 3. Apply your scale factor to go from “counts” → volts
    #    e.g. if 1 count = 0.488 µV, then:
    scale_factor = 0.488e-6  # volts per count
    data_volts = df_centered.values * scale_factor  # shape: (n_times, n_channels)
    
    # 4. Build your MNE RawArray with zero-mean, physical-unit data
    info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
    raw = mne.io.RawArray(data_volts.T, info)
    return raw

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

## === 4. Dynamic Thresholding ===
def annotate_dynamic_p2p(raw, n_mads=3, win_sec=0.2, step_sec=0.1, label='BAD_dynamic'):
    sf = raw.info['sfreq']
    data = raw.get_data()
    win_samp, step_samp = int(win_sec*sf), int(step_sec*sf)
    
    # build P2P matrix: windows × channels
    n_win = 1 + (data.shape[1] - win_samp)//step_samp
    p2p   = np.zeros((n_win, data.shape[0]))
    for w in range(n_win):
        seg = data[:, w*step_samp : w*step_samp+win_samp]
        p2p[w] = seg.ptp(axis=1)
    
    # channel-wise median & MAD on the p2p distribution
    med = np.median(p2p, axis=0)
    mad = np.median(np.abs(p2p - med[None,:]), axis=0)
    thr = med + n_mads*mad
    
    # annotate windows exceeding threshold
    onsets, durs, descs = [], [], []
    for w in range(n_win):
        if (p2p[w] > thr).any():
            onset = (w*step_samp)/sf
            onsets.append(onset)
            durs.append(win_sec)
            descs.append(label)
    
    new_ann = mne.Annotations(onset=onsets, duration=durs, description=descs)
    raw.set_annotations(raw.annotations + new_ann)
    raw = merge_overlapping_annotations(raw, label='BAD_dynamic')
    return raw


def merge_overlapping_annotations(raw, label='BAD_dynamic'):
    """Merge overlapping or contiguous annotations of a given label."""
    # Extract existing annotations
    anns = raw.annotations
    onsets   = np.array(anns.onset)
    durations= np.array(anns.duration)
    descs    = np.array(anns.description)
    
    # Filter only the annotations we want to merge
    mask = descs == label
    keep_on = onsets[mask]
    keep_du = durations[mask]
    
    # Build (start, end) pairs and sort by start time
    intervals = sorted(zip(keep_on, keep_on + keep_du))
    
    # Merge them
    merged = []
    curr_start, curr_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= curr_end:            # overlap or contiguous
            curr_end = max(curr_end, end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = start, end
    merged.append((curr_start, curr_end))
    
    # Build new annotation arrays
    new_onsets   = list(onsets[~mask])            # keep all other labels
    new_durations= list(durations[~mask])
    new_descs    = list(descs[~mask])
    
    # Add merged BAD_dynamic back in
    for start, end in merged:
        new_onsets.append(start)
        new_durations.append(end - start)
        new_descs.append(label)
    
    # Recreate and set annotations (sorted by onset)
    sorted_idx = np.argsort(new_onsets)
    raw.set_annotations(mne.Annotations(
        onset      = np.array(new_onsets)[sorted_idx],
        duration   = np.array(new_durations)[sorted_idx],
        description= np.array(new_descs)[sorted_idx]
    ))
    return raw


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
