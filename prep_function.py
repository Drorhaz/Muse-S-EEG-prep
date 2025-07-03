import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

from prep_config import (
    RAW_CHANNEL_NAMES, CH_NAMES, SFREQ, CH_TYPES,
    MUSE_POSITIONS, BASE_FILTER_SETTINGS,
)

# === 1. Load EEG data from CSV ===
def load_muse_data(csv_path):
    # === Load and extract timestamps ===
    df = pd.read_csv(csv_path)
    if not any(x.lower() == 'timestamp' for x in df.columns):
        raise ValueError("Input CSV must contain a 'TimeStamp' column.")
    
    curr_raw_names = [val for key, val in RAW_CHANNEL_NAMES.items() if key in df.columns][0]
    df_eeg = df[curr_raw_names]
    eeg_data = df_eeg.values.T * 1e-6  # Convert µV to V
    nan_prec = np.isnan(eeg_data).sum() / df_eeg.shape[0]
    print(f"Percentage of NaN values interpolated in EEG data: {nan_prec * 100:.2f}%")

    # 1. In pandas, fill all NaNs:
    df_clean = (df_eeg.interpolate(method='linear', axis=0).bfill().ffill())
    
    # 2. Remove DC offset (baseline correction) per channel
    #    Now each channel will have mean ≈ 0 counts
    df_centered = df_clean - df_clean.mean(axis=0)
    
    # 3. Apply your scale factor to go from "counts" → volts
    #    e.g. if 1 count = 0.488 µV, then:
    scale_factor = 0.488e-6  # volts per count
    data_volts = df_centered.values * scale_factor  # shape: (n_times, n_channels)
    
    # 4. Build your MNE RawArray with zero-mean, physical-unit data
    info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
    raw = mne.io.RawArray(data_volts.T, info)
    montage = mne.channels.make_dig_montage(MUSE_POSITIONS, coord_frame='head')
    raw.set_montage(montage)
    return raw

# === 2. Filtering ===
def base_filtering(raw):
    raw.filter(BASE_FILTER_SETTINGS['highpass_freq'], None, fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.notch_filter(BASE_FILTER_SETTINGS['notch_freq'], fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.filter(None, BASE_FILTER_SETTINGS['lowpass_freq'], fir_design=BASE_FILTER_SETTINGS['filter_design'])
    raw.set_eeg_reference(ref_channels=BASE_FILTER_SETTINGS['eeg_reference'])


## === 3. Dynamic Thresholding ===
def annotate_dynamic_p2p(raw, n_mads=3, win_sec=0.2, step_sec=0.1, label='BAD_dynamic'):
    sf = raw.info['sfreq']
    data = raw.get_data()
    win_samp, step_samp = int(win_sec * sf), int(step_sec * sf)
    
    # build P2P matrix: windows × channels
    n_win = 1 + (data.shape[1] - win_samp) // step_samp
    p2p   = np.zeros((n_win, data.shape[0]))
    for w in range(n_win):
        seg = data[:, w*step_samp : w*step_samp + win_samp]
        p2p[w] = np.ptp(seg, axis=1)
    
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


# === 4. ICA ===
def run_ica(raw, output_dir=None):
    """Run ICA and plot components.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to process
    output_dir : str | None
        Not used, kept for backward compatibility
    
    Returns
    -------
    ica : mne.preprocessing.ICA
        The fitted ICA object
    """
    ica = ICA(n_components=4, method='fastica', random_state=42)
    ica.fit(raw, reject_by_annotation=True)
    
    # Plot components and sources
    ica.plot_components(inst=raw)
    ica.plot_sources(raw)
    
    # Plot properties for each component
    for i in range(ica.n_components_):
        ica.plot_properties(raw, picks=i)
    
    return ica


def remove_ica_comp_and_plot(ica, raw, comps_to_remove):
    ica.exclude = comps_to_remove  
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    raw.plot(n_channels=4, title='Original')
    raw_clean.plot(n_channels=4, title='After ICA cleanup')
    return raw_clean

def plot_and_save_psd(raw, output_dir, min_duration=1.0):
    """Plot and save PSD using only segments longer than min_duration.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to plot
    output_dir : str
        Directory to save the plot
    min_duration : float
        Minimum duration in seconds for segments to include
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The PSD plot figure
    """
    import os
    
    # Store original annotations
    original_annot = raw.annotations.copy()
    
    # Get the annotations that mark bad segments
    bad_annot = raw.annotations[raw.annotations.description == 'BAD_dynamic']
    
    # Keep only annotations for segments longer than min_duration
    long_bad_annot = bad_annot[bad_annot.duration >= min_duration]
    
    # Create a new annotations object with only the long segments
    raw.set_annotations(long_bad_annot)
    
    # Plot PSD using standard parameters
    fig = raw.plot_psd(picks='all',
                    fmin=0,
                    fmax=50,
                    n_fft=256,  # 1 second window
                    reject_by_annotation=True,
                    average=False,
                    show=False)  # Don't show yet to save first
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'power_spectral_density.png'))
    
    # Now display the figure
    fig.show()
    
    # Restore original annotations
    raw.set_annotations(original_annot)
