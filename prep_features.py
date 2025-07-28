import numpy as np
import mne

def extract_frequency_bands(raw, bands=None):
    """
    Extract frequency bands from an MNE Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    bands : dict or None
        Dictionary of band names to (low, high) Hz tuples. If None, uses default bands.

    Returns
    -------
    per_channel_raws : dict
        Dictionary with channel names as keys and filtered Raw objects as values.
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}
    per_channel_raws = {}
    sfreq = raw.info['sfreq']
    for ch in raw.ch_names:
        sig = raw.copy().pick_channels([ch]).get_data()[0]
        data_list = [sig]
        ch_names = [ch]
        ch_types = ['eeg']
        for band_name, (l_hz, h_hz) in bands.items():
            filt = mne.filter.filter_data(sig[np.newaxis, :], sfreq, l_hz, h_hz, method='fir', fir_design='firwin')[0]
            data_list.append(filt)
            ch_names.append(f"{ch}_{band_name}")
            ch_types.append('misc')
        data_mat = np.vstack(data_list)
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw_chan = mne.io.RawArray(data_mat, info)
        raw_chan.set_annotations(raw.annotations)
        per_channel_raws[ch] = raw_chan
    return per_channel_raws 