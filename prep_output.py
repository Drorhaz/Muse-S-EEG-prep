import numpy as np
import pandas as pd
import mne


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
    # 1. Grab the common time axis and the shared annotations
    first_raw = per_channel_raws[next(iter(per_channel_raws))]
    times = first_raw.times  # in seconds
    annot = first_raw.annotations
    artifact = np.array([''] * len(times), dtype=object)

    # 2. Fill in the description for any timepoint inside an annotation
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
        mask = (times >= onset) & (times < (onset + duration))
        artifact[mask] = desc

    all_data = {'time_s': times, 'artifact': artifact}
    for raw_ch in per_channel_raws.values():
        data = raw_ch.get_data()  # shape (n_ch, n_samp)
        for idx, name in enumerate(raw_ch.ch_names):
            all_data[name] = data[idx]

    # 3. Make DataFrame and save
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined data to {csv_path}")

