import pandas as pd
import mne
import os
from prep_config import Config

class EEGIOHandler:
    """
    EEGIOHandler provides methods for all input/output operations:
    - Loading Muse CSV files as MNE Raw objects
    - Saving Raw objects to FIF
    - Saving DataFrames to CSV
    - Exporting all per-channel frequency band data to CSV
    - Saving all outputs (FIF and CSV) for a set of per-channel Raw objects
    """
    @staticmethod
    def load_muse_csv(csv_path):
        df = pd.read_csv(csv_path)
        if not any(x.lower() == 'timestamp' for x in df.columns):
            raise ValueError("Input CSV must contain a 'TimeStamp' column.")
        curr_raw_names = [val for key, val in Config.RAW_CHANNEL_NAMES.items() if key in df.columns][0]
        df_eeg = df[curr_raw_names]
        eeg_data = df_eeg.values.T * 1e-6  # Convert µV to V
        nan_prec = eeg_data.sum() / df_eeg.shape[0]
        print(f"Percentage of NaN values interpolated in EEG data: {nan_prec * 100:.2f}%")
        df_clean = (df_eeg.interpolate(method='linear', axis=0).bfill().ffill())
        df_centered = df_clean - df_clean.mean(axis=0)
        scale_factor = 0.488e-6  # volts per count
        data_volts = df_centered.values * scale_factor
        info = mne.create_info(Config.CH_NAMES, Config.SFREQ, Config.CH_TYPES)
        raw = mne.io.RawArray(data_volts.T, info)
        montage = mne.channels.make_dig_montage(Config.MUSE_POSITIONS, coord_frame='head')
        raw.set_montage(montage)
        return raw

    @staticmethod
    def export_all_data_to_csv(per_channel_raws, csv_path):
        """
        Export all per-channel frequency band data to a CSV file.

        Parameters
        ----------
        per_channel_raws : dict
            Dictionary of channel name to Raw object. Each value should be an MNE Raw object containing
            the original channel and its frequency band decompositions as separate channels.
        csv_path : str
            Path to save the CSV file.

        Output
        ------
        Writes a CSV file with columns:
            - 'time_s': time in seconds
            - 'artifact': annotation label for each timepoint (if any)
            - one column per channel and per band (e.g., 'AF7', 'AF7_delta', ...)
        The CSV is saved to the specified path. Prints a confirmation message on success.
        """
        first_raw = per_channel_raws[next(iter(per_channel_raws))]
        times = first_raw.times
        annot = first_raw.annotations
        artifact = pd.Series([''] * len(times), dtype=object)
        for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
            mask = (times >= onset) & (times < (onset + duration))
            artifact[mask] = desc
        all_data = {'time_s': times, 'artifact': artifact}
        for raw_ch in per_channel_raws.values():
            data = raw_ch.get_data()
            for idx, name in enumerate(raw_ch.ch_names):
                all_data[name] = data[idx]
        df_all = pd.DataFrame(all_data)
        df_all.to_csv(csv_path, index=False)
        print(f"Saved combined data to {csv_path}")

    @staticmethod
    def save_all_outputs(per_channel_raws, output_dir):
        """
        Save all per-channel Raw objects to FIF files in the output directory and export all data to a CSV file.

        Parameters
        ----------
        per_channel_raws : dict
            Dictionary of channel name to Raw object.
        output_dir : str
            Directory to save the output files.
        """
        for curr_chan, curr_raw in per_channel_raws.items():
            curr_raw.save(os.path.join(output_dir, f"{curr_chan}_decomposed_raw.fif"), overwrite=True)
        csv_path = os.path.join(output_dir, 'all_channels_with_bands.csv')
        EEGIOHandler.export_all_data_to_csv(per_channel_raws, csv_path)