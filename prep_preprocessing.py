import numpy as np
import mne
from mne.preprocessing import ICA
from prep_config import Config
from prep_io import EEGIOHandler
import os

class EEGPreprocessor:
    """
    EEGPreprocessor provides methods for filtering, annotating, and cleaning EEG data
    from Muse headbands using the scientific Python stack. The MNE Raw object is stored as self.raw.
    """
    def __init__(self, data):
        """
        Initialize the EEGPreprocessor with either a CSV file path or an existing MNE Raw object.

        Parameters
        ----------
        data : str or mne.io.Raw
            If str, treated as a CSV file path and loaded using EEGIOHandler.load_muse_csv.
            If mne.io.Raw, used directly.
        """
        if isinstance(data, str):
            self.raw = EEGIOHandler.load_muse_csv(data)
        elif hasattr(data, 'get_data') and hasattr(data, 'info'):
            self.raw = data
        else:
            raise ValueError("data must be a CSV file path or an mne.io.Raw object.")

    def base_filtering(self):
        """
        Apply high-pass, notch, and low-pass filtering, and set average EEG reference.
        Modifies self.raw in-place.
        """
        self.raw.filter(Config.BASE_FILTER_SETTINGS['highpass_freq'], None, fir_design=Config.BASE_FILTER_SETTINGS['filter_design'])
        self.raw.notch_filter(Config.BASE_FILTER_SETTINGS['notch_freq'], fir_design=Config.BASE_FILTER_SETTINGS['filter_design'])
        self.raw.filter(None, Config.BASE_FILTER_SETTINGS['lowpass_freq'], fir_design=Config.BASE_FILTER_SETTINGS['filter_design'])
        self.raw.set_eeg_reference(ref_channels=Config.BASE_FILTER_SETTINGS['eeg_reference'])

    def annotate_dynamic_p2p(self, n_mads=3, win_sec=0.2, step_sec=0.1, label='BAD_dynamic'):
        """
        Annotate dynamic peak-to-peak artifacts in the EEG data using a moving window and MAD threshold.
        Modifies self.raw in-place by adding new annotations.

        Parameters
        ----------
        n_mads : int
            Number of MADs above median for threshold.
        win_sec : float
            Window size in seconds.
        step_sec : float
            Step size in seconds.
        label : str
            Annotation label for detected artifacts.
        """
        sf = self.raw.info['sfreq']
        data = self.raw.get_data()
        win_samp, step_samp = int(win_sec * sf), int(step_sec * sf)
        n_win = 1 + (data.shape[1] - win_samp) // step_samp
        p2p = np.zeros((n_win, data.shape[0]))
        for w in range(n_win):
            seg = data[:, w*step_samp : w*step_samp + win_samp]
            p2p[w] = np.ptp(seg, axis=1)
        med = np.median(p2p, axis=0)
        mad = np.median(np.abs(p2p - med[None, :]), axis=0)
        thr = med + n_mads * mad
        onsets, durs, descs = [], [], []
        for w in range(n_win):
            if (p2p[w] > thr).any():
                onset = (w * step_samp) / sf
                onsets.append(onset)
                durs.append(win_sec)
                descs.append(label)
        new_ann = mne.Annotations(onset=onsets, duration=durs, description=descs)
        self.raw.set_annotations(self.raw.annotations + new_ann)
        self.raw = EEGPreprocessor.merge_overlapping_annotations(self.raw, label=label)

    @staticmethod
    def merge_overlapping_annotations(raw, label='BAD_dynamic'):
        """
        Merge overlapping or contiguous annotations of a given label in an MNE Raw object.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw EEG data.
        label : str
            The annotation label to merge.

        Returns
        -------
        raw : mne.io.Raw
            The raw EEG data with merged annotations.
        """
        anns = raw.annotations
        onsets = np.array(anns.onset)
        durations = np.array(anns.duration)
        descs = np.array(anns.description)
        mask = descs == label
        keep_on = onsets[mask]
        keep_du = durations[mask]
        intervals = sorted(zip(keep_on, keep_on + keep_du))
        merged = []
        if intervals:
            curr_start, curr_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= curr_end:
                    curr_end = max(curr_end, end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = start, end
            merged.append((curr_start, curr_end))
        new_onsets = list(onsets[~mask])
        new_durations = list(durations[~mask])
        new_descs = list(descs[~mask])
        for start, end in merged:
            new_onsets.append(start)
            new_durations.append(end - start)
            new_descs.append(label)
        sorted_idx = np.argsort(new_onsets)
        raw.set_annotations(mne.Annotations(
            onset=np.array(new_onsets)[sorted_idx],
            duration=np.array(new_durations)[sorted_idx],
            description=np.array(new_descs)[sorted_idx]
        ))
        return raw

    def run_ica(self):
        """
        Run ICA decomposition on the EEG data and plot components and sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            The fitted ICA object.
        """
        ica = ICA(n_components=4, method='fastica', random_state=42)
        ica.fit(self.raw, reject_by_annotation=True)
        ica.plot_components(inst=self.raw)
        ica.plot_sources(self.raw)
        for i in range(ica.n_components_):
            ica.plot_properties(self.raw, picks=i)
        return ica

    def remove_ica_comp_and_plot(self, ica, comps_to_remove):
        """
        Remove selected ICA components and plot before/after comparison.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            The ICA object.
        comps_to_remove : list
            List of component indices to remove.

        Returns
        -------
        raw_clean : mne.io.Raw
            The cleaned EEG data.
        """
        ica.exclude = comps_to_remove
        raw_clean = self.raw.copy()
        ica.apply(raw_clean)
        self.raw.plot(n_channels=4, title='Original')
        raw_clean.plot(n_channels=4, title='After ICA cleanup')
        return raw_clean

    def plot_and_save_psd(self, output_dir, min_duration=1.0):
        """
        Plot and save the power spectral density (PSD) of the EEG data, using only segments longer than min_duration.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot.
        min_duration : float
            Minimum duration in seconds for segments to include.
        """
        original_annot = self.raw.annotations.copy()
        bad_annot = self.raw.annotations[self.raw.annotations.description == 'BAD_dynamic']
        long_bad_annot = bad_annot[bad_annot.duration >= min_duration]
        self.raw.set_annotations(long_bad_annot)
        fig = self.raw.plot_psd(picks='all',
                                fmin=0,
                                fmax=50,
                                n_fft=256,
                                reject_by_annotation=True,
                                average=False,
                                show=False)
        fig.savefig(os.path.join(output_dir, 'power_spectral_density.png'))
        fig.show()
        self.raw.set_annotations(original_annot) 