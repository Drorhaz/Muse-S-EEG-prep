"""
Configuration settings for Muse-S EEG preprocessing.

This module provides a Config class containing all relevant constants and settings
for EEG data processing with the Muse headband.
"""

class Config:
    """
    Configuration class for Muse-S EEG preprocessing.

    Attributes
    ----------
    RAW_CHANNEL_NAMES : dict
        Raw channel column names in the CSV file.
    CH_NAMES : list
        Channel names used in MNE after loading.
    SFREQ : int
        Sampling frequency (Hz).
    CH_TYPES : str
        EEG channel type (used by MNE).
    MUSE_POSITIONS : dict
        Approximate 3D positions of Muse electrodes (meters).
    BASE_FILTER_SETTINGS : dict
        Base filtering settings for preprocessing EEG.
    MEDIAN_FILTER_SETTINGS : dict
        Median filter settings for artifact removal.
    DYNAMIC_THRESHOLD_SETTINGS : dict
        Dynamic threshold-based artifact removal settings.
    AMPLITUDE_REJECTION_SETTINGS : dict
        Amplitude-based automatic annotation settings.
    ICA_SETTINGS : dict
        ICA artifact annotation settings.
    DEFAULT_CHANNEL_ORDER : list
        Default channel order for plots or looping.
    FREQ_BANDS : dict
        Brain wave bands for extraction and plotting.
    OUTPUT_SETTINGS : dict
        Output control settings.
    """
    RAW_CHANNEL_NAMES = {
        'RAW_AF7': ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10'],
        'eeg1': ['eeg1', 'eeg2', 'eeg3', 'eeg4']
    }
    CH_NAMES = ['AF7', 'AF8', 'TP9', 'TP10']
    SFREQ = 256
    CH_TYPES = 'eeg'
    MUSE_POSITIONS = {
        'AF7': [-0.035, 0.065, 0.04],    # Front-left
        'AF8': [0.030, 0.060, 0.030],    # Front-right
        'TP9': [-0.072, -0.045, 0.01],   # Rear-left
        'TP10': [0.068, -0.042, 0.015],  # Rear-right
    }
    BASE_FILTER_SETTINGS = {
        'highpass_freq': 1.0,
        'lowpass_freq': 40.0,
        'notch_freq': 50.0,
        'filter_design': 'firwin',
        'eeg_reference': 'average'
    }
    MEDIAN_FILTER_SETTINGS = {
        'enabled': False,
        'kernel_size': 7,
        'annotation_label': 'median_artifact'
    }
    DYNAMIC_THRESHOLD_SETTINGS = {
        'enabled': True,
        'n_mads': 10,
        'action': 'replace',
        'annotation_label': 'mad_artifact'
    }
    AMPLITUDE_REJECTION_SETTINGS = {
        'enabled': True,
        'threshold': 400e-6,
        'annotation_label': 'amp_artifact'
    }
    ICA_SETTINGS = {
        'enabled': True,
        'n_mads': 6,
        'annotation_label': 'ica_artifact'
    }
    DEFAULT_CHANNEL_ORDER = ['AF7', 'AF8', 'TP9', 'TP10']
    FREQ_BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    OUTPUT_SETTINGS = {
        'brainwave_window_sec': 0.05,
        'output_dir': 'output',
        'enable_csv_export': True,
        'enable_global_plot': True,
        'enable_verification': True,
        'downsample_reference': True,
        'fill_missing': 'zero',  # or 'interpolate'
        'verification_metric': 'corr'  # Options: 'mse', 'corr', etc.
    }
