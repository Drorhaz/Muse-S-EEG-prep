# eeg_config.py

# ======================================
# EEG Configuration Settings for Muse Headband
# ======================================

# Raw channel column names in the CSV file
RAW_CHANNEL_NAMES = ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10']
# RAW_CHANNEL_NAMES = ['eeg1', 'eeg2', 'eeg3', 'eeg4']

# Channel names used in MNE after loading
CH_NAMES = ['AF7', 'AF8', 'TP9', 'TP10']

# Sampling frequency (Muse headbands usually operate at 256 Hz)
SFREQ = 256

# EEG channel type (used by MNE)
CH_TYPES = 'eeg'

# Approximate 3D positions of Muse electrodes (meters)
MUSE_POSITIONS = {
    'AF7': [-0.035, 0.065, 0.04],    # Front-left
    'AF8': [ 0.030, 0.060, 0.030],   # Front-right
    'TP9': [-0.072, -0.045, 0.01],   # Rear-left
    'TP10': [0.068, -0.042, 0.015],  # Rear-right
}

# Base filtering settings for preprocessing EEG
BASE_FILTER_SETTINGS = {
    'highpass_freq': 1.0,
    'lowpass_freq': 40.0,
    'notch_freq': 50.0,
    'filter_design': 'firwin',
    'eeg_reference': 'average'
}

# Median filter settings for artifact removal
MEDIAN_FILTER_SETTINGS = {
    'enabled': False,
    'kernel_size': 7,
    'annotation_label': 'median_artifact'
}

# Dynamic threshold-based artifact removal settings
DYNAMIC_THRESHOLD_SETTINGS = {
    'enabled': True,
    'n_mads': 10,
    'action': 'replace',
    'annotation_label': 'mad_artifact'
}

# Amplitude-based automatic annotation settings
AMPLITUDE_REJECTION_SETTINGS = {
    'enabled': True,
    'threshold': 400e-6,
    'annotation_label': 'amp_artifact'
}

# ICA artifact annotation settings
ICA_SETTINGS = {
    'enabled': True,
    'n_mads': 6,
    'annotation_label': 'ica_artifact'
}

# Default channel order for plots or looping
DEFAULT_CHANNEL_ORDER = ['AF7', 'AF8', 'TP9', 'TP10']

# Brain wave bands
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}  # For use in extraction and plotting

# Output control settings
OUTPUT_SETTINGS = {
    'brainwave_window_sec': 0.05,
    'output_dir': 'output',
    'enable_csv_export': True,
    'enable_global_plot': True,
    'enable_verification': True,
    'downsample_reference': True,
    'fill_missing': 'zero', # or 'interpolate'
    'verification_metric': 'corr'  # Options: 'mse', 'corr', etc.
}
