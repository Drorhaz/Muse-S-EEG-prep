Muse-S-EEG-prep
Muse S EEG preprocessor – Hackathon Project

🧠 Muse-S EEG Brainwave Pipeline – Full Processing Flow
🎯 Project Goal
Develop a modular tool to extract, clean, visualize, and optionally validate brainwave activity (Delta, Theta, Alpha, Beta, Gamma) from Muse-S EEG recordings. Users benefit from raw signal inspection, layered filtering, frequency decomposition, and detailed visualization of power spectra and annotations.

📥 Input
CSV from Mind Monitor, containing:

Raw EEG (4 channels: TP9, AF7, AF8, TP10)

Timestamps

📤 Output
Plots

Raw EEG (4 subplots) – unfiltered channel traces

Base-filtered EEG – highpass + notch + lowpass

Annotated EEG with dynamic threshold artifacts

ICA removal preview – original vs. ICA-cleaned

Brainwave decomposed signals – per-channel Raw view with 5 frequency bands

Power Spectral Density (PSD) Plot – full-spectrum analysis (0–50 Hz)

Validation plot (optional) – compares original vs. filtered band power with correlation & MSE

CSV Export

Raw EEG & filtered EEG

Brainwave band values per channel (4 × 5)

Global annotation metadata per timepoint

FIF Files

Saved filtered/decomposed Raw objects (.fif) for each channel

🧪 Pipeline Steps
Data Loading

Converts raw Muse CSV to an MNE-compatible Raw object

Handles missing data, centers channels, scales values to volts

Raw EEG Plot

Displays the original signal for each channel

Base Filtering

High-pass filter (default: 1 Hz)

Notch filter (default: 50 Hz)

Low-pass filter (default: 40 Hz)

EEG reference set to ‘average’

Filtered EEG Plot

Shows cleaned signals after filtering

Dynamic Threshold Artifact Annotation

Annotates segments exceeding a multiple of the median absolute deviation (MAD)

Visualized as overlays on the EEG trace

Annotation Reset (Optional)

Removes specific annotations (e.g., "BAD_dynamic") for clean ICA input

ICA Decomposition & Cleanup

Fits ICA model to identify blink/heartbeat/muscle artifacts

Plots components, sources, and properties

Allows interactive or scripted rejection

Reconstructs signal and plots before vs. after

Brainwave Band Extraction

Extracts Delta–Gamma bands per channel

MNE Raw object is created for each channel with original + 5 bands

Per-Channel Band Plot

Time-domain visualization of each frequency-filtered band per channel

Power Spectral Density (PSD) Plot

Shows frequency energy distribution across all channels (0–50 Hz)

Data Export

Saves decomposed .fif files for each channel

Consolidated CSV output (timestamps, bands, annotations)
