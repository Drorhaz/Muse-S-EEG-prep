🧠 Muse-S-EEG-prep
Modular Muse-S EEG Preprocessor – Hackathon Project

🎯 Goal
Extract, clean, and visualize brainwave activity (Delta, Theta, Alpha, Beta, Gamma) from Muse-S EEG recordings. The pipeline supports raw signal inspection, multi-stage filtering, frequency decomposition, annotation, and optional validation.
The pipeline empowers EEG researchers and developers with a streamlined, modular process for cleaning and analyzing Muse-S data. Transparent plots and customizable filters make it ideal for experimentation, research, and hackathons.

📥 Input
CSV exported from Mind Monitor containing:

Raw EEG: TP9, AF7, AF8, TP10

TimeStamp column

📤 Output
🔹 Visualizations
Raw EEG – 4 subplots (1 per channel)
Filtered EEG – high-pass + notch + low-pass
Artifact Annotations – MAD-based thresholding overlays
ICA Cleanup Preview – before vs. after component rejection
Brainwave Decomposition – band-filtered EEG per channel
Power Spectral Density (PSD) – 0–50 Hz energy plots
Validation Plot (optional) – original vs. filtered band power with correlation and MSE

🔹 Files
.csv with:
Raw + filtered EEG
Brainwave power (per channel and band)
Annotation metadata
.fif MNE files (per channel)

🧪 Pipeline Overview
Load CSV
Clean & scale EEG signals to volts
Create mne.Raw object
Raw Plot
Visual inspection of unfiltered EEG
Base Filtering
High-pass (1 Hz), notch (50 Hz), low-pass (40 Hz), average referencing
Filtered Plot
View the cleaned signal
Artifact Annotation
Detect high-amplitude segments (MAD-based threshold)
ICA Decomposition (Optional)\
Detect & remove blink/heartbeat artifacts
Preview component effects
Brainwave Extraction
Filter signal into Delta–Gamma bands per channel
Create decomposed MNE Raw objects
Spectral Plotting
PSD across channels (0–50 Hz)
Band decomposition plot per channel
Export Results
Combined CSV (time, bands, annotations)
.fif files for further analysis

