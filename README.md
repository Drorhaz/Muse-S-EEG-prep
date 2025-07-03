🧠 Muse-S-EEG-prep
Modular Muse-S EEG Preprocessor – Hackathon Project

🎯 Goal
Extract, clean, and visualize brainwave activity (Delta, Theta, Alpha, Beta, Gamma) from Muse-S EEG recordings.

This pipeline empowers EEG researchers and developers with a streamlined, modular process for cleaning and analyzing Muse-S data. Transparent plots and customizable filters make it ideal for experimentation, research, and hackathons.

📥 Input
CSV exported from Mind Monitor, containing:

Raw EEG channels: TP9, AF7, AF8, TP10

TimeStamp column

📤 Output
🔹 Visualizations
Raw EEG – 4 subplots (1 per channel)

Filtered EEG – high-pass + notch + low-pass

Artifact Annotations – MAD-based threshold overlays

ICA Cleanup Preview – original vs. ICA-cleaned traces

Brainwave Decomposition – 5-band signals per channel (Delta → Gamma)

Power Spectral Density (PSD) – 0–50 Hz full-spectrum plot

Validation Plot (optional) – correlation and MSE comparison (original vs. filtered)

🔹 Files
.csv containing:

Raw + filtered EEG

Brainwave power per channel × band

Annotation metadata

.fif MNE files for each decomposed channel

🧪 Pipeline Overview
1. Load CSV
Clean and scale EEG signals to volts

Convert to mne.Raw object

2. Raw Plot
Visual inspection of unfiltered EEG signals

3. Base Filtering
High-pass (1 Hz)

Notch filter (50 Hz)

Low-pass (40 Hz)

Average referencing

4. Filtered Plot
Displays cleaned EEG signals

5. Artifact Annotation
Annotate high-amplitude segments using MAD threshold

6. ICA Decomposition (Optional)
Detect and remove blink/heartbeat artifacts

Preview and reject ICA components interactively or manually

7. Brainwave Extraction
Decompose each channel into 5 frequency bands

Store each as a new MNE Raw object

8. Spectral Plotting
PSD plot across all channels (0–50 Hz)

Per-channel wave decomposition plots

9. Export Results
.csv with time, bands, and annotations

.fif files for in-depth EEG analysis

🛠️ Built With
Python

MNE

NumPy

Pandas

Matplotlib
