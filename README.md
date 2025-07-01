# Muse-S-EEG-prep
Muse S EEG preprocessor  - Hackathon 
# 🧠 Muse-S EEG Brainwave Extraction – Hackathon Guidelines

## 🎯 Project Goal
Develop a tool to extract and validate brainwave activity (Alpha, Beta, Gamma, Delta, Theta) from raw Muse-S EEG data. Filters are applied to clean the signal and compute wave power per channel and globally, with visual previews and validation against Mind Monitor reference values.

---

## 📥 Input
- **CSV from Mind Monitor**, containing:
  - Raw EEG (4 channels: TP9, AF7, AF8, TP10)
  - Precomputed brainwaves (for verification only)

---

## 📤 Output
1. **Plots**
   - Raw EEG (4 subplots)
   - Base-filtered EEG + annotation preview for each filter:
     - Median filter → highlights spikes
     - Threshold artifact → marks extreme regions
     - Amplitude rejection → flags high/low segments
     - ICA → tags blink/heartbeat artifacts
   - Final filtered vs. raw EEG (4 subplots)
   - Global brainwave graph (all 5 waves)
   - (Optional) Brainwave validation overlay

2. **CSV Export**
   - Raw EEG
   - Cleaned EEG
   - Brainwave power per channel (4 × 5)
   - Global brainwave power (5)

3. **Verification Report**
   - 20 metrics: computed vs. Mind Monitor brainwave accuracy (method TBD)

---

## 👥 Team Roles

| Group | Focus                            |
|-------|----------------------------------|
| 1     | Input, CLI/GUI, plotting         |
| 2     | Filtering pipeline (base + optional) |
| 3     | Brainwave extraction + validation |

> Shared repo on GitHub. Use Python, MNE, NumPy, Pandas, Matplotlib/Plotly.

---

## ⚙️ Filtering Flow
- **Base filter** always runs first.
- User selects optional filters:
  - Median
  - Dynamic Threshold
  - Amplitude Rejection
  - ICA (only after base)
- Plots show **base-filtered signal** with **preview annotations** of what each filter would affect—before applying.

---

## ✅ Verification
- Compute brainwaves from cleaned EEG
- Compare against Mind Monitor values
- Output 1 metric per channel × wave (20 total)
- Methods under evaluation: MSE, correlation, etc.

---

## 📌 Summary
This tool empowers users to clean, visualize, and validate EEG data from Muse-S using a modular pipeline. Each step is transparent and interactive, supporting both research and development use cases.

