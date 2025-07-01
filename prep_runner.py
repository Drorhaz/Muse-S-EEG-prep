# prep_runner.py

import os
import sys
import pandas as pd
from prep_function import (
    load_muse_data, base_filtering,
    median_filter_artifact_removal, dynamic_threshold_artifact_removal,
    auto_artifact_rejection, run_ica, annotate_ica_artifacts
)
from prep_config import ICA_SETTINGS, OUTPUT_SETTINGS
from prep_output import (
    plot_annotated_eeg, extract_brainwaves,
    export_cleaned_data, plot_global_brainwaves,
    verify_against_reference
)

def run_eeg_cleaning_pipeline(csv_path, output_dir=None):
    output_dir = output_dir or OUTPUT_SETTINGS.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # === Load and extract timestamps ===
    ref_df = pd.read_csv(csv_path)
    if 'TimeStamp' not in ref_df.columns:
        raise ValueError("Input CSV must contain a 'TimeStamp' column.")
    timestamp_list = ref_df['TimeStamp'].tolist()

    # === 1. Load and filter ===
    raw = load_muse_data(csv_path)
    raw = base_filtering(raw)
    raw = median_filter_artifact_removal(raw)
    raw = dynamic_threshold_artifact_removal(raw)
    raw = auto_artifact_rejection(raw)

    # === 2. ICA ===
    ica = run_ica(raw)
    ica.exclude = [0]  # Example setting
    if ICA_SETTINGS['enabled']:
        raw = annotate_ica_artifacts(raw, ica, label=ICA_SETTINGS['annotation_label'], n_mads=ICA_SETTINGS['n_mads'])
    ica.apply(raw)

    # === 3. Plot annotations ===
    plot_annotated_eeg(raw, output_dir)

    # === 4. Brainwave Extraction with aligned timestamps ===
    brainwave_df = extract_brainwaves(raw, timestamp_list)

    # === 5. Export CSV ===
    if OUTPUT_SETTINGS.get('enable_csv_export', True):
        export_cleaned_data(raw, brainwave_df, os.path.join(output_dir, "cleaned_output.csv"))

    # === 6. Global Brainwave Plot ===
    if OUTPUT_SETTINGS.get('enable_global_plot', True):
        plot_global_brainwaves(brainwave_df, os.path.join(output_dir, "global_brainwaves.png"))

    # === 7. Verification ===
    if OUTPUT_SETTINGS.get('enable_verification', True):
        verify_against_reference(brainwave_df, csv_path, os.path.join(output_dir, "verification_results.csv"))

    print(f"Pipeline complete. Outputs saved to '{output_dir}'")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "mindMonitor_2025-05-21--23-26-36.csv"

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
    else:
        run_eeg_cleaning_pipeline(csv_path)
