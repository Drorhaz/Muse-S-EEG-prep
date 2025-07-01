import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prep_function import (
    load_muse_data, base_filtering,
    median_filter_artifact_removal, dynamic_threshold_artifact_removal,
    auto_artifact_rejection, run_ica, annotate_ica_artifacts
)
from prep_config import CH_NAMES, ICA_SETTINGS
from prep_plot import plot_eeg_with_annotations, plot_single_channel_with_annotations

def run_eeg_cleaning_pipeline(csv_path):
    # Load raw EEG data
    raw = load_muse_data(csv_path)
    raw_before_cleaning = raw.copy()

    # Apply cleaning steps in sequence
    raw = base_filtering(raw)
    raw = median_filter_artifact_removal(raw)
    raw = dynamic_threshold_artifact_removal(raw)
    raw = auto_artifact_rejection(raw)

    # ICA step
    ica = run_ica(raw)
    ica.exclude = [0]  # Example: you can change this interactively
    if ICA_SETTINGS['enabled']:
        raw = annotate_ica_artifacts(raw, ica, label=ICA_SETTINGS['annotation_label'], n_mads=ICA_SETTINGS['n_mads'])
    ica.apply(raw)

    # === Annotation Summary ===
    print("\n=== Annotation Summary ===")
    label_counts = {}
    for ann in raw.annotations:
        label = ann['description']
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in label_counts.items():
        print(f"{label}: {count} annotations")

    # Plot the first 20 seconds of each channel with annotations
    for ch_name in raw.ch_names:
        print(f"Displaying: {ch_name}")
        plot_single_channel_with_annotations(raw, channel_name=ch_name)   
    # duration_sec = 20
    # sfreq = raw.info['sfreq']
    # n_samples = int(duration_sec * sfreq)
    # times = raw.times[:n_samples]
    # data = raw.get_data()[:, :n_samples] * 1e6  # Convert to µV for display

    # # Generate unique colors for annotation labels
    # unique_labels = sorted(set(ann['description'] for ann in raw.annotations))
    # colormap = cm.get_cmap('tab10', len(unique_labels))
    # label_colors = {label: colormap(i) for i, label in enumerate(unique_labels)}

    # for ch_idx, ch_name in enumerate(CH_NAMES):
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(times, data[ch_idx], label=f'{ch_name} Raw Signal', color='black')
    #     plt.title(f'{ch_name} - First {duration_sec} Seconds with Annotations')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Amplitude (µV)')
    #     plt.grid(True)

    #     # Overlay annotations
    #     for annotation in raw.annotations:
    #         onset = annotation['onset']
    #         duration = annotation['duration']
    #         label = annotation['description']
    #         end = onset + duration
    #         if onset < duration_sec:
    #             plt.axvspan(onset, min(end, duration_sec), color=label_colors[label], alpha=0.3, label=label)

    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys())
    #     plt.tight_layout()
    #     plt.show()

    return raw

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "mindMonitor_2025-05-21--23-26-36.csv"

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
    else:
        run_eeg_cleaning_pipeline(csv_path)
