import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import plotly.graph_objects as go

def plot_eeg_with_annotations(raw, duration_sec=20, save_pdf_path=None):

    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))
    fig.write_html("test_plot.html", auto_open=True)

    sfreq = raw.info['sfreq']
    n_samples = int(duration_sec * sfreq)
    times = raw.times[:n_samples]
    data = raw.get_data()[:, :n_samples] * 1e6  # Convert to µV for display

    ch_names = raw.ch_names
    unique_labels = sorted(set(ann['description'] for ann in raw.annotations))
    colormap = cm.get_cmap('tab10', len(unique_labels))
    label_colors = {label: colormap(i) for i, label in enumerate(unique_labels)}

    for ch_idx, ch_name in enumerate(ch_names):
        plt.figure(figsize=(12, 4))
        plt.plot(times, data[ch_idx], label=f'{ch_name} Raw Signal', color='black')
        plt.title(f'{ch_name} - First {duration_sec} Seconds with Annotations')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.grid(True)

        for annotation in raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            label = annotation['description']
            end = onset + duration
            if onset < duration_sec:
                plt.axvspan(onset, min(end, duration_sec), color=label_colors[label], alpha=0.3, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.tight_layout()
        plt.show()


def plot_single_channel_with_annotations(raw, channel_name=None, figsize=(60, 4)):
    sfreq = raw.info['sfreq']
    times = raw.times
    data = raw.get_data() * 1e6  # µV

    if channel_name is None:
        ch_idx = 0
        channel_name = raw.ch_names[0]
    else:
        ch_idx = raw.ch_names.index(channel_name)

    signal = data[ch_idx]

    unique_labels = sorted(set(ann['description'] for ann in raw.annotations))
    colormap = cm.get_cmap('tab10', len(unique_labels))
    label_colors = {label: colormap(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=figsize)  # רוחב גדול יוצר אפקט של גלילה
    plt.plot(times, signal, label=f'{channel_name} EEG', color='black')
    plt.title(f'{channel_name} - Full Duration with Annotations')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)

    for ann in raw.annotations:
        onset = ann['onset']
        duration = ann['duration']
        label = ann['description']
        end = onset + duration
        plt.axvspan(onset, end, color=label_colors[label], alpha=0.3, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    plt.show()