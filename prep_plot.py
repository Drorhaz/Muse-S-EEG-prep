import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import plotly.graph_objects as go


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