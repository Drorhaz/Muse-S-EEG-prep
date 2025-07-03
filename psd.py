import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_main_bands_psd(filepath, fs=256):
    """
    Plot PSD for Delta, Theta, Alpha, Beta bands from a Muse CSV file.

    Args:
        filepath (str): Path to the CSV file.
        fs (int): Sampling frequency in Hz (default: 256).
    """
    df = pd.read_csv(filepath)

    bands = ["Delta", "Theta", "Alpha", "Beta"]
    found_columns = {}

    # Find one representative column for each band (e.g., Delta_TP9)
    for band in bands:
        for col in df.columns:
            if col.startswith(band) and pd.api.types.is_numeric_dtype(df[col]):
                found_columns[band] = col
                break

    if not found_columns:
        raise ValueError("No valid band columns found in the file.")

    # Create subplots
    fig, axes = plt.subplots(nrows=len(found_columns), figsize=(10, 2.5 * len(found_columns)))

    if len(found_columns) == 1:
        axes = [axes]  # If only one subplot, make it iterable

    for ax, (band, col) in zip(axes, found_columns.items()):
        signal = pd.to_numeric(df[col], errors='coerce').dropna().values
        if len(signal) == 0:
            ax.set_title(f"{band}: no valid data")
            continue

        freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
        ax.semilogy(freqs, psd, label=col, color='purple')
        ax.set_title(f'{band} Band - {col}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (µV²/Hz)')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_file = "demo_eeg_data.csv"
    plot_main_bands_psd(test_file)