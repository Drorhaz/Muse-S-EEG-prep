import unittest
from prep_preprocessing import EEGPreprocessor
from prep_features import extract_frequency_bands
from prep_io import EEGIOHandler
import os
import glob

class TestEEGPreprocessor(unittest.TestCase):
    def setUp(self):
        # Use a small real CSV if available, else skip test
        self.csv_path = 'mindMonitor_2025-05-21--23-26-36.csv'
        self.output_dir = 'final_outputs'
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV {self.csv_path} not found.")

    def test_init_and_filter(self):
        preproc = EEGPreprocessor(self.csv_path)
        # Should have a raw attribute
        self.assertTrue(hasattr(preproc, 'raw'))
        # Should be able to run base_filtering without error
        try:
            preproc.base_filtering()
        except Exception as e:
            self.fail(f"base_filtering raised an exception: {e}")

    def test_output_files_created(self):
        preproc = EEGPreprocessor(self.csv_path)
        preproc.base_filtering()
        per_channel_raws = extract_frequency_bands(preproc.raw)
        EEGIOHandler.save_all_outputs(per_channel_raws, self.output_dir)
        fif_files = glob.glob(os.path.join(self.output_dir, '*_decomposed_raw.fif'))
        csv_files = glob.glob(os.path.join(self.output_dir, 'all_channels_with_bands.csv'))
        png_files = glob.glob(os.path.join(self.output_dir, '*.png'))
        self.assertEqual(len(fif_files), 4, "Should be 4 FIF files (one per channel)")
        self.assertEqual(len(csv_files), 1, "Should be 1 CSV file")
        self.assertGreaterEqual(len(png_files), 2, "Should be at least 2 PNG files (plots)")

if __name__ == '__main__':
    unittest.main()