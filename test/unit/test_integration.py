"""
Integration tests using actual example files
Tests the full workflow with real data
"""
import unittest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from mviewer import Viewer


class TestIntegrationWithExamples(unittest.TestCase):
    """Integration tests with example data files"""

    def setUp(self):
        """Check if example files exist"""
        self.example_dir = os.path.join(os.path.dirname(__file__), '../../example')
        self.mat_file = os.path.join(self.example_dir, 'F01_B01_S01_R01_N.mat')
        self.pkl_file = os.path.join(self.example_dir, 'F01_B01_S01_R01_N.pkl')

    def test_example_files_exist(self):
        """Check that example files are available"""
        has_mat = os.path.exists(self.mat_file)
        has_pkl = os.path.exists(self.pkl_file)

        if not (has_mat or has_pkl):
            self.skipTest('No example files found')

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../example/F01_B01_S01_R01_N.pkl')),
        "Example pickle file not found"
    )
    def test_load_example_pkl(self):
        """Test loading example pickle file"""
        viewer = Viewer()
        viewer.load(self.pkl_file)

        self.assertIsNotNone(viewer.data)
        self.assertEqual(viewer.loaded_data_type, 'pkl')
        self.assertIn('AUDIO', viewer.data)

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../example/F01_B01_S01_R01_N.mat')),
        "Example mat file not found"
    )
    def test_load_example_mat(self):
        """Test loading example mat file"""
        viewer = Viewer()
        viewer.load(self.mat_file)

        self.assertIsNotNone(viewer.data_orig)
        self.assertEqual(viewer.loaded_data_type, 'mat')

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../example/F01_B01_S01_R01_N.pkl')),
        "Example pickle file not found"
    )
    def test_full_workflow_pkl(self):
        """Test complete workflow: load -> convert -> verify"""
        viewer = Viewer()

        # Load pickle
        viewer.load(self.pkl_file)
        self.assertIsNotNone(viewer.data)

        # Verify data structure
        self.assertIn('AUDIO', viewer.data)
        self.assertIn('SIGNAL', viewer.data['AUDIO'])
        self.assertIn('SRATE', viewer.data['AUDIO'])

        # Check that we have expected channels
        expected_channels = ['AUDIO', 'TR', 'TB', 'TT', 'UL', 'LL', 'JAW']
        for channel in expected_channels:
            if channel in viewer.data:
                self.assertIn('SIGNAL', viewer.data[channel])


if __name__ == '__main__':
    unittest.main()
