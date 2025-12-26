"""
Unit tests for mviewer utility functions
Tests: load_mat, load_pkl, load_textgrid, check_dictionary, get_struct
"""
import unittest
import os
import sys
import tempfile
import pickle
import numpy as np

# Add parent directory to path to import mviewer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from mviewer import load_pkl, check_dictionary


class TestLoadPkl(unittest.TestCase):
    """Test load_pkl function"""

    def setUp(self):
        """Create temporary test data"""
        self.test_dir = tempfile.mkdtemp()
        self.valid_pkl = os.path.join(self.test_dir, 'valid.pkl')
        self.invalid_pkl = os.path.join(self.test_dir, 'invalid.pkl')
        self.empty_pkl = os.path.join(self.test_dir, 'empty.pkl')

        # Create valid pickle file
        valid_data = {'AUDIO': {'NAME': 'test', 'SIGNAL': np.array([1, 2, 3])}}
        with open(self.valid_pkl, 'wb') as f:
            pickle.dump(valid_data, f)

        # Create invalid pickle (not a dict)
        with open(self.invalid_pkl, 'wb') as f:
            pickle.dump([1, 2, 3], f)

        # Create empty dict pickle
        with open(self.empty_pkl, 'wb') as f:
            pickle.dump({}, f)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_load_valid_pkl(self):
        """Test loading valid pickle file"""
        data = load_pkl(self.valid_pkl)
        self.assertIsInstance(data, dict)
        self.assertIn('AUDIO', data)

    def test_load_invalid_pkl_not_dict(self):
        """Test loading pickle that's not a dictionary"""
        with self.assertRaises(AssertionError) as context:
            load_pkl(self.invalid_pkl)
        self.assertIn('not python-dictionary', str(context.exception))

    def test_load_empty_pkl(self):
        """Test loading empty dictionary"""
        with self.assertRaises(AssertionError) as context:
            load_pkl(self.empty_pkl)
        self.assertIn('no valid keys', str(context.exception))

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        with self.assertRaises(FileNotFoundError):
            load_pkl('nonexistent.pkl')


class TestCheckDictionary(unittest.TestCase):
    """Test check_dictionary function"""

    def setUp(self):
        """Create test data structures"""
        self.field_names = ['NAME', 'SRATE', 'SIGNAL']
        self.channel_names = ['TR', 'TB', 'TT']
        self.audio_channel = 'AUDIO'

        # Valid dictionary
        self.valid_dict = {
            'AUDIO': {'NAME': 'audio', 'SRATE': 44100, 'SIGNAL': np.array([])},
            'TR': {'NAME': 'tr', 'SRATE': 100, 'SIGNAL': np.array([])},
            'TB': {'NAME': 'tb', 'SRATE': 100, 'SIGNAL': np.array([])},
            'TT': {'NAME': 'tt', 'SRATE': 100, 'SIGNAL': np.array([])}
        }

    def test_valid_dictionary(self):
        """Test with valid dictionary structure"""
        # Should not raise any exception
        check_dictionary(self.valid_dict, self.field_names, self.channel_names, self.audio_channel)

    def test_missing_channel(self):
        """Test with missing channel"""
        invalid_dict = self.valid_dict.copy()
        del invalid_dict['TR']
        with self.assertRaises(AssertionError) as context:
            check_dictionary(invalid_dict, self.field_names, self.channel_names, self.audio_channel)
        self.assertIn('not matching', str(context.exception))

    def test_extra_channel(self):
        """Test with extra channel"""
        invalid_dict = self.valid_dict.copy()
        invalid_dict['EXTRA'] = {'NAME': 'extra', 'SRATE': 100, 'SIGNAL': np.array([])}
        with self.assertRaises(AssertionError) as context:
            check_dictionary(invalid_dict, self.field_names, self.channel_names, self.audio_channel)
        self.assertIn('not matching', str(context.exception))

    def test_missing_field(self):
        """Test with missing field in channel"""
        invalid_dict = self.valid_dict.copy()
        invalid_dict['TR'] = {'NAME': 'tr', 'SRATE': 100}  # Missing SIGNAL
        with self.assertRaises(AssertionError) as context:
            check_dictionary(invalid_dict, self.field_names, self.channel_names, self.audio_channel)
        self.assertIn('field_names are not matching', str(context.exception))


if __name__ == '__main__':
    unittest.main()
