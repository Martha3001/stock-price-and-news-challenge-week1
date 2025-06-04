# Test for data_loader.py
import unittest
import pandas as pd
import os
import sys
# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""

    def test_load_csv(self):
        """Test loading CSV data"""
        # Assuming 'test_data.csv' is a valid CSV file in the test directory
        df = DataLoader.load_csv('tests/test_data.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_load_csv_invalid_path(self):
        """Test loading CSV data with an invalid path"""
        with self.assertRaises(FileNotFoundError):
            DataLoader.load_csv('invalid_path.csv')