import pandas as pd

class DataLoader:
    """Handles loading of solar data from various sources"""
    
    @staticmethod
    def load_csv(filepath):
        """
        Load data from CSV file
        Args:
            filepath: Path to CSV file
        Returns:
            pandas.DataFrame
        """
        return pd.read_csv(filepath)