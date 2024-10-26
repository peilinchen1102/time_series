import pandas as pd
from numpy.typing import NDArray

def load_data(file_path: str) -> NDArray:
    """
    Loads time series data from a CSV file.
    :param file_path: Path to CSV file
    :return: Array of time series data
    """
    df = pd.read_csv(file_path)
    return df.values.flatten()