import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def extract_features(window: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Extracts features from a window of time series data.

    :param window: Array-like time series window of shape (window_size, num_channels)
    :return: Array of extracted features (e.g., mean, std) for each channel
    """
    # axis = 0 for each channel
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    
    return mean, std

def fixed_window_features(time_series: NDArray[np.float64], window_size: int, overlap: int=0) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Extracts features from time series data using a fixed window size.

    :param time_series: Array-like full time series data
    :param window_size: Size of each fixed window
    :param overlap: Number of timesteps to overlap between consecutive windows
    :return: List of tuples with extracted features for each window
    """
    features = []
    timesteps, channels = time_series.shape
    step_size = window_size - overlap
    
    for i in range(0, timesteps, step_size):
        window = time_series[i:i + window_size]
        
        # If window shorter than window size, pad it
        if window.shape[0] < window_size:
            padding = np.zeros((window_size - window.shape[0], channels))
            window = np.vstack((window, padding))

        features.append(extract_features(window))

    return features

def variable_window_features(times_series: NDArray[np.float64], change_points: List[int], overlap:int = 0) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Extracts features from multichannel time series data using variable window sizes based on specified change points.

    :param time_series: Array-like full time series data of shape (num_timesteps, num_channels)
    :param change_points: List of sorted indices indicating change points in the time series
    :param overlap: Number of overlapping timesteps between consecutive windows
    :return: List of tuples with extracted features (mean, std) for each variable window
    """

    features = []
    timesteps, channels = times_series.shape
    change_points = [0] + change_points + [timesteps]

    for start, end in zip(change_points[:-1], change_points[1:]):
        # window around change point to capture essential change pt info
        s = start - overlap if start - overlap > 0 else 0
        e = end + overlap if end + overlap < timesteps else timesteps
        window = times_series[s:e]
        if window.shape[0] > 1:
            features.append(extract_features(window))
    return features