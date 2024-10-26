import numpy as np
import torch
from src.feature_extraction import fixed_window_features, variable_window_features
from numpy.typing import NDArray
from typing import List

def tokenize(time_series: NDArray[np.float64], window_size: int, overlap: int, change_points: List[int]) -> torch.Tensor:
    """
    Tokenizes the time series into fixed-size windows with extracted features.
    :param time_series: Array of time series data
    :param window_size: Size of each token window
    :param num_features: Number of features per window
    :return: Tensor of tokens
    """
    tokens = []
    if window_size:
        features = fixed_window_features(time_series, window_size, overlap)
    else:
        features = variable_window_features(time_series, change_points, overlap)

    # TODO: change to use attention
    tokens = np.concatenate([time_series, features])
    return torch.tensor(tokens, dtype=torch.float32)