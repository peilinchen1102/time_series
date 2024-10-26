import numpy as np
import torch
from numpy.typing import NDArray
from typing import Union

def tokenize_time_series(time_series: NDArray[np.float64], window_size: int, num_features: int) -> torch.Tensor:
    """
    Tokenizes the time series into fixed-size windows with extracted features.
    :param time_series: Array of time series data
    :param window_size: Size of each token window
    :param num_features: Number of features per window
    :return: Tensor of tokens
    """
    tokens = []
    for i in range(0, len(time_series) - window_size + 1, window_size):
        window = time_series[i:i + window_size]
        features = extract_features(window)
        token = np.concatenate([window, features])
        tokens.append(token)
    
    tokens = np.array(tokens)
    return torch.tensor(tokens, dtype=torch.float32)