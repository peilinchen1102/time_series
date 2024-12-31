import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

def generate_token(window: torch.Tensor) -> torch.Tensor:
    """
    Extracts features from a window of time series data.

    :param window: Tensor time series window of shape (window_size, num_channels)
    :return: Tensor (token) of extracted features (e.g., mean, std) for each channel
    """
    # dim=0 for each channel
    mean = torch.mean(window, dim=0)
    std = torch.std(window, dim=0)
    
    combined_features = torch.cat((window, mean.unsqueeze(0), std.unsqueeze(0)), dim=0)
    return combined_features

def fixed_window_tokens(time_series: torch.Tensor, window_size: int, overlap: int=0) -> torch.Tensor:
    """
    Extracts features from time series data using a fixed window size.

    :param time_series: Tensor full time series data
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
            padding = torch.zeros((window_size - window.shape[0], channels))
            window = torch.vstack((window, padding))
        features.append(generate_token(window))

    return torch.stack(features)

def variable_window_tokens(time_series: torch.Tensor, change_points: List[int], overlap: int=0) -> torch.Tensor:
    """
    Extracts features from multichannel time series data using variable window sizes based on specified change points.

    :param time_series: Tensor full time series data of shape (num_timesteps, num_channels)
    :param change_points: List of sorted indices indicating change points in the time series
    :param overlap: Number of overlapping timesteps between consecutive windows
    :return: List of tuples with extracted features (mean, std) for each variable window
    """

    features = []
    timesteps, channels = time_series.shape
    change_points = [0] + change_points + [timesteps]
    max_length = 0
    for start, end in zip(change_points[:-1], change_points[1:]):
        # Window around change point to capture essential change point info
        s = max(start - overlap, 0)
        e = min(end + overlap, timesteps)
        window = time_series[s:e]
        
        if window.shape[0] > 1:
            token = generate_token(window)
            max_length = max(max_length, token.shape[0])
            features.append(token)
    return pad_sequence(features, batch_first=True)

def tokenize(time_series: torch.Tensor, window_size: int, overlap: int, change_points: List[int]) -> torch.Tensor:
    """
    Tokenizes the time series into fixed-size windows with extracted features.
    :param time_series: Array of time series data
    :param window_size: Size of each token window
    :param num_features: Number of features per window
    :return: Tensor of tokens
    """
    tokens = []
    if window_size:
        return fixed_window_tokens(time_series, window_size, overlap)
    else:
        return variable_window_tokens(time_series, change_points, overlap)
    
