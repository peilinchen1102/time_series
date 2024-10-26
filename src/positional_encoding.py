import numpy as np
import torch

def positional_encoding(sequence_length: int, d_model: int) -> torch.Tensor:
    """
    Generates sinusoidal positional encodings for a sequence.
    :param sequence_length: Length of the sequence
    :param d_model: Embedding dimension
    :return: Positional encoding tensor
    """
    position = np.arange(sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pos_encoding, dtype=torch.float32)