import pandas as pd
from src.tokenization import tokenize
from src.positional_encoding import positional_encoding
from src.transformer_model import TimeSeriesTransformer
from src.change_points import detect_changes
import torch
from src.cpd import cpd_segmentation

def preprocess(time_series, window_size, overlap):
    # change_points = detect_changes(time_series)
    
    # tokens = tokenize(time_series, window_size, overlap, change_points) # fixed

    model = 'binseg'
    segment_length = 60
    pen = 21
    n_bkps= 10
    min_size = 30
    tokens, change_points, intervals = cpd_segmentation(torch.tensor(time_series), model, segment_length, pen, min_size, n_bkps)

    pos_encoding = positional_encoding(tokens.shape[0], tokens.shape[1]).unsqueeze(-1)
    model_input = torch.cat((tokens, pos_encoding), dim=-1)
    return model_input


def remove_outliers():
    pass

def interpolate():
    pass

def normalize():
    pass

def smoothing():
    pass


