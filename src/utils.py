import pandas as pd
from numpy.typing import NDArray
from typing import List
import numpy as np
import torch
import ruptures as rpt
import matplotlib.pyplot as plt
import wfdb
import ast
import torch.nn.functional as F

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def load_data(path, sampling_rate):
    """
    Loads time series data.
    :param file_path: Path to data
    :return: Array of time series data
    """
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return X_train, y_train, X_test, y_test

def pad_window_interval(model_inputs: List[torch.Tensor]):
    padded_sequences = []
    max_seq_length = max(tensor.shape[1] for tensor in model_inputs)
    for tensor in model_inputs:
        seq_len = tensor.shape[1]
        padding_size = max_seq_length - seq_len
        if padding_size > 0:
            padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), mode='constant', value=0)
            padded_sequences.append(padded_tensor)
        else:
            padded_sequences.append(tensor)

    return padded_sequences