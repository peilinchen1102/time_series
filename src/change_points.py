from numpy.typing import NDArray
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

def detect_changes(time_series: NDArray):

    # use 'rbf', better for sudden jumps
    algo = rpt.Pelt(model="rbf", min_size=1, jump=10).fit(time_series)
    result = algo.predict(pen=5)

    change_points = [i for i in result if i < len(time_series)]
    return change_points