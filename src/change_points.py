from numpy.typing import NDArray
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

def detect_changes(time_series: NDArray):

    # use 'l2', better for multivariate data
    algo = rpt.Pelt(model="l2", min_size=1, jump=10).fit(time_series)
    result = algo.predict(pen=2)

    change_points = [i for i in result if i < len(time_series)]
    return change_points