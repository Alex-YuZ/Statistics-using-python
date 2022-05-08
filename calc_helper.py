import numpy as np
import pandas as pd


def ecdf(data):
    """Map the given data to (x, y) for plotting

    Args:
        data (list): the data series to be plotted on the ecdf plot
    """
    # Get the data length
    dt_length = len(data)
    
    # Sort the given data in ascending order
    x = np.sort(data)
    
    # Mark each data point as percentage
    y = np.arange(1, dt_length+1) / dt_length
    
    return x, y


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]