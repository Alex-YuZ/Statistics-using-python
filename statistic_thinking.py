import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def ecdf(df, x_label):
    """Plot ecdf based on a given dataframe and column name"""
    x = np.sort(df[col_name])
    length = len(df[col_name])
    y = np.arange(1, length+1) / length
    
    # Plot and set properties
    plt.plot(x, y, marker='.', linestyle='')
    plt.xlabel("{}".format(x_label))
    plt.ylabel('ECDF')
    
    # Keep data off plot edges
    plt.margins(.05);