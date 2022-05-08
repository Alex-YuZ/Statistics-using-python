import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bernoulli_trials_plot(simulation, col_name, normed=True):
    """Plot the distribution of bernoulli trials

    Args:
        simulation (numpy.ndarray): result from bernoulli simulation
        col_name (str): customized name for the bernoulli result
        normed (bool, optional): display relative/absolute frequencies. Defaults to True.
    """
    simulation_df = pd.DataFrame(simulation, columns=[col_name], dtype='int')
    counts = simulation_df[col_name].value_counts()
    counts_total = counts.sum()
    
    if normed==True:
        res_df = pd.DataFrame(counts/counts_total)
        res_df.reset_index(inplace=True)
        res_df.rename(columns={col_name: 'density', 'index': col_name}, inplace=True)
        sns.barplot(data=res_df, 
                    x=col_name, 
                    y='density', 
                    color=sns.color_palette()[0]);
    else:
        res_df = pd.DataFrame(counts)
        res_df.reset_index(inplace=True)
        res_df.rename(columns={col_name: 'counts', 'index': col_name}, inplace=True)
        sns.barplot(data=res_df, 
                    x=col_name, 
                    y='counts', 
                    color=sns.color_palette()[0]);
        
        
def print_eq(slope, intercept, x_var, y_var):
    formula = """
    {3} = {0:.4f}*{2} + {1:.4f}
    """
    print(formula.format(slope, intercept, x_var, y_var))

