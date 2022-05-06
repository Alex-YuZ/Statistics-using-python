import numpy as np
from viz_helper import bernoulli_trials_plot
from probability_test import perform_bernoulli_trials
import pandas as pd

def bank_defaults(num_trials, prob_default, num_repeats=1000):
    """Simulate bank defaults using Bernoulli trials

    Args:
        num_trials (int): number of trials in Bernoulli trial
        prob_default (float): threshold for determining the binary result
        num_repeats (int, optional): number of repeats for the Bernoulli trials. Defaults to 1000.

    Returns:
        pd.DataFrame: Bernoulli trials results
    """
    n_defaults = np.empty(num_repeats)
    for i in range(num_repeats):
        n_defaults[i] = perform_bernoulli_trials(num_trials, prob_default)

    bernoulli_trials_plot(n_defaults, 'num of defaults in 100 loans')
    return pd.DataFrame({'num_defaults': n_defaults})