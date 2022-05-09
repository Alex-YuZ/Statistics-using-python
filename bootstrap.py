import numpy as np
import pandas as pd


# def bootstrap_replicate_1d(data, func):
#     bs_sample = np.random.choice(data, len(data))
#     return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_sample = np.random.choice(data, len(data))
        bs_replicates[i] = func(bs_sample)

    return bs_replicates

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression.

    Args:
        x (numpy.array): predictor variable array
        y (numpy.array): response varaible array
        size (int, optional): _description_. Defaults to 1.

    Returns:
        bs_slope_reps (numpy.array): bootstrapped samples of slopes
        bs_intercept_reps (numpy.array): bootstrapped samples of intercepts
    """

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps