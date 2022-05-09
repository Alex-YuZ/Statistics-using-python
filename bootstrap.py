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
        
    -----
    This can be used to find confidence interval for slopes and intercepts.
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

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets.

    Args:
        data1 (numpy.array): 1d numpy array for dataset1 to be permutated.
        data2 (numpy.array): 1d numpy array for dataset2 to be permutated.

    Returns:
        perm_sample_1 (numpy.array): data array 1 afte permutation
        perm_sample_2 (numpy.array): data array 2 afte permutation
    -----
    This function can be used in the following scenario such as hypthesis testing:
        H_0 Hypothesis: data1 and data2 have the identical distribution
    """

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2