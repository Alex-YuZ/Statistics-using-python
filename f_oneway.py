import pandas as pd
import numpy as np
from scipy.stats import f


def cal_f(*args, alpha_level=0.05):
    group_means = []
    concate_np = []
    sample_sizes = []
    ss_within = 0
    num_groups = 0
    
    for sample in args:
        num_groups += 1
        sample_sizes = np.append(sample_sizes, len(sample))
        single_mean = np.mean(sample)
        group_means = np.append(group_means, single_mean)
        
        concate_np = np.concatenate([concate_np, sample])
        
        ss_within += np.sum((np.subtract(sample, single_mean))**2)
        
    dof_between = num_groups - 1
    dof_within = np.size(concate_np) - num_groups
    grand_mean = np.mean(concate_np)
    ss_between = np.sum(sample_sizes*(group_means - grand_mean)**2)
    
    ms_between = ss_between/dof_between
    ms_within = ss_within/dof_within
    f_statistic = ms_between/ms_within
    f_critical = f.ppf(1-alpha_level, dof_between, dof_within)