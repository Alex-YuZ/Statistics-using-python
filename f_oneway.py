import pandas as pd
import numpy as np
from scipy.stats import f
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
    ss_total = ss_between + ss_within
    
    ms_between = ss_between/dof_between
    ms_within = ss_within/dof_within
    
    f_statistic = ms_between/ms_within
    f_critical = f.ppf(1-alpha_level, dof_between, dof_within)
    p_value = f.sf(f_statistic, dof_between, dof_within)
    
    
    # Calculate eta2
    explained_var = ss_between/ss_total
    
    group_means_ = np.around(group_means, 2)
    
    conclusions = ['Reject the Null as statistically significant.', 'Fail to reject the Null.']
    
    res = conclusions[0] if f_statistic >= f_critical else conclusions[1]
    
    print_out = """
    =================ONE-WAY ANOVA TEST REPORT=================
    
      Size in Each Sample: {0}
      Mean in Each Sample: {1}
      Grand Mean: {2: .2f}
      Total Sum of Squares: {14: .4f}
      
      Between-Groups
        Sum of Squares: {3: .4f}
        Degree of Freedom: {4}
        Mean Squares: {5: .4f}
        
      Within-Groups
        Sum of Squares: {6: .4f}
        Degree of Freedom: {7}
        Mean Squares: {8: .4f}
        
      F Statistic: {9: .4f}
      P Value: {10: .4f}
      F Critical: {11: .4f}
      
      Explained Variance (eta_sqd): {12: .4f}
      
      ---------------------------------
      Conclusion: {13}
      
    ============================END============================
      
    
    
    """
    
    print(print_out.format(sample_sizes, 
                           group_means_, 
                           grand_mean, 
                           ss_between, 
                           dof_between, 
                           ms_between, 
                           ss_within, 
                           dof_within, 
                           ms_within, 
                           f_statistic, 
                           p_value, 
                           f_critical, 
                           explained_var, 
                           res, 
                           ss_total))

    
    labels_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    labels_collections = []
    for i in range(len(sample_sizes)):
        label = labels_choices.pop(0)
        labels_collections.append(label)
    group_labels = np.repeat(labels_collections, repeats=sample_sizes[0])
    
    tukey_hsd = pairwise_tukeyhsd(concate_np, group_labels)
    print(tukey_hsd)