import pandas as pd
import numpy as np
from scipy.stats import t

def within_t_test(to_compare, alpha_level=0.05, test_type='two_tail', 
           sample=None, sample_mean=None, sample_std=None, sample_size=None):
    
    if sample is not None:
        sample_size = len(sample)
        dof = sample_size -1

        # Calculate sample statistics
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
    else:
        dof = sample_size - 1

    # Calculate t statistic
    se = sample_std/np.sqrt(sample_size)
    t_statistic = (sample_mean - to_compare)/se
    
    # Define conlusions context
    conlusions = ["Reject the NULL as statistically significant!", "Fail to reject the NULL."]

    # Calculate t critical value
    if test_type == 'left_tail':
        test_kind = "Left-Tailed"
        t_critical = t.ppf(alpha_level, dof)
        p_value = t.sf(abs(t_statistic), dof)
        
        if t_statistic <= t_critical:
            res = conlusions[0]
        else:
            res = conlusions[1]

    elif test_type == 'right_tail':
        test_kind = "Right-Tailed"
        t_critical = t.ppf(1-alpha_level, dof)
        p_value = t.sf(abs(t_statistic), dof)
        if t_statistic >= t_critical:
            res = conlusions[0]
        else:
            res = conlusions[1]
            

    elif test_type == 'two_tail':
        test_kind = "Two-Tailed"
        p_value = t.sf(abs(t_statistic), dof)*2
        if sample_mean >= to_compare:
            t_critical = t.ppf(1-alpha_level/2, dof)
            
            if t_statistic >= t_critical:
                res = conlusions[0]
            else:
                res = conlusions[1]
                

        else:
            t_critical = t.ppf(alpha_level/2, dof)
            
            if t_statistic <= t_critical:
                res = conlusions[0]
            else:
                res = conlusions[1]
                
    # Cohen's d
    cohen = (sample_mean - to_compare)/sample_std
    
    # r_sqrd
    r2 = t_statistic**2/(t_statistic**2 + dof)
    
    # margin of error
    t_critical1 = t.ppf(alpha_level/2, dof)
    moe = abs(t_critical1)*se
    
    # CI
    lower = sample_mean - moe
    upper = sample_mean + moe
    ci = "{0:.1%} Confidence Interval=({1:.2f}, {2:.2f})".format(1-alpha_level, lower, upper)

    print_out = """
    =============== Reports ==============
    
        **Descriptive Statistics Summary**

          sample size: {5}
          sample mean: {0:.3f}
          sample SD: {1:.3f}
      
        **Inferential Statistics Summary**

          Test Type: One-Sample {9} t-test
          degree of freedom: {6}
          p-value: {7:.5f}
          t-statistic: {2:.3f}
          t-critical: {3:.3f}
          alpha-level: {8}
          margin of error: {13:.2f}
          {12}
      
        **Effect Size**
          Cohen's d: {10:.3f}
          r2: {11: .3f}
            
          ---------------------------------
          
    Conclusion: {4}
    
    ================== END =================
    """

    print(print_out.format(sample_mean, 
                           sample_std, 
                           t_statistic, 
                           t_critical, 
                           res, 
                           sample_size, 
                           dof, p_value, 
                           alpha_level, 
                           test_kind, 
                           cohen, 
                           r2, 
                           ci, 
                           moe))
    
    
    
    
def between_t_test(group1, group2, alpha_level=0.05, test_type='two_tail'):

    size1 = group1.shape[0]
    size2 = group2.shape[0]

    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    ddof1 = size1 - 1
    ddof2 = size2 - 1
    ddof = size1 + size2 -2
    
    ssq1 = np.sum((group1 - mean1)**2)
    ssq2 = np.sum((group2 - mean2)**2)
    
    pooled_var = (ssq1 + ssq2)/(ddof1 + ddof2)

    se = np.sqrt(std1**2/size1 + std2**2/size2)
    corrected_se = np.sqrt(pooled_var/size1 + pooled_var/size2)
    
    t_statistic = (mean1 - mean2)/se
    corr_t_statistic = (mean1 - mean2)/corrected_se
    
    # Define conlusions context
    conlusions = ["Reject the NULL as statistically significant!", "Fail to reject the NULL."]
    
    if test_type == 'left_tail':
        test_kind = 'Left-Tailed'
        t_critical = t.ppf(alpha_level, ddof)
        p_value = t.sf(abs(t_statistic), ddof)
        
        if t_statistic <= t_critical:
            res = conclusions[0]
        else:
            res = conclusions[1]
            
    elif test_type == 'right_tail':
        test_kind = 'Right-Tailed'
        t_critical = t.ppf(1-alpha_level, ddof)
        p_value = t.sf(abs(t_statistic), ddof)
        if t_statistic >= t_critical:
            res = conlusions[0]
        else:
            res = conlusions[1]
            
    elif test_type == 'two_tail':
        test_kind = "Two-Tailed"
        p_value = t.sf(abs(t_statistic), ddof)*2
        t_critical = abs(t.ppf(alpha_level/2, ddof))
            
        if t_statistic >= 0 and t_statistic >= t_critical:
            res = conlusions[0]
            
        elif t_statistic <= 0 and t_statistic <= -t_critical:
            t_critical = -t_critical
            res = conlusions[0]
            
        else:
            if t_statistic <=0:
                t_critical = -t_critical
            
            res = conlusions[1]
            
    # r_sqrd
    r2 = t_statistic**2/(t_statistic**2 + ddof)
    
    # margin of error
    t_critical1 = t.ppf(alpha_level/2, ddof)
    moe = abs(t_critical1)*se
    
    # mean diffs
    mean_diff = mean1 - mean2
    
    # CI
    lower = mean_diff - moe
    upper = mean_diff + moe
    ci = "{0:.1%} Confidence Interval = ({1:.2f}, {2:.2f})".format(1-alpha_level, lower, upper)
                
    print_out = """
    =============== Reports ==============
    
        **Descriptive Statistics Summary**

          sample-1 size: {0}
          sample-1 mean: {1:.3f}
          sample-1 SD: {2:.4f}
          sample-1 dof: {20}
          
          sample-2 size: {3}
          sample-2 mean: {4:.3f}
          sample-2 SD: {5:.4f}
          sample-2 dof: {21}
          
          Pooled Variance: {18: .4f}
          Standard Error: {6:.4f}
          Standard Error (Corrected): {17: .4f}
      
        **Inferential Statistics Summary**

          Test Type: Between-Group {13} t-Test
          degree of freedom: {7}
          p-value: {8:.5f}
          t-statistic: {9:.3f}
          t-statistic (corrected): {19: .3f}
          t-critical: {10:.3f}
          alpha-level: {11}
          margin of error: {15:.2f}
          {14}
      
        **Effect Size**
          r2: {16: .3f}
      
            
          ---------------------------------
          
    Conclusion: {12}
    
    ================== END =================
    """

    print(print_out.format(size1, 
                           mean1, 
                           std1, 
                           size2, 
                           mean2, 
                           std2, 
                           se, 
                           ddof, 
                           p_value, 
                           t_statistic, 
                           t_critical, 
                           alpha_level, 
                           res, 
                           test_kind, 
                           ci, 
                           moe, 
                           r2, 
                           corrected_se, 
                           pooled_var, 
                           corr_t_statistic, 
                           ddof1, 
                           ddof2
                           ))