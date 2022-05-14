import pandas as pd
import numpy as np
from scipy.stats import t

def t_test_1sample(to_compare, alpha_level=0.05, alternative='two_sided', 
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
    if alternative == 'less':
        test_kind = "Left-Tailed"
        t_critical = t.ppf(alpha_level, dof)
        p_value = t.sf(abs(t_statistic), dof)
        
        if t_statistic <= t_critical:
            res = conlusions[0]
        else:
            res = conlusions[1]

    elif alternative == 'greater':
        test_kind = "Right-Tailed"
        t_critical = t.ppf(1-alpha_level, dof)
        p_value = t.sf(abs(t_statistic), dof)
        if t_statistic >= t_critical:
            res = conlusions[0]
        else:
            res = conlusions[1]
            

    elif alternative == 'two_sided':
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


def t_test_paired_sample(group1, group2, alpha_level=.05, alternative="two_sided"):
    """execute paired-sample t-test

    Args:
        group1 (array-like): _sample data 1
        group2 (array-like): _sample data 2
        alpha_level (float, optional): probability threshold to 
            reject/retain the null_. Defaults to .05.
        alternative (str, {'two_sided', 'less', 'greater'}): direction of t-test. Defaults to "two_sided".
    """
    # Convert to numpy arrays
    gp1_np = np.array(group1)
    gp2_np = np.array(group2)
    
    # length of samples
    size = len(gp1_np)
    
    # degree of freedom
    dof = size - 1
    
    # mean of each sample
    mean_gp1 = gp1_np.mean()
    mean_gp2 = gp2_np.mean()
    
    # mean of differences (MD)
    mean_diff = np.mean(gp1_np - gp2_np)
    
    # standard deviation of MD
    s_d = np.std(gp1_np - gp2_np, ddof=1)
    
    # standard error of MD
    se = s_d/np.sqrt(size)
    
    # calculate t-statistic
    t_statistic = mean_diff / se
    
    # calcluate effect size in cohen's d
    cohens_d = mean_diff / s_d
    
    # find t critical value for calculating CI
    t_critical_ci = t.ppf(.975, df=dof)
    
    # margin of error
    moe = t_critical_ci*se
    
    # CI intervals
    lower, upper = mean_diff - moe, mean_diff + moe
    
    ci = "{0:.1%} Confidence Interval = ({1:.2f}, {2:.2f})".format(1-alpha_level, lower, upper)
    
    # calculate effect size in r2
    r2 = t_statistic**2/(t_statistic**2 + dof)
    
    # Define conlusions context
    conclusions = ["Reject the NULL as statistically significant!", "Fail to reject the NULL."]
    
    # logic for 'two-tailed' t-test
    if alternative == 'two_sided':
        t_critical = t.ppf(1-alpha_level/2, df=dof)
        test_kind = 'Two-Tailed'
        p = t.sf(abs(t_statistic), df=dof)*2
        if abs(t_statistic) >= t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        t_critical_formatted = "(+/-){:.3f}".format(t_critical)
    
    # when alternative hypothesis is 'less than'
    elif alternative == 'less':
        t_critical = t.ppf(alpha_level, df=dof)
        test_kind = 'Left-Tailed'
        p = t.sf(abs(t_statistic), df=dof)
        if t_statistic < t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        t_critical_formatted = "{:.3f}".format(t_critical)
    
    # when alternative hypothesis is 'greater than'       
    elif alternative == 'greater':
        t_critical = t.ppf(1 - alpha_level, df=dof)
        test_kind = 'Right-Tailed'
        p = t.sf(abs(t_statistic), df=dof)
        if t_statistic > t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        t_critical_formatted = "{:.3f}".format(t_critical)
            
    
    # result formatting
    print_out = """
    =============== Reports ==============
    
        **Descriptive Statistics Summary**

          sample size: {0}
          degree of freedom: {1}
          
          mean of sample-1: {2:.3f}
          mean of sample-2: {3:.3f}
          
          difference of means: {4:.3f}
          SD of differences of means: {5:.3f}
      
        **Inferential Statistics Summary**

          Test Type: Paired-Sample {6} t-test
          p-value: {7:.5f}
          t-statistic: {8:.3f}
          t-critical: {9}
          alpha-level: {10}
          margin of error: {11:.2f}
          {12}
      
        **Effect Size**
          Cohen's d: {13:.3f}
          r2: {14: .3f}
            
          ---------------------------------
          
    Conclusion: {15}
    
    ================== END =================
    """
    
    print(print_out.format(size, 
                           dof, 
                           mean_gp1, 
                           mean_gp2, 
                           mean_diff, 
                           s_d, 
                           test_kind, 
                           p, 
                           t_statistic, 
                           t_critical_formatted, 
                           alpha_level, 
                           moe, 
                           ci, 
                           cohens_d, 
                           r2, 
                           res))


 
def t_test_2sample(group1, group2, alpha_level=0.05, alternative='two_sided'):

    size1 = group1.shape[0]
    size2 = group2.shape[0]

    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    dof1 = size1 - 1
    dof2 = size2 - 1
    dof = size1 + size2 -2
    
    ssq1 = np.sum((group1 - mean1)**2)
    ssq2 = np.sum((group2 - mean2)**2)
    
    pooled_var = (ssq1 + ssq2)/(dof1 + dof2)

    se = np.sqrt(std1**2/size1 + std2**2/size2)
    corrected_se = np.sqrt(pooled_var/size1 + pooled_var/size2)
    
    t_statistic = (mean1 - mean2)/se
    corr_t_statistic = (mean1 - mean2)/corrected_se
    
    # Define conlusions context
    conclusions = ["Reject the NULL as statistically significant!", "Fail to reject the NULL."]
    
    if alternative == 'less':
        test_kind = 'Left-Tailed'
        t_critical = t.ppf(alpha_level, dof)
        p_value = t.sf(abs(t_statistic), dof)
        t_critical_formatted = "{:.3f}".format(t_critical)
        
        if t_statistic <= t_critical:
            res = conclusions[0]
        else:
            res = conclusions[1]
            
    elif alternative == 'greater':
        test_kind = 'Right-Tailed'
        t_critical = t.ppf(1-alpha_level, dof)
        t_critical_formatted = "{:.3f}".format(t_critical)
        p_value = t.sf(abs(t_statistic), dof)
        if t_statistic >= t_critical:
            res = conclusions[0]
        else:
            res = conclusions[1]
            
    elif alternative == 'two_sided':
        test_kind = "Two-Tailed"
        p_value = t.sf(abs(t_statistic), dof)*2
        t_critical = abs(t.ppf(alpha_level/2, dof))
        t_critical_formatted = "(+/-){:.3f}".format(t_critical)            

        if abs(t_statistic) > abs(t_critical):
            res = conclusions[0]
            
        else:
            res = conclusions[1]
            
    # r_sqrd
    r2 = t_statistic**2/(t_statistic**2 + dof)
    
    # margin of error
    t_critical1 = t.ppf(alpha_level/2, dof)
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
          
          mean difference: {22: .2f}
          
          Pooled Variance: {18: .4f}
          Standard Error: {6:.4f}
          Standard Error (using pooled variance): {17: .4f}
      
        **Inferential Statistics Summary**

          Test Type: Between-Group {13} t-Test
          degree of freedom: {7}
          p-value: {8:.5f}
          t-statistic: {9:.3f}
          t-statistic (using pooled variance): {19: .3f}
          t-critical: {10}
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
                           dof, 
                           p_value, 
                           t_statistic, 
                           t_critical_formatted, 
                           alpha_level, 
                           res, 
                           test_kind, 
                           ci, 
                           moe, 
                           r2, 
                           corrected_se, 
                           pooled_var, 
                           corr_t_statistic, 
                           dof1, 
                           dof2, 
                           mean_diff
                           ))
    
    
    
def t_test_ind_from_stats(x_bar, y_bar, x_size, y_size, 
                          ss_x=None, ss_y=None, s2_p=None, 
                          pop_mean_diff=0., dof=None, alpha=.05, 
                          alternative='two_sided'):
    """execute independent sample t-test given descriptive statistics

    Args:
        x_bar (float): _description_
        y_bar (float): _description_
        x_size (int): _description_
        y_size (int): _description_
        ss_x (float, optional): _description_. Defaults to None.
        ss_y (float, optional): _description_. Defaults to None.
        s2_p (float, optional): _description_. Defaults to None.
        pop_mean_diff (float, optional): _description_. Defaults to 0.
        dof (_type_, optional): _description_. Defaults to None.
        alpha (float, optional): _description_. Defaults to .05.
        alternative (str, optional): _description_. Defaults to 'two_sided'.
    """
    
    mean_diff = x_bar - y_bar - pop_mean_diff
    x_dof = x_size - 1
    y_dof = y_size - 1
    
    if s2_p is None and ss_x is not None and ss_y is not None:
        s2_p = (ss_x + ss_y) / (x_dof + y_dof)
    
    if dof is None:
        degree_of_freedom = x_size + y_size -2
    else:
        degree_of_freedom = dof
    
    se = np.sqrt(s2_p*(1/x_size + 1/y_size))
    t_statistic = mean_diff / se
    conclusions = ["Reject the Null.", "Fail to Reject the Null."]
    # Find t critical value
    if alternative == 'two_sided':
        test_type = 'Two-Tailed'
        t_critical = t.ppf(1-alpha/2, degree_of_freedom)
        t_critical_formatted = "(+/-){: .3f}".format(t_critical)
        p_value = t.sf(abs(t_statistic), degree_of_freedom)*2
        
        if abs(t_statistic) > t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        
    elif alternative == 'less':
        test_type = 'Left-Tailed'
        t_critical = t.ppf(alpha, degree_of_freedom)
        t_critical_formatted = "{: .3f}".format(t_critical)
        p_value = t.sf(abs(t_statistic), degree_of_freedom)
        
        if t_statistic < t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        
    elif alternative == 'greater':
        test_type = 'Right-Tailed'
        # Better need a try-except logic here
            
        t_critical = t.ppf(1-alpha, degree_of_freedom)
        t_critical_formatted = "{: .3f}".format(t_critical)
        p_value = t.sf(t_statistic, degree_of_freedom)
        
        if t_statistic > t_critical:
            res = conclusions[0]
            
        else:
            res = conclusions[1]
        
    
    
    output = """
    =============== Reports ==============
    
        **Descriptive Statistics Summary**

          sample-1 size: {0}
          sample-1 mean: {1:.3f}
          sample-1 dof: {2}
          
          sample-2 size: {3}
          sample-2 mean: {4:.3f}
          sample-2 dof: {5}
          
          mean difference: {6: .2f}
          
          Pooled Variance: {7: .4f}
      
        **Inferential Statistics Summary**

          Test Type: Independent Samples {8} t-Test
          degree of freedom: {9}
          p-value: {10:.5f}
          t-statistic (using pooled variance): {11: .3f}
          t-critical: {12}
          alpha-level: {13}
      
          ---------------------------------
          
    Conclusion: {14}
    
    ================== END =================
    """
    
    print(output.format(x_size, 
                        x_bar, 
                        x_dof, 
                        y_size, 
                        y_bar, 
                        y_dof, 
                        mean_diff, 
                        s2_p, 
                        test_type, 
                        degree_of_freedom, 
                        p_value, 
                        t_statistic, 
                        t_critical_formatted, 
                        alpha, 
                        res))