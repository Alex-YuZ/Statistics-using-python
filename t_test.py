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