import textwrap
import pandas as pd
import numpy as np
from scipy.stats import f
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import qsturng



def cal_f(*args, alpha_level=0.05):
  """execute one-way ANOVA
     execute post-hoc Tukey's HSD test if AVONA result 
     statistically significant

  Args: a series of sample arrays
      alpha_level (float, optional): confidence level. Defaults to 0.05.
  """
    # Store mean of each sample
    group_means = []
    
    # Concatenate all sample as a 1D numpy array
    concate_np = []
    
    # Store the size of each sample
    sample_sizes = []
    
    # Sum of squared deviation
    ss_within = 0
    
    # number of samples collected
    num_groups = 0
    
    # Iterate each sample and calculate its descriptive statisstics
    for sample in args:
        num_groups += 1
        sample_sizes = np.append(sample_sizes, len(sample))
        single_mean = np.mean(sample)
        group_means = np.append(group_means, single_mean)
        
        concate_np = np.concatenate([concate_np, sample])
        
        ss_within += np.sum((np.subtract(sample, single_mean))**2)
    
    # Calc degree of freedom for between-subject
    dof_between = num_groups - 1
    
    # Calc degree of freedom for within-subject
    dof_within = np.size(concate_np) - num_groups
    
    # Calc grand mean of combined samples
    grand_mean = np.mean(concate_np)
    
    # Calc sum of squares for between-subject and the total SS
    ss_between = np.sum(sample_sizes*(group_means - grand_mean)**2)
    ss_total = ss_between + ss_within
    
    # Mean of SS for between and within subject respectively
    ms_between = ss_between/dof_between
    ms_within = ss_within/dof_within
    
    # Calc f statistic
    f_statistic = ms_between/ms_within
    
    # Look up f critical on given alpha level and dof
    f_critical = f.ppf(1-alpha_level, dof_between, dof_within)
    
    # Look up probability on calculated f statistic and dof
    p_value = f.sf(f_statistic, dof_between, dof_within)
    
    # Calculate effect size (eta2)
    explained_var = ss_between/ss_total
    
    # Rounded group means with precision=2
    group_means_ = np.around(group_means, 2)
    
    # Define hypothesis test result
    conclusions = ['Reject the Null as statistically significant.', 'Fail to reject the Null.']
    
    # Judge the result
    res = conclusions[0] if f_statistic >= f_critical else conclusions[1]
    
    # Report contents
    print_out = """
    ================= ONE-WAY ANOVA TEST REPORT =================
    
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
      
    ==================== ONE-WAY ANOAVA END =======================   
    
    """
    # Get rid of indentation of in report
    formatted_print = textwrap.dedent(print_out)
    
    # Print the result
    print(formatted_print.format(sample_sizes, 
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
    
    # -------- Multiple Comparison Test Part ----------- #
    
    if res == conclusions[1]:
        print("No Need for Tukey's HSD as non-statistically significant from Aone-way ANOVA.")
    else:
        # Check each sample size is the same or not
        flag = True
        first = sample_sizes[0]
        for i in sample_sizes:
            if first != i:
                flag = False
                break
            else:
                continue
        
        # Construct labels for multiple comparison results    
        if flag == True:
            labels_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            labels_collections = []
            for i in range(len(sample_sizes)):
                label = labels_choices.pop(0)
                labels_collections.append(label)
            group_labels = np.repeat(labels_collections, repeats=sample_sizes[0])
            
            # Use statsmodels.stats.libqsturng.qsturng() to calculate q value
            # studentized range statistic table:
            # https://www2.stat.duke.edu/courses/Spring98/sta110c/qtable.html
            q_cirtical = qsturng(1-alpha_level, len(group_means), dof_within)
            
            # Calculate Tukey's hsd value
            hsd_value = q_cirtical*np.sqrt(ms_within/sample_sizes[0])
            
            # Generate tukey's hsd result table
            tukey_hsd = pairwise_tukeyhsd(concate_np, group_labels)
            
            # Report format
            hsd_res = """
            =============== Multiple Comparison Test REPORT ===============
            
            Critical Q Value: {0:.2f}
            Critical Tukey's HSD: {1:.2f}
            
            {2}
            
            ================= Multiple Comparison Test END ================
            
            """
            
            # Get rid of indentation in the report
            formatted_res = textwrap.dedent(hsd_res)
            print(formatted_res.format(q_cirtical, hsd_value, tukey_hsd))
            
        else:
            print("This version does not support Tukey's HSD analysis if sample sizes are different.")