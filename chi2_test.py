from scipy.stats import chisquare, chi2, chi2_contingency
import numpy as np
import pandas as pd

def chi2_goodness_of_fit(obs, exp, alpha=.05):
    """Execute chi-square goodness of fit test on obs. and exp. data"""
    # Calculate Chi-Square statistic and P-value
    chi_stat, p_val = chisquare([41,59], [33,67])
    
    # Calculate degree of freedom
    dof = len(obs) - 1
    
    # Calculate Chi-Square critical value given alpha-level
    chi_critical = chi2.ppf(1-alpha, dof)
    
    # Format result string
    print_out = """
    ============== Pearson's Chi-Square Test =============
        Chi Square is: {0:.4f}
        P value is {1:.4f}
        
        Chi2 Critical Value at alpha={2}: {3:.4f}
    ----------------------------------
        {4}
    ======================== END =========================
    """
    conc = "Reject the NULL" if chi_stat >= chi_critical else "Fail to reject the NULL"
    
    print(print_out.format(chi_stat, 
                       p_val, 
                       alpha, 
                       chi_critical, 
                       conc))