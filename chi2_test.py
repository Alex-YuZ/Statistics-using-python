from scipy.stats import chisquare, chi2, chi2_contingency
import numpy as np
import pandas as pd

def chi2_goodness_of_fit(obs, exp, alpha=.05):
    """Execute chi-square goodness of fit test on obs. and exp. data"""
    # Calculate Chi-Square statistic and P-value
    chi_stat, p_val = chisquare(obs, exp)
    
    # Calculate degree of freedom
    dof = len(obs) - 1
    
    # Calculate Chi-Square critical value given alpha-level
    chi_critical = chi2.ppf(1-alpha, dof)
    
    # Format result string
    print_out = """
    ============== Pearson's Chi-Square Test =============
        DoF: {5}
        Chi Square is: {0:.4f}
        P value is {1:.4f}
        
        Chi2 Critical Value at alpha={2}: {3:.4f}
    ----------------------------------
        CONCLUSION: {4}
    ======================== END =========================
    """
    conc = "Reject the NULL" if chi_stat >= chi_critical else "Fail to reject the NULL"
    
    print(print_out.format(chi_stat, 
                           p_val, 
                           alpha, 
                           chi_critical, 
                           conc, 
                           dof))
    
    
def chi2_independence(data, alpha=0.05):
    """Execute chi-square pearson's test on obs. and exp. data"""
    chi_sq_statistic, p, dof, expected = chi2_contingency(data)
    chi_sq_critical = chi2.ppf(1-alpha, dof)
    print_out="""
    ============== Pearson's Chi-Square Test =============
        DoF: {0}
        
        Chi-Square Statistic: {1:.4f}
        Chi-Square Critical Value (at \u03B1={3}): {5: .4f}
        
        P value: {2:.4f}
        \u03B1 level: {3}
        
    ----------------------------------
        Conclusion: {4}
    ======================== END =========================
    """

    expected_df = pd.DataFrame(expected)
    conc = "Reject the NULL" if p <= alpha else "Fail to reject the NULL"
    
    print(print_out.format(dof, 
                           chi_sq_statistic, 
                           p, 
                           alpha, 
                           conc, 
                           chi_sq_critical))
