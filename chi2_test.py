from scipy.stats import chisquare, chi2, chi2_contingency
import numpy as np
import pandas as pd
import textwrap

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
    
    
def chi2_independence(data, inds, cols, alpha=0.05):
    """Execute chi-square pearson's test on obs. and exp. data"""
    data = np.array(data)
    n_size = np.sum(data)
    k = min(data.shape)
    
    
    chi2_statistic, p, dof, expected_np = chi2_contingency(data)
    chi2_critical = chi2.ppf(1-alpha, dof)
    
    cramer_v = np.sqrt(chi2_statistic/(n_size*(k-1)))
    
    
    obs_df = pd.DataFrame(data, index=inds, columns=cols)
    exp_df = pd.DataFrame(expected_np, index=inds, columns=cols).round(2)
    
    concl = "Reject the NULL" if p < alpha else "Fail to reject the NULL"
    
    print_out="""
    ============== Pearson's Chi-Square Test =============
    *** Descriptive Summary ***
    
        Total Size: {0}
        R x C: {1} x {2}
        DoF: {3}
        
    *** Chi2 Statistics *** 
    
        Chi-Square Statistic: {4:.4f}
        Chi-Square Critical Value (at \u03B1={5}): {6: .4f}
        P value (when chi2={4:.2f}): {7:.4f}
    
    *** Effect Size ***
    
        Cramer's V (\u03D5): {8:.2f}
        
    *** Observation and Expectation Values (Contigency Table) ***
    
    Observation:
        {9}
    
    Expectation:
        {10}
    
    ----------------------------------
    
    Conclusion: {11}
    
    ======================== END =========================
    """

    fmt_print = textwrap.dedent(print_out)
    
    
    
    print(fmt_print.format(n_size, 
                           data.shape[0], 
                           data.shape[1], 
                           dof, 
                           chi2_statistic, 
                           alpha, 
                           chi2_critical, 
                           p, 
                           cramer_v, 
                           obs_df, 
                           exp_df, 
                           concl))
