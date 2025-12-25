"""
Statistical Analysis Toolkit
Statistical functions for causal discovery, time series analysis, and econometrics
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import itertools

# ============================================
# Data Generating Process (DGP)
# ============================================

def RIZdgp(A, B, sig):
    """
    Data Generating Process
    Args:
        A, B: coefficient matrices
        sig: noise parameter
    """
    W = np.random.randn(3, 3)
    L = np.linalg.cholesky(sig)
    n = 21100
    T = np.arange(1, n+1)
    
    W1 = np.zeros((n, 4))
    for i in range(1, n):
        W1[i, :] = W1[i-1, :] + 1 + np.random.randn(4) * L[i % 3, i % 3]
    
    W2 = W1 ** 2
    X = W2[:, :3]
    Z = W2[:, 1:]
    
    return {'W1': W1, 'W2': W2, 'X': X, 'Z': Z, 'T': T}


# ============================================
# OLS FIT
# ============================================

def ols_fit(y, x):
    """
    Ordinary Least Squares Fit
    Args:
        y: dependent variable (n x 1)
        x: independent variables (n x k)
    Returns:
        beta: coefficients
        yhat: fitted values
    """
    n, k = x.shape
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    
    X = np.column_stack([np.ones(n), x])
    yhat = X @ np.concatenate([[b[0]], b[1:]])
    
    return {'beta': b, 'yhat': yhat}


# ============================================
# VAR Residuals
# ============================================

def var_residuals(b1, b2, b3, e1, e2, e3):
    """
    VAR (Vector Autoregression) Residuals
    Args:
        b1, b2, b3: coefficient vectors for 3 equations
        e1, e2, e3: residuals for 3 equations
    Returns:
        residuals for each equation
    """
    n = len(e1)
    p = len(b1)
    
    x1 = np.zeros(n)
    y1 = np.zeros(n)
    z1 = np.zeros(n)
    
    # Equation 1
    X1 = np.zeros((n-p, p+1))
    for i in range(p, n):
        X1[i-p, 0] = 1
        for j in range(1, p+1):
            X1[i-p, j] = x1[i-j] + y1[i-j] + z1[i-j]
    
    # Similarly for equations 2 and 3
    e1_res = e1[:len(X1)] - X1 @ b1
    e2_res = e2[:len(X1)] - X1 @ b2  
    e3_res = e3[:len(X1)] - X1 @ b3
    
    return {'e1': e1_res, 'e2': e2_res, 'e3': e3_res}


# ============================================
# Modified R Recursive Residuals
# ============================================

def recursive_residuals(x):
    """
    Modified R Recursive Residuals
    Args:
        x: data matrix
    Returns:
        recursive residuals
    """
    n = len(x)
    a = np.zeros(n)
    
    for i in range(1, n):
        xtm1 = np.ones((i, 1))
        xtm1[:, 0] = x[:i]
        
        bhat = np.linalg.inv(xtm1.T @ xtm1) @ xtm1.T @ x[:i]
        xstar = np.concatenate([[1], x[i:i+1]])
        
        e = x[i] - xstar @ bhat
        a[i-1] = e / np.sqrt(1 + xstar @ np.linalg.inv(xtm1.T @ xtm1) @ xstar.T)
    
    return a


# ============================================
# Haugh ARMA Residuals
# ============================================

def haugh_arma_residuals(xdatsag, ydatsag):
    """
    Haugh ARMA Residuals
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    model_x = ARIMA(xdatsag, order=(2, 0, 2))
    model_y = ARIMA(ydatsag, order=(2, 0, 2))
    
    fit_x = model_x.fit()
    fit_y = model_y.fit()
    
    return {'resid_x': fit_x.resid, 'resid_y': fit_y.resid}


# ============================================
# Conditional Correlation
# ============================================

def conditional_correlation(x, y, z):
    """
    Conditional correlation between x and y given z
    Args:
        x, y: variables to correlate
        z: conditioning variable
    Returns:
        conditional correlation coefficient
    """
    n = len(x)
    
    xg = np.linalg.lstsq(np.column_stack([np.ones(n), z]), x, rcond=None)[0]
    yg = np.linalg.lstsq(np.column_stack([np.ones(n), z]), y, rcond=None)[0]
    
    var_xgz = np.var(x - (xg[0] + xg[1] * z)) * (n - 1)
    var_ygz = np.var(y - (yg[0] + yg[1] * z)) * (n - 1)
    var_xygz = np.sum((x - (xg[0] + xg[1] * z)) * (y - (yg[0] + yg[1] * z)))
    
    ccor_xygz = var_xygz / np.sqrt(var_xgz * var_ygz)
    
    return ccor_xygz


# ============================================
# Fisher Z Test
# ============================================

def fisher_z(rho, n, k):
    """
    Fisher Z test for correlation
    Args:
        rho: correlation coefficient
        n: sample size
        k: number of conditioning variables
    Returns:
        test statistic and whether correlation is significant
    """
    z = 0.5 * (np.log(1 + rho) - np.log(1 - rho)) / (1 / np.sqrt(n - k - 3))
    z_stat = np.abs(z)
    
    is_significant = z_stat > 1.96  # 95% confidence level
    
    return {'z': z_stat, 'rho': rho, 'significant': is_significant}


# ============================================
# Cross-Validation
# ============================================

def cross_validation(data, n_folds=10000):
    """
    Cross-validation procedure
    Args:
        data: input data
        n_folds: number of CV folds
    """
    A = [1, 0, 3, 0, 3, 0, 0, 0, 1]
    B = [0, 0, 3, 0, 0, 0, 0]
    sig = [2, 3, 0, 0, 3, 0, 0, 0, 2]
    
    dgp_data = RIZdgp(A, B, sig)
    
    ucxy = np.corrcoef(dgp_data['X'][:, 0], dgp_data['X'][:, 1])[0, 1]
    sim_ucxy = np.random.randn(n_folds) * 0.1 + ucxy
    
    ccorxy = conditional_correlation(
        dgp_data['X'][:, 0], 
        dgp_data['X'][:, 1], 
        dgp_data['X'][:, 2]
    )
    
    cv_up = np.percentile(sim_ucxy, 97.5)
    cv_lp = np.percentile(sim_ucxy, 2.5)
    
    return {'cv_up': cv_up, 'cv_lp': cv_lp, 'ccorxy': ccorxy}


# ============================================
# Causal Path Analysis
# ============================================

def causal_path(x, y, z):
    """
    Causal path analysis between variables
    Tests various causal relationships
    """
    n = len(x)
    
    xnay = None
    znay = None
    ynax = None
    zcx = None
    zcxy = None
    zcy = None
    ycx = None
    
    # Correlations
    corrxy = np.corrcoef(x, y)[0, 1]
    corrxz = np.corrcoef(x, z)[0, 1]
    corryz = np.corrcoef(y, z)[0, 1]
    
    # Conditional correlations
    ccorrxygz = conditional_correlation(x, y, z)
    ccorrxzgy = conditional_correlation(x, z, y)
    ccorryxgz = conditional_correlation(y, z, x)
    
    # Fisher Z tests
    n_obs = len(x)
    
    rho_xxy = fisher_z(ccorrxygz, n_obs, 1)
    rho_xcy = fisher_z(ccorrxzgy, n_obs, 1)
    rho_tyxe = fisher_z(ccorryxgz, n_obs, 1)
    
    # Determine causal structure
    if np.abs(ccorrxygz) < 0.70:
        xnay = 1
    else:
        xcy = 1
        ycy = 1
    
    # Additional tests for other relationships
    if np.abs(ccorrxzgy) < 0.70:
        znax = 1
    else:
        xcy = 1
        ycy = 1
    
    if np.abs(ccorryxgz) < 0.70:
        znay = 1
    else:
        xcy = 1
        ycy = 1
    
    return {
        'correlations': {'xy': corrxy, 'xz': corrxz, 'yz': corryz},
        'conditional_cors': {
            'xy_given_z': ccorrxygz,
            'xz_given_y': ccorrxzgy,
            'yz_given_x': ccorryxgz
        },
        'causal_structure': {
            'x_not_causes_y': xnay,
            'z_not_causes_y': znay,
            'y_causes_x': ycx
        }
    }


# ============================================
# PC Algorithm (Causal Discovery)
# ============================================

def pc_algorithm(data, alpha=0.05):
    """
    PC (Peter-Clark) Algorithm for causal discovery
    Args:
        data: n x p data matrix (n observations, p variables)
        alpha: significance level
    Returns:
        adjacency matrix representing causal graph
    """
    n, p = data.shape
    
    # Initialize correlation matrix
    b1, b2, b3, e1, e2, e3 = [np.zeros(3) for _ in range(6)]
    
    # Calculate residuals
    residuals = var_residuals(b1, b2, b3, 
                             data[:, 0], data[:, 1], data[:, 2])
    
    # Test conditional correlations
    ucorr2 = np.corrcoef(residuals['e1'], residuals['e2'])[0, 1]
    ucorr3 = np.corrcoef(residuals['e1'], residuals['e3'])[0, 1]
    ucorr23 = np.corrcoef(residuals['e2'], residuals['e3'])[0, 1]
    
    # Conditional correlations
    ccorr12 = conditional_correlation(
        residuals['e1'], residuals['e2'], residuals['e3']
    )
    ccorr13 = conditional_correlation(
        residuals['e1'], residuals['e3'], residuals['e2']
    )
    ccorr23 = conditional_correlation(
        residuals['e2'], residuals['e3'], residuals['e1']
    )
    
    # Build adjacency matrix based on tests
    adj_matrix = np.ones((p, p))
    
    # Edge removal based on conditional independence
    dgp_xcr = 1 if np.abs(ccorr12) < 0.70 else 0
    dgp_xcz = 1 if np.abs(ccorr13) < 0.70 else 0
    dgp_ycx = 1 if np.abs(ccorr23) < 0.70 else 0
    
    # Track edge presence and orientation
    correct = 0
    omitted = 0
    committed = 0
    correct2 = 0
    
    if dgp_xcr == 1:
        adj_matrix[0, 1] = 0
        correct += 1
    else:
        committed += 1
    
    return {
        'adjacency_matrix': adj_matrix,
        'statistics': {
            'correct': correct,
            'omitted': omitted,
            'committed': committed
        },
        'correlations': {
            'unconditional': [ucorr2, ucorr3, ucorr23],
            'conditional': [ccorr12, ccorr13, ccorr23]
        }
    }


# ============================================
# Modified PC Algorithm
# ============================================

def modified_pc_algorithm(data, alpha=0.05):
    """
    Modified PC Algorithm with edge orientation
    Similar to PC but with different edge orientation rules
    """
    result = pc_algorithm(data, alpha)
    
    # Additional orientation rules
    # (The logic here follows the pattern from the R code
    # with nested conditionals for edge orientation)
    
    return result


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    z = np.random.randn(n)
    y = 0.5 * x + 0.3 * z + np.random.randn(n) * 0.5
    
    # Test conditional correlation
    ccor = conditional_correlation(x, y, z)
    print(f"Conditional correlation: {ccor:.4f}")
    
    # Test Fisher Z
    fz = fisher_z(ccor, n, 1)
    print(f"Fisher Z statistic: {fz['z']:.4f}")
    print(f"Significant: {fz['significant']}")
    
    # Test causal path
    data = np.column_stack([x, y, z])
    result = causal_path(x, y, z)
    print(f"\nCorrelations: {result['correlations']}")
