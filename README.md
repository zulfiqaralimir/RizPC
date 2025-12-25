# RizPC
Statistical functions for causal discovery and time series analysis in R and Python

# RizPC - Statistical Analysis Toolkit

Statistical functions for causal discovery, time series analysis, and econometrics in R and Python.

## Features

- Data Generating Processes (DGP)
- OLS Regression
- VAR Residuals
- Recursive Residuals
- Conditional Correlation
- Fisher Z Test
- Causal Path Analysis
- PC Algorithm for Causal Discovery
- Cross-Validation

## Quick Start

### Python
```python
from statistical_analysis import conditional_correlation
import numpy as np

x = np.random.randn(1000)
y = np.random.randn(1000)
z = np.random.randn(1000)

ccor = conditional_correlation(x, y, z)
print(f"Conditional correlation: {ccor:.4f}")
```

### R
```r
source("statistical_analysis.R")

x <- rnorm(1000)
y <- rnorm(1000)
z <- rnorm(1000)

ccor <- ccorrelation(x, y, z)
print(ccor)
```

## Installation

### Python Requirements
```bash
pip install numpy scipy scikit-learn statsmodels
```

### R Requirements
```r
install.packages("forecast")  # Optional, for ARMA models
```

## License
MIT License
