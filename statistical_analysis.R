# Statistical Analysis Toolkit
# Statistical functions for causal discovery, time series analysis, and econometrics

# ============================================
# Data Generating Process (DGP)
# ============================================

RIZdgp <- function(A, B, sig) {
  # Data Generating Process
  # Args:
  #   A, B: coefficient matrices
  #   sig: noise parameter
  
  W <- matrix(rnorm(9), 3, 3)
  L <- chol(sig)
  n <- 21100
  
  W1 <- matrix(0, nrow=n, ncol=4)
  for (i in 2:n) {
    W1[i, ] <- W1[i-1, ] + 1 + rnorm(4) * L[i %% 3 + 1, i %% 3 + 1]
  }
  
  W2 <- W1^2
  X <- W2[, 1:3]
  Z <- W2[, 2:4]
  T <- 1:n
  
  return(list(W1=W1, W2=W2, X=X, Z=Z, T=T))
}


# ============================================
# OLS FIT
# ============================================

olsfit <- function(y, x) {
  # Ordinary Least Squares Fit
  # Args:
  #   y: dependent variable (n x 1)
  #   x: independent variables (n x k)
  # Returns:
  #   list with beta coefficients and fitted values
  
  n <- length(y)
  X <- cbind(1, x)
  betahat <- solve(t(X) %*% X) %*% t(X) %*% y
  yhat <- X %*% betahat
  
  return(list(beta=betahat, yhat=yhat))
}


# ============================================
# VAR Residuals
# ============================================

varsl13 <- function(x, y, z) {
  # VAR (Vector Autoregression) Residuals
  # Args:
  #   x, y, z: time series variables
  # Returns:
  #   list with coefficients and residuals for 3 equations
  
  n <- length(x)
  p <- 3  # lag order
  
  x1 <- rep(0, n)
  y1 <- rep(0, n)
  z1 <- rep(0, n)
  
  # Create lagged matrices
  X1 <- matrix(0, nrow=n-p, ncol=p+1)
  X2 <- matrix(0, nrow=n-p, ncol=p+1)
  X3 <- matrix(0, nrow=n-p, ncol=p+1)
  
  for (i in (p+1):n) {
    X1[i-p, 1] <- 1
    X2[i-p, 1] <- 1
    X3[i-p, 1] <- 1
    
    for (j in 1:p) {
      X1[i-p, j+1] <- y[i-j] + z[i-j]
      X2[i-p, j+1] <- x[i-j] + z[i-j]
      X3[i-p, j+1] <- x[i-j] + y[i-j]
    }
  }
  
  # Estimate coefficients
  b1 <- solve(t(X1) %*% X1) %*% t(X1) %*% y[(p+1):n]
  b2 <- solve(t(X2) %*% X2) %*% t(X2) %*% x[(p+1):n]
  b3 <- solve(t(X3) %*% X3) %*% t(X3) %*% z[(p+1):n]
  
  # Calculate residuals
  e1 <- y[(p+1):n] - X1 %*% b1
  e2 <- x[(p+1):n] - X2 %*% b2
  e3 <- z[(p+1):n] - X3 %*% b3
  
  return(list(b1=b1, b2=b2, b3=b3, e1=e1, e2=e2, e3=e3))
}


# ============================================
# Modified R Recursive Residuals
# ============================================

recursive_residuals <- function(x) {
  # Modified R Recursive Residuals
  # Args:
  #   x: data vector
  # Returns:
  #   vector of recursive residuals
  
  n <- length(x)
  a <- rep(0, n-1)
  
  for (i in 2:n) {
    xtm1 <- cbind(1, x[1:(i-1)])
    bhat <- solve(t(xtm1) %*% xtm1) %*% t(xtm1) %*% x[1:(i-1)]
    xstar <- c(1, x[i])
    
    e <- x[i] - sum(xstar * bhat)
    a[i-1] <- e / sqrt(1 + t(xstar) %*% solve(t(xtm1) %*% xtm1) %*% xstar)
  }
  
  return(a)
}


# ============================================
# Haugh ARMA Residuals
# ============================================

haugh_arma_residuals <- function(xdatsag, ydatsag) {
  # Haugh ARMA Residuals
  # Requires: forecast package
  # Args:
  #   xdatsag, ydatsag: time series data
  # Returns:
  #   list with residuals from ARIMA models
  
  if (!require("forecast")) {
    stop("Please install the 'forecast' package: install.packages('forecast')")
  }
  
  model_x <- arima(xdatsag, order=c(2, 0, 2))
  model_y <- arima(ydatsag, order=c(2, 0, 2))
  
  return(list(resid_x=residuals(model_x), resid_y=residuals(model_y)))
}


# ============================================
# Conditional Correlation
# ============================================

ccorrelation <- function(x, y, z) {
  # Conditional correlation between x and y given z
  # Args:
  #   x, y: variables to correlate
  #   z: conditioning variable
  # Returns:
  #   conditional correlation coefficient
  
  n <- length(x)
  
  # Regress x on z
  xg <- lm(x ~ z)$coefficients
  # Regress y on z
  yg <- lm(y ~ z)$coefficients
  
  # Calculate residuals
  x_resid <- x - (xg[1] + xg[2] * z)
  y_resid <- y - (yg[1] + yg[2] * z)
  
  # Variance and covariance
  var_xgz <- var(x_resid) * (n - 1)
  var_ygz <- var(y_resid) * (n - 1)
  var_xygz <- sum(x_resid * y_resid)
  
  # Conditional correlation
  ccor_xygz <- var_xygz / sqrt(var_xgz * var_ygz)
  
  return(ccor_xygz)
}


# ============================================
# Fisher Z Test
# ============================================

fishrZ <- function(rho, n, k) {
  # Fisher Z test for correlation
  # Args:
  #   rho: correlation coefficient
  #   n: sample size
  #   k: number of conditioning variables
  # Returns:
  #   list with z statistic and significance
  
  z <- 0.5 * (log(1 + rho) - log(1 - rho)) / (1 / sqrt(n - k - 3))
  z_stat <- abs(z)
  
  is_significant <- z_stat > 1.96  # 95% confidence level
  
  return(list(z=z_stat, rho=rho, significant=is_significant))
}


# ============================================
# Cross-Validation
# ============================================

cross_validation <- function(n_folds=10000) {
  # Cross-validation procedure
  # Args:
  #   n_folds: number of CV folds
  # Returns:
  #   list with CV results
  
  A <- matrix(c(1, 0, 3, 0, 3, 0, 0, 0, 1), 3, 3)
  B <- c(0, 0, 3, 0, 0, 0, 0)
  sig <- matrix(c(2, 3, 0, 0, 3, 0, 0, 0, 2), 3, 3)
  
  dgp_data <- RIZdgp(A, B, sig)
  
  ucxy <- cor(dgp_data$X[, 1], dgp_data$X[, 2])
  sim_ucxy <- rnorm(n_folds) * 0.1 + ucxy
  
  ccorxy <- ccorrelation(dgp_data$X[, 1], dgp_data$X[, 2], dgp_data$X[, 3])
  
  cv_up <- quantile(sim_ucxy, 0.975)
  cv_lp <- quantile(sim_ucxy, 0.025)
  
  return(list(cv_up=cv_up, cv_lp=cv_lp, ccorxy=ccorxy))
}


# ============================================
# Causal Path Analysis
# ============================================

causalpath <- function(x, y, z) {
  # Causal path analysis between variables
  # Tests various causal relationships
  # Args:
  #   x, y, z: variables
  # Returns:
  #   list with causal structure information
  
  n <- length(x)
  
  xnay <- NULL
  znay <- NULL
  ynax <- NULL
  zcx <- NULL
  zcxy <- NULL
  zcy <- NULL
  ycx <- NULL
  
  # Correlations
  corrxy <- cor(x, y)
  corrxz <- cor(x, z)
  corryz <- cor(y, z)
  
  # Conditional correlations
  ccorrxygz <- ccorrelation(x, y, z)
  ccorrxzgy <- ccorrelation(x, z, y)
  ccorryzgx <- ccorrelation(y, z, x)
  
  # Fisher Z tests
  rho_xxy <- fishrZ(ccorrxygz, n, 1)
  rho_xcy <- fishrZ(ccorrxzgy, n, 1)
  rho_tyxe <- fishrZ(ccorryzgx, n, 1)
  
  # Determine causal structure
  if (abs(ccorrxygz) < 0.70) {
    xnay <- 1
  } else {
    xcy <- 1
    ycy <- 1
  }
  
  # Additional tests
  if (abs(ccorrxzgy) < 0.70) {
    znax <- 1
  }
  
  if (abs(ccorryzgx) < 0.70) {
    znay <- 1
  }
  
  return(list(
    correlations = list(xy=corrxy, xz=corrxz, yz=corryz),
    conditional_cors = list(
      xy_given_z=ccorrxygz,
      xz_given_y=ccorrxzgy,
      yz_given_x=ccorryzgx
    ),
    causal_structure = list(
      x_not_causes_y=xnay,
      z_not_causes_y=znay,
      y_causes_x=ycx
    )
  ))
}


# ============================================
# PC Algorithm (Causal Discovery)
# ============================================

pc_algorithm <- function(data, alpha=0.05) {
  # PC (Peter-Clark) Algorithm for causal discovery
  # Args:
  #   data: n x p data matrix
  #   alpha: significance level
  # Returns:
  #   list with adjacency matrix and statistics
  
  n <- nrow(data)
  p <- ncol(data)
  
  # Initialize
  b1 <- rep(0, 3)
  b2 <- rep(0, 3)
  b3 <- rep(0, 3)
  
  # Calculate residuals
  residuals <- varsl13(data[, 1], data[, 2], data[, 3])
  
  # Test conditional correlations
  ucorr2 <- cor(residuals$e1, residuals$e2)
  ucorr3 <- cor(residuals$e1, residuals$e3)
  ucorr23 <- cor(residuals$e2, residuals$e3)
  
  # Conditional correlations
  ccorr12 <- ccorrelation(residuals$e1, residuals$e2, residuals$e3)
  ccorr13 <- ccorrelation(residuals$e1, residuals$e3, residuals$e2)
  ccorr23 <- ccorrelation(residuals$e2, residuals$e3, residuals$e1)
  
  # Build adjacency matrix
  adj_matrix <- matrix(1, p, p)
  
  # Edge removal based on conditional independence
  dgp_xcr <- ifelse(abs(ccorr12) < 0.70, 1, 0)
  dgp_xcz <- ifelse(abs(ccorr13) < 0.70, 1, 0)
  dgp_ycx <- ifelse(abs(ccorr23) < 0.70, 1, 0)
  
  # Track statistics
  correct <- 0
  omitted <- 0
  committed <- 0
  
  if (dgp_xcr == 1) {
    adj_matrix[1, 2] <- 0
    correct <- correct + 1
  } else {
    committed <- committed + 1
  }
  
  return(list(
    adjacency_matrix = adj_matrix,
    statistics = list(
      correct = correct,
      omitted = omitted,
      committed = committed
    ),
    correlations = list(
      unconditional = c(ucorr2, ucorr3, ucorr23),
      conditional = c(ccorr12, ccorr13, ccorr23)
    )
  ))
}


# ============================================
# Example Usage
# ============================================

if (interactive()) {
  # Generate sample data
  set.seed(42)
  n <- 1000
  x <- rnorm(n)
  z <- rnorm(n)
  y <- 0.5 * x + 0.3 * z + rnorm(n) * 0.5
  
  # Test conditional correlation
  ccor <- ccorrelation(x, y, z)
  cat("Conditional correlation:", ccor, "\n")
  
  # Test Fisher Z
  fz <- fishrZ(ccor, n, 1)
  cat("Fisher Z statistic:", fz$z, "\n")
  cat("Significant:", fz$significant, "\n")
  
  # Test causal path
  result <- causalpath(x, y, z)
  cat("\nCorrelations:\n")
  print(result$correlations)
}
