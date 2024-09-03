"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts (responsibilities)
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    log_post = np.zeros((n, K))
    log_likelihood = 0.0

    for j in range(K):
        # Compute the log of the Gaussian probability for each observed entry
        log_pj = np.log(mixture.p[j] + 1e-16)  # log(π_j) with numerical stability
        log_N = np.zeros(n)
        
        for i in range(n):
            # Determine the observed coordinates for user i
            observed = X[i] != 0
            
            if np.sum(observed) > 0:  # If there are observed entries
                diff = X[i, observed] - mixture.mu[j, observed]
                log_N[i] = -0.5 * np.sum(diff**2) / mixture.var[j] - 0.5 * np.sum(observed) * np.log(2 * np.pi * mixture.var[j])
            else:
                log_N[i] = 0  # No observed data, log of the Gaussian is 0
            
            log_post[i, j] = log_pj + log_N[i]

    # Compute the log-likelihood using logsumexp for numerical stability
    log_likelihood = np.sum(logsumexp(log_post, axis=1))
    
    # Convert log posterior to actual posterior probabilities
    post = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))

    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Update the mixing proportions
    n_hat = np.sum(post, axis=0)  # Sum of responsibilities for each component
    p = n_hat / n  # Mixing proportions

    # Update the means
    mu = np.dot(post.T, X) / n_hat.reshape(-1, 1)

    # Update the variances
    var = np.zeros(K)
    for j in range(K):
        diff = X - mu[j]
        var[j] = np.dot(post[:, j], np.sum(diff**2, axis=1)) / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = -np.inf
    log_likelihood = 0
    
    while True:
        # E-step: Calculate responsibilities and log-likelihood
        post, log_likelihood = estep(X, mixture)
        
        # M-step: Update the parameters based on responsibilities
        mixture = mstep(X, post, mixture)  # Pass the mixture object here
        
        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) <= 1e-6 * np.abs(log_likelihood):
            break
        
        prev_log_likelihood = log_likelihood
    
    return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    X_pred = np.copy(X)
    
    for i in range(n):
        observed_indices = X[i] > 0
        missing_indices = X[i] == 0
        
        if np.any(missing_indices):  # If there are missing values to fill
            log_probs = np.zeros(K)
            for j in range(K):
                # Compute log likelihood only on the observed entries
                log_prob = -0.5 * np.sum(((X[i, observed_indices] - mixture.mu[j, observed_indices]) ** 2) / mixture.var[j])
                log_prob -= 0.5 * np.sum(observed_indices) * np.log(2 * np.pi * mixture.var[j])
                log_prob += np.log(mixture.p[j])
                log_probs[j] = log_prob
            
            # Normalize the log probabilities to get responsibilities
            logsumexp_val = logsumexp(log_probs)
            responsibilities = np.exp(log_probs - logsumexp_val)
            
            # Estimate missing values
            X_pred[i, missing_indices] = responsibilities @ mixture.mu[:, missing_indices]
    
    return X_pred
 
