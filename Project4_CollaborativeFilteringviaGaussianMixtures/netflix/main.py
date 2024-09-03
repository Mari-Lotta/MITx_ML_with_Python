# Import necessary modules and functions
import numpy as np
import matplotlib.pyplot as plt
import common 
import kmeans
import em  # Ensure you import the run function from naive_em


# Load the dataset
X = np.loadtxt("toy_data.txt")

# Define number of clusters and seeds
Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]


### lowest cost for each K
"""
best_costs = []

for K in Ks:
    lowest_cost = None
    best_mixture = None
    best_post = None

    for seed in seeds:
        # Initialize the mixture model and post
        mixture, post = common.init(X, K, seed)  # Properly unpack the tuple

        # Run the K-means algorithm
        mixture, post, cost = kmeans.run(X, mixture, post)

        # Check if the current run has the lowest cost
        if lowest_cost is None or cost < lowest_cost:
            lowest_cost = cost
            best_mixture = mixture
            best_post = post

    best_costs.append(lowest_cost)

    # Plot the best solution for this K
    plot(X, best_mixture, best_post, f"K={K}_best_solution.png")
    plt.show()

# Report the lowest cost for each K
for i, K in enumerate(Ks):
    print(f"Cost|K={K} = {best_costs[i]}")
"""
### maximum log-likelihood for each K
"""
best_log_likelihoods = []

for K in Ks:
    best_log_likelihood = None
    best_mixture = None
    best_post = None

    for seed in seeds:
        # Initialize the mixture model
        mixture, post = common.init(X, K, seed)

        # Run the EM algorithm
        mixture, post, log_likelihood = em.run(X, mixture, post)

        # Check if the current run has the highest log-likelihood
        if best_log_likelihood is None or log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_mixture = mixture
            best_post = post

    best_log_likelihoods.append(best_log_likelihood)

    # Plot the best solution for this K using EM
    plot(X, best_mixture, best_post, f"EM_K={K}_best_solution.png")
    plt.show()

# Report the maximum log-likelihood for each K
for i, K in enumerate(Ks):
    print(f"Log-likelihood|K={K} = {best_log_likelihoods[i]}")

"""
### Compare BIC Scores
best_bic = None
best_K = None

for K in Ks:
    max_log_likelihood = None
    best_mixture = None
    best_post = None
    
    for seed in range(5):  # Using seeds 0 to 4
        # Initialize mixture
        mixture, post = common.init(X, K, seed)
        
        # Run EM algorithm
        mixture, post, log_likelihood = em.run(X, mixture, post)
        
        # Track the maximum log-likelihood for this K
        if max_log_likelihood is None or log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            best_mixture = mixture
            best_post = post
    
    # Compute BIC for the best mixture model found for this K
    bic_value = common.bic(X, best_mixture, max_log_likelihood)
    
    # Track the best BIC score and corresponding K
    if best_bic is None or bic_value > best_bic:
        best_bic = bic_value
        best_K = K

# Report the best K and corresponding BIC
print(f"Best K = {best_K}")
print(f"Best BIC = {best_bic}")