import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the k-path problem
def simulate_k_path(m, k, C_star_inf, n):
    # Generate random arm means for two problem instances P1 and P2
    gap = np.linspace(0, C_star_inf / n, m // k)  # Varying the mean for the arms
    P1_means = np.concatenate([np.full(k, 1 / 2 - g) for g in gap])  # Instance P1 with decreasing means
    P2_means = np.concatenate([np.full(k, 1 / 2 + g) for g in gap])  # Instance P2 with increasing means
    # Simulate the reward difference between optimal (S*) and action (Ŝ)
    reward_P1 = np.random.binomial(1, P1_means)
    reward_P2 = np.random.binomial(1, P2_means)
    # Calculate suboptimality gap
    optimal_action = reward_P1.sum()
    selected_action = reward_P2.sum()
    suboptimality_gap = optimal_action - selected_action

    return suboptimality_gap

# Parameters
m = 100  # Number of arms
k = 10   # Arms per path
C_star_inf = 2  # Data coverage coefficient
n = 1000  # Number of samples

# Run simulation to calculate the suboptimality gap
suboptimality_gaps = []
for _ in range(1000):  # Simulate 1000 trials to average the results
    suboptimality_gap = simulate_k_path(m, k, C_star_inf, n)
    suboptimality_gaps.append(suboptimality_gap)

# Plot the results
plt.figure(figsize=(8, 6))
plt.hist(suboptimality_gaps, bins=50, color='skyblue', edgecolor='black')
plt.title(f"Distribution of Suboptimality Gap (n={n}, m={m}, k={k})", fontsize=14)
plt.xlabel("Suboptimality Gap", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()
# Analyze the mean and variance of the suboptimality gap
mean_gap = np.mean(suboptimality_gaps)
variance_gap = np.var(suboptimality_gaps)
mean_gap, variance_gap