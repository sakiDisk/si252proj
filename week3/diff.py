# Enhance the simulation by adding more complexity to the environment.
# Specifically, we'll test how the CLCB algorithm scales with:
# - a larger number of arms (m),
# - non-uniform reward distributions (e.g., normal distribution),
# - varying triggering probabilities for each arm.

import numpy as np
import matplotlib.pyplot as plt

# Set new parameters for the complex environment
m_values = [10, 20, 30, 40, 50]  # Different number of arms to test scaling
reward_distributions = ['uniform', 'normal']  # Two types of reward distributions
delta = 0.05  # Confidence level (failure probability)
B1 = 1  # Smoothness coefficient (Assumption in the theorem)
C_inf_star = 2  # Coverage coefficient (Assumption in the theorem)
alpha = 1  # Approximation ratio

# Simulate the reward of each arm (uniform or normal random)
def simulate_rewards(m, dist_type='uniform'):
    if dist_type == 'uniform':
        return np.random.uniform(0, 1, m)
    elif dist_type == 'normal':
        return np.random.normal(0.5, 0.15, m)  # Mean=0.5, std=0.15
    return np.random.uniform(0, 1, m)

# Calculate the suboptimality gap
def suboptimality_gap(selected_action, optimal_action, rewards):
    return rewards[optimal_action] - rewards[selected_action]

# CLCB Algorithm simulation
def c_lcb_algorithm(m, n, rewards, delta):
    # Simulate dataset
    dataset = np.random.choice(m, size=n, replace=True)  # Simulate arm selection
    # Count the number of selections for each arm
    counts = np.bincount(dataset, minlength=m)
    # Compute empirical means for each arm
    means = np.zeros(m)
    for i in range(m):
        means[i] = np.mean(np.random.choice(rewards, size=counts[i]))
    # Calculate the Lower Confidence Bound (LCB) for each arm
    lcb = means - np.sqrt((4 * np.log(4 * m * n / delta)) / (2 * counts))
    # Select the arm with the maximum LCB (maximizing reward under pessimism)
    selected_action = np.argmax(lcb)
    optimal_action = np.argmax(rewards)
    # Calculate the suboptimality gap
    gap = suboptimality_gap(selected_action, optimal_action, rewards)
    return gap, selected_action, optimal_action

# Simulate the scaling for different numbers of arms and reward distributions
suboptimality_gaps_scaling = {}
for m in m_values:
    suboptimality_gaps_scaling[m] = {}
    for dist in reward_distributions:
        gaps = []
        for n in [100, 500, 1000, 5000, 10000]:
            rewards = simulate_rewards(m, dist)
            gap, selected, optimal = c_lcb_algorithm(m, n, rewards, delta)
            gaps.append(gap)
        suboptimality_gaps_scaling[m][dist] = gaps

# Visualization of the simulation process
# Plotting the suboptimality gaps for different environments
plt.figure(figsize=(10, 6))

for m in m_values:
    for dist in reward_distributions:
        plt.plot([100, 500, 1000, 5000, 10000], suboptimality_gaps_scaling[m][dist], label=f"m={m}, dist={dist}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Suboptimality Gap')
plt.title('Suboptimality Gap Analysis for CLCB Algorithm with Different Dataset Sizes and Reward Distributions')
plt.legend()
plt.grid(True)
plt.show()