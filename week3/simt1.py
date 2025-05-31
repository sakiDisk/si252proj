import numpy as np
import matplotlib.pyplot as plt

# Set parameters
m = 10  # Number of arms
n_values = [10, 50, 100, 500, 1000]  # Different sample sizes to analyze
delta = 0.05  # Confidence level (failure probability)
B1 = 1  # Smoothness coefficient (Assumption in the theorem)
C_inf_star = 2  # Coverage coefficient (Assumption in the theorem)
alpha = 1  # Approximation ratio

# Simulate the reward of each arm (uniform random between 0 and 1)
def simulate_rewards(m):
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

# Analyze the suboptimality gap over different sample sizes
suboptimality_gaps = []
for n in n_values:
    rewards = simulate_rewards(m)
    gap, selected, optimal = c_lcb_algorithm(m, n, rewards, delta)
    suboptimality_gaps.append(gap)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(n_values, suboptimality_gaps, marker='o', label='Suboptimality Gap')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Suboptimality Gap')
plt.title('Suboptimality Gap Analysis for CLCB Algorithm')
plt.grid(True)
plt.legend()
plt.show()