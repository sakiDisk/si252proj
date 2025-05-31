import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the CMAB-T environment with scaling for more complex situations
def generate_feedback(arm_idx, true_rewards):
    """
    Simulate Bernoulli feedback for the selected arm with the true reward probabilities.
    The feedback is based on whether the arm is triggered, and the reward is drawn from a Bernoulli distribution.
    """
    return np.random.binomial(1, true_rewards[arm_idx])

def clcb_algorithm(n, delta, num_arms, true_rewards, trigger_probabilities):
    """
    Run the CLCB algorithm on a CMAB-T environment with larger and more complex datasets.
    """
    # Initialize arm statistics
    arm_counts = np.zeros(num_arms)
    arm_rewards = np.zeros(num_arms)
    arm_lcb = np.zeros(num_arms)

    # Track the process for visualization
    arm_lcb_history = []

    for t in range(n):
        # Select an arm based on the lowest LCB (Lower Confidence Bound)
        arm_idx = np.argmin(arm_lcb)
        # Generate feedback for the selected arm
        reward = generate_feedback(arm_idx, true_rewards)
        # Update the statistics for the selected arm
        arm_counts[arm_idx] += 1
        arm_rewards[arm_idx] += reward
        # Update the LCB for the selected arm based on the empirical mean and counts
        arm_lcb[arm_idx] = (arm_rewards[arm_idx] / arm_counts[arm_idx]) - np.sqrt(
            (2 * np.log(4 * num_arms * n / delta)) / arm_counts[arm_idx])
        # Store the history of LCBs for visualization
        arm_lcb_history.append(arm_lcb.copy())
    return arm_counts, arm_rewards, arm_lcb, arm_lcb_history

def run_simulation(num_arms, n_samples, delta, true_rewards, trigger_probabilities):
    """
    Run the simulation with the given parameters and evaluate the results.
    """
    # Run the CLCB algorithm and get the results
    arm_counts, arm_rewards, arm_lcb, arm_lcb_history = clcb_algorithm(n_samples, delta, num_arms, true_rewards,
                                                                       trigger_probabilities)
    # Calculate the optimal arm based on the highest true reward
    optimal_arm = np.argmax(true_rewards)
    optimal_reward = true_rewards[optimal_arm]
    # Calculate the reward of the arm selected by the algorithm
    selected_arm = np.argmin(arm_lcb)
    selected_reward = arm_rewards[selected_arm] / arm_counts[selected_arm] if arm_counts[selected_arm] > 0 else 0
    # Compute the suboptimality gap
    suboptimality_gap = optimal_reward - selected_reward

    # Display the results
    print(f"Optimal arm index: {optimal_arm}, Reward: {optimal_reward}")
    print(f"Selected arm index: {selected_arm}, Estimated Reward: {selected_reward}")
    print(f"Suboptimality gap: {suboptimality_gap}")

    # Visualization of the LCB and reward evolution
    plt.figure(figsize=(12, 6))
    # Plot the LCB history
    arm_lcb_history = np.array(arm_lcb_history)
    for arm in range(num_arms):
        plt.plot(arm_lcb_history[:, arm], label=f"Arm {arm} LCB", linestyle='--')
    # Highlight the true rewards
    plt.axhline(y=optimal_reward, color='r', linestyle='-', label="Optimal Reward")
    plt.axhline(y=selected_reward, color='g', linestyle='-', label="Selected Reward (Estimated)")
    # Labels and title
    plt.xlabel("Round")
    plt.ylabel("Lower Confidence Bound (LCB) / Reward")
    plt.title(f"CLCB Algorithm: LCB Evolution, Optimal and Selected Rewards for {num_arms} Arms")
    plt.legend()
    plt.show()

# Simulate with a larger number of arms and more complex reward distributions
num_arms = 10  # Increase the number of arms
n_samples = 1000  # Larger sample size to simulate more rounds
delta = 0.05  # 95% confidence level
# True rewards are now drawn from a normal distribution to simulate more realistic and complex scenarios
true_rewards = np.random.normal(0.5, 0.1, num_arms)  # Normally distributed rewards, mean 0.5 and std 0.1
# Trigger probabilities are still randomly generated
trigger_probabilities = np.random.rand(num_arms)
# Run the simulation
run_simulation(num_arms, n_samples, delta, true_rewards, trigger_probabilities)