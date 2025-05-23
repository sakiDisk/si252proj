In Part 2, authors introduce the combinatorial multi-armed bandit (**CMAB**) framework with probabilistically triggered arms (**CMAB-T**). This problem is formalized by a tuple $I := ([m], D, S, D_{\text{trig}}, R)$, where:

* **Base arms**: A set of $m$ base arms, each having its distribution from which random outcomes are drawn.
* **Combinatorial actions**: The learner selects a combinatorial action $S_t \in S$ at each round. This action can include a subset of base arms, referred to as super arms.
* **Probabilistic arm triggering feedback**: After an action is selected, each base arm may or may not be triggered probabilistically. If triggered, feedback for that arm is observed, and the reward for the chosen action is computed based on the outcomes of the triggered arms.
* **Reward function**: The reward is a non-negative value determined by the selected action, the outcomes of the triggered arms, and the probabilistic feedback mechanism.

Key reward conditions, including monotonicity and smoothness (the 1-norm triggering probability modulated (TPM) condition), are introduced to ensure that the reward function behaves predictably, even in the case of probabilistic arm triggering.

Additionally, the paper defines the **offline data collection** setting, where the learner has access only to pre-collected datasets of actions and corresponding feedback, instead of actively exploring actions. This offline setting is analyzed using performance metrics like the **suboptimality gap**, which quantifies the difference between the rewards of the optimal action and the action chosen by the learner.

Finally, the **data coverage conditions** (infinity-norm and 1-norm TPM) are discussed to evaluate the quality of the dataset. These conditions assess how well the data covers the possible outcomes and whether it allows the learner to accurately estimate the reward for optimal actions. These coverage conditions are critical for ensuring that the learner can find near-optimal solutions even without active exploration.