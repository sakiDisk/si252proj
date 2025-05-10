# Project Title: OFFLINE LEARNING FOR COMBINATORIAL MULTI-ARMED BANDITS

**Team Members:**
* Meng Lingyao - Student ID: 2022533108

## Project Overview

This project introduces Off-CMAB, the first offline learning framework designed specifically for combinatorial multi-armed bandits (CMAB). Traditional CMAB algorithms are primarily tailored for online settings, requiring real-time interactions with the environment. However, in many applications, such as healthcare or autonomous driving, online interactions are costly, unsafe, or impractical. To address these limitations, Off-CMAB allows for decision-making based on pre-collected offline datasets, eliminating the need for live exploration. The core innovation of this framework is the combinatorial lower confidence bound (CLCB) algorithm, which incorporates pessimistic reward estimations combined with combinatorial solvers to efficiently handle the uncertainty inherent in offline data.

The primary objective of the project is to develop a robust algorithm that can efficiently learn from offline datasets containing combinatorial actions and their feedback. This includes tackling challenges such as nonlinear reward functions, handling large combinatorial action spaces, and dealing with out-of-distribution actions that are not part of the optimal or feasible actions within the dataset. The framework is applied to practical problems like learning to rank, large language model (LLM) caching, and social influence maximization, demonstrating its broad applicability and superior performance through extensive experiments with both synthetic and real-world datasets.


## Background and Motivation

Combinatorial multi-armed bandits (CMAB) represent a core area in sequential decision-making, focusing on choosing a set of actions (or a combination of base arms) to maximize rewards in a dynamic environment. This framework has been extensively explored over the past decade and has found applications in real-world domains such as recommendation systems, healthcare, and cyber-physical systems. Traditional CMAB approaches primarily rely on online learning, where actions are chosen and evaluated through active exploration in real-time environments.

However, online exploration in CMAB faces two significant challenges. First, it often incurs substantial costs, such as degraded user experiences in recommendation systems or ethical concerns in healthcare when making decisions that directly impact individuals' well-being. Second, online learning neglects the potential benefits of utilizing existing offline datasets, which often contain valuable historical information that could guide learning agents without the need for further costly or ethically questionable interactions.

## Core Idea: Combinatorial Lower Confidence Bound (CLCB)

The **Combinatorial Lower Confidence Bound (CLCB)** algorithm is part of the Off-CMAB framework, which is designed for **offline learning** in **combinatorial multi-armed bandit (CMAB)** problems. In the CLCB algorithm, the primary goal is to identify an optimal combinatorial action by using a dataset of pre-collected samples and performing **pessimistic reward estimations** combined with **combinatorial solvers**.

1. **Pessimism in Estimation**: The algorithm employs a pessimistic approach by calculating **lower confidence bounds (LCBs)** on the estimated reward of base arms. These bounds help prevent the selection of actions with high fluctuations in their rewards, especially when they are based on limited data.

2. **Base Arms and Combinatorial Actions**:

   * A **base arm** is a fundamental unit of the problem (e.g., an individual action in a multi-arm setting).
   * **Combinatorial actions** are combinations of base arms. For example, selecting multiple arms at once or performing a combination of tasks simultaneously. The problem becomes combinatorially complex because of the many possible combinations.

3. **Combining LCBs with a Combinatorial Solver**: Once the LCBs for each base arm are computed, the algorithm uses a combinatorial solver to select an action from a set of possible actions (combinatorial actions). This solver aims to minimize the suboptimality gap, which is the difference between the selected action's expected reward and the optimal action's expected reward.


## Project Objective 1: Implement the algorithm

### Part 1: Generate the model of this algorithm

The main algorithm proposed is the **CLCB** (Combinatorial Lower Confidence Bound) algorithm, which is designed to handle combinatorial multi-armed bandit (CMAB) problems in an offline setting. It proceeds as follows:

1. **Input**:

   * Dataset $D = \{(S_t, \tau_t, (X_{t,i})_{i \in \tau_t})\}_{t=1}^n$, where $S_t$ is the selected combinatorial action at time $t$, $\tau_t$ is the set of triggered arms, and $(X_{t,i})$ are the rewards observed for the triggered arms.
   * A computational oracle, `ORACLE`, that helps select the best action based on the mean estimates.

2. **Procedure**:

   * For each arm $i \in [m]$:

     * Calculate the number of times $N_i$ the arm was triggered in the dataset.
     * Calculate the empirical mean $\hat{\mu}_i$ for each arm.
     * Compute the lower confidence bound (LCB) for each arm $\overline{\mu}_i = \hat{\mu}_i - \sqrt{\frac{\log(4mn/\delta)}{2N_i}}$, where $\delta$ is the probability of failure.

   * Call the oracle with the lower confidence bounds for all arms to select the combinatorial action $\hat{S} = \text{ORACLE}(\overline{\mu}_1, \overline{\mu}_2, \dots, \overline{\mu}_m)$.

3. **Return**:

   * The selected action $\hat{S}$.

### Part 2: Theoretical Analysis

The theoretical performance of the CLCB algorithm is analyzed in terms of the suboptimality gap:

1. The **suboptimality gap** is the difference between the reward of the optimal action $S^*$ and the reward of the chosen action $\hat{S}$.

2. Theoretical results show that the CLCB algorithm achieves a near-optimal suboptimality gap with the following upper bounds:

   * **Infinity-norm TPM Data Coverage**: $\text{SubOpt}(\hat{S}; \alpha, I) \leq 2\alpha B_1 \sqrt{2C^\infty \log(2mn/\delta)} / n$
   * **1-norm TPM Data Coverage**: $\text{SubOpt}(\hat{S}; \alpha, I) \leq 2\alpha B_1 \sqrt{2K^* C^1_1 \log(2mn/\delta)} / n$

3. The gap is minimized as the number of samples $n$ increases and depends on factors like the smoothness of the reward function and the data coverage coefficient.

### Part 3: Performance Evaluation

After implementing the algorithm, the next step is to evaluate its performance. The paper provides extensive experiments on both synthetic and real-world datasets to validate the effectiveness of the CLCB algorithm:

1. **Synthetic Dataset**:

   * Performance metrics like suboptimality gap are used to compare CLCB with baseline algorithms (e.g., CUCB-Offline, EMP).
   * Results show that CLCB consistently outperforms these baselines in minimizing the suboptimality gap.

2. **Real-World Dataset**:

   * Experiments on real-world data (e.g., Yelp, SciQ) demonstrate that CLCB achieves significant improvements, including cost reductions in LLM cache scenarios.

## Project Objective 2: Innovations (If possible)

1. **Incorporating Dynamic Data**:

   One potential innovation could be to extend the algorithm to handle **non-stationary data**, where the distribution of rewards changes over time. In this case, the algorithm would need to adapt not just based on observed data, but also on the evolution of the environment, perhaps leveraging **online learning** or **dynamic modeling**.

2. **Incorporating Multi-objective Optimization**:

   In real-world applications, decisions often need to optimize for multiple objectives simultaneously. Extending CLCB to handle **multi-objective bandit problems** could open up new possibilities, such as optimizing for both cost-efficiency and user satisfaction in recommendation systems or LLM caching.

