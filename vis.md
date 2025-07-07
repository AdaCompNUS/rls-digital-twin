### The Complete MPC Cost Function

The objective of the MPC is to find the optimal control sequence $\mathbf{U}^*$ that minimizes the following cost function $J$:

$$
\begin{align*}
\min_{\mathbf{U}} J(\mathbf{x}_0, \mathbf{U}) = \underbrace{ \sum_{k=0}^{N-1} \left[
    \begin{aligned}
        & (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})^T \mathbf{Q}_{\text{state}} (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}) \\
        + & (\mathbf{u}_k - \mathbf{u}_k^{\text{ref}})^T \mathbf{R}_{\text{ff}} (\mathbf{u}_k - \mathbf{u}_k^{\text{ref}}) \\
        + & \mathbf{u}_k^T \mathbf{R}_{\text{control}} \mathbf{u}_k
    \end{aligned}
\right]}_{\text{Running Cost}}
+ \underbrace{ (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})^T \mathbf{Q}_{\text{terminal}} (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}}) }_{\text{Terminal Cost}} \\
+ \underbrace{ w_{\text{dyn}} \sum_{k=0}^{N-1} ||\mathbf{s}_k^{\text{dyn}}||^2 + w_{\text{cbf}} \sum_{j=1}^{P} \sum_{k=0}^{M-1} ||\mathbf{s}_{j,k}^{\text{cbf}}||^2 }_{\text{Slack Costs}}
\end{align*}
$$

---

### Breakdown of the Equation

Here is a detailed explanation of each component of the cost function.

#### 1. Running Cost
This is the cost accumulated at each step $k$ over the prediction horizon (from $k=0$ to $N-1$).

*   **State Tracking Cost:** This term penalizes deviation from the reference trajectory.
    $$
    (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})^T \mathbf{Q}_{\text{state}} (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})
    $$

*   **Feedforward Control Tracking Cost:** This term encourages the controller to adhere to the pre-calculated ideal velocities.
    $$
    (\mathbf{u}_k - \mathbf{u}_k^{\text{ref}})^T \mathbf{R}_{\text{ff}} (\mathbf{u}_k - \mathbf{u}_k^{\text{ref}})
    $$

*   **Control Regularization Cost:** This term penalizes large control inputs to ensure smooth actions.
    $$
    \mathbf{u}_k^T \mathbf{R}_{\text{control}} \mathbf{u}_k
    $$

#### 2. Terminal Cost
This is a single, large cost applied only at the final step $N$ of the horizon to ensure high final-pose accuracy.

$$
(\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})^T \mathbf{Q}_{\text{terminal}} (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})
$$

#### 3. Slack Costs
These terms penalize the violation of "soft" constraints, making the solver more robust.

*   **Dynamics Slack Cost:** Penalizes any deviation from the robot's motion model.
    $$
    w_{\text{dyn}} \sum_{k=0}^{N-1} ||\mathbf{s}_k^{\text{dyn}}||^2
    $$

*   **CBF Slack Cost:** This is the term you asked about. It penalizes any predicted violation of the safe distance from obstacles.
    $$
    w_{\text{cbf}} \sum_{j=1}^{P} \sum_{k=0}^{M-1} ||\mathbf{s}_{j,k}^{\text{cbf}}||^2
    $$
    -   $w_{\text{cbf}}$ is the scalar weight (`slack_obstacle_weight`).
    -   $\sum_{j=1}^{P}$ sums the penalties over all $P$ detected obstacles.
    -   $\sum_{k=0}^{M-1}$ sums the penalties over the first $M$ future time steps.
    -   $\mathbf{s}_{j,k}^{\text{cbf}}$ is the slack variable: it is zero if the robot is safe from obstacle $j$ at time $k$, and non-zero if the safety distance is violated.
    -   $||\cdot||^2$ is the squared norm, which heavily penalizes larger violations.

---

### Variable Definitions

-   $J$: The total cost to be minimized.
-   $N$: The prediction horizon length.
-   $M$: The horizon length for applying CBF constraints ($M \le N$).
-   $P$: The total number of detected obstacles.
-   $k$: The discrete time step index, $k \in [0, N]$.
-   $\mathbf{x}_k$: The predicted state vector at step $k$.
-   $\mathbf{u}_k$: The control input (velocity) vector at step $k$.
-   $\mathbf{x}_k^{\text{ref}}$: The reference state from the trajectory at step $k$.
-   $\mathbf{u}_k^{\text{ref}}$: The reference (feedforward) control input at step $k$.
-   $\mathbf{Q}_{\text{state}}, \mathbf{R}_{\text{control}}, \mathbf{R}_{\text{ff}}, \mathbf{Q}_{\text{terminal}}$: Diagonal weight matrices.
-   $\mathbf{s}_k^{\text{dyn}}, \mathbf{s}_{j,k}^{\text{cbf}}$: Slack variables for dynamics and CBF constraints.
-   $w_{\text{dyn}}, w_{\text{cbf}}$: Scalar weights for the slack variables.