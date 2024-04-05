> ### Overall TODOs:
> - [ ] Add references to relevant Papers (Curi, Rothfuss, Jaksch)
> - [ ] Convert headers and paragraphs to HTML
> - [ ] Convert LaTeX equations to MathJax syntax
> - [ ] 

# Introduction
Traditional reinforcement learning (RL) algorithms excel in learning policies tailored to specific tasks by maximizing cumulative rewards.
Most common approaches in control, such as model predictive control (MPC) or model-based reinforcement learning (MBRL), assume that the system dynamics $f^*(\cdot,\cdot)$ are known. Let's consider the standard optimal control (OC) problem and optimize for some specific cumulative reward $J(\cdot)$ or cost $C(\cdot)$ over a state trajectory horizon:

$$\pi^* = \argmin_{\pi \in \Pi} C(\pi, f^*) = \argmin_{\pi \in \Pi} \int_{0}^Tc(x, \pi(x)) \,dt$$
$$\text{s.t.} \quad \dot{x} = f^{*}(x, \pi(x))+w, \quad x(0) = x_0,$$

where $\pi^*\in\Pi$ denotes the optimal policy. Existing RL methods primarily rely on discrete-time dynamics $f^*_d(\cdot,\cdot)$. Commonly, this problem is discretized using approximate methods such as Euler-forward emulation::

$$\pi^* = \argmin_{\pi\in\Pi} C_d(\pi, f^*) = \argmin_{\pi\in\Pi} \sum_{t=0}^{T-1} \, c_d(x_t, \pi(x_t))$$
$$\text{s.t.}  \quad x_{t+1} = f^*_d(x_t, u_t) + w_t, \quad x(0)= x_0,\quad u_t = \pi(x_t),$$

where $f^*_d(\cdot,\cdot)$ and $C_d(\cdot)$ denotes the discretized dynamics and cost, respectively. Again, the goal is to find the optimal policy $\pi^*$. 

So, not only do many OC formulations assume that the system dynamics are known; many RL formulations assume that the system behaves in such a discrete manner and directly assign discrete rewards accordingly. 
However, in many real-world scenarios, the system is fundamentally continuous in time, and an accurate dynamics model $f^*$ is not available.

> TODO: Highlight previous work by Curi, Bergkamp etc.

In this blog post, we highlight two recent contributions to the 2023 NeurIPS conference from our group, which adress the two issues of leveraging the continuous-time aspect of real systems as well as dealing with unknown system dynamics.

### Digression: System Identification
Identifying the unknown system dynamics by taking measurements is not a new field. Famously, Åström and Eyckhoff published a survey on system identification back in 1970 [reference]. 
In the traditional setting, parameters are estimated by measuring the system, which however requires some prior knowledge about the underlying equations of motion. 
Nowadays, the system dynamics are often learned non-parametrically from data, and RL has proven to be a flexible paradigm to learn about said dynamics and solve a task at hand using the cost function.


# The Need for Continuous-time Modeling
The conventional RL approach of using discrete-time dynamics and discretized rewards simplifies the modeling process and speeds up computation time. However, it introduces inaccuracies and even instability, especially when dealing with systems that exhibit complex, nonlinear behaviours over time. Thus, using discretized dynamics is actually limiting to many real-world applications.

Given the advantages of continuous-time modeling, we propose an optimistic continuous-time model-based RL algorithm – OCORL. OCORL has the ability to handle the dual challenge of deciding how to explore the environment, but at the same time when to make observations of the unknown underlying system.

Moreover, in the paper (?), we theoretically analyze OCORL and show a general regret bound that holds for any choice of measurement selection strategy (MSS). We further show that for common choices of MSSs, such as equidistant sampling, the regret is sublinear when we model the dynamics with GPs. 
>To our knowledge, we are the first to give a no-regret algorithm for a rich class of nonlinear dynamical systems in the continuous-time RL setting.

### The Measurement Selection Strategy (MSS)
> TODO: Explain what a MSS is. Mention state-action pair $z=(x,u)$.

### Optimistic and Greedy Exploration Strategies
OCORL is a variant of the Hallucinated Upper-Confidence RL (H-UCRL) strategy introduced by [reference: Chowdhury and Gopalan (2017); Curi et al. (2020)]. Opposed to greedy exploration, optimistic exploration strategies like H-UCRL or OCORL actively seek out areas of high uncertainty within the model. 

H-UCRL introduces a form of exploration that specifically targets the gaps in the model's knowledge, instead of exploiting the current best-known actions as a greedy approach would. This enables better long-term performance by ensuring a more comprehensive understanding of the environment, particularly in complex or penalty-based scenarios where greedy methods might otherwise lead to suboptimal exploration and learning outcomes. For a detailed exploration of these concepts, you can read [Felix Berkenkamp's blog post on the topic](https://berkenkamp.me/blog/2020-12-06-mbrl-exploration/).

### OCORL: An Optimistic Exploration Algorithm
Recall that we are treating the case where the continuous-time dynamical system $f^*$ is unknown. We assume an episodic RL setting, meaning that at an episode $n$, we optimize for a given cost function $C(\pi_n,f_n)$. 
We not only want to optimize over the policy $\pi_n$ and then deploy it, but also want to find a plausible system $f_n$ which approximates the true system $f^*$.
In general, the true system dynamics for a continuous-time system are given by the following ODE:
$$\dot{x} = f^{*}(x, \pi(x))+w, \quad x(0) = x_0,$$

which we want to approximate with $\dot{x}_n\approx f_n(x,\pi_n(x))$. After every episode, we reset to $x_n(0)=x_0$. So far, this gives us the following OC problem to solve at each episode $n$:

$$\pi_n = \argmin_{\pi_n \in \Pi} C(\pi_n, f_n) = \argmin_{\pi_n \in \Pi} \int_{0}^Tc(x_n, \pi(x_n)) \,dt$$
$$\text{s.t.} \quad \dot{x}_n = f_n(x_n, \pi_n(x_n)), \quad x_n(0) = x_0,$$

Now, how do we approximate the unknown system $f^*$ with $f_n$? 
We want to do so as well as possible based on the observations we have made up to the episode $n$. 
By deploying the policy $\pi_n$ on the real system after each episode, we get a trajectory $\mathcal{D}_n$. 

$$\mathcal{D}_n=\{ \text{Here: Either a mathematical or visual description of }\mathcal{D}_n\}$$

> Note: add how dataset is made from $\mathcal{D}_{1:n}$. Also add a visual description (see Lenarts Powerpoint).

The previously gathered data up to the current episode, denoted $\mathcal{D}_{1:n}$, gives us a basis from which we want to estimate the system dynamics. From our dataset and due to epistemic (and aleatoric) uncertainty, we receive a belief over $f_n-$ a distribution over possible dynamics. More concretely, we build a set of all models that are compatible with the data collected up to the current episode. 
This set is defined as follows:

> Note: following eqn is simplified. Is this acceptable?

$$\mathcal{M_n}(\delta) = \{f \text{ s.t. } \forall z \in \mathcal{Z}: \quad\mid\mid\mu_{n}(z) - f(z)\mid\mid \le \beta_n(\delta) \sigma_{n}(z)\},$$

which is the set of all models $f$ within a confidence set spanned by $\mu_n$ and $\sigma_n$. In other words, for a given state-action pair $z=(x,u)$, our learned model predicts a mean estimate $\mu_n(z)$ and quantifies our epistemic uncertainty $\sigma_n(z)$ about the function $f^*$ with a certain confidence. The cofidence level is given by an appropriate choice of the constants $\delta$ and $\beta_n(\delta)$.

We thus say that with a probability of at least $1-\delta$, our true model lies within the intersection of all compatible model sets over all episodes up to $n$:

$$\Pr\bigg( f^* \in \bigcap_{n \ge 0}\mathcal{M_n}(\delta) \bigg) \geq 1-\delta.$$

>Note: Add a visual description of eqn above (Lenarts Powerpoint)

Now, for each episode $n$, we co-optimize over our goal and our model:

$$(\pi_n, f_n) = \arg \min_{\pi \in \Pi} \min_{f \in \mathcal{M}_{n-1} \cap \mathcal{F}} C(\pi, f).$$

$$\text{s.t.} \quad \dot{x}_n = f_n(x_n, \pi_n(x_n)), \quad x_n(0) = x_0,$$

> Note: There is a bug somewhere here that makes the rendering of the following EQNs unreadable on gitlab!

Here, $f_n$ is a dynamical system such that the cost by controlling $f_n$ with its optimal policy $\pi_n$ is the lowest among all the plausible systems after the previous iteration, given by the set $\mathcal{M}_{n-1}$. The additional constraint on $f$ given by the set $\mathcal{F}$ denotes the Lipschitz-continuity of $f^*$, which is a common assumption in control theory.
> (Maybe add a footnote: Assuming [Lipschitz-continuity](https://example.org) on our system dynamics means that we have no infinite slopes or discontinuities in our dynamics. Additionally, the derivative of our function has a continuous and bounded derivative. References: Khalil, 2015 and Curi, 2021).

With this method, we have reduced the problem to an optimal control problem, where we not only optimize over the policy, but also over plausible models of our system dynamics. This method is inherently optimistic, which drives exploration in a setting where the true dynamics $f^*$ is unknown.


### Theory
The co-optimization problem above is infinite-dimensional, in general nonlinear, and thus hard to solve $-$ or even completely intractable.
In our paper (Appendix B), we present details on how we solve it in practice using a reparametrization trick.

> Note: Maybe add some details about reparametrization trick.

For now, we assume that we can solve the OC problem. By doing so and deploying a policy $\pi_n$ at episode $n$ instead of the optimal policy $\pi^*$, we incur a regret,

$$r_n=C(\pi_n,f^*) - C(\pi^*,f^*).$$

We analyze OCORL using the notion of regret, which is a measure of the difference between the actual performance and the optimal performance under the best policy $\pi^*$ from the class $\Pi$. We evaluate the cumulative regret that sums the gaps between the performance of the policy $\pi_n$ and the optimal policy $\pi^*$ over all the episodes:

$$R_N = \sum_{n=1}^Nr_n.$$

If the cumulative regret $R_N$ is sublinear in $N$, then the average cost of the policy $C(\pi_n, f^*)$ converges to the optimal cost $C(\pi^*, f^*)$ [add footnote below]. 

> Footnote: When the cumulative regret $R_N$​ grows sublinearly with $N$, it means that as $N$ becomes large, the average regret per decision point $R_N/N$ tends to zero. This implies that the difference in performance between the policy being followed ($\pi_n$) and the optimal policy ($\pi$) diminishes over time.

In our paper, we show that when modeling the dynamics with Gaussian Processes (GP), the cumulative regret can be bounded and is sublinear in $N$ for common MSS choices. We do this by relating the regret bound to the following model complexity: 
$$\mathcal{I}_N(f^*, S) = \max_{\substack{\pi_1, \ldots, \pi_N \\ \pi_n \in \Pi}} \sum_{n=1}^N\int_0^T || \sigma_{n-1}(z_n(t))||^2 \,dt.$$

We expect that the regret of any model-based continuous-time RL algorithm depends both on the hardness of learning the underlying true dynamics model $f^*$ and the MSS. The model complexity captures both aspects.

Intuitively, for a given $N$, the more complicated the dynamics $f^*$, the larger the epistemic uncertainty and thereby the model complexity. In the continuous-time setting, we do not observe the state at every time step, but only at a finite number of times wherever the MSS proposes to measure the system. Accordingly, the MSS influences how we collect data and update our calibrated model. Therefore, the model complexity depends on the MSS.

In our paper, we prove that by running OCORL, we have with probability at least $1-\delta$ that the cumulative regret is bounded by the following:

$$R_N(S) \le 2 \beta_{N} L_c (1 + L_{\pi}) T^\frac{3}{2} e^{L_f (1 + L_{\pi}) T} \sqrt{N\mathcal{I}_N(f^*, S)}.$$

If the model complexity term $\mathcal{I}_N(f^*, S)$ and $\beta_{N}$ grow at a rate slower than $N$, the regret is sublinear and the average performance of OCORL converges to $C(\pi^*, f^*)$. We show that this is the case for different MSS by bounding the model complexity and modeling the dynamics with Gaussian processes (GP). 

To learn $f^*$ we fit a GP model with a zero mean and kernel $k$ to the collected data $\mathcal{D}_{1:n}$. We show sublinear regret for the proposed MSSs for the case when we model dynamics with GPs. Finally, the regret can be bounded as follows:

$$\textrm{TODO: Add regret bounds}$$

Here, $\gamma_n$ is the maximum information gain after observing $n$ points [cite Srinivas]. We define $\gamma_n$ in the appendix of this paper, where we also provide the rates for common kernels.
For example, for the RBF kernel, $\gamma_n = \mathcal{O}\left(\log(n)^{d_x + d_u + 1}\right)$, where $d_x$ and $d_u$ refers to the state and input dimensionality.

To conclude, we related the regret bound $R_N$ to the model complexity $\mathcal{I}_N$, and then bounded the model complexity as well as the measurement uncertainty of GPs. This shows that the cumulative regret is sublinear in $N$, which assures convergence of our optimized policy $\pi_n$ to the optimal policy $\pi^*$. This theoretical framework lays the foundation for the practical application of OCORL, ensuring that the algorithm not only converges to optimal policies but does so efficiently, making it suitable for a wide range of continuous-time systems.


### Applications
> Mention some things. Maybe there is a nice figure?


# Active Exploration in RL
> Here's some text. Mention exploration aspect by using an inherent reward function.

### OPAX: An Optimistic Active Exploration Algorithm
> Here's some text.


# Conclusion
> Here's some text.
