---
layout: post
title:  Reinforcement Learning Notes Sutton
date:   2023-06-16 15:09:00 -0000
description: Detailed notes and summaries from Sutton and Barto's Reinforcement Learning textbook.
tags: [ml]
math: true
draft: true
---

Notes for the book ["Reinforcement Learning" by Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

## Table of Contents

1.  [The Reinforcement Learning Problem](#org0aa0346)
    1.  [Terms](#org1c3f54e)
    2.  [Elements of Reinforcement Learning](#org72c4eb4)
    3.  [Limitations and Scope](#org5b1867e)
2.  [Multi-arm Bandits](#orgc4593d0)
    1.  [The n-Armed Bandit Problem](#org4d4d191)
    2.  [Action-Value Methods](#orgbf3cbce)
    3.  [Incremental Implementation](#orgccb0a8b)
    4.  [Tracking a Non-stationary Problem](#org170d560)
    5.  [Optimistic Initial Values](#org20c1418)
    6.  [Upper-Confidence-Bound Action Selection](#orgf610985)
    7.  [Gradient Bandits](#org5bfac35)
3.  [Finite Markov Decision Processes](#org6a6f048)
    1.  [Agent-Environment Interface](#org02432cc)
    2.  [Goals and Rewards](#orgd30daf6)
    3.  [Returns](#orgb70fa84)
    4.  [The Markov Property](#org3aa6f79)
    5.  [Markov Decision Processes](#org48c49f2)
    6.  [Value Functions](#org886d6af)
4.  [Dynamic Programming](#org9c82917)
    1.  [Policy Evaluation](#org17a598b)
    2.  [Policy Improvement](#orged48075)
    3.  [Policy Iteration](#org5c218d2)
    4.  [Asynchronous Dynamic Programming](#orgbcce2f3)
5.  [Monte Carlo Methods](#org6a8f574)
    1.  [Monte Carlo Prediction](#orgee69425)
    2.  [Monte Carlo Estimation of Action Values](#orgcab3bb9)
    3.  [Off-policy Prediction via Importance Sampling](#org3c2102b)
    4.  [Incremental Implementation](#org8b8d9c2)
6.  [Temporal Difference Learning](#org93fc6cb)
    1.  [TD Prediction](#org164e6dc)
    2.  [Sarsa: On-Policy TD Control](#org9bf1fe8)
    3.  [Q-Learning: Off-Policy TD Control](#org5d2fcd3)
7.  [Eligibility Traces](#orge24d76b)
    1.  [*n*-Step TD Prediction](#org21b1b70)
    2.  [TD(&lambda;)](#org6dc06c3)
8.  [Planning and Learning with Tabular Methods](#orgf8e0391)
    1.  [Models and Planning](#org2f453f9)
    2.  [Integrated Planning, Acting, and Learning](#org57600f2)
    3.  [When the Model is Wrong](#org7b8b423)
    4.  [Trajectory Sampling](#org760ae04)
    5.  [Heuristic Search](#orgc7deb08)
9.  [On-policy Approximation of Action Values](#org0d36e8a)
    1.  [Value Prediction with Function Approximation](#org61c880f)
10. [Policy Approximation](#org5f7a2ad)
    1.  [Actor-Critic Methods](#orgb4dea9b)
    2.  [Eligibility Traces for Actor-Critic Methods](#org83f18eb)



<a id="org0aa0346"></a>

## The Reinforcement Learning Problem


<a id="org1c3f54e"></a>

### Terms

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Term</th>
<th scope="col" class="org-left">Definition</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">\(s\)</td>
<td class="org-left">state</td>
</tr>


<tr>
<td class="org-left">\(a\)</td>
<td class="org-left">action</td>
</tr>


<tr>
<td class="org-left">\(\Delta\)</td>
<td class="org-left">set of all nonterminal states</td>
</tr>


<tr>
<td class="org-left">\(\Delta^+\)</td>
<td class="org-left">set of all states</td>
</tr>


<tr>
<td class="org-left">\(A(s)\)</td>
<td class="org-left">set of all possible actions in state \(s\)</td>
</tr>


<tr>
<td class="org-left">\(R\)</td>
<td class="org-left">set of possible rewards</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">\(t\)</td>
<td class="org-left">discrete time step</td>
</tr>


<tr>
<td class="org-left">\(T\)</td>
<td class="org-left">final time step</td>
</tr>


<tr>
<td class="org-left">\(S_t\)</td>
<td class="org-left">state at \(t\)</td>
</tr>


<tr>
<td class="org-left">\(A_t\)</td>
<td class="org-left">action at \(t\)</td>
</tr>


<tr>
<td class="org-left">\(R_t\)</td>
<td class="org-left">reward at \(t\), dependent on \(A_{t-1}\) and \(S_{t-1}\)</td>
</tr>


<tr>
<td class="org-left">\(G_t\)</td>
<td class="org-left">cumulative discounted reward following \(t\)</td>
</tr>


<tr>
<td class="org-left">\(G_t^{(n)}\)</td>
<td class="org-left"><i>n</i>-step return</td>
</tr>


<tr>
<td class="org-left">\(G_t^\lambda\)</td>
<td class="org-left">&lambda;-return</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">\(\pi\)</td>
<td class="org-left">policy</td>
</tr>


<tr>
<td class="org-left">\(\pi(s)\)</td>
<td class="org-left">action taken in state \(s\)</td>
</tr>


<tr>
<td class="org-left">\(\pi(a \mid s)\)</td>
<td class="org-left">probability of taking action \(a\) in state \(s\)</td>
</tr>


<tr>
<td class="org-left">\(p(s', r \mid s)\)</td>
<td class="org-left">probability of transitioning to \(s'\) with reward \(r\)</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">\(v_\pi(s)\)</td>
<td class="org-left">value of \(s\) under policy \(\pi\) (expected return)</td>
</tr>


<tr>
<td class="org-left">\(v_*(s)\)</td>
<td class="org-left">value of state \(s\) under optimal policy</td>
</tr>


<tr>
<td class="org-left">\(q_\pi(s, a)\)</td>
<td class="org-left">value of taking action \(a\) under policy \(\pi\)</td>
</tr>


<tr>
<td class="org-left">\(q_*(s, a)\)</td>
<td class="org-left">value of taking action \(a\) under optimal policy \(\pi\)</td>
</tr>


<tr>
<td class="org-left">\(V_t(s)\)</td>
<td class="org-left">estimate of \(v_pi(s)\) or \(v_*(s)\)</td>
</tr>


<tr>
<td class="org-left">\(Q_t(s, a)\)</td>
<td class="org-left">estimate of \(q_\pi(s, a)\) or \(q_*(s, a)\)</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">\(\gamma\)</td>
<td class="org-left">discount-rate parameter</td>
</tr>


<tr>
<td class="org-left">\(\epsilon\)</td>
<td class="org-left">probability of random action in &epsilon;-greedy</td>
</tr>


<tr>
<td class="org-left">\(\alpha\), \(\beta\)</td>
<td class="org-left">step-size parameters</td>
</tr>


<tr>
<td class="org-left">\(\lambda\)</td>
<td class="org-left">decay-rate parameter</td>
</tr>
</tbody>
</table>


<a id="org72c4eb4"></a>

### Elements of Reinforcement Learning


<a id="orgfd279d6"></a>

#### Policy

A *policy* defines the way the agent behaves. It is a mapping from perceived
states to actions to be taken. A policy can be anything from a DNN to a look up
table.


<a id="org377deee"></a>

#### Reward signal

A *reward signal* defines the goal of a reinforcement learning problem. It
should be given after every action taken in the environment. The goal of
reinforcement learning is to find a policy $$\pi$$ that maximizes the *reward
signal*. Reward is an immediate indication of what is good.


<a id="orgd04ec5c"></a>

#### Value function

A *value function* specifies what is good in the long run. The *value* of a
state is the total amount of reward an agent can get in the future from that
state.


<a id="org3ea02a2"></a>

#### Model

A model is something that mimics the behavior of the environment or allows
inference to be made about how the environment will behave. Models are used for
planning. Methods that use models and planning are called *model-based* methods
compared to *model-free* methods.


<a id="org5b1867e"></a>

### Limitations and Scope

Other methods such as genetic algorithms can be used when there is a lot of time
and the space of policies is small. Additionally, they can be used when the
learning agent cannot accurately sense the state of the environment.


<a id="orgc4593d0"></a>

## Multi-arm Bandits


<a id="org4d4d191"></a>

### The n-Armed Bandit Problem

You have $$n$$ different actions. After each action you receive a reward chosen
from a stationary probability distribution that depends on the action you chose.
Your objective is to maximize the expected total reward over a time period of
$$x$$ *time steps*. You do not know the probability distributions of any of the
actions.

This is similar to a slot machine with $$n$$ levers. The rewards are the payoffs
for hitting the jackpot. Since you do not know the probability distributions of
any of the arms, you need to balance being *greedy* (*exploiting* your
knowledge) and selecting non-greedy actions (*exploring*).


<a id="orgbf3cbce"></a>

### Action-Value Methods

The actual value of an action $$a$$ is $$q(a)$$ and the predicted value is $$Q_t(a)$$.
The true value of an action is the mean reward received when that action is
selected.

\begin{equation}
Q_t(a)=\frac{R_1+R_2+...+R_{N_{t(a)}}}{N_t(a)}
\end{equation}

At the start we define $$Q_t(a)=0$$ or some other default value until we have some
data. As we go through each action more and more, we can begin to get a more
accurate estimate of the $$Q$$ function. As $$N_t(a)\to\infty$$ we will get the true
$$q(a)$$ (law of large numbers).

The *greedy* action selection policy for this approach is simple:

\begin{equation}
A_t=\mathop{\text{argmax }}_{a}Q_t(a)
\end{equation}

This approach has the flaw that it is only exploiting current knowledge and is
not spending any time to explore. This will get stuck on local maximums, not
finding out if other policies would give higher rewards.

The better approach is to have a small chance $$\epsilon$$ to do a random action.
This guarantees that $$N_t(a)\to\infty$$ for all $$a$$ ensuring $$Q_t(a)$$ will converge
to $$q(a)$$.

This becomes even more necessary when the rewards are non-stationary - the reward
distributions can change over time.


<a id="orgccb0a8b"></a>

### Incremental Implementation

We can write equation 1 more efficiently.

\begin{equation}
Q_{k+1}=Q_k + \frac{1}{k}[R_k-Q_k]
\end{equation}

In code this would be

    q = q + a [ r - q ]

where `q` is $$Q_k$$, `a` is the step size $$\frac{1}{k}$$, and `r` is the rewards
$$R_k$$.


<a id="org170d560"></a>

### Tracking a Non-stationary Problem

When our problem is non-stationary, we don&rsquo;t want to decrease the amount we are
incrementing by as in equation 3 ($$\frac{1}{k}$$ will decrease to 0). So we
change it to $$\alpha\in(0,1]$$, the step-size parameter


<a id="org20c1418"></a>

### Optimistic Initial Values

The methods above have the initial action-value estimates as $$0$$. This causes
the training to be highly dependent on the first chosen action as that one will
be exploited until the rest are visited by epsilon exploration. While we are
guaranteed to converge as $$t\to\infty$$, we can do better by assigning all of the
initial action-value estimates as something high. This will cause the learner to
be &ldquo;disappointed&rdquo; in the ones it sees, guaranteeing it will explore each action
early on.


<a id="orgf610985"></a>

### Upper-Confidence-Bound Action Selection

&epsilon;-greedy action selection will force all of the non-greedy actions to be tried, but it does not have any preference for choosing which one. A better approach would be to try the actions which would choose actions based on uncertainty.

\begin{equation}
A_t=\mathop{\text{argmax }}\left[Q_t(a)+c\sqrt{\frac{\ln t}{N_t(a)}}\right]
\end{equation}

In this equation we will choose actions that maximize the Q value, however, as
time goes on, the $$c\sqrt{\frac{\ln t}{N_t(a)}}$$ will grow and overwrite the Q
value. This will cause those actions to be explored. Every time an action is
chosen, the denominator increases, decreasing the uncertainty of that action.
However, every time another action is chosen, the numerator increases
logarithmically causing the exploring term to increase.


<a id="org5bfac35"></a>

### Gradient Bandits

The previous examples of Q-learning all attempt to estimate reward from a given
action. However, for policy gradients, we learn a numerical *preference*
$$H_t(a)$$ for each action $$a$$. The larger the preference, the more action $$a$$
gets chosen. This preferences does not have any connection to reward. If we add
$$x$$ to every preference, nothing will change.

In this we use the notation $$\pi_t(a)$$ for the probability of taking action $$a$$
at time $$t$$. At the start, all actions have the same probability of being
chosen. The policy is determined according to a soft-max distribution:

\begin{equation}
\pi_t(a)=\frac{e^{H_t(a)}}{\sum_{b=1}^n e^{H_t(b)}}
\end{equation}

We can update the preferences with stochastic gradient ascent. After selecting
action $$A_t$$ and receiving the reward $$R_t$$ we update the preferences with
equations 6 and 7.

\begin{equation}
H_{t+1}(A_t)=H_t(A_t)+\alpha(R_t-\bar{R_t})(1-\pi_t(A_t))
\end{equation}

and

\begin{equation}
H_{t+1}(a)=H_t(a)-\alpha(R_t-\bar{R_t})(\pi_t(A_t)), \forall a \ne A_t
\end{equation}

where $$\alpha>0$$ is the step-size parameter and $$\bar{R_t}\in \mathbb{R}$$ is the
average of all the rewards up to and including time $$t$$. Essentially, what this
does is if the reward gained is higher than the baseline, then the probability
of taking $$A_t$$ is increased and if it is lower, the probability is decreased.
All the other actions move in an opposite direction.

This is a stochastic approximation to gradient ascent. The exact *gradient ascent* would be increasing everything proportional to its effect on performance with:

\begin{equation}
H_{t+1}(a)=H_t(a)+\alpha\frac{\delta \mathbb{E}[R_t]}{\delta H_t(a)}
\end{equation}

where the measure of performance is the expected reward:

\begin{equation\*}
\mathbb{E}[R_t]=\sum_{b} \pi_t(b)q(b)
\end{equation\*}


<a id="org6a6f048"></a>

## Finite Markov Decision Processes


<a id="org02432cc"></a>

### Agent-Environment Interface

The reinforcement learning problem is a framing of learning from interaction to
achieve a goal. The learner and decision-maker is called the *agent*. The thing
it interacts with is called the *environment*.

The agent and the environment interact in a sequence of discrete time steps
$$t = 1, 2, 3, \ldots$$. At each time step $$t$$ the agent receives a representation of
the environment&rsquo;s *state* $$S_t$$. One time step later it receives a numerical
reward $$R_{t+1}$$ and a new state $$S_{t+1}$$.


<a id="orgd30daf6"></a>

### Goals and Rewards

Sutton recommends that rewards only be given for what we actually want achieved.
For example, for robot escaping a maze, the reward at each time step will
decrease by $$-1$$ as we want it to escape as quickly as possible. For a chess
robot, it will get $$1$$ for winning, $$-1$$ for losing, and $$0$$ for drawing and any
non-terminal position. Sutton states that giving a reward for subgoals like
capturing pieces might cause it to learn to capture pieces at the cost of
losing.


<a id="orgb70fa84"></a>

### Returns

A formal definition of the agents goal is to maximize the cumulative reward it
receives in the long run. If the sequence of rewards received after step $$t$$ is
denoted as $$R_{t+1}, R_{t+2}, R_{t+3},...R_T$$, then the expected return (in
simplest form) $$G_t$$ is defined with the following equation:

\begin{equation}
G_t=R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
\end{equation}

If the environment continues forever, this would infinite causing a problem. Therefore, we will *discount* the later rewards:

\begin{equation}
G_t=R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{ k=0}^{\infty} \gamma^k R_{t+k+1}
\end{equation}


<a id="org3aa6f79"></a>

### The Markov Property

An environment has the Markov property if the environment&rsquo;s response at $$t+1$$
depends only on the state and action $$S_t$$ and $$A_t$$. If it depends on previous
actions and states, then it does not have the Markov property.

However, even when the state signal is non-Markov, we can still think of the
state in reinforcement learning as an approximation to a Markov state.


<a id="org48c49f2"></a>

### Markov Decision Processes

If a reinforcement learning task satisfies the Markov property, it is called a
*Markov decision process* or *MDP*. If the state and action spaces are finite,
then it is called a *finite MDP*. Given any state and action $$s$$ and $$a$$, the probability of each possible pair of next state and reward $$s'$$ and $$r$$ is:

\begin{equation}
p(s', r \mid s, a) = Pr\{S_{t+1}=s', R_{t+1}=r\mid S_t=s, A_t=a\}
\end{equation}

Given these dynamics, we can compute anything else about the environment such as
the expected rewards for state-action pairs:

\begin{equation}
r(s, a) = \mathbb{E}\left[R_{t+1}\mid S_t=s, A_t=a\right]=\sum_{r\in R} r \sum_{s'\in S} p(s', r \mid s, a)
\end{equation}


<a id="org886d6af"></a>

### Value Functions

The policy $$\pi$$ is a map from each state $$s$$ and action $$a$$ to the probability
$$\pi(a\mid s)$$ of taking $$a$$ when in $$s$$. The *value* of state $$s$$ under policy
$$\pi$$, $$v_\pi(s)$$ is the expected return when starting in $$s$$ and following
$$\pi$$ after. For an MDP we can define this formally with:

\begin{equation}
v\_\pi(s)=\mathbb{E}\_\pi[G_t \mid S_t=s]=\mathbb{E}\_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R\_{t+k+1}\big| S\_t=s\right]
\end{equation}

where $$\mathbb{E}_\pi [ \cdot ]$$ denotes the expected value of a random variable given that
the agent follows policy $$\pi$$.


<a id="org9c82917"></a>

## Dynamic Programming


<a id="org17a598b"></a>

### Policy Evaluation

As the value of any state $$s\in\mathcal{S}$$ under policy $$\pi$$ can be described
by equation 13,

\begin{equation}
\begin{split}
v\_\pi(s)&=\mathbb{E}\_\pi\left[\sum\_{k=0}^{\infty} \gamma^k R_{t+k+1}\big| S_t=s\right] \\\\\
&=\mathbb{E}\_\pi[R\_{t+1}+\gamma v\_\pi(S\_{t+1})\mid S\_t=s] \\\\\
&=\sum_a \pi(a\mid s) \sum\_{s', r} p(s', r \mid s, a)[r+\gamma v\_\pi(s')]
\end{split}
\end{equation}

where $$\pi(a\mid s)$$ is the probability of taking action $$a$$ in state $$s$$ under
$$\pi$$. In my understanding, what we are doing is finding the probability of
getting each next state (the $$\pi(a\mid s))$$ part, and then multiplying the
value of the state that action would give us by that probability. So if an
action has a $$0%$$ chance of occurring, whatever value the state that action
would bring doesn&rsquo;t matter as we would never get there. And if we have a 50-50
chance of picking 2 actions, our value would be the average of those 2 next
state&rsquo;s values.

We can get the value function $$v_\pi(s)$$ through an iterative computation. If we
have a sequence of value functions $$v_0, v_1, v_2, \ldots$$ mapping
$$\mathcal{S}\mapsto\mathbb{R}$$, we can use the following iterative equation to
converge to $$v_\pi$$:

\begin{equation}
v\_{k+1}(s) =\mathbb{E}\_\pi[R_{t+1}+\gamma v\_k(S\_{t+1})\mid S_t=s]
\end{equation}

To produce each next approximation $$v_{k+1}$$, it replaces the old value of $$s$$
with the new value and the expected immediate rewards and all one-step
transitions possible. This is called a *full backup* as it *backs up* the value
of every state once to produce a new value function.


<a id="orged48075"></a>

### Policy Improvement

If we have the value function for a current policy, we can use it to determine if we should change our policy. We can do this by choosing an action $$a\ne\pi(s)$$ and using it for the next step and then following the current policy for the rest. If this gives a greater expected value, then our new policy would be better. The value of this method would be:

\begin{equation}
\begin{split}
q\_\pi(s,a)&=\mathbb{E}\_\pi[R\_{t+1}+\gamma v_\pi(S_{t+1})\mid S\_t=s, A_t=a] \\\\\
&=\sum_{s',r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s') \right]
\end{split}
\end{equation}

To my understanding, what we are doing is finding the probability of every
possible next reward and state given our action and multiplying it by its value
under our current policy.

Given that this is only for changing one policy for a single state, we can change the policy at *all* states and *all* possible actions:

\begin{equation}
\begin{split}
\pi'(s)&=\mathop{\text{argmax }}\_{a} q\_\pi(s, a) \\\\\
&=\mathop{\text{argmax }}\_{a} \mathbb{E}[R_{t+1}+\gamma v\_\pi(S_{t+1})\mid S\_t=s, A\_t=a] \\\\\
&=\mathop{\text{argmax }}\_{a} \sum\_{s', r} p(s', r|s, a)\left[r + \gamma v_\pi(s') \right]
\end{split}
\end{equation}

This is finding all values of $$a$$ at which the value under our policy of that
action are at their maximum. This new policy is going to be as good as or better
than our current policy. If it is just as good, that means our current policy is
the optimal policy.


<a id="org5c218d2"></a>

### Policy Iteration

We can use the previous two sections to find the value function for our current
policy, then use that value function to improve our current policy, then find
the value function for that new policy and repeat until we converge on the
optimal policy.


<a id="orgbcce2f3"></a>

### Asynchronous Dynamic Programming

Our current algorithm makes progress in successive sweeps of the entire state
space. This locks it up in these sweeps before making any improvements.

Asynchronous algorithms allow the agent to update the policy at the same time as
it is using it. This allows it to back up parts of the state set that are most
relative to the agent.


<a id="org6a8f574"></a>

## Monte Carlo Methods

Previous sections require complete knowledge of the environment, using Monte
Carlo methods, we can avoid this problem and only require *experience*.


<a id="orgee69425"></a>

### Monte Carlo Prediction

If we want to find $$v_\pi(s)$$ given a set of episodes obtained by following
$$\pi$$ and passing through $$s$$. Each occurrence of state $$s$$ is called a *visit*
to $$s$$. $$s$$ might be visited many times, however the first time will be called
the *first visit* to $$s$$.

*first-visit MC method* estimates $$v_\pi(s)$$ as the average of the returns
following first visits to $$s$$. The *ever-visit MC method* averages returns
following all visits to $$s$$.

The first-visit MC method goes as follows:

    env = Enviornment()
    p = Policy()
    V = dict()
    returns = dict()
    
    while True:
        episode = []
        state = env.reset()
        done = False
        while not done:
            state, rewards, done = env.act(p(state))
            episode.push((state, rewards))
    
        occurences = {}
        for state, rewards in episode:
            G ← return following the first occurrence of s
            Append G to Returns(s)
            V (s) ← average(Returns(s))


<a id="orgcab3bb9"></a>

### Monte Carlo Estimation of Action Values

If a model is not available, we can estimate *action* values instead of *state*
values. We can estimate $$q_*$$ using Monte Carlo methods. Our goal is to estimate
$$q_\pi(s,a)$$ - the expected return when starting in state $$s$$, taking action
$$a$$, and then following policy $$\pi$$. We can use the first-visit MC method for
this as well

One major flaw of this method, is if we have a deterministic policy, in each
state only 1 action will ever be taken. This will cause us to never converge on
$$q_\pi$$. In order for this to work, we must also balance out exploration.


<a id="org3c2102b"></a>

### Off-policy Prediction via Importance Sampling

All previous methods use episodes generated with the current policy $$\pi$$.
However, what if we only have data collected by another policy $$\mu$$ such that
$$\mu\ne\pi$$ - can we estimate $$v_\pi$$ and $$q_\pi$$? In this, we call $$\pi$$ the
*target policy* because we are learning its value function and we call $$\mu$$ the
*behavior policy* because it is the policy controlling the agent and generating
behavior.

To do this, we require that every action taken under $$\pi$$ is also taken under
$$\mu$$, in other words $$\pi(a\mid s) > 0 \implies \mu(a\mid s) > 0$$. This is because we can not
learn the value of actions that never occur under $$\mu$$.

Importance sampling is a general technique for estimating expected values under
one distribution given samples from another. Given a starting state $$S_t$$, the
probability of the subsequent state-action sequence, $$A_t, S_{t+1},
A_{t+1},\ldots,S_T$$ occurring under any policy $$\pi$$ is:

\begin{equation\*}
\prod_{k=t}^{T-1} \pi(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)
\end{equation\*}

Since each event occurring is independent (by definition of our assumption of a
MDP), we can use the product rule of statistics and multiply the chance of each
event happening together. The chance of each action $$A_k$$ happening is given by
the policy $$\pi$$ with $$\pi(A_k\mid S_k)$$. The probability of $$S_k$$ happening is
given by the Markov chain with $$p(S_{k+1}\mid S_k, A_k)$$.

We define the importance-sampling ratio as:

\begin{equation}
\rho_t^T=\frac{\prod_{k=t}^{T-1} \pi(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)}{\prod_{k=t}^{T-1} \mu(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k\mid S_k)}{\mu({A_k \mid S_k})}
\end{equation}

This makes intuitive sense as well. If our target policy $$\pi$$ has a high
probability of doing an action and so does our behavior policy, then that action
has a high importance.

We can use this to then calculate the value function $$V(s)$$ with:

\begin{equation}
V(s)=\frac{\sum_{t\in \mathcal{J}(s)} \rho_t^{T(t)}G_t}{|\mathcal{J}(s)|}
\end{equation}

where $$\mathcal{J}(s)$$ is the set of all the times $$s$$ has been visited, $$G_t$$
is the return after $$t$$ through $$T(t)$$.

<a id="org8b8d9c2"></a>

### Incremental Implementation

We can make our algorithm incremental updating the value every time new data is
found. Given returns $$G_1, G_2, \ldots, G_{n-1}$$ all starting with the same
state and each with a random weight $$W_i$$ (where $$W_i=\rho_t^{T(t)}$$), we want
to estimate:

\begin{equation}
V_n=\frac{\sum_{k=1}{n-1} W_k G_k}{\sum_{k=1}^{n-1} W_k}
\end{equation}

To update $$V_n$$ as we get more returns $$G_n$$, we use the following equation:

\begin{equation}
V_{n+1}=V_n+\frac{W_n}{C_n}[G_n-V_n]
\end{equation}

and

\begin{equation\*}
C_{n+1}=C_n+W_{n+1}
\end{equation\*}


<a id="org93fc6cb"></a>

## Temporal Difference Learning


<a id="org164e6dc"></a>

### TD Prediction

Both TD and Monte Carlo use experience to to solve their prediction problem and
update their estimate of $$v$$ and of $$v_\pi$$. Monte Carlo must wait until the end
of the episode to know $$G_t$$, however, TD methods can update the value function
immediately. The simplest TD method known as *TD(0)* uses:

\begin{equation}
V(S_t)\leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
\end{equation}

My understanding of this is that we can use our previous knowledge to estimate
the return as $$G_t = \gamma V(S_{t+1}) - V(S_t)$$ because we know what the value
of our next state is and we just see how much that would have increased by
subtracting the value of the current state.

We know from section 3 that $$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s] =
\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1)} \mid S_t=s]$$ (Equation 13).


<a id="org9bf1fe8"></a>

### Sarsa: On-Policy TD Control

Sarsa uses the idea of TD learning to estimate $$q_\pi(s, a)$$. It uses the
following equation:

\begin{equation}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
\end{equation}

Sarsa gets its name from the elements it uses to update Q function, $$S_t, A_t,
R_{t+1}, S_{t+1}, A_{t+1}$$.


<a id="org5d2fcd3"></a>

### Q-Learning: Off-Policy TD Control

In its simplest form, *one-step Q-learning* is defined by:

\begin{equation}
Q(S\_t, A_t) \leftarrow Q(S\_t, A\_t) + \alpha \left[R\_{t+1} + \gamma \mathop{\text{ max }}\_{a} Q(S\_{t+1}, a) - Q(S_t,A_t) \right]
\end{equation}

The difference between this and Sarsa, is that we no longer need to know
$$A_{t+1}$$ and instead choose whatever would be the best next action. It directly
approximates $$q_*$$ independent of the policy being followed.


<a id="orge24d76b"></a>

## Eligibility Traces


<a id="org21b1b70"></a>

### *n*-Step TD Prediction

*n-Step TD Prediction* is an intermediate between Monte Carlo learning and TD
learning. Instead of backing up every step like in TD learning or at the end of
an episode like in Monte Carlo learning, we back up after *n* steps.


<a id="org6dc06c3"></a>

### TD(&lambda;)

One doesn&rsquo;t have to do a backup just after *n*-step return, but toward the
*average* of *n*-step returns. For example, one could do the backup of the
average of a 2-step return and 4-step return with: $$\frac{1}{2}
G_t^{t+2}(V_t(S_{t+2}))+\frac{1}{2}G_t^{t+4}(V_t(S_{t+4}))$$.

Using this idea, we can define TD(&lambda;) as a way of averaging *n*-step
returns called *&lambda;-return* defined by:

\begin{equation}
L_t=(1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{t+n}(V_t(S_{t+n}))
\end{equation}

One way of implementing this is by keeping a memory associated with every state
called its *eligibility trace*. On each step, we update this with the following
equation:

\begin{equation}
E_t(s)=\gamma\lambda E_{t-1}(s), \forall s \in \mathcal{S},s\ne S_t
\end{equation}

and

\begin{equation}
E_t(S_t)=\gamma\lambda E_{t-1}(S_t)
\end{equation}

where $$\gamma$$ is the discount rate and $$\lambda$$ is the parameter introduced in
Equation 25. Now using this we can change the value with the following
equations:

\begin{equation}
\delta_t = R_{t+1} + \gamma V_t(S_{t+1})-V_t(S_t)
\end{equation}

\begin{equation}
\Delta V_t(s)=\alpha\delta_t E_t(s), \forall s \in \mathcal{S}
\end{equation}

These increments can be done on each step to form an online algorithm or at the
end to form an offline algorithm.


<a id="orgf8e0391"></a>

## Planning and Learning with Tabular Methods


<a id="org2f453f9"></a>

### Models and Planning

A *model* of the environment is anything that can tell us how the environment
will respond to actions. Given a state and an action, a model produces a
prediction of the resultant next state and next reward. There are two types of
models: a *distribution model* which produces all possible next states and their
probabilities and a *sample model* which produces only one of the possibilities.

There are two approaches for *planning*, or producing/improving a policy given a
model: *state-space planning* where planning is a search through the state space
for an optimal policy or path to a goal or *plan-space planning* which is a
search through the space of plans.


<a id="org57600f2"></a>

### Integrated Planning, Acting, and Learning

With a planning agent, there are 2 ways you can use real experiences, to
directly improve the model (*model-learning*) or to directly improve the value
function (*direct reinforcement learning*).

    Initialize Q(s, a) and M odel(s, a) for all s ∈ S and a ∈ A(s)
    Do forever:
        (a) S ← current (nonterminal) state
        (b) A ← ε-greedy(S, Q)
        (c) Execute action A; observe resultant reward, R, and state, S′
        (d) Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S′, a) − Q(S, A)]
        (e) Model(S,A) ← R,S′ (assuming deterministic environment)
        (f) Repeat n times:
            S ← random previously observed state
            A ← random action previously taken in S
            R,S′ ← Model(S,A)
            Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S', a) - Q(S, A)]

In this code, step (d) is direct RL while steps (e) and (f) are model learning.


<a id="org7b8b423"></a>

### When the Model is Wrong

When the environment is stochastic and only a limited number of samples have
been observed, the model can be inaccurate and the planning process will compute
a suboptimal policy. However, the suboptimal policy computer by the policy
quickly leads to the discovery and correction of the model. This tends to happen
when the model is optimistic.


<a id="org760ae04"></a>

### Trajectory Sampling

Instead of sampling state-action pairs in the model uniformly like two sections
ago, we can instead follow a trajectory using our current policy. This is simple
to implement. We start from our starting state and then use our current policy
to make a sample episode with our model.


<a id="orgc7deb08"></a>

### Heuristic Search

Interesting.


<a id="org0d36e8a"></a>

## On-policy Approximation of Action Values


<a id="org61c880f"></a>

### Value Prediction with Function Approximation

Instead of the value function being a table, it can be a parameterized function
with input vector $$\textbf{w} \in \mathbb{R}^n$$ We can write $$\hat{v}(s,\textbf{w})
\approx v_\pi(s)$$ with the approximated value of state $$s$$ given by weight
vector $$\textbf{w}$$.

All of the backups used before can now be interpreted as making the function
more like the value being backed up. Before this was moving a value in a table
by a small amount, but now we can do this by changing the weights.


<a id="org5f7a2ad"></a>

## Policy Approximation


<a id="orgb4dea9b"></a>

### Actor-Critic Methods

Actor-critic methods are TD methods that have a separate memory structure to
explicitly represent the policy independent of the value function. The policy
structure is known as the *actor* because it selects actions and the estimated
value function is known as the *critic* because it criticizes the actions made
by the actor. Learning is always on-policy: the critic must learn about and
critique whatever policy is currently being followed by the actor. The critique
takes the form of a TD error (a scalar signal) and is the only output of the
critic:

\begin{equation\*}
\delta_t = R_{t+1} + \gamma V_t(S_{t+1}) - V(S_t)
\end{equation\*}

where $$V_t$$ is the value function implemented by the critic at time $$t$$. The TD
error can be used to evaluate the action taken. If the TD error is positive, the
tendency to select $$A_t$$ should be strengthened, conversely, if the TD error is
negative, the tendency should be weakened. We can denote this with the following
equation:

\begin{equation\*}
H_{t+1}(S_t, A_t) = H_t(S_t, A_t)+\beta \delta_t
\end{equation\*}

where $$H_t(s, a)$$ is the preference for taking action $$a$$ at time $$t$$ in state
$$s$$.


<a id="org83f18eb"></a>

### Eligibility Traces for Actor-Critic Methods

We can add an eligibility trace to this with the following equation:

\begin{equation\*}
H_{t+1}(s, a) = H_t(s, a)+\alpha\delta_t E_t(s,a)
\end{equation\*}

