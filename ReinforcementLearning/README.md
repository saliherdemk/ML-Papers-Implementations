
# Implementation Results

[Snake AI with Deep Q-Network](https://github.com/saliherdemk/snake-rl/tree/master)

<center>
<img src="./media/demo.gif" width="600" height="500"></img>
</center>

### A Couple of Things to Note:

####  Creating a Separate Target Network Significantly Improved Model Quality

In standard Q-learning, the same network is used to predict Q-values for the current state and calculate target Q-values for the next state. This creates a moving target problem, as the network is updated continuously, which causes the targets to shift as well. It's like trying to hit a target that keeps moving while you're shooting at it.

By introducing a target network, the main network is updated continuously during training, while the target network is updated periodically (matching the main network). This stabilizes the training process and significantly improves the model's performance.

#### Reward Scaling is Important

For smaller networks, providing large rewards or penalties (e.g., +10 or -10) can cause significant issues, resulting in large losses. To avoid this, we need to clip the target Q-values to prevent extremely high or low rewards from causing instability in the learning process.

#### Representing the Entire State with Features is Valid

Unless you're using a CNN, giving the entire grid to the model as an input might not be the most effective approach. Instead, it’s more efficient to extract meaningful features from the state, such as `danger_straight`, `danger_up`, `food_position`, etc., and pass those features to the model. This approach leads to better model performance.

#### Going to Dead End Issue

I've tried to represent the state with those 11 features:

- head direction -> 4
- will collide if it keeps goind straight, left or right -> 3
- where is the food - left, right, top, down -> 4

Since the state can provide information only for the next move, snake can't know if it's going to the dead end or not. And since we want to use ai to solve that rather than some algorithm, we have to provide more information.

I provide 3 more input `straight_safe`, `left_safe`, `right_safe` which we check if there is an enough space that snake can fit.

# Reinforcement Learning (RL)

Reinforcement Learning is a computational approach where an agent learns to make decisions by interacting with an environment to maximize a reward signal.

At each time step $t$, the agent:
- Observes a state $s_t$
- Chooses an action $a_t$
- Receives a reward $r_t$ (time-deleyed label: Reward might be delayed until the problem is solved.) 
- Moves to a new state $s_{t+1}$

This process continues over time and the agent's goal is to learn a policy $\pi(a|s)$ that maximizes cumulative reward.


- State $s$: Snapshot of the environment at a time 
- Action $a$: Choice the agent makes 
- Reward $r$: Scalar feedback signal 
- Policy $\pi(a\|s)$: A strategy that maps states to actions
- Value Function: Measures how good it is to be in a state (or state-action pair)
- Environment: The world the agent interacts with


### Return: Measuring Long-term Success

The total reward the agent wants to maximize is called the return:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^\infty \gamma^k r_{t+k}
$$

Where $\gamma \in [0, 1)$ is the discount factor — it prioritizes immediate rewards over distant ones. 

If the discount factor $\gamma$ is larger, the agent considers future rewards more.


### Value Functions

Value function is the expected total future reward starting from a given state (or state-action pair), considering all future steps under a policy.

#### State-Value Function $V^\pi(s)$

Expected return starting from state $s$, following policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]
$$

#### Action-Value Function $Q^\pi(s, a)$

Expected return starting from state $s$, taking action $a$, and following policy $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

Given a chess position, the state-value function is the expected total reward from the current position, considering all possible moves and outcomes for the next 10 moves.

The action-value function is the expected total reward if you first play a specific move, then consider all possible outcomes for the remaining 9 moves.

---

## Bellman Equations

These express recursive relationships between values.

### Bellman Equation for $V^\pi(s)$

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r + \gamma V^\pi(s') \right]
$$

- $\pi(a∣s)$: probability of taking action $a$ in state $s$

- $P(s'|s,a)$: probability of transitioning to state $s'$ after $a$

- $r(s,a,s′)$: reward received after taking action $a$ in $s$ and landing in $s'$

- $\gamma$: discount factor

The Bellman equation tells us the value of a state equals the immediate reward plus the value of the next state.

### Optimality

#### Optimal state-value function

$$
V^*(s) = \max_\pi V^\pi(s)
$$

It's the best possible value you can get from state s among all policies $\pi$.

#### Optimal action-value function

$$
Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

This is the maximum expected return the agent can get if it starts in state $s$, takes action $a$,
then follows the best possible policy afterward.

#### Bellman Optimality Equation

This equation gives us a way to compute $Q∗(s,a)$ recursively:

$$
Q^*(s,a) = E \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
$$

This says:

If I take action $a$ in state $s$, the best total reward I can get equals:

the immediate reward $r$ plus the best possible total reward from the next state $s'$ which is the max over all possible next actions.

That’s the optimality, we always assume we’ll behave perfectly from now on.

## Policies

### Deterministic Policy

$$
\pi(s) = a
$$

Always selects the same action for a state.

### Stochastic Policy

$$
\pi(a | s) = P(a \mid s)
$$

Gives probabilities for each action.

---

## Exploration vs Exploitation

The agent must explore unknown actions to discover better strategies, but also exploit what it knows to earn rewards.

### $\varepsilon$-Greedy Strategy

- With probability $\varepsilon$, take a random action.
- With probability $1 - \varepsilon$, take the best action.

---

## Learning Methods

There are 3 main RL families:

| Type | Description |
|------|-------------|
| Value-based | Learn a value function (e.g. Q-learning, DQN) |
| Policy-based | Learn the policy directly (e.g. REINFORCE) |
| Actor-Critic | Learn both (e.g. A2C, PPO) |

---

## Algorithms (Preview)

| Algorithm | Uses Value? | Uses Policy? | Uses Neural Net? |
|----------|-------------|--------------|------------------|
| Q-learning | Yes | No | No |
| DQN | Yes | No | Yes |
| REINFORCE | No | Yes | Yes |
| A2C / PPO | Yes | Yes | Yes |

---

## Markov Decision Process (MDP)

Formally, an RL problem is modeled as an MDP:

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

Where:
- $\mathcal{S}$: Set of states
- $\mathcal{A}$: Set of actions
- $P(s'|s,a)$: Transition probability
- $R(s,a)$: Reward function
- $\gamma$: Discount factor

## Example

For snake game example,

| Concept             | In Snake                                                  |
|---------------------|-----------------------------------------------------------|
| State $s$       | Positions of snake, food, direction, etc.                |
| Action $a$      | Up, Down, Left, Right                                     |
| Reward $r$      | +1 (food), -100 (death), 0 otherwise                      |
| Policy $\pi$    | Strategy for where to move (If food is above, and there’s no wall, go up)|
| Transition $P(s'\|s,a)$ | How snake grows/moves                  |
| Return $G_t$    | Total food eaten before death                             |
| $V^\pi(s)$      | How promising a state is                                  |
| $Q^\pi(s,a)$    | How good a specific move is                               |
| Discount Factor $\gamma$ | How much future food matters                    |
| Bellman Equation    | Used to update action values (If I go right now, how much food can I expect in total)|



Lets say we're on this this which `o` is our snake tail, `h` is our snake head and `F` is the food.
```
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . o o o o .
. . . . . . o .
. . . . . . h .
. . . . . . F .
. . . . . . . .
```

We will represent this state with the 14 features as we mentioned before.

Our head facing down so `dir_down` true, others false. We can't die in the next state so all three are false. Our snake can fit wherever we go so all three is true. Food is on down relative to our head so `foodOnDown` true others false. 

- dir_up -> false
- dir_down -> true
- dir_left -> false
- dir_right -> false
- danger_straight -> false
- danger_left -> false
- danger_right -> false
- straight_safe -> true
- left_safe -> true
- right_safe -> true
- foodOnLeft -> false
- foodOnRight -> false
- foodOnUp -> false
- foodOnDown -> true

Now, we move to the next step. Let's assume our $\epsilon = 1$ which mean we're picking a random move, meaning we're exploring. Suppose the chosen random move is to go right. Next state is:

```
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . o o o .
. . . . . . o .
. . . . . h o .
. . . . . . F .
. . . . . . . .
```

At this step, we calculate the reward of that action and check whether the game is over. Our reward function is implemented as:

```
getReward() {
    if (this.game.gameOver) return -1;
    else if (this.game.justAteFood) return 1;
    else return -0.005;
}
```

Since our game is not over and we didn't eat the food, our reward is -0.005. Now, we're adding this MDP process to our `replayBuffer`

```
this.replayBuffer.push({
    state: currentState,
    action: relativeAction,
    reward,
    nextState: nextState,
    done,
});
```

We continue this process until the snake dies. At each step we decrease $\epsilon$, meaning we gradually reduce exploration and increase exploitation. For the first few steps, we wait until the replayBuffer has enough entries. After that at each step, we train our model using a random sample from the replayBuffer. Training process looks like this,

We're taking MDP consist of  `state`, `action`, `reward`, `nextState`, `done`

```
const currentQ = this.model.forward(state);
const nextQ = this.targetModel.forward(nextState);
const targetQ = reward + (done ? 0 : this.gamma * Math.max(...nextQ));.
```

The model outputs the Q-values (estimated future rewards) for each possible action.
`Math.max(...nextQ)` selects the best possible next action according to the target model.
So, currentQ is the model's prediction for the current state, and `nextQ` is its prediction for the future state (which we haven’t seen yet).

With this setup, our problem becomes a supervised learning task:
We train the model to output the targetQ for the given currentQ.

```
const error = predQ[action] - targetQ;
```

We only update the Q-value for the action that was actually taken, because that’s the only outcome we observed.

Assume:

- currentQ = $[0.1, -0.3, 0.2]$ corresponds to [left, right, straight]
- nextQ = $[-1, 0.3, -0.2]$
- The action taken was right (index 1)
- The reward was -0.005, and gamma = 0.99

We calculate:

$$
targetQ = reward + \gamma *max(nextQ) = −0.005 + 0.99 *0.3 = 0.292
$$

Now, our model had previously estimated $currentQ[1] = -0.3$ for going right, but we now believe (based on what happened after taking that action) that it should’ve been closer to 0.292. So we train the model to move that Q-value up toward 0.292.

This works because if going right leads to a bad outcome, then the reward is low, and so is targetQ.This causes the model to reduce $Q[state][right]$.

Then at inference, we select the action with the highest predicted Q-value. Since going right now has a lower Q-value due to poor experience, it's less likely to be chosen. 

The rest of the process is standard backpropagation using mse.

## Resources

- https://www.youtube.com/watch?v=0MNVhXEX9to 
- https://www.youtube.com/watch?v=9hbQieQh7-o& 
