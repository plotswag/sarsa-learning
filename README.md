# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Train agent with SARSA in Gym environment, making sequential decisions for maximizing cumulative rewards.

## SARSA LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with random values for all state-action pairs.

### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.

## SARSA LEARNING FUNCTION
### Name: JEEVANESH S
### Register Number: 212222243002
```PYTHON
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000,
          max_steps=200):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # decay schedules
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # epsilon-greedy action selection
    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q[state])

    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset()
        action = select_action(state, epsilons[e])

        for t in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, epsilons[e])

            # TD target and update
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alphas[e] * td_error

            state, action = next_state, next_action

            if done:
                break

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:

### optimal policy, optimal value function , success rate for the optimal policy.
<img width="1117" height="691" alt="image" src="https://github.com/user-attachments/assets/aed18da5-e4f7-490e-a5f3-529f65a84685" />

<img width="1827" height="662" alt="image" src="https://github.com/user-attachments/assets/e405411c-9830-4bcf-91a5-ece01e7127c1" />

### plot comparing the state value functions of Monte Carlo method and SARSA learning.
<img width="1077" height="757" alt="image" src="https://github.com/user-attachments/assets/098cbe94-9879-4298-bded-12de613689fd" />
<img width="1817" height="673" alt="image" src="https://github.com/user-attachments/assets/5c7d4e44-6b9e-4f78-bd8c-9d8cf9c0c4b4" />

## RESULT:

Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
