#!/usr/bin/python
print(
'''
                      .__    .__                   
  _____ _____    ____ |  |__ |__| ____   ____      
 /     \\__  \ _/ ___\|  |  \|  |/    \_/ __ \     
|  Y Y  \/ __ \\  \___|   Y  \  |   |  \  ___/     
|__|_|  (____  /\___  >___|  /__|___|  /\___  >    
      \/     \/     \/     \/        \/     \/     
.__machine learning with A7MD0V..__                
|  |   ____ _____ _______  ____ |__| ____    ____  
|  | _/ __ \\__  \\_  __ \/    \|  |/    \  / ___\ 
|  |_\  ___/ / __ \|  | \/   |  \  |   |  \/ /_/  >
|____/\___  >____  /__|  |___|  /__|___|  /\___  / 
          \/     \/           \/        \//_____/
using TensorFlow 1.40 and Python 3.5 ----2017-->

72 65 69 6E 66 6F 72 63 65 6D 65 6E 74 
6C 65 61 72 6E 69 6E 67           

'''
)

'''
There are two functions used in Reinforcement Learning namely: policy iteration and value iteration. These are stored in 'functions'. We use OpenAI's Gym to load a Frozen Lake environment, which is similar to the classical AI environment of Wumpus World. The agent uses RL called from our dynamic programming functions to reach from a start state to a goal state. The program will return the decisions that need to be taken by the agent.
'''
# ----------=: import libraries :=-----------------------------------------
import gym
import numpy as np

from functions import policy_iteration, value_iteration

# ----------=: MDP Actions :=----------------------------------------------
# Unicode symbols
action_mapping = {
    0: '\u2191',  # UP
    1: '\u2192',  # RIGHT
    2: '\u2193',  # DOWN
    3: '\u2190',  # LEFT
}

# ----------=: MDP Parameters :=-------------------------------------------
# Number of episodes to play
n_episodes = 10000

# ----------=: Reward Calculation :=---------------------------------------
def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):

        terminated = False
        state = environment.reset()

        while not terminated:

            # Select best action to perform in a current state
            action = np.argmax(policy[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            # Summarize total reward
            total_reward += reward

            # Update current state
            state = next_state

            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1

    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward

# Functions to find best policy
solvers = [
    ('Policy Iteration', policy_iteration),
    ('Value Iteration', value_iteration)
]

for iteration_name, iteration_func in solvers:

    # Load a Frozen Lake environment
    environment = gym.make('FrozenLake8x8-v0')

    # Search for an optimal policy using policy iteration
    policy, V = iteration_func(environment.env)

    print('\n Final policy derived using {}:'.format(iteration_name))
    print(' '.join([action_mapping[action] for action in np.argmax(policy, axis=1)]))

    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)

    print('{} :: number of wins over {} episodes = {}'.format(iteration_name, n_episodes, wins))
    print('{} :: average reward over {} episodes = {} \n\n'.format(iteration_name, n_episodes, average_reward))

