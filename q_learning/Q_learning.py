#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import gym
from matplotlib import pyplot as plt
import time
from IPython.display import clear_output


# In[53]:


env = gym.make("FrozenLake-v0")
n_observations = env.observation_space.n
n_actions = env.action_space.n

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))

#number of episode we will run
n_episodes = 10000

#maximum of iteration per episode
max_iter_episode = 100

#initialize the exploration probability to 1
exploration_proba = 1

#exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001

# minimum of exploration proba
min_exploration_proba = 0.01

#discounted factor
gamma = 0.99

#learning rate
lr = 0.1


# In[54]:


rewards_per_episode = list()
#we iterate over episodes
for e in range(n_episodes):
    #we initialize the first state of the episode
    current_state = env.reset()
    done = False
    
    #contains the sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(max_iter_episode):
        # we sample a float from a uniform distribution between 0 and 1
        # if the sampled flaot is less than the exploration probability
        #    the agent explore the environment by choosing a random action
        # else
        #    he exploit his knowledge using the bellman equation 
        if np.random.uniform(0,1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
        
        # The environment runs the choen action and returns the next state,
        # the reward for that action and true if the epiosed is ended.
        next_state, reward, done, _ = env.step(action)
        
        # We update our Q-table using the Q-learning iteration equation
        Q_table[current_state, action] = (1-lr)*Q_table[current_state, action] + lr*(reward + gamma*max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    # We update our exploration probability using exponential decay formula 
    exploration_proba = max(min_exploration_proba,np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)

print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000,": mean espiode reward: ",np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
print("\n\n")

