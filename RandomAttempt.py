import gym
from gym import Env
import numpy as np
import random
import matplotlib.pyplot as plt

"""
For the visualization part:
  env = gym.make('MountainCar-v0', render_mode="human")
  env.render()
"""

def moving_average(a, n):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

n_episodes = 500

position_queue = []

env = gym.make('MountainCar-v0')

# list used for the moving average (look at the plot code)
mov_average = []

for episode in range(n_episodes):
    
    observation = env.reset()
    observation = observation[0]
    observation[0] = round(observation[0],2)
    observation[1] = round(observation[1],2)
    
    done = False
    
    mov_average.append(observation[0])

    while not done:
        action = np.random.randint(0, env.action_space.n)

        new_observation, reward, terminated, truncated, info = env.step(action)
        mov_average.append(new_observation[0])
        new_observation[0] = round(new_observation[0],2)
        new_observation[1] = round(new_observation[1],2)

        # IF YOU WANT TO CHANGE THE REWARDS
        """
        if new_observation[0] >= 0.5:
          reward = 100
        if new_observation[0] - observation[0] > 0 and action == 2: 
            reward = reward + 1
        if new_observation[0] - observation[0] < 0 and action == 0: 
            reward = reward + 1
        """

        done = terminated or truncated

        if done and observation[0] >= 0.5:
           print('episode n: ' + str(episode) + ' Flag reached')
        elif done and observation[0] < 0.5:
           print('episode n: ' + str(episode) + ' Flag not reached')

        observation = new_observation

position_queue = moving_average(mov_average, 4020) #201*20

# PLOT CODE
"""
rolling_length = 500
fig, axs = plt.subplots(ncols=1, figsize=(15, 5))

axs.set_title("Positions Per Episode (Training)")
episode_position_t = np.convolve(np.array(position_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs.plot(range(len(episode_position_t)), episode_position_t)

plt.show()
"""