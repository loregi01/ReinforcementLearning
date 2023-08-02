import gym
from gym import Env
import numpy as np
import random
from collections import defaultdict
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

learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.0
eps = start_epsilon
discount_factor = 0.95

position_queue = []

env = gym.make('MountainCar-v0')

Q = defaultdict(lambda: np.zeros(env.action_space.n))

mm = []

for episode in range(n_episodes):
    observation = env.reset()
    observation = observation[0]
    observation[0] = round(observation[0],2)
    observation[1] = round(observation[1],2)
    done = False
    print('episode :' + '' + str(episode))
    mm.append(observation[0])

    while not done:
        action = np.random.randint(0, env.action_space.n)

        new_observation, reward, terminated, truncated, info = env.step(action)
        mm.append(new_observation[0])
        new_observation[0] = round(new_observation[0],2)
        new_observation[1] = round(new_observation[1],2)

        done = terminated or truncated

        observation = new_observation

    eps = max(final_epsilon, eps - epsilon_decay)
    #print(eps)

position_queue = moving_average(mm, 4020) #201*20

"""
for episode in range(n_episodes):
  observation = env.reset()
  observation = observation[0]
  observation[0] = round(observation[0],2)
  observation[1] = round(observation[1],2)
  done = False
  print('episode :' + '' + str(episode))
  position_queue.append(observation[0])
  velocity_queue.append(observation[1])

  while not done:
    action = np.argmax(Q[(observation[0], observation[1])])
    
    new_observation, reward, terminated, truncated, info = env.step(action)
    
    position_queue.append(new_observation[0])
    velocity_queue.append(new_observation[1])
    
    new_observation[0] = round(new_observation[0],2)
    new_observation[1] = round(new_observation[1],2)
    
    done = terminated or truncated
    
    observation = new_observation
"""

rolling_length = 500
fig, axs = plt.subplots(ncols=1, figsize=(15, 5))

axs.set_title("Positions Per Episode (Training)")
episode_position_t = np.convolve(np.array(position_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs.plot(range(len(episode_position_t)), episode_position_t)

plt.show()