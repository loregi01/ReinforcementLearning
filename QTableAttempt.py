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

learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.0
eps = start_epsilon
discount_factor = 0.95

position_queue_t = []
velocity_queue_t = []
position_queue = []
velocity_queue = []

env = gym.make('MountainCar-v0')

Q = defaultdict(lambda: np.zeros(env.action_space.n))

for episode in range(n_episodes):

    observation = env.reset()
    observation = observation[0]
    observation[0] = round(observation[0],2)
    observation[1] = round(observation[1],2)
    position_queue_t.append(observation[0])
    velocity_queue_t.append(observation[0])
    done = False
    print('episode :' + '' + str(episode))

    while not done:
        if np.random.random() < eps:
          action = np.random.randint(0, env.action_space.n)
        else:
          action = np.argmax(Q[(observation[0], observation[1])])

        new_observation, reward, terminated, truncated, info = env.step(action)
        new_observation[0] = round(new_observation[0],2)
        new_observation[1] = round(new_observation[1],2)
        position_queue_t.append(new_observation[0])
        velocity_queue_t.append(new_observation[1])

        done = terminated or truncated

        Q[(observation[0],observation[1])][action] += learning_rate * (reward + discount_factor * np.max(Q[(new_observation[0],new_observation[1])]) - Q[(observation[0], observation[1])][action])

        observation = new_observation

    eps = max(final_epsilon, eps - epsilon_decay)
    #print(eps)

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
fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

axs[0].set_title("Positions Per Episode (Training)")
episode_position_t = np.convolve(np.array(position_queue_t).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs[0].plot(range(len(episode_position_t)), episode_position_t)

axs[1].set_title("Cart Velocity Per Episode (Training)")
episode_velocity_t = np.convolve(np.array(velocity_queue_t).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs[1].plot(range(len(episode_velocity_t)), episode_velocity_t)

plt.show()

fig2, axs2 = plt.subplots(ncols=2, figsize=(15, 5))

axs2[0].set_title("Positions Per Episode")
episode_position = np.convolve(np.array(position_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs2[0].plot(range(len(episode_position)), episode_position)

axs2[1].set_title("Cart Velocity Per Episode")
episode_velocity = np.convolve(np.array(velocity_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs2[1].plot(range(len(episode_velocity)), episode_velocity)

plt.show()