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
n_episodes = 500
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.0
eps = start_epsilon
discount_factor = 0.95

reward_queue = []

env = gym.make('MountainCar-v0')

Q = defaultdict(lambda: np.zeros(env.action_space.n))

mov_average = []

for episode in range(n_episodes):
    
    observation = env.reset()
    observation = observation[0]
    observation[0] = round(observation[0],2)
    observation[1] = round(observation[1],2)

    done = False

    print('episode :' + '' + str(episode))

    while not done:
        if np.random.random() < eps:
          action = np.random.randint(0, env.action_space.n)
        else:
          action = np.argmax(Q[(observation[0], observation[1])])

        new_observation, reward, terminated, truncated, info = env.step(action)

        mov_average.append(reward)
        
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

        Q[(observation[0],observation[1])][action] += learning_rate * (reward + discount_factor * np.max(Q[(new_observation[0],new_observation[1])]) - Q[(observation[0], observation[1])][action])

        observation = new_observation

    eps = max(final_epsilon, eps - epsilon_decay)
    #print(eps)

print("Exploiting the training...")
reward_queue = moving_average(mov_average, 4020) #201*20

observation = env.reset()
observation = observation[0]
observation[0] = round(observation[0],2)
observation[1] = round(observation[1],2)

done = False

best_observation = observation
  
while not done:
    action = np.argmax(Q[(observation[0], observation[1])])
    
    new_observation, reward, terminated, truncated, info = env.step(action)
    
    new_observation[0] = round(new_observation[0],2)
    new_observation[1] = round(new_observation[1],2)

    if observation[0] >= best_observation[0]:
      best_observation = observation
    
    done = terminated or truncated
    
    observation = new_observation

    #env.render()

print(best_observation)

if best_observation[0] >= 0.5:
    print('Flag reached')
else:
   print('Flag not reached')


#PLOT CODE
"""
rolling_length = 500
fig, axs = plt.subplots(ncols=1, figsize=(15, 5))

axs.set_title("Reward Per Episode (Training)")
episode_reward_t = np.convolve(np.array(reward_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
axs.plot(range(len(episode_reward_t)), episode_reward_t)

plt.show()
"""