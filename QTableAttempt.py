import gym
from gym import Env
import numpy as np
import random
from collections import defaultdict

learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1
eps = start_epsilon
discount_factor = 0.95

env = gym.make('MountainCar-v0')

Q = defaultdict(lambda: np.zeros(env.action_space.n))

for episode in range(n_episodes):

    observation = env.reset()
    observation = observation[0]
    done = False
    print('episode :' + '' + str(episode))

    while not done:
        if np.random.random() < eps:
          action = np.random.randint(0, env.action_space.n)
        else:
          action = np.argmax(Q[(observation[0], observation[1])])

        new_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        Q[(observation[0],observation[1])][action] += learning_rate * (reward + discount_factor * np.max(Q[(new_observation[0],new_observation[1])]) - Q[(observation[0], observation[1])][action])

        observation = new_observation

    print(observation[0])
    eps = max(final_epsilon, eps - epsilon_decay)
    print(eps)

