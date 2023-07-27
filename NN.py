import gym
from gym import Env
import numpy as np
import random
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense




def get_model(input_shape, actions):

  model = Sequential()
  model.add(Dense(24, activation='relu', input_shape=input_shape))
  model.add(Dense(24, activation='relu'))
  model.add(Dense(actions, activation='linear'))
  model.compile(loss='mse', optimizer='adam')
  return model



def main():
   
  env = gym.make('MountainCar-v0', render_mode="human")

  input_shape = env.observation_space.shape
  actions = env.action_space.n
  
  model = None

  if not os.path.exists("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Progetto/model"):
    print('Training...')
    model = get_model(input_shape, actions)

    n_episodes = 50
    start_epsilon = 1
    epsilon_decay = 0.95
    final_epsilon = 0.1
    eps = start_epsilon
    discount_factor = 0.95

    for episode in range(n_episodes):
    
      observation = env.reset()
      observation = observation[0]
      eps = eps*epsilon_decay
      observation = observation.reshape(1,2)
      #round(observation[0], 2) per il round a 2 cifre decimali
      done = False
      print('episode :' + '' + str(episode))
      #time.sleep(2)

      while not done:
        action = -1

        if np.random.random() < eps:
          action = np.random.randint(0, env.action_space.n)
        else:
          input = np.array(observation, dtype = np.float32)
          out = model.predict(input)
          action = np.argmax(out)

        new_observation, reward, terminated, truncated, info = env.step(action)
        new_observation = new_observation.reshape(1,2)
        done = terminated or truncated

        #new_observation = np.array([new_observation], dtype = np.float32)
        input = np.array(new_observation, dtype = np.float32)

        r = np.max(model.predict(input))
        target = reward + discount_factor *r

        target_vector = model.predict(np.array(observation, dtype = np.float32))[0]
        target_vector[action] = target

        x = np.array(observation, dtype = np.float32)
        y = np.array([target_vector]) #target_vector.reshape(-1, env.action_space.n)

        model.fit(x, y, epochs=1, verbose = 0) #verbose=0
        observation = new_observation
  
    model.save("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Progetto/model")
  
  model = keras.models.load_model("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Progetto/model")

  observation = env.reset()
  observation = observation[0]
  print(observation)
  
  done = False
  best_observation = observation
  
  position_queue = []

  while not done :

    position_queue.append(observation[0])
    
    observation = observation.reshape(1,2)
    input = np.array(observation, dtype = np.float32)
    out = model.predict(input)
    action = np.argmax(out[0])
    new_observation, reward, terminated, truncated, info = env.step(action) 
    done = terminated or truncated
    observation = new_observation
    if observation[0] >= best_observation[0]:
      best_observation = observation

    env.render()

  print(best_observation)

  rolling_length = 500
  fig2, axs2 = plt.subplots(ncols=1, figsize=(15, 5))

  axs2.set_title("Positions Per Episode")
  episode_position = np.convolve(np.array(position_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length
  axs2.plot(range(len(episode_position)), episode_position)

  plt.show()

  if best_observation[0] >= 0.5:
    print('flag reached')


main()