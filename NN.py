import gym
from gym import Env
import numpy as np
import random
from collections import deque
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
  
  EXP_MAX_SIZE = 10000 # Max batch size of past experience
  BATCH_SIZE = EXP_MAX_SIZE//10 # Training set size
  experience = deque([],EXP_MAX_SIZE) # Past experience arranged as a queue
  model = None

  if not os.path.exists("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Project/ReinforcementLearning/model"):
    print('Training...')
    model = get_model(input_shape, actions)

    n_episodes = 500
    start_epsilon = 1
    final_epsilon = 0.05
    eps = start_epsilon
    discount_factor = 0.95

    for episode in range(n_episodes):
    
      observation = env.reset()
      observation = observation[0]
      initial_position = observation[0]

      done = False
      print('episode :' + '' + str(episode))
      #time.sleep(2)

      while not done:
        action = -1

        temp = np.random.random()
        if temp < eps:
          action = np.random.randint(0, env.action_space.n)
        else:
          input = np.array(observation.reshape(1,2), dtype = np.float32)
          out = model.predict_on_batch(input)
          action = np.argmax(out)

        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        #reward = reward + abs(new_observation[0] - initial_position)
        #if new_observation[0] >= 0.5:
          #reward = 100
        #else:
          #reward = (new_observation[0] - 0.5) / (abs(new_observation[1])*1000)

        #input = np.array(new_observation.reshape(1,2), dtype = np.float32)
        #r = np.max(model.predict_on_batch(input))
        #target = reward + discount_factor*r

        #target_vector = model.predict_on_batch(np.array(observation.reshape(1,2), dtype = np.float32))[0]
        #target_vector[action] = target

        if len(experience) >= EXP_MAX_SIZE:
          experience.popleft()
        
        item = np.array([np.array(observation), action, new_observation, reward], dtype=object)
        experience.append(item)
        observation = new_observation

        if done:
          if len(experience) >= BATCH_SIZE and (episode+1) % 5 == 0:
            batch = random.sample(experience, BATCH_SIZE)
            t1 = list()
            t2 = list()
            for e in batch:
              obs = e[0]
              act = e[1]
              new_obs = e[2]
              rew = e[3]

              input = np.array(new_obs.reshape(1,2), dtype = np.float32)
              out = model.predict_on_batch(input)
              r = np.max(out)
              target = rew + discount_factor*r
              target_vector = model.predict_on_batch(np.array(obs.reshape(1,2), dtype = np.float32))[0]
              target_vector[act] = target
              t1.append(obs)
              t2.append(target_vector)

            model.fit(tf.constant(t1), tf.constant(t2), epochs = 1, verbose = 0, validation_split = 0.2)
            model.save("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Project/ReinforcementLearning/model")
            eps -= 1/200

            if eps < final_epsilon:
              eps = final_epsilon
        
  model = keras.models.load_model("C:/Users/39377/Desktop/MasterDegree/AI&ML/ML/Project/ReinforcementLearning/model")

  observation = env.reset()
  observation = observation[0]
  print(observation)
  
  done = False
  best_observation = observation
  

  while not done :
    
    observation = observation.reshape(1,2)
    input = np.array(observation, dtype = np.float32)
    out = model.predict_on_batch(input)
    action = np.argmax(out[0])
    new_observation, reward, terminated, truncated, info = env.step(action) 
    done = terminated or truncated
    observation = new_observation
    if observation[0] >= best_observation[0]:
      best_observation = observation

  print(best_observation)

  if best_observation[0] >= 0.5:
    print('flag reached')


main()


