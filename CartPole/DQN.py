# -*- coding: utf-8 -*-

import random
import gym
import math
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


EPOCHS = 1000
THRESHOLD = 100
MONITOR = True

class DQN:   
    def __init__(self, env_string, batch_size=64):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string)
        self.input_size = self.env.observation_space.shape[0]
        print('input size ', self.input_size)
        self.action_size = self.env.action_space.n
        print('action size ', self.action_size)
        self.batch_size = batch_size
        print('batch size ', self.batch_size)
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        alpha = 0.01
        alpha_decay = 0.01
        
        if MONITOR:
            self.env = gym.wrappers.Monitor(self.env,
                                            '../data/'+env_string,
                                            force=True)
        # init model
        self.model = Sequential()
        self.model.add(Dense(24,
                             input_dim=self.input_size,
                             activation='tanh'))
        self.model.add(Dense(48,
                             activation='tanh'))
        self.model.add(Dense(self.action_size,
                             activation='linear'))
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=alpha,
                                          decay=alpha_decay))
    
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    
    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory,
                                  min(len(self.memory),
                                      batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
            self.model.fit(np.array(x_batch),
                           np.array(y_batch),
                           batch_size=len(x_batch),
                           verbose=0)
    
    #epsilon greedy policy
    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
    
    def train(self):
        scores = deque(maxlen=100)
        avg_scores = []
        for e in range(EPOCHS):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            i = 0
            
            #play an episode of cartpole
            while not done:
                #choose action with epsilon greedy policy
                action = self.choose_action(state, self.epsilon)
                
                next_state, reward, done, _ = self.env.step(action)

                next_state = self.preprocess_state(next_state)
                
                #store in memory buffer for training network
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                
                #epsilon decay
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon_decay*self.epsilon) # decrease epsilon
                
                #in episode timer
                i += 1
                scores.append(i)
                mean_score = np.mean(scores)
                print(mean_score)
                avg_scores.append(mean_score)
                if mean_score >= THRESHOLD:
                    print('Ran {} episodes. SOlved after {} trials'.format(e, e - 100))
                    return avg_scores
                    break;
                
            if e % 100 == 0 and e > 0:
                print('[Episode {} - Mean surival time over last 100 episodes was {} ticks'.format(e, mean_score))
                print('Did not solve after {} episodes :'.format(e))
                return avg_scores
                break;
            #train the network at the end of each episode
            self.replay(self.batch_size)

    
    def preprocess_state(self, state):
        return np.reshape(state, [1, self.input_size])


env_string = 'CartPole-v0'
agent = DQN(env_string)
scores = agent.train()

plt.plot(scores)
plt.show()

