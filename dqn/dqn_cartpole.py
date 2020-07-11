import os
import math
import time
import numpy as np
import random
import pandas as pd
from collections import deque
from logger.score import ScoreLogger
import gym

import tensorflow as tf
from tensorflow import keras

# Reward decay
GAMMA = 0.9
# Learning Rate
ALPHA = 0.001

# Experience-Replay Memory Parameters
MEMORY_SIZE = 100000
BATCH_SIZE = 20

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1
EPSILON_DECAY = 0.9

class DQNAgent:

    def __init__(self, environment):
        super().__init__()
        self.obs_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_MAX
        self.q_net = self.network()

    def network(self):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(
            24,
            input_shape=(self.obs_space,),
            activation="relu"
        ))
        self.model.add(keras.layers.Dense(
            24,
            activation="relu"
        ))
        self.model.add(keras.layers.Dense(
            self.action_space,
            activation="linear"
        ))
        self.model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(
                learning_rate=ALPHA
            )
        )
        return self.model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.q_net.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = reward
                + GAMMA * np.amax(self.q_net.predict(next_state)[0])
            q_values = self.q_net.predict(state)
            q_values[0][action] = q_update
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq='batch')
            self.q_net.fit(state, q_values, verbose=0)
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)

def main():
    env = gym.make("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0")
    
    dqn_agent = DQNAgent(env)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        step = 0
        done = False
        while not done:
            step += 1
            # env.render()
            action = dqn_agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            dqn_agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Run: {0}\nEpsilon: {1}\tScore: {2}".format(run, dqn_agent.epsilon, step))
                score_log.add_score(step, run)
                break
            dqn_agent.experience_replay()


if __name__ == "__main__":
    main()



