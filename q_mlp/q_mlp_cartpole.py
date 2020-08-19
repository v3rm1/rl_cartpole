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
BATCH_SIZE = 25

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1
EPSILON_DECAY = 0.9

# Number Of Episodes to run
EPISODES = 2500

class DQNAgent:
    """ """
    def __init__(self, environment):
        super().__init__()
        self.obs_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_MAX
        self.q_net = self.network()

    def network(self):
        """ """
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Dense(100,
                               input_shape=(self.obs_space, ),
                               activation="sigmoid"))
        self.model.add(
            keras.layers.Dense(self.action_space, activation="linear"))
        self.model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(learning_rate=ALPHA))
        return self.model

    def memorize(self, state, action, reward, next_state, done):
        """

        :param state: 
        :param action: 
        :param reward: 
        :param next_state: 
        :param done: 

        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """

        :param state: 

        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.q_net.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        """ """
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            print("q_update before discount:{}".format(q_update))
            if not done:
                q_update = reward + GAMMA * np.amax(self.q_net.predict(next_state)[0])
                print("q_update after discount:{}".format(q_update))
            q_values = self.q_net.predict(state)
            q_values[0][action] = q_update
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
            self.q_net.fit(state, q_values, verbose=0)
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)


def main():
    """ """
    env = gym.make("CartPole-v0")
    # score_log = ScoreLogger("CartPole-v0")

    dqn_agent = DQNAgent(env)
    for ep in range(EPISODES):
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
            print(reward)
            next_state = np.reshape(next_state,
                                    [1, env.observation_space.shape[0]])
            dqn_agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                    ep, dqn_agent.epsilon, step))
                # score_log.add_score(step, ep)
                break
            dqn_agent.experience_replay()


if __name__ == "__main__":
    main()
