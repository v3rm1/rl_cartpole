import math
import time
import gym
import numpy as np
import pandas as pd
from logger.score import ScoreLogger

# Discretization
BUCKETS = (
    1,
    1,
    6,
    12,
)

# Learning Rate
ALPHA_MIN = 0.1

# Exploration Exploitation Parameters
EPSILON_MIN = 0.1
EPSILON_MAX = 1
EPSILON_DECAY = 0.9

ADAPTIVITY_DIVISOR = 25

# Episode limit
EPISODES = 5000


class QLAgent():
    """
    Creates an instance of a q Learning agent for a given Open AI Gym Environment.
    """
    def __init__(self, environment, discrete_buckets, min_alpha, min_epsilon,
                 adaptivity_div):
        self.action_space = environment.action_space
        self.obs_space = environment.observation_space
        self.discrete_buckets = discrete_buckets
        self.alpha = min_alpha
        self.epsilon = min_epsilon
        self.adaptivity_div = adaptivity_div
        self.gamma = 1
        self.Q = np.zeros(self.discrete_buckets + (self.action_space.n, ))

    def discretize_inputs(self, observation):
        """
        Discretize Inputs (SPECIFICALLY FOR CARTPOLE ENVIRONMENT)
        :param observation: Current State as a 4*1 Vector

        """
        bound_diff = [
            self.obs_space.high[0] - self.obs_space.low[0], 1,
            self.obs_space.high[2] - self.obs_space.low[2],
            2 * math.radians(50)
        ]
        lower_bounds = [
            self.obs_space.low[0], -0.5, self.obs_space.low[2],
            -math.radians(50)
        ]
        ratios = [(observation[i] + abs(lower_bounds[i])) / bound_diff[i]
                  for i in range(len(observation))]
        discretized_obs = [
            min(
                self.discrete_buckets[i] - 1,
                max(0, (int(round(
                    (self.discrete_buckets[i] - 1 * ratios[i]))))))
            for i in range(len(observation))
        ]
        return tuple(discretized_obs)

    def choose_action(self, state):
        """

        :param state: 

        """
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def q_update(self, state, action, reward, new_state, alpha):
        """

        :param state: 
        :param action: 
        :param reward: 
        :param new_state: 
        :param alpha: 

        """
        self.Q[state][action] += alpha * (
            reward + self.gamma * np.max(self.Q[new_state]) -
            self.Q[state][action])
        return self.Q[state][action]

    def epsilon_update(self):
        """ """
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)

    def alpha_update(self, run):
        """

        :param run: 

        """
        return max(self.alpha,
                   min(1, 1 - math.log10((run + 1) / self.adaptivity_div)))


def main():
    """ """

    env = gym.make('CartPole-v0')
    score_log = ScoreLogger("CartPole-v0")

    ql_agent = QLAgent(env, BUCKETS, ALPHA_MIN, EPSILON_MIN,
                       ADAPTIVITY_DIVISOR)
    for eps in range(EPISODES):
        alpha = ql_agent.alpha_update(eps)
        state = env.reset()
        state = ql_agent.discretize_inputs(state)
        step = 0
        done = False
        while not done:
            step += 1
            action = ql_agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = ql_agent.discretize_inputs(next_state)
            q_latest = ql_agent.q_update(state, action, reward, next_state,
                                         alpha)
            reward = reward if not done else -reward
            state = next_state
            if done:
                print("Run: {0}\nEpsilon: {1}\tScore: {2}".format(
                    eps, ql_agent.epsilon, step))
                score_log.add_score(step, eps)
                break
            ql_agent.epsilon_update()


if __name__ == "__main__":
    main()
