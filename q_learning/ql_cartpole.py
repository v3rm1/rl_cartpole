import math
import time
import gym
import numpy as np
import pandas as pd


BUCKETS = (1, 1, 6, 12,)
MAX_EPISODES = 1000
MIN_ALPHA = 0.1
MIN_EPSILON = 0.1
ADAPTIVITY_DIVISOR = 25


class Agent():

    def __init__(self, environment, discrete_buckets,
                 min_alpha, min_epsilon, adaptivity_div):
        self.action_space = environment.action_space
        self.obs_space = environment.observation_space
        self.discrete_buckets = discrete_buckets
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.adaptivity_div = adaptivity_div
        self.gamma = 1
        self.Q = np.zeros(self.discrete_buckets +
                          (self.action_space.n, ))

    def discretize_inputs(self, observations):
        bound_diff = [self.obs_space.high[0] - self.obs_space.low[0],
                      1,
                      self.obs_space.high[2] - self.obs_space.low[2],
                      2 * math.radians(50)]
        lower_bounds = [self.obs_space.low[0], -0.5,
                        self.obs_space.low[2], -math.radians(50)]
        ratios = [(observations[i] + abs(lower_bounds[i]))
                  / bound_diff[i] for i in range(len(observations))]
        discretized_obs = [min(self.discrete_buckets[i] - 1,
                           max(0, (int(round((self.discrete_buckets[i] - 1
                                              * ratios[i]))))
                               )) for i in range(len(observations))]
        return tuple(discretized_obs)

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def q_update(self, state, action, reward, new_state, alpha):
        self.Q[state][action] += alpha * (reward + self.gamma *
                                          np.max(self.Q[new_state]) -
                                          self.Q[state][action])
        return self.Q[state][action]

    def epsilon_update(self, episode_number):
        return max(self.min_epsilon,
                   min(1, 1 -
                       math.log10((episode_number + 1) /
                                  self.adaptivity_div)))

    def alpha_update(self, episode_number):
        return max(self.min_alpha,
                   min(1, 1 -
                       math.log10((episode_number + 1) /
                                  self.adaptivity_div)))


def main():

    env = gym.make('CartPole-v0')
    cols = ['Episode', 'CartPosition', 'CartVelocity',
            'PoleAngle', 'PoleTipVelocity', 'Action', 'QValue', 'Reward']
    proc_df = pd.DataFrame(columns=cols)
    # env = gym.wrappers.Monitor(env, 'monitors/cartpole1', force=True)
    agent = Agent(env, BUCKETS, MIN_ALPHA, MIN_EPSILON, ADAPTIVITY_DIVISOR)
    for episode in range(0, MAX_EPISODES):
        print("Ongoing episode: {eps:d}".format(eps=episode))
        current_state = agent.discretize_inputs(env.reset())
        alpha = agent.alpha_update(episode)
        epsilon = agent.alpha_update(episode)
        is_done = False
        idx = 0
        while not is_done:
            # env.render()
            action = agent.choose_action(current_state, epsilon)
            obs, reward, is_done, _ = env.step(action)
            next_state = agent.discretize_inputs(obs)
            q_latest = agent.q_update(current_state, action, reward,
                                      next_state, alpha)
            current_state = next_state
            idx += 1
            proc_dict = {
                'Episode': episode,
                'CartPosition': obs[0],
                'CartVelocity': obs[1],
                'PoleAngle': obs[2],
                'PoleTipVelocity': obs[3],
                'Action': action,
                'QValue': q_latest,
                'Reward': reward
            }
            proc_df = proc_df.append(proc_dict, ignore_index=True)
    env.close()
    record_file_path = "./q_learning/records/ql_rec_" \
        + time.strftime("%m%d_%H%M") \
        + ".csv"
    proc_df.to_csv(record_file_path)


if __name__ == "__main__":
    main()
