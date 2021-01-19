from os import path
import yaml
import numpy as np
import random
import math
from collections import deque
from time import strftime
import csv
import gym
from logger.score import ScoreLogger
from discretizer import CustomDiscretizer
from debug_plot_functions import DebugLogger
from itertools import starmap, repeat

# On LAPTOP
# from pyTsetlinMachine.tm import QRegressionTsetlinMachine
# On PEREGRINE
from q_rtm import QRegressionTsetlinMachine


# Path to file containing all configurations for the variables used by the q-rtm system
CONFIG_PATH = path.join(path.dirname(path.realpath(__file__)), 'config.yaml')
#
CONFIG_TEST_SAVE_PATH = path.join(path.dirname(path.realpath(__file__)), 'tested_configs.csv')
class RTMQL:
    def __init__(self, environment, config, eps_decay_config="EDF"):
        super().__init__()
        # Since we represent each value in the vector as a 2 bit string
        self.obs_space = 2 * environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=config['memory_params']['memory_size'])
        self.replay_batch = config['memory_params']['batch_size']

        self.episodes = config['game_params']['episodes']
        self.reward = config['game_params']['reward']
        self.max_score = config['game_params']['max_score']

        self.gamma = config['learning_params']['gamma']
        self.learning_rate = config['learning_params']['learning_rate']
        
        self.weighted_clauses = config['qrtm_params']['weighted_clauses']
        self.incremental = config['qrtm_params']['incremental']

        
        self.epsilon_max = config['learning_params']['EDF']['epsilon_max']
        self.eps_decay = eps_decay_config
        self.epsilon_min = config['learning_params']['EDF']['epsilon_min']

        self.epsilon = self.epsilon_max

        self.T = config['qrtm_params']['T']
        self.s = config['qrtm_params']['s']
        self.number_of_clauses = config['qrtm_params']['number_of_clauses']
        self.number_of_features = config['qrtm_params']['number_of_features']

        if eps_decay_config == "SEDF":
            self.sedf_alpha = config['learning_params']['SEDF']['tail']
            self.sedf_beta = config['learning_params']['SEDF']['slope']
            self.sedf_delta = config['learning_params']['SEDF']['tail_gradient']
            print("Agent configured to use Stretched Exponential Decay Function for Epsilon value.\nAlpha (tail): {}\nBeta (slope): {}\nDelta (tail_gradient): {}".format(self.sedf_alpha, self.sedf_beta, self.sedf_delta))
        else:
            self.epsilon_min = config['learning_params']['EDF']['epsilon_min']
            self.epsilon_max = config['learning_params']['EDF']['epsilon_max']
            self.epsilon_decay = config['learning_params']['EDF']['epsilon_decay']
            print("Agent configured to use Exponential Decay Function for Epsilon value.\nDecay: {}\nMax Epsilon: {}\nMin Epsilon: {}".format(self.epsilon_decay, self.epsilon_max, self.epsilon_min))

        self.agent_1 = self.tm_model()
        self.agent_2 = self.tm_model()

    def exp_eps_decay(self, current_ep):
        self.epsilon = self.epsilon_max * pow(self.epsilon_decay, current_ep)
        return max(self.epsilon_min, self.epsilon)

    def stretched_exp_eps_decay(self, current_ep):
        self.epsilon = 1.1 - (1 / (np.cosh(math.exp(-(current_ep - self.sedf_alpha * self.episodes) / (self.sedf_beta * self.episodes)))) + (current_ep * self.sedf_delta / self.episodes))
        return max(min(self.epsilon_max, self.epsilon), self.epsilon_min)


    def tm_model(self):
        self.tm_agent = QRegressionTsetlinMachine(number_of_clauses=self.number_of_clauses, T=self.T, s=self.s, reward=self.reward, gamma=self.gamma, max_score=self.max_score, number_of_actions=self.action_space, weighted_clauses=self.weighted_clauses)
        self.tm_agent.number_of_patches = 2
        self.tm_agent.number_of_ta_chunks = int(((self.number_of_features - 1) / 32) + 1)
        self.tm_agent.number_of_features = self.number_of_features
        return self.tm_agent

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.action_space)
            # print("Randomized Action: {}".format(a))
            return a
        q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
        # print("Q value based Action: {}".format(np.argmax(q_values)))
        if q_values[0] != q_values[1]:
            return np.argmax(q_values)
        else:
            return 0 if q_values[0] > 0 else 1

    def experience_replay(self, episode):
        if len(self.memory) < self.replay_batch:
            return [0,0]
        batch = random.sample(self.memory, self.replay_batch)
        # batch = list(starmap(self.memory.pop, repeat((), self.replay_batch)))
        for state, action, reward, next_state, done in batch:
            q_values = [np.random.random() + self.agent_1.predict(state), np.random.random() + self.agent_2.predict(state)]
            target_q = [np.random.random() + self.agent_1.predict(next_state), np.random.random() + self.agent_2.predict(next_state)]
            q_update = reward
            if not done:
                q_update = self.learning_rate * ( reward + self.gamma * target_q[action] - q_values[action] )
            # print("Q Values: {}".format(q_values))
            q_values[action] = q_update
            self.agent_1.fit(state, q_values[0], incremental=self.incremental)
            self.agent_2.fit(state, q_values[1], incremental=self.incremental)
        if self.eps_decay == "SEDF":
            # STRETCHED EXPONENTIAL EPSILON DECAY
            self.epsilon = self.stretched_exp_eps_decay(episode)
        else:
            # EXPONENTIAL EPSILON DECAY
            self.epsilon = self.exp_eps_decay(episode)
        return q_values if len(q_values)>0 else [0,0]
        
def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def store_config_tested(config_data, win_count, run_date, tested_configs_file_path=CONFIG_TEST_SAVE_PATH):
    run_dt = run_date
    # Defining dictionary key mappings
    field_names = ['decay_fn', 'epsilon_min', 'epsilon_max', 'epsilon_decay', 'alpha', 'beta', 'delta', 'reward_discount', 'mem_size', 'batch_size', 'episodes', 'reward', 'max_score', 'action_space', 'qrtm_n_clauses', 'T', 's', 'wins', 'win_ratio', 'run_date', 'bin_length', 'incremental', 'weighted_clauses', 'binarizer']
    decay_fn = config_data['learning_params']['epsilon_decay_function']
    if decay_fn == "SEDF":
        alpha = config_data['learning_params']['SEDF']['tail']
        beta = config_data['learning_params']['SEDF']['slope']
        delta = config_data['learning_params']['SEDF']['tail_gradient']
        eps_min = 0
        eps_max = 0
        eps_decay = 0
    else:
        alpha = 0
        beta = 0
        delta = 0
        eps_min = config_data['learning_params']['EDF']['epsilon_min']
        eps_max = config_data['learning_params']['EDF']['epsilon_max']
        eps_decay = config_data['learning_params']['EDF']['epsilon_decay']
    store_config = {
        'decay_fn': decay_fn,
        'epsilon_min': eps_min,
        'epsilon_max': eps_max,
        'epsilon_decay': eps_decay,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'reward_discount': config_data['learning_params']['gamma'],
        'mem_size': config_data['memory_params']['memory_size'],
        'batch_size': config_data['memory_params']['batch_size'],
        'episodes': config_data['game_params']['episodes'],
        'reward': config_data['game_params']['reward'],
        'max_score': config_data['game_params']['max_score'],
        'action_space': config_data['game_params']['action_space'],
        'qrtm_n_clauses': config_data['qrtm_params']['number_of_clauses'],
        'T': config_data['qrtm_params']['T'],
        's': config_data['qrtm_params']['s'],
        'wins': win_count,
        'win_ratio': win_count/config_data['game_params']['episodes'],
        'run_date': run_date,
        'bin_length':config_data['qrtm_params']['feature_length'],
        'incremental':config_data['qrtm_params']['incremental'],
        'weighted_clauses': config_data['qrtm_params']['weighted_clauses'],
        'binarizer': config_data['preproc_params']['binarizer']
    }
    # Write to file. Mode a creates file if it does not exist.
    if not path.exists(tested_configs_file_path):
        with open(tested_configs_file_path, 'w', newline='') as write_obj:
            header_writer = csv.writer(write_obj)
            header_writer.writerow(field_names)
    with open(tested_configs_file_path, 'a+', newline='') as write_obj:
        dict_writer = csv.DictWriter(write_obj, fieldnames=field_names)
        dict_writer.writerow(store_config)
    return

def main():
    config = load_config(CONFIG_PATH)
    gamma = config['learning_params']['gamma']
    episodes = config['game_params']['episodes']
    run_dt = strftime("%Y%m%d_%H%M%S")
    epsilon_decay_function = config['learning_params']['epsilon_decay_function']
    feature_length = config['qrtm_params']['feature_length']
    print("Configuration file loaded. Creating environment.")
    env = gym.make("CartPole-v0")
    
    # Initializing loggers and watchers
    debug_log = DebugLogger("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0", episodes)

    print("Initializing custom discretizer.")
    discretizer = CustomDiscretizer()
    print("Initializing Q-RTM Agent.")
    rtm_agent = RTMQL(env, config, epsilon_decay_function)
    binarized_length = int(config['qrtm_params']['feature_length'])
    binarizer = config['preproc_params']['binarizer']
    
    prev_actions = []
    
    win_ctr = 0
    q_list_0 = []
    q_list_1 = []
    q_list_total = []
    for curr_ep in range(episodes):
        q_0 = []
        q_1 = []
        state = env.reset()
        state = discretizer.cartpole_binarizer(input_state=state, n_bins=binarized_length-1, bin_type=binarizer)
        state = np.reshape(state, [1, feature_length * env.observation_space.shape[0]])
        step = 0
        done = False
        while not done:
            step += 1
            # env.render()
            action = rtm_agent.act(state)
            prev_actions.append(action)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            next_state = discretizer.cartpole_binarizer(next_state, n_bins=binarized_length-1, bin_type=binarizer)
            next_state = np.reshape(next_state,
                [1, feature_length * env.observation_space.shape[0]])
            rtm_agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                if step > 195:
                    win_ctr += 1
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(curr_ep, rtm_agent.epsilon, step))
                score_log.add_score(step,
                curr_ep,
                gamma,
                epsilon_decay_function,
                consecutive_runs=episodes,
                sedf_alpha=config['learning_params']['SEDF']['tail'],
                sedf_beta=config['learning_params']['SEDF']['slope'],
                sedf_delta=config['learning_params']['SEDF']['tail_gradient'],
                edf_epsilon_decay=config['learning_params']['EDF']['epsilon_decay'])
                break
        q_vals = rtm_agent.experience_replay(curr_ep)
        q_0.append(q_vals[0])
        q_1.append(q_vals[1])
        q_list_0.append(np.sum(q_0))
        q_list_1.append(np.sum(q_1))
        q_list_total.append(np.sum(q_0) + np.sum(q_1)/2)
    
    debug_log.add_watcher(q_list_0,
                          q_list_1,
                          q_list_total,
                          n_clauses=config["qrtm_params"]["number_of_clauses"],
                          T=config["qrtm_params"]["T"],
                          feature_length=feature_length)
    print("win_ctr: {}".format(win_ctr))
    store_config_tested(config, win_ctr, run_dt)


if __name__ == "__main__":
    main()
