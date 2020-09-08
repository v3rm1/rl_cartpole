from os import path
import yaml
import numpy as np
import random
import math
from collections import deque
import gym
from logger.score import ScoreLogger
from pyTsetlinMachine.tm import QRegressionTsetlinMachine
from discretizer import CustomDiscretizer

# Path to file containing all configurations for the variables used by the q-rtm system
CONFIG_PATH = path.join(path.dirname(path.realpath(__file__)), 'config.yaml')

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
        
        self.epsilon = config['learning_params']['EDF']['epsilon_max']
        self.eps_decay = eps_decay_config
        self.epsilon_min = config['learning_params']['EDF']['epsilon_min']

        self.T = config['qrtm_params']['T']
        self.s = config['qrtm_params']['s']
        self.number_of_clauses = config['qrtm_params']['number_of_clauses']

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
        return min(1, self.epsilon)


    def tm_model(self):
        self.tm_agent = QRegressionTsetlinMachine(number_of_clauses=self.number_of_clauses, T=self.T, s=self.s, reward=self.reward, gamma=self.gamma, max_score=self.max_score, number_of_actions=self.action_space)
        self.tm_agent.number_of_patches = 2
        self.tm_agent.number_of_ta_chunks = 2
        self.tm_agent.number_of_features = 16
        return self.tm_agent

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.action_space)
            print("Randomized Action: {}".format(a))
            return a
        q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
        print("Q value based Action: {}".format(np.argmax(q_values)))
        return np.argmax(q_values)

    def experience_replay(self, episode):
        if len(self.memory) < self.replay_batch:
            return
        batch = random.sample(self.memory, self.replay_batch)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = reward + self.gamma * np.amax([self.agent_1.predict(next_state), self.agent_2.predict(next_state)])
            q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
            q_values[action] = q_update
            self.agent_1.fit(state, q_values[0])
            self.agent_2.fit(state, q_values[1])
        if self.eps_decay == "SEDF":
            # STRETCHED EXPONENTIAL EPSILON DECAY
            self.epsilon = self.stretched_exp_eps_decay(episode)
        else:
            # EXPONENTIAL EPSILON DECAY
            self.epsilon = self.exp_eps_decay(episode)
        
def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def main():
    config = load_config(CONFIG_PATH)
    gamma = config['learning_params']['gamma']
    episodes = config['game_params']['episodes']
    epsilon_decay_function = config['learning_params']['epsilon_decay_function']
    print("Configuration file loaded. Creating environment.")
    env = gym.make("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0", episodes)
    print("Initializing custom discretizer.")
    discretizer = CustomDiscretizer()
    print("Initializing Q-RTM Agent.")
    rtm_agent = RTMQL(env, config, epsilon_decay_function)
    prev_actions = []
    for curr_ep in range(episodes):
        state = env.reset()
        state = discretizer.cartpole_discretizer(input_state=state)
        state = np.reshape(state, [1, 2*env.observation_space.shape[0]])
        step = 0
        done = False
        while not done:
            step += 1
            # env.render()
            action = rtm_agent.act(state)
            prev_actions.append(action)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            next_state = discretizer.cartpole_discretizer(next_state)
            next_state = np.reshape(next_state,
                [1, 2 * env.observation_space.shape[0]])
            rtm_agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                curr_ep, rtm_agent.epsilon, step))
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
            rtm_agent.experience_replay(curr_ep)


if __name__ == "__main__":
    main()
