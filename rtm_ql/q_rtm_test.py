import numpy as np
import random
import math
from collections import deque
import gym
from logger.score import ScoreLogger
from pyTsetlinMachine.tm import QRegressionTsetlinMachine
from discretizer import CustomDiscretizer

# Reward decay
GAMMA = 1

# Experience-Replay Memory Parameters
MEMORY_SIZE = 100000
BATCH_SIZE = 25

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.9

# Number Of Episodes to run
EPISODES = 300


class RTMQL:
    def __init__(self, environment):
        super().__init__()
        # Since we represent each value in the vector as a 2 bit string
        self.obs_space = 2 * environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_MAX
        self.agent_1 = self.tm_model()
        self.agent_2 = self.tm_model()

    def linear_eps_decay(self, eps_decay=EPSILON_DECAY):
        self.epsilon = self.epsilon * eps_decay
        return max(EPSILON_MIN, self.epsilon)

    def stretched_exp_eps_decay(self, current_ep):
        A = 0.1
        B = 0.3
        C = 0.1
        standardized_time = (current_ep-A*EPISODES)/(B*EPISODES)
        cosh = np.cosh(math.exp(-standardized_time))
        self.epsilon = 1.1-(1/cosh+(current_ep*C/EPISODES))
        return self.epsilon

    def tm_model(self):
        self.tm_agent = QRegressionTsetlinMachine(number_of_clauses=100, T=100*8, s=2.75, reward=1, gamma=GAMMA, max_score=200, number_of_actions=2)
        self.tm_agent.number_of_patches = 2
        self.tm_agent.number_of_ta_chunks = 2
        self.tm_agent.number_of_features = 16
        return self.tm_agent

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # print("Epsilon: {}".format(self.epsilon))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
        # print("Q Value: {}".format(q_values))
        return np.argmax(q_values)

    def experience_replay(self, episode):
        # self.epsilon = EPSILON_MAX
        if len(self.memory) < BATCH_SIZE:
            return
        batch_ctr = 0
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            # print("q_update before discount:{}".format(q_update))
            if not done:
                q_update = reward + GAMMA * np.amax([self.agent_1.predict(next_state), self.agent_2.predict(next_state)])
                # print("q_update after discount:{}".format(q_update))
            q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
            q_values[action] = q_update
            self.agent_1.fit(state, q_values[0])
            self.agent_2.fit(state, q_values[1])
        batch_ctr += 1
        print("Batch_Ctr: {}".format(batch_ctr))
        print("Epsilon: {}".format(self.epsilon))
        # LINEAR EPSILON DECAY
        # self.epsilon = self.linear_eps_decay()
        # STRETCHED EXPONENTIAL EPSILON DECAY
        self.epsilon = self.stretched_exp_eps_decay(episode)



def main():
    env = gym.make("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0")
    discretizer = CustomDiscretizer()
    rtm_agent = RTMQL(env)
    prev_actions = []
    for curr_ep in range(EPISODES):
        # print("EPISODE NUMBER: {}".format(ep))
        state = env.reset()
        state = discretizer.cartpole_discretizer(input_state=state)
        state = np.reshape(state, [1, 2*env.observation_space.shape[0]])
        step = 0
        done = False
        while not done:
            step += 1
            # env.render()
            action = rtm_agent.act(state)
            # print(action)
            prev_actions.append(action)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            # print(reward)
            next_state = discretizer.cartpole_discretizer(next_state)
            next_state = np.reshape(next_state,
                [1, 2 * env.observation_space.shape[0]])
            rtm_agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                curr_ep, rtm_agent.epsilon, step))
                score_log.add_score(step, curr_ep, GAMMA, EPSILON_DECAY)
                break
            rtm_agent.experience_replay(curr_ep)


if __name__ == "__main__":
    main()
