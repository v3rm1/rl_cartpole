import numpy as np
import random
from collections import deque
import gym
from logger.score import ScoreLogger
from pyTsetlinMachine.tm import QRegressionTsetlinMachine
from discretizer import CustomDiscretizer

# Reward decay
GAMMA = 0.9

# Experience-Replay Memory Parameters
MEMORY_SIZE = 100000
BATCH_SIZE = 25

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.9

# Number Of Episodes to run
EPISODES = 200


class RTMQL:
    def __init__(self, environment):
        super().__init__()
        # Since we represent each value in the vector as a 2 bit string
        self.obs_space = 2 * environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_MAX
        self.agent = self.tm_model()

    def tm_model(self):
        self.tm_agent = QRegressionTsetlinMachine(number_of_clauses=100, T=100*8, s=2.75, reward=1, gamma=0.9, max_score=200, number_of_actions=2)
        self.tm_agent.number_of_patches = 2
        self.tm_agent.number_of_ta_chunks = 2
        self.tm_agent.number_of_features = 16

        return self.tm_agent

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        print("Epsilon: {}".format(self.epsilon))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = [self.agent.predict(state)]
        print("Q Value: {}".format(q_values))
        return np.argmax(q_values)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        q_update = np.zeros((2,))
        for state, action, reward, next_state, done in batch:
            q_update[action] = reward
            print("q_update before discount:{}".format(q_update))
            if not done:
                q_update[action] = reward + GAMMA * self.agent.predict(state)[action]
                print("q_update after discount:{}".format(q_update))
        q_values = self.agent.predict(state)
        q_values[action] = np.max(q_update)
        self.agent.fit(state, q_values)
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)



def main():
    env = gym.make("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0")
    discretizer = CustomDiscretizer()
    rtm_agent = RTMQL(env)
    prev_actions = []
    total_iter = 0
    for ep in range(EPISODES):
        print("EPISODE NUMBER: {}".format(ep))
        state = env.reset()
        state = discretizer.cartpole_discretizer(input_state=state)
        state = np.reshape(state, [1, 2*env.observation_space.shape[0]])
        step = 0
        done = False
        while not done:
            step += 1
            total_iter += 1
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
                ep, rtm_agent.epsilon, step))
                score_log.add_score(step, ep)
                break
            rtm_agent.experience_replay()
    print("TOTAL_ITER: {}".format(total_iter))


if __name__ == "__main__":
    main()
