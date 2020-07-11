import gym
import numpy as np



class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if np.random.rand(0,1) < self.epsilon:
            return self.env.action_space.sample()
        return action
    

def main():
    env = RandomActionWrapper(gym.make('CartPole-v0'))
    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    done = False
    for episode in range(0,100):
        env.render()
        print("Episode: {}".format(episode))
        while not done:
            obs, reward, done, _ = env.step(0)
            total_reward += reward
            total_steps += 1
        episode += 1
        print("Episode ended.\nTotal reward: {}\nTotal steps: {}".format(total_reward, total_steps))
    env.close()
    
    return 0

if __name__ == "__main__":
    main()