import gym


def main():
    """ """
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, "recording", force=True)
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    done = False
    for episode in range(0, 5000):
        env.render()
        print("Episode: {}".format(episode))
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
        episode += 1
        print("Episode ended.\nTotal reward: {}\nTotal steps: {}".format(
            total_reward, total_steps))
    env.close()
    env.env.close()

    return 0


if __name__ == "__main__":
    main()
