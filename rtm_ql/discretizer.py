import numpy as np
import gym


class CustomDiscretizer:
    def __init__(self):
        super().__init__()
    
    def discretize_position(self, input_position):
        if input_position < 0:
            return np.fromiter(np.binary_repr(3, width=2), int)
        elif input_position > 0:
            return np.fromiter(np.binary_repr(1, width=2), int)
        else:
            return np.fromiter(np.binary_repr(0, width=2), int)

    def discretize_cart_velocity(self, input_cart_velocity):
        if input_cart_velocity < 0:
            return np.fromiter(np.binary_repr(3, width=2), int)
        elif input_cart_velocity > 0:
            return np.fromiter(np.binary_repr(1, width=2), int)
        else:
            return np.fromiter(np.binary_repr(0, width=2), int)

    def discretize_angle(self, input_angle):
        if input_angle < 0:
            return np.fromiter(np.binary_repr(3, width=2), int)
        elif input_angle > 0:
            return np.fromiter(np.binary_repr(1, width=2), int)
        else:
            return np.fromiter(np.binary_repr(0, width=2), int)

    def discretize_angular_velocity(self, input_angular_velocity):
        if input_angular_velocity < 0:
            return np.fromiter(np.binary_repr(3, width=2), int)
        elif input_angular_velocity > 0:
            return np.fromiter(np.binary_repr(1, width=2), int)
        else:
            return np.fromiter(np.binary_repr(0, width=2), int)

    def cartpole_discretizer(self, input_state):
        op_1 = self.discretize_position(input_state[0])
        op_2 = self.discretize_cart_velocity(input_state[1])
        op_3 = self.discretize_angle(input_state[2])
        op_4 = self.discretize_angular_velocity(input_state[3])
        return [op_1, op_2, op_3, op_4]


def test():
    env = gym.make("CartPole-v0")
    state = env.reset()
    print("Original State: {}".format(state))
    disc = CustomDiscretizer()
    disc_state = disc.cartpole_discretizer(state)
    print("Discretized State: {}".format(disc_state))
    print("END")


if __name__ == "__main__":
    test()
