import numpy as np
import gym

class CustomDiscretizer:
    def discretize_position(self, input_position):
        if input_position < 0:
            return np.binary_repr(3).zfill(2)
        elif input_position > 0:
            return np.binary_repr(1).zfill(2)
        else:
            return np.binary_repr(0).zfill(2)

    def discretize_cart_velocity(self, input_cart_velocity):
        if input_cart_velocity < 0:
            return np.binary_repr(3).zfill(2)
        elif input_cart_velocity > 0:
            return np.binary_repr(1).zfill(2)
        else:
            return np.binary_repr(0).zfill(2)


    def discretize_angle(self, input_angle):
        if input_angle < 0:
            return np.binary_repr(3).zfill(2)
        elif input_angle > 0:
            return np.binary_repr(1).zfill(2)
        else:
            return np.binary_repr(0).zfill(2)


    def discretize_angular_velocity(self, input_angular_velocity):
        if input_angular_velocity < 0:
            return np.binary_repr(3).zfill(2)
        elif input_angular_velocity > 0:
            return np.binary_repr(1).zfill(2)
        else:
            return np.binary_repr(0).zfill(2)

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
