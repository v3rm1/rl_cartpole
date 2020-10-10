import numpy as np
import gym


class CustomDiscretizer:
    def __init__(self):
        super().__init__()
    

    def _binarize(self, fp_num, range_min, range_max, n_bins=15):
        binary_rep = np.zeros(shape=n_bins+1, dtype=int)
        bin_delta = (np.absolute(range_max) + np.absolute(range_min))/n_bins
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(n_bins):
            bin_min = range_min + i * bin_delta
            bin_max = range_min + (i+1) * bin_delta
            if bin_min <= fp_num <= bin_max:
                binary_rep[i] = 1
        return binary_rep

    def cartpole_binarizer(self, input_state, n_bins=16):
        op_1 = self._binarize(input_state[0], -3, 3, n_bins)
        op_2 = self._binarize(input_state[1], -500, 500, n_bins)
        op_3 = self._binarize(input_state[2], -42, 42, n_bins)
        op_4 = self._binarize(input_state[3], -500, 500, n_bins)
        return [op_1, op_2, op_3, op_4]

    # def cartpole_debinarizer(self, input_state):
    #     op_1 = self._debinarize(input_state[0])
    #     op_2 = self._debinarize(input_state[1])
    #     op_3 = self._debinarize(input_state[2])
    #     op_4 = self._debinarize(input_state[3])
    #     return [op_1, op_2, op_3, op_4]


def test():
    env = gym.make("CartPole-v0")
    state = env.reset()
    print("Original State: {}".format(state))
    disc = CustomDiscretizer()
    disc_state = disc.cartpole_binarizer(state)
    print("Discretized State: {}".format(disc_state))
    # print("Discretized State: {}".format(disc.cartpole_debinarizer(disc_state)))
    print("END")


if __name__ == "__main__":
    test()
