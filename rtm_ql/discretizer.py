import numpy as np
# import gym


class CustomDiscretizer:
    def __init__(self):
        super().__init__()


    def _binned_binarizer(self, fp_num, range_min, range_max, n_bins=15):
        binary_rep = np.zeros(shape=n_bins+1, dtype=int)
        bin_delta = (np.absolute(range_max) + np.absolute(range_min))/n_bins
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(1,n_bins+1):
            bin_min = range_min + (i-1) * bin_delta
            bin_max = range_min + (i) * bin_delta
            if bin_min <= fp_num <= bin_max:
                binary_rep[i] = 1
        return binary_rep

    def _greater_than_binarizer(self, fp_num, n_places=15):
        binary_rep = np.zeros(shape=n_places+1, dtype=int)
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(1, n_places+1):
            if np.absolute(fp_num) > (i-1)/100:
                binary_rep[n_places-i+1] = 1
        return binary_rep


    def cartpole_binarizer(self, input_state, n_bins=15, bin_type="GT"):
        if bin_type == "BN":
            # binned binarizer
            op_1 = self._binned_binarizer(input_state[0], 0, 3, n_bins)
            op_2 = self._binned_binarizer(input_state[1], 0, 500, n_bins)
            op_3 = self._binned_binarizer(input_state[2], 0, 42, n_bins)
            op_4 = self._binned_binarizer(input_state[3], 0, 500, n_bins)
        else:
            # greater_than binarizer:
            op_1 = self._greater_than_binarizer(input_state[0], n_places=n_bins)
            op_2 = self._greater_than_binarizer(input_state[1], n_places=n_bins)
            op_3 = self._greater_than_binarizer(input_state[2], n_places=n_bins)
            op_4 = self._greater_than_binarizer(input_state[3], n_places=n_bins)
        return [op_1, op_2, op_3, op_4]


# def test():
#     env = gym.make("CartPole-v0")
#     state = env.reset()
#     print("Original State: {}".format(state))
#     disc = CustomDiscretizer()
#     disc_state = disc.cartpole_binarizer(state)
#     print("Discretized State: {}".format(disc_state))
#     print("END")


# if __name__ == "__main__":
#     test()
