import numpy as np
import gym
import struct


class CustomDiscretizer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def _binarize(float_rep):
        [bin_str] = struct.unpack(">Q", struct.pack(">d", float_rep))
        return str(f"{bin_str:064b}")

    @staticmethod
    def _debinarize(binary_rep):
        float_num = int(binary_rep, 2).to_bytes(8, byteorder="big")
        return struct.unpack('>d', float_num)[0]

    @staticmethod
    def _binarize(float_rep):
        [bin_str] = struct.unpack(">Q", struct.pack(">d", float_rep))
        return np.fromiter(np.binary_repr(bin_str, width=64), int)

    @staticmethod
    def _debinarize(binary_rep):
        float_num = np.base_repr(binary_rep)
        return float_num

    def cartpole_binarizer(self, input_state):
        op_1 = self._binarize(input_state[0])
        op_2 = self._binarize(input_state[1])
        op_3 = self._binarize(input_state[2])
        op_4 = self._binarize(input_state[3])
        return [op_1, op_2, op_3, op_4]

    def cartpole_debinarizer(self, input_state):
        op_1 = self._debinarize(input_state[0])
        op_2 = self._debinarize(input_state[1])
        op_3 = self._debinarize(input_state[2])
        op_4 = self._debinarize(input_state[3])
        return [op_1, op_2, op_3, op_4]


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
