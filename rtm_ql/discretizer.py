import numpy as np
import gym
import struct


class CustomDiscretizer:
    def __init__(self):
        super().__init__()
    

    def _binarize(self, fp_num, places=32):
        bin_str = []
        fp_num = np.format_float_positional(fp_num, precision=8)
        whole, dec = str(fp_num).split(".")
        dec = 0 if len(str(dec)) == 0 else dec
        
        if whole[0] == "-":
            bin_str.append(1) 
        else:
            bin_str.append(0)

        whole = int(whole[1:]) if whole[0] == "-" else int(whole)
        dec = int(dec)

        # res = bin(whole).lstrip("0b") + "."
        res = bin(whole).lstrip("0b")

        if len(bin(whole).lstrip("0b")) <= places:
            for bin_len_ctr in range(places - len(bin(whole).lstrip("0b")) - 1):
                bin_str.append(0)

        for dec_len_ctr in range(places):
            whole, dec = str(np.format_float_positional((self._decimal_converter(dec))* 2, precision=8)).split(".")
            dec = 0 if len(str(dec)) == 0 else int(dec)
            dec = int(dec)
            res += whole

        for bin_res in res:
            bin_str.append(bin_res)
        bin_str = np.fromiter(bin_str, int)
        return bin_str

    @staticmethod
    def _decimal_converter(num):
        while num > 1:
            num /= 10
        return num


    # @staticmethod
    # def _binarize(float_rep):
    #     [bin_str] = struct.unpack(">Q", struct.pack(">d", float_rep))
    #     return np.fromiter(np.binary_repr(bin_str, width=64), int)

    # @staticmethod
    # def _debinarize(binary_rep):
    #     float_num = np.base_repr(binary_rep)
    #     return float_num

    def cartpole_binarizer(self, input_state, places=16):
        op_1 = self._binarize(input_state[0], places)
        op_2 = self._binarize(input_state[1], places)
        op_3 = self._binarize(input_state[2], places)
        op_4 = self._binarize(input_state[3], places)
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
