import numpy as np
import ctypes as C
import os
from pyTsetlinMachine.tm import _lib

class QRegressionTsetlinMachine():
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=16, weighted_clauses=False, s_range=False, reward=1, gamma=0.9, max_score=100, number_of_actions=2):
		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.rtm = None
		self.reward = reward
		self.gamma = gamma
		self.max_score=max_score
		self.weighted_clauses = weighted_clauses
		self.n_actions = number_of_actions
		if s_range:
			self.s_range = s_range
		else:
			self.s_range = s
		super().__init__()
	
	def __del__(self):
		if self.rtm != None:
			_lib.tm_destroy(self.rtm)

	def fit(self, X, Y, epochs=100, incremental=True):
		number_of_examples = X.shape[0]

		self.max_y = (np.ceil(self.reward * (1 - np.power(self.gamma, self.max_score)) / ((1 - self.gamma)))) if self.gamma<1 else self.max_score
		self.min_y = 1

		if self.rtm == None:
			self.number_of_features = X.shape[1]*2
			self.number_of_patches = 2
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.rtm = _lib.CreateTsetlinMachine(self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
		elif incremental == False:
			_lib.tm_destroy(self.rtm)
			self.rtm = _lib.CreateTsetlinMachine(self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		if self.max_y == self.max_score:
			Ym = np.ascontiguousarray((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
		else:
			Ym = np.ascontiguousarray((Y - self.min_y)/(self.max_y - self.min_y)).astype(np.int32)

		_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//4, 1, 1, self.number_of_features//4, 1, 1, 0)
		print("Fitting X={} to Y={}".format(Xm, Ym))
		_lib.tm_fit_regression(self.rtm, self.encoded_X, Ym, number_of_examples, epochs)

		return

	def predict(self, X, epochs=100):
		number_of_examples = X.shape[0]
		self.max_y = (np.ceil(self.reward * (1 - np.power(self.gamma, self.max_score)) / ((1 - self.gamma)))) if self.gamma<1 else self.max_score
		self.min_y = 1
		if self.rtm == None:
			self.number_of_features = X.shape[1]*2
			self.number_of_patches = 2
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.rtm = _lib.CreateTsetlinMachine(self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//4, 1, 1, self.number_of_features//4, 1, 1, 0)
	
		Y = np.zeros(number_of_examples, dtype=np.int32)
		_lib.tm_predict_regression(self.rtm, self.encoded_X, Y, number_of_examples)
		print("Prediction by tm: {}".format(Y))
		

		# if self.max_y == self.max_score:
		# 	Ym = np.ascontiguousarray((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
		# 	print("Y: {}\tYm: {}".format(Y, Ym))
		# else:
		# 	Ym = np.ascontiguousarray((Y - self.min_y)/(self.max_y - self.min_y)).astype(np.int32)
		# _lib.tm_fit_regression(self.rtm, self.encoded_X, Ym, number_of_examples, epochs)

		return (1.0*(Y[0])*(self.max_y - self.min_y)/(self.T) + self.min_y) if self.max_y == self.max_score else (1.0*(Y[0])*(self.max_y - self.min_y) + self.min_y)
