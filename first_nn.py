"""
First neural network attempt
"""

import re
import argparse
import numpy as np
import tensorflow as tf

n_hidden_1 = 26
n_hidden_2 = 26
n_hidden_3 = 26
n_input = 7 * 6 * 3
n_classes = 7

# Process arguments for perceptron
def handle_args():
    parser = argparse.ArgumentParser(description=
                 'Fit perceptron model and make predictions on test data.')
    parser.add_argument('--rows', type=int,   default=6, help='Rows')
    parser.add_argument('--cols', type=int,   default=7, help='Columns')
    parser.add_argument('--team',  type=int, default=1, help='Team number')
    parser.add_argument('train',    help='Training data file')
    # parser.add_argument('test',     help='Test data file')

    return parser.parse_args()

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  header = f.readline().strip()
  varnames = p.split(header)
  f.close()

  # Read data
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  x = data[:, 0:-1]
  y = data[:, -1]
  return ((x, y), varnames)

def train_nn(D, n, K, MaxIter):
	# Set initial weights equal to zero
	# Two layer network train

	W = np.array(np.multiply([0.2], D[0]))
	v = np.array(np.multiply([0.2], D[0]))
	a = np.array(np.multiply([0], K))
	h = np.array(np.multiply([0], K))
	x2 = np.array(np.multiply([0], K))
	for i in range(MaxIter):
		G = np.array(np.multiply(np.array(np.multiply([0], K)), np.size(D)))
		g = np.array(np.multiply([0], D[0]))
		count = 0

		np.seterr(divide='ignore', invalid='ignore')
		for j in range(len(D[0])):
			x2[j] = np.divide(np.multiply(W[j], D[j]), np.multiply(np.dot(W[j], D[j]), D[j]))
		print(len(x2))
		# x2[np.isnan(x2)] = 0
		# print(len(W))
		# for j in range(len(x2)):
		# 	a[j] = np.dot(W, x2[j])
		# print("LA", len(a))
		# 	h = np.tanh(a)
		# 	print(np.multiply(v, h))
		# 	y2[count] = np.multiply(v, h)
		# 	count += 1
		# e = np.subtract(K, y2)
		# print("E:", len(e))
		# print("H:", len(h))
		# print("g:", len(g))
			# g = np.subtract(g, np.multiply(e, h))

			# 	for j in range(np.size(K)):
			# 		G[i] = np.subtract(g[i], np.multiply(e, np.multiply(np.multiply(v[i],
			# 			np.subtract(1, np.square(np.tanh(a[i])))), x[i])))
		# W = np.subtract(W, np.multiply(n, G))
		# v = np.subtract(v, np.multiply(n, g))
		print(MaxIter - i)
	return W, v

def predict_nn(set, x):
	pass

def main():
	args = handle_args()
	((train_x, train_y), varnames) = read_data(args.train)
	hi, hey = train_nn(train_x, 0.01, train_y, 100)
	print(hi)
	print()
	print(hey)

	# Make predictions, compute accuracy
	# correct = 0
	# for (x,y) in zip(test_x, test_y):
	# 	activation = predict_nn((w,b), x)
	# if activation * y > 0:
	# 	correct += 1
	# acc = float(correct)/len(test_y)
	# print("Accuracy: ",acc)


if __name__ == "__main__":
	main()
	