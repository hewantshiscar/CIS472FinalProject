"""
First neural network attempt
"""

import re
import argparse
import numpy as np
import tensorflow as tf

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
	W = np.array(np.multiply(np.array(np.multiply([0], K)), np.size(D)))
	v = np.array(np.multiply([0], K))
	a = np.array(np.multiply([0], K))
	h = np.array(np.multiply([0], K))
	for i in range(MaxIter):
		G = np.array(np.multiply(np.array(np.multiply([0], K)), np.size(D)))
		g = np.array(np.multiply([0], K))
		for x,y in D:
			for j in range(1, K):
				x2 = np.divide(np.dot(w[i], x), np.multiply(np.dot(w[i], x)), x)
				a[i] = np.dot(W[i], x2)
				h[i] = np.tanh(a[i])
			y2 = np.dot(v, h)
			e = np.subtract(y, y2)
			g = np.subtract(g, np.multiply(e, h))
			for j in range(1, k):
				G[i] = np.subtract(g[i], np.multiply(e, np.multiply(np.multiply(v[i],
					np.subtract(1, np.square(np.tanh(a[i])))), x)))
		W = W - np.multiply(n, G)
		v = np.subtract(v, np.multiply(n, g))

	return W, v

def predict_place():
	pass

def main():
	args = handle_args()
	((train_x, train_y), varnames) = read_data(args.train)
	hi, hey = train_nn(train_x, 0.01, train_y, 100)
	print(hi)
	print()
	print(hey)

if __name__ == "__main__":
	main()
	