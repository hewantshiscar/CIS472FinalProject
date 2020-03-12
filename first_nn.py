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

def train_nn():
	pass

def predict_place():
	pass

def main():
	args = handle_args()
	((train_x, train_y), varnames) = read_data(args.train)

if __name__ == "__main__":
	main()
	