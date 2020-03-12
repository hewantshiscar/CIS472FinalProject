"""
First neural network attempt
"""

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
    parser.add_argument('test',     help='Test data file')

    return parser.parse_args()


def main():
	args = handle_args()
	



if __name__ == "__main__":
	main()
	