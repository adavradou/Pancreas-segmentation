"""
Created on Sun Jul 25 2021
@author: Agapi Davradou

This module reads the parameters from the command line.
"""


import argparse
from numpy.distutils.fcompiler import str2bool


parser = argparse.ArgumentParser(
    description="2D U-Net model.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_path", default='../data',
                    help="The path to the training dataset")
parser.add_argument("--batch_size", type=int, default=128,
                    help="The batch size for training")
parser.add_argument("--regenerate", type=str2bool,
                    default=True,
                    help="Create the .npy files for training")
parser.add_argument("--image_size", type=int, default=256,
                    help="Image size used for training (only the one dimension needed)")
parser.add_argument("--folds", type=int, default=4,
                    help="Number of folds to use for cross validation")
parser.add_argument("--seed", type=int, default=2,
                    help="Seed for random number generation")
parser.add_argument("--epochs", type=int,
                    default=1000,
                    help="Number of epochs to train the model")
parser.add_argument("--learningrate", type=float,
                    default=0.0001,
                    help="The learning rate for the training")
parser.add_argument("--channels", type=int,
                    default=6,
                    help="Number of channels of U-Net model")
parser.add_argument("--depth", type=int,
                    default=5,
                    help="Depth of U-Net model")
parser.add_argument("--print_model", type=str2bool,
                    default=True,
                    help="Print the model's configuration")
parser.add_argument("--batchnorm", type=str2bool,
                    default=True,
                    help="Use batch normalization")
parser.add_argument("--maxpool", type=str2bool,
                    default=True,
                    help="Use max pooling")
parser.add_argument("--residual", type=str2bool,
                    default=True,
                    help="Use residual connections")
parser.add_argument("--dropout", type=float,
                    default=0.5,
                    help="Percentage of dropout")
parser.add_argument("--verbose", type=int,
                    default=2,
                    help="Specifies verbosity mode (0 = silent, 1= progress bar, 2 = one line per epoch)")

args = parser.parse_args()