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
subparsers = parser.add_subparsers(help='sub-command help')

# Create the parser for the "train" command
parser_train = subparsers.add_parser('train', help='Use train command to start training the model')

parser_train.add_argument("--input_path", default='../data',
                    help="The path to the training dataset")
parser_train.add_argument("--dataset_list", default=['NIH'],
                    help="The name(s) of the dataset(s) used")
parser_train.add_argument("--output_path", default='../results',
                    help="The output path")
parser_train.add_argument("--batch_size", type=int, default=128,
                    help="The batch size for training")
parser_train.add_argument("--regenerate", type=str2bool,
                    default=True,
                    help="Create the .npy files for training")
parser_train.add_argument("--image_size", type=int, default=256,
                    help="Image size used for training (only the one dimension needed)")
parser_train.add_argument("--folds", type=int, default=4,
                    help="Number of folds to use for cross validation")
parser_train.add_argument("--seed", type=int, default=2,
                    help="Seed for random number generation")
parser_train.add_argument("--epochs", type=int,
                    default=1000,
                    help="Number of epochs to train the model")
parser_train.add_argument("--learningrate", type=float,
                    default=0.0001,
                    help="The learning rate for the training")
parser_train.add_argument("--channels", type=int,
                    default=6,
                    help="Number of channels of U-Net model")
parser_train.add_argument("--depth", type=int,
                    default=5,
                    help="Depth of U-Net model")
parser_train.add_argument("--print_model", type=str2bool,
                    default=True,
                    help="Print the model's configuration")
parser_train.add_argument("--batchnorm", type=str2bool,
                    default=True,
                    help="Use batch normalization")
parser_train.add_argument("--maxpool", type=str2bool,
                    default=True,
                    help="Use max pooling")
parser_train.add_argument("--residual", type=str2bool,
                    default=True,
                    help="Use residual connections")
parser_train.add_argument("--dropout", type=float,
                    default=0.5,
                    help="Percentage of dropout")
parser_train.add_argument("--verbose", type=int,
                    default=2,
                    help="Specifies verbosity mode (0 = silent, 1= progress bar, 2 = one line per epoch)")



# Create the parser for the "test" command
parser_test = subparsers.add_parser('test', help='Use test command to test the model')

parser_test.add_argument("--test_path", default='../data',
                    help="The path to the testing dataset")
parser_test.add_argument("--output_path", default='../results',
                    help="The output path")
parser_test.add_argument("--volume_name", default='.nii.gz',
                    help="The name of the testing volume")
parser_test.add_argument("--weights_name", default='weights.h5',
                    help="The name of the weights file")
parser_test.add_argument("--image_size", type=int, default=256,
                    help="Image size used for training (only the one dimension needed)")
parser_test.add_argument("--plot_results", type=str2bool,
                    default=True,
                    help="Plot test results")


args = parser.parse_args()