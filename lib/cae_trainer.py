import tensorflow as tf
import argparse
from cae import CAE


class CAEtrainer():
    """A trainer class for training a Convolutional Autoencoder."""

    def __init__(self, CAE, optimizer, loss_func, args):
        """Initializing a CAE trainer object.
        Args:
            - CAE: a Convolutional Autoencoder. An instance of cae.CAE
            - optimizer: A tensorflow.keras.optimizers instance
            - lass_func: A tensorflow.keras.losses instance
        """

        self._CAE = CAE
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._set_up_from_args(args)

    
    @staticmethod
    def get_arguments():
        """static method for parsing the arguments before instantiating a CAEtrainer"""

        parser = argparse.ArgumentParser()

        # training related
        parser.add_argument('--epochs', type=int, default=200,
                            help='number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size')
        
        # workspace related
        parser.add_argument('--gen_workspace', type=bool, default=False, 
                            help='If gen_workspace==False, saved workspaces are used.')
        parser.add_argument('--workspace_dir', type=str, default='../workspaces',
                            help='folder where the generated workspaces are stored')

        # CAE related:
        parser.add_argument('--pooling', type=str, default='max',
                            help='pooling type of the CAE')
        parser.add_argument('--latent_dim', type=int, default=16,
                            help='latent dimension of the CAE')
        parser.add_argument('--conv_filters', type=int, nargs='+', default=[4, 8, 16],
                            help='number of filters in the conv layers.')
        
        return parser

    
    def _set_up_from_args(self, args):
        """setting up some variables from the parsed arguments."""

        self._epochs = args.epochs
        self._batch_size = args.batch_size
        self._gen_workspace = args.gen_workspace
        self._workspace_dir = args.workspace_dir
        self._pooling = args.pooling
        self._latent_dim = args.latent_dim
        self._conv_filters = args.conv_filters
