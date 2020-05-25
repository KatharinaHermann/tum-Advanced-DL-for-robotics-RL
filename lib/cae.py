import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, Conv2DTranspose
import numpy as np


class Encoder(Layer):
    """Encoder of a Convolutional Autoencoder"""

    def __init__(self, pooling, padding, latent_dim, input_dim, conv_filters):

        super().__init__()

        # convolutional layers:
        self._conv1 = Conv2D(filters=conv_filters[0], kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv2 = Conv2D(filters=conv_filters[1], kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv3 = Conv2D(filters=conv_filters[2], kernel_size=(3, 3), padding=padding, activation='relu')

        # pooling layers:
        if pooling == 'max':
            self._pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding=padding)
            self._pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding=padding)
            self._pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding=padding)
        else:
            self._pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding=padding)
            self._pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding=padding)
            self._pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding=padding)

        # flattening:
        self._flatten = Flatten()

        # dense layer:
        self._dense = Dense(units=latent_dim,  activation='relu')

    def call(self, x):
        """forward pass of the encoder"""

        x = self._pool1(self._conv1(x))
        x = self._pool2(self._conv2(x))
        x = self._pool3(self._conv3(x))
        x = self._dense(self._flatten(x))

        return x

class Decoder(Layer):
    """Decoder of a Convolutional Autoencoder."""

    def __init__(self, padding, latent_dim, input_shape, conv_filters):

        super().__init__()
        
        self._input_shape = input_shape
        self._conv_filters = conv_filters

        # calculating the number of units for the dense layer:
        num_pool_layers = len(conv_filters)
        dense_units = input_shape[0] / (4 ** num_pool_layers) * conv_filters[-1]

        # dense layer:
        self._dense = Dense(units=dense_units,  activation='relu')

        # deconvolution layers:
        self._deconv1 = Conv2DTranspose(filters=conv_filters[-1], kernel_size=(3, 3), strides=(2, 2), padding=padding, activation='relu')
        self._deconv2 = Conv2DTranspose(filters=conv_filters[-2], kernel_size=(3, 3), strides=(2, 2), padding=padding, activation='relu')
        self._deconv3 = Conv2DTranspose(filters=conv_filters[-3], kernel_size=(3, 3), strides=(2, 2), padding=padding, activation='relu')

        # convolution layers:
        self._conv1 = Conv2D(filters=conv_filters[-2], kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv2 = Conv2D(filters=conv_filters[-3], kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding=padding, activation='relu')


    def call(self, x):
        """forward pass of the decoder"""

        x = self._dense(x)
        x = self._reshape(x)
        x = self._conv1(self._deconv1(x))
        x = self._conv2(self._deconv2(x))
        x = self._conv3(self._deconv3(x))

        return x



    def _reshape(self, x):
        """for reshaping the dense layer into a tensor that can be given
        to the transpose convolutional layers.
        """
        num_of_layers = len(self._conv_filters)
        height = self._input_shape[0] / (2 ** num_of_layers)
        width = self._input_shape[1] / (2 ** num_of_layers)
        filter_size = self._conv_filters[-1]
        
        return tf.reshape(x, [-1, height, width, filter_size])


class CAE(tf.Module):
    """Convolutional Auto Encoder class"""

    def __init__(self, pooling, padding, latent_dim,
                 input_shape=(32, 32), conv_filters=[4, 8, 16]):
        """Initializing a Convolutional Auto Encoder.
        Args:
            - pooling: type of pooling layer to be used for downsampleing.
                       should be either 'max' or 'average'.
            - padding: type of padding to be used for the convolutional and pooling layers.
                       should be either 'valid' or 'same'.
            - latent_dim: The dimension of the latent space where the input should be projected.
            - input_shape: tuple with the shape of the input
            - conv_filters: list containing the number of filters to be used in the convolutions in the encoder respectively.
                            In the decoder we use the same number of filters but in the exact opposite order.
        """

        assert pooling in ['max', 'average'] , 'pooling should be either \'max\' or \'average\' but received {}'.format(pooling)
        assert padding in ['valid', 'same'] , 'padding should be either \'valid\' or \'same\' but received {}'.format(padding)
        
        super().__init__()

        self._encoder = Encoder(pooling=pooling, padding=padding, latent_dim=latent_dim,
                                input_shape=input_shape, conv_filters=conv_filters)
        self._decoder = Decoder(padding=padding, latent_dim=latent_dim,
                                input_shape=input_shape, conv_filters=conv_filters)

        
    def call(self, x):
        """forward pass of the CAE"""

        x = self._encoder(x)
        x = self._decoder(x)

        return x



        













