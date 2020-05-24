import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten
import numpy as np


class Encoder(Layer):
    """Encoder of a Convolutional Autoencoder"""

    def __init__(self, pooling, padding, latent_dim):

        super().__init__()

        # convolutional layers:
        self._conv1 = Conv2D(filters=4, kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv2 = Conv2D(filters=8, kernel_size=(3, 3), padding=padding, activation='relu')
        self._conv3 = Conv2D(filters=16, kernel_size=(3, 3), padding=padding, activation='relu')

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

    def __init__(self, pooling, padding, latent_dim):

        super().__init__()

        # deconvolution layers:











