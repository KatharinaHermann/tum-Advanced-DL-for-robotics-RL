import numpy as np
import tensorflow as tf
import os

from hwr.cae.cae import CAE


def test_weight_loading():
    model = CAE(pooling='max',
                    latent_dim=16,
                    input_shape=(32, 32),
                    conv_filters=[4, 8, 16])
    model.build(input_shape=(1, 32, 32, 1))
    model.load_weights(filepath='../models/cae/model_num_5_size_8.h5')

    for k, _ in model._get_trainable_state().items():
        k.trainable = False


if __name__ == '__main__':

    test_weight_loading()
    print('All tests have run successfully!')

