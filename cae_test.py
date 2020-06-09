import os
import sys
import numpy as np
import tensorflow as tf

from lib.cae.cae import CAE

model = CAE(pooling='max',
            latent_dim=16,
            input_shape=(32, 32),
            conv_filters=[4, 8, 16])

model.summary()

#model.load_weights(os.path.join('models/cae', 'model_num_5_size_8.h5'))