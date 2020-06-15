import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from hwr.cae.cae import CAE
from hwr.random_workspace import visualize_workspace


"""Visualize the output of the trained Convolutional Autoencoder"""

pooling = 'max'
latent_dim = 16
input_shape = (32, 32)
conv_filters = [4, 8, 16]
model = CAE(
    pooling=pooling,
    latent_dim=latent_dim, 
    input_shape=input_shape,
    conv_filters=conv_filters,
    )
model.build(input_shape=(1, 32, 32, 1))
model.load_weights(filepath='../models/cae/model_num_5_size_8.h5')

# Plot results on an unseen workspace: #
path = os.path.join('../workspaces/', ('ws_' + str(9500) + '.csv'))
x = np.expand_dims(np.loadtxt(path), axis=2).astype('float32')
x = np.expand_dims(x, axis=0)
x = tf.convert_to_tensor(x)

x_hat = tf.cast(model(x) >= 0.5, tf.float32)

fig2 = visualize_workspace(x.numpy()[0, :, :, 0], fignum=2)
fig3 = visualize_workspace(x_hat.numpy()[0, :, :, 0], fignum=3)

plt.show()
