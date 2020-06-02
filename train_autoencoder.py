import tensorflow as tf
import tensorflow.keras.optimizers as opt
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.data import Dataset
from tensorflow.keras.losses import BinaryCrossentropy
from lib.cae.cae import CAE
from lib.cae.cae_trainer import CAEtrainer, weighted_cross_entropy
from lib.random_workspace import visualize_workspace


"""Train a Convolutional Autoencoder."""

parser = CAEtrainer.get_arguments()
args = parser.parse_args()

input_shape = (args.grid_size, args.grid_size)
model = CAE(args.pooling, args.latent_dim, input_shape, args.conv_filters)
optimizer = opt.Adam(learning_rate=args.learning_rate,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-7)
print('optimizer: {}'.format(optimizer))

# loss function. Calculating the positive weights for it:
mean_obj_num = (args.num_obj_max + 1) / 2
ratio = args.grid_size ** 2 / (mean_obj_num * (args.obj_size_avg ** 2))
beta = ratio
loss_func = weighted_cross_entropy(beta=beta)
print('Loss function: WCE with beta: {}'.format(beta))

trainer = CAEtrainer(CAE=model,
                        optimizer=optimizer,
                        loss_func=loss_func,
                        args=args)

trainer()

# Plot results on an unseen workspace: #

fig = plt.figure(num=1, figsize=(10, 5))
plt.plot(trainer._train_losses)
plt.plot(trainer._val_losses)

# check out the model:

path = os.path.join('workspaces/', ('ws_' + str(args.num_workspaces - 1) + '.csv'))
x = np.expand_dims(np.loadtxt(path), axis=2).astype('float32')
x = np.expand_dims(x, axis=0)
x = tf.convert_to_tensor(x)

x_hat = tf.cast(trainer._CAE(x) >= 0.5, tf.float32)

fig2 = visualize_workspace(x.numpy()[0, :, :, 0], fignum=2)
fig3 = visualize_workspace(x_hat.numpy()[0, :, :, 0], fignum=3)

plt.show()