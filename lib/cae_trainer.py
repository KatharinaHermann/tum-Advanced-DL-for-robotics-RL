import tensorflow as tf
import tensorflow.keras.optimizers as opt
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import random_workspace

from tensorflow.data import Dataset
from tensorflow.keras.losses import BinaryCrossentropy
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
        self._train_losses, self._val_losses = [], []
        self._train_accs, self._val_accs = [], []
        self._epoch_train_losses, self._epoch_val_losses = [], []
        self._epoch_train_accs, self._epoch_val_accs = [], []
        self._set_up_from_args(args)


    def __call__(self):
        """Training loop for CAE.
        It first either loads pre-generated or generates workspaces to train on.
        Then it trains the CAE.
        """

        if self._gen_workspace:
            self._generate_new_workspaces()
        
        self._load_workspaces()

        best_val_loss = 1e6
        best_val_acc = 0

        # Training Loop #
        print('-' * 5 + 'TRAINING HAS STARTED' + '-' * 5)
        for epoch in range(self._epochs):
            self._epoch_train_losses, self._epoch_val_losses = [], []
            self._epoch_train_accs, self._epoch_val_accs = [], []

            for X in self._train_data:
                self._train_on_batch(X)
            
            for X in self._val_data:
                self._validate_on_batch(X)

            # losses and accuracy of the epoch:
            self._train_losses.append(np.mean(self._epoch_train_losses))
            self._train_accs.append(np.mean(self._epoch_train_accs))
            self._val_losses.append(np.mean(self._epoch_val_losses))
            self._val_accs.append(np.mean(self._epoch_val_accs))

            print('EPOCH {}'.format(epoch))
            print('Train loss / Val loss : {} / {}'.format(self._train_losses[-1], self._val_losses[-1]))
            print('Train acc / Val acc : {} / {}'.format(self._train_accs[-1], self._val_accs[-1]))
        
            # saving the model, if it is the best so far:
            if self._val_losses[-1] < best_val_loss:
                best_val_loss = self._val_losses[-1]
                self._save_model()
            
            if self._val_accs[-1] >= best_val_acc:
                best_val_acc = self._val_accs[-1]
                #self._save_model()

        print('-' * 5 + 'TRAINING HAS ENDED' + '-' * 5)
        print('best validation loss: {}'.format(best_val_loss))
        print('best validation accuracy: {}'.format(best_val_acc))

        # loading the best model:
        self._CAE.load_weights(os.path.join(self._model_dir, 'model.h5'))


    #@tf.function
    def _train_on_batch(self, X):
        """carries out a gradient step on a mini-batch."""
        with tf.GradientTape() as tape:
            out = self._CAE(X)
            loss = self._loss_func(X, out) 

        self._epoch_train_losses.append(loss.numpy())
        self._epoch_train_accs.append(self._calc_accuracy(X, out))

        grads = tape.gradient(loss, self._CAE.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self._CAE.trainable_weights))

    #@tf.function
    def _validate_on_batch(self, X):
        """carries out a validation step on a mini-batch."""

        out = self._CAE(X)
        loss = self._loss_func(X, out)

        self._epoch_val_losses.append(loss.numpy())
        self._epoch_val_accs.append(self._calc_accuracy(X, out))

        
    def _calc_accuracy(self, X, out):
        """calculates the accuracy for a mini-batch."""

        # if an entry is bigger than 0.5, it is considered as 1:
        out_rounded = tf.cast(out >= 0.5, tf.float32)
        metric = tf.keras.metrics.Accuracy()
        _ = metric.update_state(X, out_rounded)

        return metric.result().numpy()


    def _save_model(self):
        """checking whether the path where the model has to be saved exists or not and sace the model."""

        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        file_name = 'model.h5'
        path = os.path.join(self._model_dir, file_name)
        self._CAE.save_weights(path)

        print('model was saved to ' + self._model_dir)


    def _generate_new_workspaces(self):
        """Generating new workspaces."""

        # creating the workspace saving folder if it does not exist yet:
        if not os.path.exists(self._workspace_dir):
            os.mkdir(self._workspace_dir)

        for i in range(self._num_workspaces):
            workspace = random_workspace.random_workspace(self._grid_size, self._num_obj_max, self._obj_size_avg)

            file_name = 'ws_' + str(i) + '.csv'
            path = os.path.join(self._workspace_dir, file_name)
            np.savetxt(path, workspace)

        print('generated {} workspaces and saved them into {}'.format(self._num_workspaces, self._workspace_dir))
        

    def _load_workspaces(self):
        """Loadeing pre-saved workspaces."""
        
        # list of file names in the workspace directory:
        files = [os.path.join(self._workspace_dir, name) for name in os.listdir(self._workspace_dir)]
        num_of_files = len(files)

        # read in either self._num_workspaces or num_of_files number of workspaces, whichewer is smaller:
        num_to_read = num_of_files if num_of_files < self._num_workspaces else self._num_workspaces
        
        # reading in the workspaces into a list of np.arrays:
        workspace_list = []
        for i in range(num_to_read):
            path = files[i]
            # loading and adding an extra dimension to the numpy array.
            # neede because the Conv2D layer waits for shape (batch_size, height, width, channel_size)
            # batch size will be added by the tf.data.Dataset object.
            workspace = np.expand_dims(np.loadtxt(path), axis=2).astype('float32')
            workspace_list.append(workspace)

        # creating the Datasets from the list:
        val_size = int(self._num_workspaces * 0.2)
        test_size = int(self._num_workspaces * 0.2)
        train_size = self._num_workspaces - val_size - test_size

        self._train_data = Dataset.from_tensor_slices(workspace_list[ :train_size]).batch(self._batch_size)
        self._val_data = Dataset.from_tensor_slices(workspace_list[train_size : (train_size + val_size)]).batch(self._batch_size)
        self._test_data = Dataset.from_tensor_slices(workspace_list[(train_size + val_size): ]).batch(self._batch_size)

        # setting up shuffleing for training if it is needed:
        if self._train_shuffle:
            self._train_data = self._train_data.shuffle(buffer_size=train_size)


    @staticmethod
    def get_arguments():
        """static method for parsing the arguments before instantiating a CAEtrainer"""

        parser = argparse.ArgumentParser()

        # training related
        parser.add_argument('--epochs', type=int, default=200,
                            help='number of epochs to train. default: 200')
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='number of epochs to train. default: 1e-3')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size. default: 32')
        parser.add_argument('--train_shuffle', type=bool, default=True,
                            help='Whether to shuffle or not during training. default: True')
        #parser.add_argument('--pos_weight', type=float, default=2,
        #                    help='weight for positive weighting in cross entropy loss. default: 2')
        parser.add_argument('--model_dir', type=str, default='../models/cae',
                            help='directory to save the best trained model. default: ../models/cae')
        
        # workspace related
        parser.add_argument('--gen_workspace', type=bool, default=False, 
                            help='If gen_workspace==False, saved workspaces are used. default: False')
        parser.add_argument('--workspace_dir', type=str, default='../workspaces',
                            help='folder where the generated workspaces are stored. default: ../workspaces')
        parser.add_argument('--num_workspaces', type=int, default=1000,
                            help='number of workspaces to use for training. default: 1000')
        parser.add_argument('--grid_size', type=int, default=32,
                            help='number of grid points in the workspace. default: 32')
        parser.add_argument('--num_obj_max', type=int, default=5,
                            help='maximum number of objects in the workspace. default: 5')
        parser.add_argument('--obj_size_avg', type=int, default=8,
                            help='average size of the objects in the workspace. default: 8')

        # CAE related:
        parser.add_argument('--pooling', type=str, default='max',
                            help='pooling type of the CAE. default: max')
        parser.add_argument('--latent_dim', type=int, default=16,
                            help='latent dimension of the CAE. default: 16')
        parser.add_argument('--conv_filters', type=int, nargs='+', default=[4, 8, 16],
                            help='number of filters in the conv layers. default: [4, 8, 16]')
        
        return parser

    
    def _set_up_from_args(self, args):
        """setting up some variables from the parsed arguments."""

        # training related:
        self._epochs = args.epochs                      # number of training epochs
        self._batch_size = args.batch_size              # batch size
        self._train_shuffle = args.train_shuffle        # whether to shuffle or not during training.
        self._model_dir = args.model_dir                # directory to save the best model during training.
        # workspace related
        self._gen_workspace = args.gen_workspace        # whether to newly generate workspaces (True) or use saved ones (False)
        self._workspace_dir = args.workspace_dir        # folder from which saved workspaces can be loaded
        self._num_workspaces = args.num_workspaces       # numbr of worksapces to train on
        self._grid_size = args.grid_size                # number of grid points in the workspace
        self._num_obj_max = args.num_obj_max            # maximum number of objects in the workspace
        self._obj_size_avg = args.obj_size_avg          # average size of the objects in the workspace


def weighted_cross_entropy(beta):
    """returns a weighted cross entropy loss function
    weighted by beta.
    """

    def loss(y_true, y_pred):
        # getting logits from sigmoid output:
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), (1 - tf.keras.backend.epsilon()))
        y_logits = tf.math.log(y_pred / (1 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_logits,
                                                        pos_weight=beta)
        
        return tf.reduce_mean(loss)

    return loss


if __name__ == '__main__':

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
    path = '../workspaces/ws_' + str(args.num_workspaces - 1) + '.csv'
    x = np.expand_dims(np.loadtxt(path), axis=2).astype('float32')
    x = np.expand_dims(x, axis=0)
    x = tf.convert_to_tensor(x)

    x_hat = tf.cast(trainer._CAE(x) >= 0.5, tf.float32)

    fig2 = random_workspace.visualize_workspace(x.numpy()[0, :, :, 0], fignum=2)
    fig3 = random_workspace.visualize_workspace(x_hat.numpy()[0, :, :, 0], fignum=3)
    
    plt.show()
