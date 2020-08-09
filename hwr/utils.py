import numpy
import joblib
import os
import matplotlib.pyplot as plt
import json


def normalize(feature, feature_space):
    """normalizes a feature vector according to the space boundaries."""
    norm_feature = feature - feature_space.low
    norm_feature /= (feature_space.high - feature_space.low)
    norm_feature = norm_feature * 2. - 1.
    return norm_feature


def rescale(feature, feature_space):
    """rescales a normalized feature vector according to the space boundaries."""
    rescaled_feature = (feature + 1.) / 2.
    rescaled_feature *= (feature_space.high - feature_space.low)
    rescaled_feature += feature_space.low
    return rescaled_feature


def load_params(param_file):
    """loads a json parameter file."""
    with open(param_file) as json_file:
        return json.load(json_file)


def visualize_trajectory(filename):
    """Visualizes a complete saved trajectory that was stored 
    during training in filename.
    """

    # loading the trajectory from the pickle file.
    assert os.path.exists(filename), "File: {} does not exist.".format(filename)
    trajectory = joblib.load(filename)

    workspace = trajectory[0]['workspace']
    x = [point['position'][0] for point in trajectory]
    y = [point['position'][1] for point in trajectory]
    goal = trajectory[0]['goal']

    # plotting:
    fig = plt.figure(1)
    plt.matshow(workspace, fignum=1)
    plt.plot(x, y, figure=fig)
    circle = plt.Circle((goal[1], goal[0]), 0.5, figure=fig, color='#d347a8')
    ax = fig.gca()
    ax.add_artist(circle)

    plt.show()
