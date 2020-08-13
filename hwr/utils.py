import numpy as np
import math
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


def get_random_params(params):
    """gets random hyperparams according to the info stored in params["hyper_tuning"]"""

    for key in params["hyper_tuning"]:
        if "param" in key:
            assert params["hyper_tuning"][key]["scale"] in ["linear", "log"]
            assert params["hyper_tuning"][key]["low_bound"] < params["hyper_tuning"][key]["high_bound"]
            group = params["hyper_tuning"][key]["group"]
            if params["hyper_tuning"][key]["scale"] == "linear":
                for name in params["hyper_tuning"][key]["name"]:
                    random_param = np.random.uniform(
                        low=params["hyper_tuning"][key]["low_bound"],
                        high=params["hyper_tuning"][key]["high_bound"])
                    params[group][name] = random_param
            else:
                for name in params["hyper_tuning"][key]["name"]:
                    low = math.log10(params["hyper_tuning"][key]["low_bound"])
                    high = math.log10(params["hyper_tuning"][key]["high_bound"])
                    random_param = np.random.uniform(
                        low=low,
                        high=high)
                    params[group][name] = math.pow(10, random_param)

    return params


def export_params(params, info_file):
    """"exports the actual hyperparams to a json file."""
    for key in params["hyper_tuning"]:
        if "param" in key:
            group = params["hyper_tuning"][key]["group"]
            for name in params["hyper_tuning"][key]["name"]:
                with open(info_file, 'a') as f:
                    param = params[group][name]
                    f.write("{}/{} : {}".format(group, name, param) + '\n')


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
