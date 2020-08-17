import numpy as np
import math
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


def visualize_trajectory(trajectory, fig, env):
    """Visualizes a trajectory."""

    if trajectory == []:
        fig.clf()
        ax = fig.gca()
        ax.matshow(np.zeros((env.grid_size, env.grid_size), dtype=int))

        return fig

    workspace = trajectory[0]['workspace']
    if env.normalize:
        rescaled_points = [rescale(point['position'], env.pos_bounds) for point in trajectory]
        y = [point[1] for point in rescaled_points]
        x = [point[0] for point in rescaled_points]
        goal = rescale(trajectory[0]['goal'], env.pos_bounds)
        start = rescale(trajectory[0]['position'], env.pos_bounds)
        end = rescale(trajectory[-1]['position'], env.pos_bounds)
    else:
        y = [point['position'][1] for point in trajectory]
        x = [point['position'][0] for point in trajectory]
        goal = trajectory[0]['goal']
        start = trajectory[0]['position']
        end = trajectory[-1]['position']

    # plotting:
    fig.clf()
    plt.plot(x, y, figure=fig, color='#EC66BA', linewidth=2)
    ax = fig.gca()
    cmap = ListedColormap(['#240B3B', '#81BEF7'])
    ax.matshow(workspace, cmap=cmap)

    circle_end = plt.Circle((end[0], end[1]), env.robot_radius, figure=fig, color="w")
    ax.add_artist(circle_end)
    
    circle_start = plt.Circle((start[0], start[1]), 0.3, figure=fig, color="w")
    ax.add_artist(circle_start)

    circle_goal = plt.Circle((goal[0], goal[1]), env.robot_radius, figure=fig, color="#37EC52")
    ax.add_artist(circle_goal)

    return fig


def load_trajectory_from_file(filename):
    """loads a saved trajectory from a pickle file."""
    assert os.path.exists(filename), "File: {} does not exist.".format(filename)
    return joblib.load(filename)

def straight_line_feasible(workspace, start, goal, env):
    goal = rescale(goal, env.pos_bounds)
    start = rescale(start, env.pos_bounds)
    
    pos = start.copy() 
    action = (goal-start) / (np.linalg.norm(goal-start))

    while np.linalg.norm(goal - pos) > env.robot_radius:
        
        x = int(pos[0])
        y = int(pos[1])

        if workspace[y-2: y+3, x-2: x+3].any():
            return False
        pos += action
    
    return True
    

def set_up_benchmark_params(params, key):
    """sets up the parameters according to params["benchmark"]["key"]"""

    for setting in params["benchmark"][key]:
        group = params["benchmark"][key][setting]["group"]
        name = params["benchmark"][key][setting]["name"]
        value = params["benchmark"][key][setting]["value"]
        params[group][name] = value

    return params
