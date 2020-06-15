import numpy
import joblib
import os
import matplotlib.pyplot as plt


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
