import numpy as np


class PointrobotRelabeler:
    """Class that implements a workspace relabeler object for a pointrobot.
    It generates a workspace for a given failed training trajectory for which
    the trajectory would have been a successful one. There are several strategies
    implemented for this purpose:
        - erease: Simply removes the object with which the agent has collided into. 
        - random: Randomly tries to throw in obstacles, and generates a workspace
                 in which the trajectory is feasible, however possibly not very effective.
        - sliding: Slides the obstacles of the original workspace and hence creates a workspace,
                  where the pointrobot was successful and the trajectory was a somewhat effective
                  solution to the workspace.
    """

    def __init__(self, ws_shape=(32, 32), mode='random'):
        """ Initialization of a workspace relabeler for a Pointrobot
        Args:
            - ws_shape: tuple, (ws_height, ws_width)
            - mode: str, determines the mode of the workspace generation.
                    possible values are: 'random', ...
        """

        assert mode in ['random', 'sliding'] , 'mode should be either \'random\' or \'sliding\'. Received {}'.format(mode)

        self._ws_shape = ws_shape
        self._mode = mode


    def relabel(self, workspace, trajectory):
        """creates a new workspace and goal for the given trajectory."""

        if self._mode == 'erease':
            new_ws, new_goal = self._erease_relabel(workspace, trajectory)
        elif self._mode == 'random':
            new_ws, new_goal = self._random_relabel(workspace, trajectory)
        elif self._mode == 'slding':
            new_ws, new_goal = self._sliding_relabel(workspace, trajectory)

        return new_ws, new_goal


    def _erease_relabel(self, workspace, trajectory):
        """Simply removes the obsticle into which the agent has collided."""
        new_ws, new_goal = None, None

        return new_ws, new_goal


    def _random_relabel(self, workspace, trajectory):
        """Relabels a workspace with 'random' method."""
        new_ws, new_goal = None, None

        return new_ws, new_goal


    def _sliding_relabel(self, workspace, trajectory):
        """Relabels a workspace with 'sliding' method"""
        new_ws, new_goal = None, None
    
        return new_ws, new_goal






