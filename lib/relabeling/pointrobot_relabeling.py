import numpy as np
import math


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

        assert mode in ['erease', 'random', 'sliding'] , 'mode should be either \'erease\', \'random\' or \'sliding\'. Received {}'.format(mode)

        self._ws_shape = ws_shape
        self._mode = mode


    def relabel(self, trajectory, env):
        """creates a new workspace and goal for the given trajectory."""

        if self._mode == 'erease':
            relabeled_trajectory = self._erease_relabel(trajectory)
        elif self._mode == 'random':
            relabeled_trajectory = self._random_relabel(trajectory)
        elif self._mode == 'slding':
            relabeled_trajectory = self._sliding_relabel(trajectory)

        return relabeled_trajectory


    def _erease_relabel(self, trajectory, env):
        """Simply removes the obsticle into which the agent has collided."""

        relabeled_trajectory = trajectory
        workspace = trajectory[0]['workspace']
        last_pos = trajectory[-1]['position']
        # find the entries of the matrix where the robot has collided.:
        obstacle_entries = self._find_collision_entries(trajectory, env)
        for obstacle_entry in obstacle_entries:
            workspace = self._remove_obstacle(workspace=workspace, obstacle_entry=obstacle_entry)

        for data_point in trajectory:
            data_point['workspace'] = workspace
            data_point['goal'] = last_pos

        # giving the last state the reward +1:
        trajectory[-1]['reward'] = 1
        trajectory[-1]['done'] = True

        return relabeled_trajectory


    def _random_relabel(self, trajectory, env):
        """Relabels a workspace with 'random' method."""
        relabeled_trajectory = trajectory

        return relabeled_trajectory


    def _sliding_relabel(self, trajectory, env):
        """Relabels a workspace with 'sliding' method"""
        relabeled_trajectory = trajectory
    
        return relabeled_trajectory


    def _remove_obstacle(self, workspace, obstacle_entry):
        """reomoves an obsticle from a workspaces.
        start_position should represent an entry of the workspace as a tuple. 
        The method removes an obstacle, which's part is start position.
        """

        # queue of cells in the workspace which are part of the obsticle and were not eraesed yet.
        queue = []  
        queue.append(obstacle_entry)
        # initializing neighboring cells list:
        neighbours = [(0, 0) for _ in range(4)]

        max_index = workspace.shape[0] - 1

        # placing 0 instead of 1 for start_position:
        workspace[obstacle_entry] = 0

        while queue:
            act_entry = queue.pop()
            # places to check:
            neighbours[0] = (act_entry[0] + 1, act_entry[1]) if (act_entry[0] + 1) <= max_index else (max_index, act_entry[1])
            neighbours[1] = (act_entry[0], act_entry[1] + 1) if (act_entry[1] + 1) <= max_index else (act_entry[0], max_index)
            neighbours[2] = (act_entry[0] - 1, act_entry[1]) if (act_entry[0] - 1) >= 0 else (0, act_entry[1])
            neighbours[3] = (act_entry[0], act_entry[1] - 1) if (act_entry[1] - 1) >= 0 else (act_entry[0], 0)
            
            # check the neighbouring entries and place the ones which are part of the obstacle to the queue:
            for entry in neighbours:
                if workspace[entry] == 1:
                    workspace[entry] = 0
                    queue.append(entry)

        return workspace


    def _find_collision_entries(self, trajectory, env):
        """Finds the entries of the matrix, where the agent has collided into an obstacle."""

        workspace = trajectory[0]['workspace']
        last_pos = trajectory[-1]['position']
        radius = env.radius
        # range of distances to check in every direction:
        # (it is possible that the radius of the robot is bigger than the grid size, this is why this is necessary.)
        distances = list(range(int(radius)))
        distances.append(radius)

        directions_to_check = [np.array([1, 0]), 
                               np.array([math.sqrt(2), math.sqrt(2)]), 
                               np.array([0, 1]),
                               np.array([-math.sqrt(2), math.sqrt(2)]), 
                               np.array([-1, 0]),
                               np.array([-math.sqrt(2), -math.sqrt(2)]), 
                               np.array([0, -1]),
                               np.array([math.sqrt(2), -math.sqrt(2)])]
        
        collision_entries = []
        for direction in directions_to_check:
            for distance in distances:
                pos = last_pos + distance * direction
                entry = (int(pos[0]), int(pos[1]))
                if workspace[entry] == 1:
                    collision_entries.append(entry)

        return collision_entries



