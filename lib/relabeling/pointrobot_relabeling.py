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
                    possible values are: 'erease', 'random', 'sliding'
        """

        assert mode in ['erease', 'random', 'sliding'] ,\
            'mode should be either \'erease\', \'random\' or \'sliding\'. Received {}'.format(mode)

        self._ws_shape = ws_shape
        self._mode = mode


    def relabel(self, trajectory, env):
        """creates a new workspace and goal for the given trajectory."""

        if self._mode == 'erease':
            relabeled_trajectory = self._erease_relabel(trajectory, env)
        elif self._mode == 'random':
            relabeled_trajectory = self._random_relabel(trajectory, env)
        elif self._mode == 'slding':
            relabeled_trajectory = self._sliding_relabel(trajectory, env)

        return relabeled_trajectory


    def _erease_relabel(self, trajectory, env):
        """Simply removes the obsticle into which the agent has collided.
        If the episonde has ended unsuccessfully, because the robot has collided 
        into an obstacle (or possibly more than one obstacles at the same time.),
        these obstacles are removed.
        If the episode has ended, because the agent has left the workspace, (that is,
        no obstacle has been found, with thich it has collided.), the workspace, the trajectory
        points and the goal is shifted away from the boarder with a random value.
        """

        workspace = trajectory[0]['workspace']

        # find the entries of the matrix where the robot has collided.:
        obstacle_entries = self._find_collision_entries(trajectory, env)

        if obstacle_entries:
            # if the robot really has collided.
            for obstacle_entry in obstacle_entries:
                workspace = self._remove_obstacle(workspace=workspace, obstacle_entry=obstacle_entry)
            for data_point in trajectory:
                data_point['workspace'] = workspace
        else:
            # if the obstacle has just left the workspace without collision.
            # choosing a distance with which the ws and the trajectory will be shifted away from the boarder:
            shift_distance = np.random.randint(low=1, high=4)
            trajectory = self._shift_from_boarder(trajectory=trajectory,
                                     env=env,
                                     shift_distance=shift_distance)

        # filling in the relabeled trajectory with the new goal state:
        new_goal = trajectory[-1]['position']
        for data_point in trajectory:
            data_point['goal'] = new_goal
        trajectory[-1]['reward'] = 1
        trajectory[-1]['done'] = True

        return trajectory


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
        distances = list(range(int(radius) + 1))
        distances.append(radius)

        directions_to_check = [np.array([1, 0]), 
                               np.array([math.sqrt(2) / 2, math.sqrt(2) / 2]), 
                               np.array([0, 1]),
                               np.array([-math.sqrt(2) / 2, math.sqrt(2) / 2]), 
                               np.array([-1, 0]),
                               np.array([-math.sqrt(2) / 2, -math.sqrt(2) / 2]), 
                               np.array([0, -1]),
                               np.array([math.sqrt(2) / 2, -math.sqrt(2) / 2])]
        
        collision_entries = []
        for direction in directions_to_check:
            for distance in distances:
                pos = last_pos + distance * direction
                entry = (int(pos[0]), int(pos[1]))
                if workspace[entry] == 1:
                    collision_entries.append(entry)

        return collision_entries


    def _shift_from_boarder(self, trajectory, env, shift_distance):
        """Shifts the total workspace and trajectory so that it does not end at the boarder of the workspace."""

        shift_distance = int(shift_distance)
        last_pos = trajectory[-1]['position']
        workspace = trajectory[0]['workspace']
        radius = env.radius

        # shifting the workspace. With the if conditions it is decided to which wall is the robot near:
        if last_pos[0] <= radius:
            new_workspace = np.zeros_like(workspace)
            new_workspace[shift_distance: , :] = workspace[ :(-shift_distance), :]
            workspace = new_workspace
            for point in trajectory:
                point['position'][0] += shift_distance
                point['goal'][0] += shift_distance

        elif last_pos[0] >= (workspace.shape[0] - 1 - radius):
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :(-shift_distance), :] = workspace[shift_distance: , :]
            workspace = new_workspace
            for point in trajectory:
                point['position'][0] -= shift_distance
                point['goal'][0] -= shift_distance

        if last_pos[1] <= radius:
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :, shift_distance:] = workspace[ :, :(-shift_distance)]
            workspace = new_workspace
            for point in trajectory:
                point['position'][1] += shift_distance
                point['goal'][1] += shift_distance
                
        elif last_pos[1] >= (workspace.shape[1] - 1 - radius):
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :, :(-shift_distance)] = workspace[ :, shift_distance: ]
            workspace = new_workspace
            for point in trajectory:
                point['position'][1] -= shift_distance
                point['goal'][1] -= shift_distance

        for point in trajectory:
            point['workspace'] = workspace

        # remove points from trajectory which have out of bunds coordinates after shifting:
        trajectory_to_return = []
        for point in trajectory:
            if (point['position'] > radius).all() and\
                (point['position'] < workspace.shape[0] - 1 - radius).all():
                trajectory_to_return.append(point)

        return trajectory_to_return