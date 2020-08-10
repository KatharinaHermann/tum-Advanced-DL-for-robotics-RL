import numpy as np
import math

from hwr.utils import normalize, rescale


class PointrobotRelabeler:
    """Class that implements a workspace relabeler object for a pointrobot.
    It generates a workspace for a given failed training trajectory for which
    the trajectory would have been a successful one. There are several strategies
    implemented for this purpose:
        In all cases if the episode has ended because the agent has left the workspace,
        the workspace, the trajectory points and the goal is shifted away from the boarder
        with some value. (Possibly a random value)
        - erease: Simply removes the object with which the agent has collided. 
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

        # rescale the trajectory if normalization is used in the environment:
        if env.normalize:
            trajectory = self._rescale_trajectory(trajectory, env)

        if self._mode == 'erease':
            relabeled_trajectory = self._erease_relabel(trajectory, env)
        elif self._mode == 'random':
            relabeled_trajectory = self._random_relabel(trajectory, env)
        elif self._mode == 'slding':
            relabeled_trajectory = self._sliding_relabel(trajectory, env)

        # normalize the trajectory if normalization is used in the environment:
        if env.normalize:
            relabeled_trajectory = self._normalize_trajectory(relabeled_trajectory, env)

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

        if trajectory[-1]["reward"] == env.collision_reward:

            # find the entries of the matrix where the robot has collided.:
            obstacle_entries = self._find_collision_entries(trajectory, env)

            if obstacle_entries:
                # if the robot has collided.
                for obstacle_entry in obstacle_entries:
                    workspace = self._remove_obstacle(workspace=workspace, obstacle_entry=obstacle_entry)
                for data_point in trajectory:
                    data_point['workspace'] = workspace
            else:
                # if the obstacle has just left the workspace without collision.
                # choosing a distance with which the ws and the trajectory will be shifted away from the boarder:
                erase_length = np.random.randint(low=1, high=4)
                trajectory = self._erase_from_boarder(trajectory=trajectory,
                                        env=env, erase_length=erase_length)

                #shift_distance = np.random.randint(low=1, high=4)
                #trajectory = self._shift_from_boarder(trajectory=trajectory,
                #                        env=env,
                #                        shift_distance=shift_distance)

            # add new goal state to the trajectory:
            if len(trajectory) != 0:
                return self._set_new_goal(trajectory, env)
            else:
                return []
        else:
            return self._set_new_goal(trajectory, env)


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
        radius = env.robot_radius
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
                entry = (int(pos[1]), int(pos[0]))
                if workspace[entry] == 1:
                    collision_entries.append(entry)

        return collision_entries

    def _erase_from_boarder(self, trajectory, env, erase_length):
        """Erases the last part of the trajectory, that crashed with the boarder."""
        erase_length = int(erase_length)

        trajectory_to_return = []
        for point in trajectory[:(-erase_length)]:
            trajectory_to_return.append(point)

        return trajectory_to_return


    def _shift_from_boarder(self, trajectory, env, shift_distance):
        """Shifts the total workspace and trajectory so that it does not end at the boarder of the workspace."""

        shift_distance = int(shift_distance)
        last_pos = trajectory[-1]['position']
        workspace = trajectory[0]['workspace']
        radius = env.robot_radius

        # shifting the workspace. With the if conditions it is decided to which wall is the robot near:
        if last_pos[1] <= radius:
            new_workspace = np.zeros_like(workspace)
            new_workspace[shift_distance: , :] = workspace[ :(-shift_distance), :]
            workspace = new_workspace
            for point in trajectory:
                point['position'][1] += shift_distance
                point['next_position'][1] += shift_distance
                point['goal'][1] += shift_distance

        elif last_pos[1] >= (workspace.shape[0] - 1 - radius):
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :(-shift_distance), :] = workspace[shift_distance: , :]
            workspace = new_workspace
            for point in trajectory:
                point['position'][1] -= shift_distance
                point['next_position'][1] -= shift_distance
                point['goal'][1] -= shift_distance

        if last_pos[0] <= radius:
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :, shift_distance:] = workspace[ :, :(-shift_distance)]
            workspace = new_workspace
            for point in trajectory:
                point['position'][0] += shift_distance
                point['next_position'][0] += shift_distance
                point['goal'][0] += shift_distance
                
        elif last_pos[0] >= (workspace.shape[1] - 1 - radius):
            new_workspace = np.zeros_like(workspace)
            new_workspace[ :, :(-shift_distance)] = workspace[ :, shift_distance: ]
            workspace = new_workspace
            for point in trajectory:
                point['position'][0] -= shift_distance
                point['next_position'][0] -= shift_distance
                point['goal'][0] -= shift_distance

        for point in trajectory:
            point['workspace'] = workspace

        # remove points from trajectory which have out of bunds coordinates after shifting:
        trajectory_to_return = []
        for point in trajectory:
            if (point['position'] > radius).all() and\
                    (point['position'] < workspace.shape[0] - 1 - radius).all() and\
                    (point['next_position'] > radius).all() and\
                    (point['next_position'] < workspace.shape[0] - 1 - radius).all():
                trajectory_to_return.append(point)

        return trajectory_to_return


    def _set_new_goal(self, trajectory, env):
        """Sets a new goal for the trajectory.
        The goal is going to be slightly closer to the last state than the radius of the robot.
        It also has to be assured, that the goal lies closer than the radius only
        to the last state of the trajectory.
        If the algorithm can not find a sufficient goal respecting the previous constraint,
        it will return an empty trajectory.
        (In this case any goal would be misleading to the agent during learning.)
        """

        max_iters = 1000
        current_iter = 0
        feasible_goal = False

        last_pos = trajectory[-1]['position']
        # the first goal candidate is in the direction of the last motion:
        if len(trajectory) >= 2:
            last_last_pos = trajectory[-2]['position']
            goal_direction_vect = (last_pos - last_last_pos) / (np.linalg.norm(last_pos - last_last_pos)) * env.robot_radius * 0.99
        else:
            goal_direction_vect = np.random.uniform(low=-1, high=1, size=(2,))
            goal_direction_vect = goal_direction_vect / np.linalg.norm(goal_direction_vect) * env.robot_radius * 0.99
        new_goal = last_pos + goal_direction_vect

        # matrix containing every state except for the last one for effectively calculate distance from them.
        every_other_pos = np.zeros((len(trajectory)-1, last_pos.shape[0]))
        for i, point in enumerate(trajectory[ : -1]):
            every_other_pos[i, :] = point['position']

        while not feasible_goal:
            # checking the distance from every goal
            distances = np.linalg.norm(every_other_pos - new_goal, axis=1)

            if (distances < env.robot_radius).any():
                # new random goal to try:
                goal_direction_vect = np.random.uniform(low=-1, high=1, size=(2,))
                goal_direction_vect = goal_direction_vect / np.linalg.norm(goal_direction_vect) * env.robot_radius * 0.99
                new_goal = last_pos + goal_direction_vect
            else:
                feasible_goal = True
                break

            if current_iter >= max_iters:
                break               
            current_iter += 1

        if feasible_goal:
            for data_point in trajectory:
                data_point['goal'] = new_goal
            trajectory[-1]['reward'] = env.goal_reward
            trajectory[-1]['done'] = True
        else:
            trajectory = []

        return trajectory


    def _rescale_trajectory(self, trajectory, env):
        """rescales a trajectory to the original range."""
        for point in trajectory:
            point["position"] = rescale(point["position"], env.pos_bounds)
            point["next_position"] = rescale(point["next_position"], env.pos_bounds)
            point["goal"] = rescale(point["goal"], env.pos_bounds)
            point["action"] = rescale(point["action"], env.action_bounds)

        return trajectory


    def _normalize_trajectory(self, trajectory, env):
        """normalizes a trajectory."""
        for point in trajectory:
            point["position"] = normalize(point["position"], env.pos_bounds)
            point["next_position"] = normalize(point["next_position"], env.pos_bounds)
            point["goal"] = normalize(point["goal"], env.pos_bounds)
            point["action"] = normalize(point["action"], env.action_bounds)

        return trajectory