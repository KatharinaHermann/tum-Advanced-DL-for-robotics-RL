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
        - random: Randomly tries to throw in obstacles, close to the robot's trajectory, and generates a workspace
                 in which the trajectory is feasible. Obstacle parts which result in a collision are removed, suchthat often small corridors are created.
        - sliding: NOT IMPLEMENTED! Slides the obstacles of the original workspace and hence creates a workspace,
                  where the pointrobot was successful and the trajectory was a somewhat effective
                  solution to the workspace.
    """

    def __init__(self, ws_shape=(32, 32), mode='no_relabeling', remove_zigzaging=False):
        """ Initialization of a workspace relabeler for a Pointrobot
        Args:
            - ws_shape: tuple, (ws_height, ws_width)
            - mode: str, determines the mode of the workspace generation.
                    possible values are: 'erease', 'random', 'sliding'
        """

        assert mode in ['no_relabeling', 'erease', 'random', 'sliding', 'straight_line'] ,\
            'mode should be either \'no_relabeling\', \'erease\', \'random\' or \'sliding\' or \'straight_line\'. Received {}'.format(mode)

        self._ws_shape = ws_shape
        self._mode = mode
        self._remove_zigzaging = remove_zigzaging


    def relabel(self, trajectory, env):
        """creates a new workspace and goal for the given trajectory."""
        
        relabeled_trajectory = []
        # rescale the trajectory if normalization is used in the environment:
        if env.normalize:
            trajectory = self._rescale_trajectory(trajectory, env)

        if self._mode == 'no_relabeling':
            return relabeled_trajectory

        if self._mode == 'erease':
            if self._remove_zigzaging == True:
                if not self._zig_zag_path(trajectory):
                    relabeled_trajectory = self._erease_relabel(trajectory, env)
            else: relabeled_trajectory = self._erease_relabel(trajectory, env)

        elif self._mode == 'random':
            if self._remove_zigzaging == True:
                if not self._zig_zag_path(trajectory):
                    relabeled_trajectory = self._random_relabel(trajectory, env)
            else: relabeled_trajectory = self._random_relabel(trajectory, env)

        elif self._mode == 'slding':
            if self._remove_zigzaging == True:
                if not self._zig_zag_path(trajectory):
                    relabeled_trajectory = self._sliding_relabel(trajectory, env)
            else: relabeled_trajectory = self._sliding_relabel(trajectory, env)

        elif self._mode == 'straight_line':
            relabeled_trajectory = self. _straight_line_relabel(trajectory, env)

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
        if len(trajectory) > 1:
            workspace = trajectory[0]['workspace']

            if trajectory[-1]["reward"] == env.collision_reward:

                # find the entries of the matrix where the robot has collided.:
                obstacle_entries = self._find_collision_entries(trajectory[-1], env)

                if obstacle_entries:
                    #print("Collision with obstacle!!!!!!")
                    # if the robot has collided.
                    erase_length = env.robot_radius+2
                    trajectory = self._cut_trajectory(trajectory=trajectory,
                                        env=env, erase_length=erase_length)
                    #for obstacle_entry in obstacle_entries:
                        #workspace = self._remove_obstacle(workspace=workspace, obstacle_entry=obstacle_entry)
                    for data_point in trajectory:
                        data_point['workspace'] = workspace
                else:
                    #print("Collision with boundary!!!!!!")
                    # if the obstacle has just left the workspace without collision.
                    # choosing a distance with which the ws and the trajectory will be shifted away from the boarder:
                    #erase_length = np.random.randint(low=1, high=4)
                    #trajectory = self._cut_trajectory(trajectory=trajectory,
                    #                        env=env, erase_length=erase_length)

                    shift_distance = np.random.randint(low=1, high=4)
                    trajectory = self._shift_from_boarder(trajectory=trajectory,
                                            env=env,
                                            shift_distance=shift_distance)

                # add new goal state to the trajectory:
                if len(trajectory) > 1:
                    return self._set_new_goal(trajectory, env)
                else:
                    return []
            else:
                #print("No Collision!!!!!!")
                return self._set_new_goal(trajectory, env)
        else:
            #print("Only 1 point!!!!!!")
            return[]


    def _random_relabel(self, trajectory, env):
        """Relabels a workspace with 'random' method.
        Random means, that obstacles are sampled in the range of the trajectory.
        Obstacle parts which result in a collision are removed, suchthat often small corridors are created."""
        if len(trajectory) > 1:
            workspace = trajectory[0]['workspace']

            if trajectory[-1]["reward"] == env.collision_reward:

                # find the entries of the matrix where the robot has collided.:
                obstacle_entries = self._find_collision_entries(trajectory[-1], env)

                if obstacle_entries:
                    #print("Collision with obstacle!!!!!!")
                    # if the robot has collided.
                    for obstacle_entry in obstacle_entries:
                        workspace = self._remove_obstacle(workspace=workspace, obstacle_entry=obstacle_entry)

                else:
                    #print("Collision with boundary!!!!!!")
                    # if the obstacle has just left the workspace without collision.
                    # choosing a distance with which the ws and the trajectory will be shifted away from the boarder:
                    #erase_length = np.random.randint(low=1, high=4)
                    #trajectory = self._cut_trajectory(trajectory=trajectory,
                    #                        env=env, erase_length=erase_length)

                    shift_distance = np.random.randint(low=1, high=4)
                    trajectory = self._shift_from_boarder(trajectory=trajectory,
                                    env=env,
                                    shift_distance=shift_distance)
            #else:
                #print("No Collision!!!!!!")
            #Sample new obstacles in workspace
            workspace = self._sample_objects(workspace=workspace, trajectory=trajectory, num_objects=env.num_obj_max, avg_object_size=env.obj_size_avg, env =env)
                
            for data_point in trajectory:
                data_point['workspace'] = workspace
            
            # add new goal state to the trajectory:
            if len(trajectory) > 1:
                return self._set_new_goal(trajectory, env)
            else:
                return []
        else:
            return[]


    def _straight_line_relabel(self, trajectory, env):
        """Relabels the trajectory as a 'straight line'"""
        start = trajectory[0]['position']
        goal = trajectory[0]['goal']
        workspace = trajectory[0]['workspace']

        trajectory_to_return = []

        pos = start.copy()
        action = (goal-start) / (np.linalg.norm(goal-start))
        next_pos = pos + action
        reward = env.step_reward

        while np.linalg.norm(goal - pos) > env.robot_radius:
            trajectory_to_return.append({'workspace': workspace, 'position': pos.copy(),
                'next_position': next_pos.copy(),'goal': goal, 'action': action, 'reward': reward, 'done': False})
            pos = next_pos.copy()
            next_pos += action

        if trajectory_to_return:
            trajectory_to_return[-1]['reward'] = env.goal_reward
            trajectory_to_return[-1]['done'] = True

        return trajectory_to_return
    
    
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


    def _find_collision_entries(self, traj_point, env):
        """Finds the entries of the matrix, where the agent has collided into an obstacle."""

        workspace = traj_point['workspace']
        last_pos = traj_point['position']
        radius = env.robot_radius
        # range of distances to check in every direction:
        # (it is possible that the radius of the robot is bigger than the grid size, this is why this is necessary.)
        distances = list(range(int(radius) + 3))
        #distances.append(radius)

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
                if (pos >= env.grid_size-1).any() or (pos <= 1).any():
                    return collision_entries
                entry = (int(pos[1]), int(pos[0]))
                if workspace[entry] == 1:
                    collision_entries.append(entry)

        return collision_entries

    def _cut_trajectory(self, trajectory, env, erase_length):
        """Erases the last part of the trajectory, that crashed with the boarder."""
        erase_length = int(erase_length)

        if erase_length < len(trajectory):
            trajectory_to_return = []
            for point in trajectory[:(-erase_length)]:
                trajectory_to_return.append(point)

            return trajectory_to_return
        else:
            return []


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

    def _sample_objects(self, workspace, trajectory, num_objects, avg_object_size, env):

        if (num_objects >= 1) and trajectory != []:

            #Calculate the lowest and highest positions of the trajectory in each direction, in order to sample next to it. 
            low_x = trajectory[0]['position'][0]
            high_x = trajectory[0]['position'][0]

            low_y = trajectory[0]['position'][1]
            high_y = trajectory[0]['position'][1]

            for point in trajectory:
                if point['position'][0] < low_x:
                    low_x = point['position'][0]
                if point['position'][0] > high_x:
                    high_x = point['position'][0]
                if point['position'][1] < low_y:
                    low_y = point['position'][1]
                if point['position'][1] > high_y:
                    high_y = point['position'][1]
            
            low_x = low_x - (avg_object_size) - env.robot_radius
            high_x = high_x + env.robot_radius

            low_y = low_y - (avg_object_size) - env.robot_radius
            high_y = high_y + env.robot_radius 

            
            low_x = np.clip(low_x, a_min=1., a_max=float(env.grid_size-2))
            high_x = np.clip(high_x, a_min=1., a_max=float(env.grid_size-2))  

            low_y = np.clip(low_y, a_min=1., a_max=float(env.grid_size-2))
            high_y = np.clip(high_y, a_min=1., a_max=float(env.grid_size-2)) 

            #Generate a width and height from a Gaussian distribution for each object
            width =np.random.normal(loc=avg_object_size, scale=2, size=(num_objects,1))
            width=np.asarray(width, dtype=int, order=None)
            
            height =np.random.normal(loc=avg_object_size, scale=2, size=(num_objects,1))
            height =np.asarray(height, dtype=int, order=None)   
             
            #Generate an origin from a uniform distribution for each object
            origin_y= np.random.randint(low=low_y, high=high_y, size=(num_objects,1))
            origin_x= np.random.randint(low=low_x, high=high_x, size=(num_objects,1))
            #origin = (np.array([origin_y, origin_x])).transpose()
            

            #Assign each entry with an object a 1. 
            for i in range(num_objects):
                if origin_x[i,0]+width[i] > env.grid_size:
                    right_bound=env.grid_size+1
                else: right_bound = (origin_x[i,0]+width[i]).item()

                if origin_y[i,0]+height[i] > env.grid_size:
                    upper_bound=env.grid_size+1
                else: upper_bound = (origin_y[i,0]+height[i]).item()
                
                workspace[origin_y[i,0]:upper_bound, origin_x[i,0]:right_bound] = 1

                
                for point in trajectory: 
                        x = int(point['position'][0])
                        y = int(point['position'][1])

                        point_blocked = workspace[y-2: y+3, x-2: x+3].any()
                        
                        if point_blocked:
                            workspace[y-2: y+3, x-2: x+3] = 0


            return workspace
            
        else:
            return workspace


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
        workspace =trajectory[0]['workspace']
        # the first goal candidate is in the direction of the last motion:
        if len(trajectory) >= 2:
            last_last_pos = trajectory[-2]['position']
            goal_direction_vect = (last_pos - last_last_pos) / (np.linalg.norm(last_pos - last_last_pos)) * env.robot_radius * 0.99
        else:
            goal_direction_vect = np.random.uniform(low=-1, high=1, size=(2,))
            goal_direction_vect = goal_direction_vect / np.linalg.norm(goal_direction_vect) * env.robot_radius * 0.99
        new_goal = last_pos + goal_direction_vect
        new_goal = np.clip(new_goal, a_min=1., a_max=float(env.grid_size-2))

        # matrix containing every state except for the last one for effectively calculate distance from them.
        every_other_pos = np.zeros((len(trajectory)-1, last_pos.shape[0]))
        for i, point in enumerate(trajectory[ : -1]):
            every_other_pos[i, :] = point['position']

        while not feasible_goal:
            # checking the distance from every goal
            distances = np.linalg.norm(every_other_pos - new_goal, axis=1)

            x = int(new_goal[0])
            y = int(new_goal[1])

            if ((distances <= env.robot_radius).any() or workspace[y-2: y+3, x-2: x+3].any()):
                # new random goal to try:
                goal_direction_vect = np.random.uniform(low=-1, high=1, size=(2,))
                goal_direction_vect = goal_direction_vect / np.linalg.norm(goal_direction_vect) * env.robot_radius * 0.99
                new_goal = last_pos + goal_direction_vect
                new_goal = np.clip(new_goal, a_min=1., a_max=float(env.grid_size-2))
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


    def _zig_zag_path(self, trajectory):
        """returns True, if the trajectory contains some serious zig zaging.
        It checks the average orientation difference among the points.
        If it is bigger then a specified threshold, it returns True.
        """
        threshold = math.pi / 2
        zig_zag = False
        if len(trajectory) > 1:
            angle_sum = 0
            for i, point in enumerate(trajectory[1:]):
                angle_sum += abs(self._calc_angle(trajectory[i]["action"], point["action"]))
            
            angle_diff_average = angle_sum / (len(trajectory) - 1)
            if angle_diff_average > threshold:
                zig_zag = True

        return zig_zag


    def _calc_angle(self, action1, action2):
        """calculates the angle between two successive actions."""
        length1 = np.linalg.norm(action1)
        length2 = np.linalg.norm(action2)
        cos_theta = action1 @ action2 / (length1 * length2)
        cross = action1[0] * action2[1] - action1[1] * action2[0]
        sin_theta = cross / (length1 * length2)
        return math.atan2(sin_theta, cos_theta)


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