import numpy as np

from pointrobot_relabeling import PointrobotRelabeler


class test_env():
    """helping class, that has some values that a real gym environment would have."""

    def __init__(self, radius):
        self.radius = radius


def test_remove_obstacle():
    """Tests the remove_obstacle method of the PointrobotRelabeler class"""
    workspace = np.zeros((32, 32))
    workspace[5:10, 12:19] = 1
    etalon_workspace = workspace
    workspace[22:, :5] = 1

    obstacle_entries = []
    obstacle_entries.append((22, 0))
    obstacle_entries.append((22, 4))
    obstacle_entries.append((31, 4))
    obstacle_entries.append((31, 0))
    obstacle_entries.append((5, 2))

    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    for i, obstacle_entry in enumerate(obstacle_entries):
        cleaned_workspace = relabeler._remove_obstacle(workspace=workspace,
                                                      obstacle_entry=obstacle_entry)
        assert (cleaned_workspace == etalon_workspace).all(), "workspace did not match at test position {}".format(i)


def test_find_collision_entries():
    """Tests the find_collision_entries method of the PointrobotRelabeler class."""

    workspace = np.zeros((32, 32))
    workspace[5:8, 5] = 1
    workspace[5:8, 7] = 1
    workspace[5, 5:8] = 1
    workspace[7, 5:8] = 1

    solution_workspaces = []
    solution_workspaces.append(workspace)
    solution_workspaces.append(workspace)
    solution_workspaces[1][(5, 6)], solution_workspaces[1][(7, 6)] = 0, 0
    solution_workspaces[1][(6, 5)], solution_workspaces[1][(6, 7)] = 0, 0
    solution_workspaces.append(np.zeros((32, 32)))
    solution_workspaces.append(np.zeros((32, 32)))
    
    radiuses = [0.3, 0.6, 0.9, 1.8]
    trajectory = [{'workspace': workspace, 
                  'position' : np.array([6.5, 6.5])}]
    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    for i, radius in enumerate(radiuses):
        env = test_env(radius)
        collision_entries = relabeler._find_collision_entries(trajectory, env)
        modified_workspace = workspace
        for entry in collision_entries:
            modified_workspace[entry] = 0

        assert (modified_workspace == solution_workspaces[i]).all(), "test with radius {} has failed.".format(radius)



def test_erease_relabeling():
    """Tests the relabeling method of the PointrobotRelabeler class based 
    simply on ereasing the obstacle into which the robot has collided.
    """

    workspace = np.zeros((32, 32))
    workspace[5:10, 12:19] = 1
    etalon_workspace = workspace
    workspace[22:, :5] = 1

    env = test_env(radius=0.5)
    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    trajectory = []
    trajectory.append({'workspace': workspace, 'position': np.array([21.0, 5.0]), 'done': False, 'reward': -0.01})
    trajectory.append({'workspace': workspace, 'position': np.array([21.3, 4.7]), 'done': False, 'reward': -0.01})
    trajectory.append({'workspace': workspace, 'position': np.array([21.6, 4.4]), 'done': True, 'reward': -1})
    etalon_relabeled_traj = []
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.0, 5.0]), 'done': False, 'reward': -0.01})
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.3, 4.7]), 'done': False, 'reward': -0.01})
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.6, 4.4]), 'done': True, 'reward': 1})

    relabeled_traj = relabeler.relabel(trajectory, env)

    for i in range(len(trajectory)):
        assert (etalon_relabeled_traj[i]['workspace'] == relabeled_traj[i]['workspace']).all(), \
                                                    "workspace in point {} is not correct".format(i)
        assert (etalon_relabeled_traj[i]['position'] == relabeled_traj[i]['position']).all(), \
                                                    "postion in point {} is not correct".format(i)
        assert etalon_relabeled_traj[i]['done'] == relabeled_traj[i]['done'], \
                                                    "done in point {} is not correct".format(i)
        assert etalon_relabeled_traj[i]['reward'] == relabeled_traj[i]['reward'], \
                                                    "reward in point {} is not correct".format(i)


if __name__ == '__main__':

    test_remove_obstacle()
    test_find_collision_entries()
    test_erease_relabeling()
    print('All tests has run successfully!')

