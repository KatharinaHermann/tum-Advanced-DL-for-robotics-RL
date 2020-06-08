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
    workspace[22:25, 14:20] = 1

    obstacle_entries = [(0, 0) for _ in range(5)]
    obstacle_entries[0] = (22, 14)
    obstacle_entries[1] = (22, 19)
    obstacle_entries[2] = (24, 19)
    obstacle_entries[3] = (24, 14)
    obstacle_entries[4] = (23, 17)

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

    obstacle_entries = [(0, 0) for _ in range(5)]
    obstacle_entries[0] = (22, 0)
    obstacle_entries[1] = (22, 4)
    obstacle_entries[2] = (31, 4)
    obstacle_entries[3] = (31, 0)
    obstacle_entries[4] = (25, 2)

    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')




if __name__ == '__main__':

    test_remove_obstacle()
    test_find_collision_entries()
    print('All tests has run successfully!')

