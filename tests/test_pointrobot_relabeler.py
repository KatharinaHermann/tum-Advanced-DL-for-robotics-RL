import numpy as np

from hwr.relabeling.pointrobot_relabeling import PointrobotRelabeler


class test_env():
    """helping class, that has some values that a real gym environment would have."""

    def __init__(self,
        radius,
        goal_reward=5,
        collision_reward=-1,
        step_reward=-0.01
        ):

        self.radius = radius
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.step_reward = step_reward


def test_remove_obstacle():
    """Tests the remove_obstacle method of the PointrobotRelabeler class"""
    workspace = np.zeros((32, 32))
    workspace[5:10, 12:19] = 1
    # etalon solution:
    etalon_workspace = workspace.copy()
    workspace[22:, :5] = 1

    obstacle_entries = []
    obstacle_entries.append((22, 0))
    obstacle_entries.append((22, 4))
    obstacle_entries.append((31, 4))
    obstacle_entries.append((31, 0))
    obstacle_entries.append((22, 2))

    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    for i, obstacle_entry in enumerate(obstacle_entries):
        cleaned_workspace = relabeler._remove_obstacle(workspace=workspace.copy(),
                                                      obstacle_entry=obstacle_entry)
        assert (cleaned_workspace == etalon_workspace).all(),\
             "workspace did not match at test round {}".format(i)


def test_find_collision_entries():
    """Tests the find_collision_entries method of the PointrobotRelabeler class."""

    workspace = np.zeros((32, 32))
    workspace[5:8, 5] = 1
    workspace[5:8, 7] = 1
    workspace[5, 5:8] = 1
    workspace[7, 5:8] = 1

    # etalon solutions:
    solution_workspaces = []
    solution_workspaces.append(workspace.copy())
    solution_workspaces.append(workspace.copy())
    solution_workspaces[1][(5, 6)], solution_workspaces[1][(7, 6)] = 0, 0
    solution_workspaces[1][(6, 5)], solution_workspaces[1][(6, 7)] = 0, 0
    solution_workspaces.append(np.zeros((32, 32)))
    solution_workspaces.append(np.zeros((32, 32)))
    
    radiuses = [0.3, 0.6, 0.9, 1.8]
    trajectory = [{'workspace': workspace, 
                  'position' : np.array([6.5, 6.5]),
                  'next_position': np.array([7.0, 7.0])}]
    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    for i, radius in enumerate(radiuses):
        env = test_env(radius)
        collision_entries = relabeler._find_collision_entries(trajectory, env)
        modified_workspace = workspace.copy()
        for entry in collision_entries:
            modified_workspace[entry] = 0

        assert (modified_workspace == solution_workspaces[i]).all(), "test with radius {} has failed.".format(radius)


def test_shifting():
    """Tests the workspace shifting method of the PointrobotRelabeler class."""

    workspace = np.zeros((32, 32))
    workspace[10:15, 10:15] = 1

    shift_distance = 1
    env = test_env(radius=0.5)
    relabeler = PointrobotRelabeler(ws_shape=(32, 32), mode='erease')

    trajs = []
    trajs.append([{'workspace': workspace, 'position': np.array([10, 20]),
        'next_position': np.array([10, 20]), 'goal': np.array([10, 21])}])
    trajs.append([{'workspace': workspace, 'position': np.array([10, 30.6]),
        'next_position': np.array([10, 30.6]), 'goal': np.array([10, 21])}])
    trajs.append([{'workspace': workspace, 'position': np.array([10, 0.4]),
        'next_position': np.array([10, 0.4]), 'goal': np.array([10, 21])}])
    trajs.append([{'workspace': workspace, 'position': np.array([30.6, 20]),
        'next_position': np.array([30.6, 20]), 'goal': np.array([10, 21])}])
    trajs.append([{'workspace': workspace, 'position': np.array([0.4, 20]),
        'next_position': np.array([0.4, 20]), 'goal': np.array([10, 21])}])

    # etalon solutions:
    etalon_wss = [np.zeros_like(workspace) for _ in range(5)]
    etalon_wss[0] = workspace.copy()
    etalon_wss[1][10:15, 9:14] = 1
    etalon_wss[2][10:15, 11:16] = 1
    etalon_wss[3][9:14, 10:15] = 1
    etalon_wss[4][11:16, 10:15] = 1

    etalon_trajs = []
    etalon_trajs.append([{'workspace': etalon_wss[0], 'position': np.array([10, 20]),
        'next_position': np.array([10, 20]), 'goal': np.array([10, 21])}])
    etalon_trajs.append([{'workspace': etalon_wss[1], 'position': np.array([10, 29.6]),
        'next_position': np.array([10, 29.6]), 'goal': np.array([10, 20])}])
    etalon_trajs.append([{'workspace': etalon_wss[2], 'position': np.array([10, 1.4]),
        'next_position': np.array([10, 1.4]), 'goal': np.array([10, 22])}])
    etalon_trajs.append([{'workspace': etalon_wss[3], 'position': np.array([29.6, 20]),
        'next_position': np.array([29.6, 20]), 'goal': np.array([9, 21])}])
    etalon_trajs.append([{'workspace': etalon_wss[4], 'position': np.array([1.4, 20]),
        'next_position': np.array([1.4, 20]), 'goal': np.array([11, 21])}])

    for i, trajectory in enumerate(trajs):
        relabeled_traj = relabeler._shift_from_boarder(trajectory=trajectory,
                                                             env=env,
                                                             shift_distance=shift_distance)
        assert (relabeled_traj[0]['workspace'] == etalon_trajs[i][0]['workspace']).all(),\
            "Workspace did not match at test case {}.".format(i)
        assert (relabeled_traj[0]['position'] == etalon_trajs[i][0]['position']).all(),\
            "Position did not match at test case {}.".format(i)


def test_goal_setting():
    """Tests the _set_new_goal() method of the PointrobotRelabeler class."""

    workspace = np.zeros((32, 32))
    env = test_env(radius=0.5)

    traj_0 = [{'workspace': workspace, 'position': np.array([0, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])}]
    
    traj_1 = [
        {'workspace': workspace, 'position': np.array([-0.5, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])}
        ]

    traj_2 = [
        {'workspace': workspace, 'position': np.array([0.4, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, 0.4]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([-0.4, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])}
    ]
    
    traj_3 = [
        {'workspace': workspace, 'position': np.array([0.4, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, 0.4]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([-0.4, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, -0.4]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])},
        {'workspace': workspace, 'position': np.array([0, 0]), 'next_position': np.array([0, 0]), 'goal': np.array([10, 21])}
    ]

    # outputs:
    relabeler = PointrobotRelabeler(ws_shape=(32, 32), mode='erease')
    sol_traj_0 = relabeler._set_new_goal(traj_0, env)
    sol_traj_1 = relabeler._set_new_goal(traj_1, env)
    sol_traj_2 = relabeler._set_new_goal(traj_2, env)
    sol_traj_3 = relabeler._set_new_goal(traj_3, env)

    # tests:
    # traj_0:
    assert len(sol_traj_0) == len(traj_0)
    assert (sol_traj_0[0]['position'] == traj_0[0]['position']).all()
    assert np.linalg.norm(sol_traj_0[0]['goal'] - traj_0[0]['position']) == (env.radius * 0.99)

    # traj_1:
    assert len(sol_traj_1) == len(traj_1)
    for i, point in enumerate(sol_traj_1):
        assert (sol_traj_1[i]['position'] == traj_1[i]['position']).all()
    
    pos_0, pos_1 = traj_1[0]['position'], traj_1[1]['position']
    vect = (pos_1 - pos_0) / np.linalg.norm(pos_1 - pos_0) * env.radius * 0.99
    new_goal_etalon = pos_1 + vect
    assert (sol_traj_1[0]['goal'] == new_goal_etalon).all()

    #traj_2:
    assert len(sol_traj_2) == len(traj_2)
    for i, point in enumerate(sol_traj_1[ : -1]):
        assert (sol_traj_2[i]['position'] == traj_2[i]['position']).all()
        assert np.linalg.norm(sol_traj_2[i]['goal'] - sol_traj_2[i]['position']) > env.radius

    assert (sol_traj_2[-1]['position'] == traj_2[-1]['position']).all()
    assert np.linalg.norm(sol_traj_2[-1]['goal'] - sol_traj_2[-1]['position']) == env.radius * 0.99

    # traj_3:
    assert len(sol_traj_3) == 0


def test_erease_relabeling():
    """Tests the relabeling method of the PointrobotRelabeler class based 
    simply on ereasing the obstacle into which the robot has collided.
    """

    workspace = np.zeros((32, 32))
    workspace[5:10, 12:19] = 1
    etalon_workspace = workspace.copy()
    workspace[22:, :5] = 1

    env = test_env(radius=0.5)
    relabeler = PointrobotRelabeler(ws_shape=(32, 32),
                                    mode='erease')

    trajectory = []
    trajectory.append({'workspace': workspace, 'position': np.array([21.0, 5.0]),
        'done': False, 'reward': env.step_reward})
    trajectory.append({'workspace': workspace, 'position': np.array([21.3, 4.7]),
        'done': False, 'reward': env.step_reward})
    trajectory.append({'workspace': workspace, 'position': np.array([21.6, 4.4]),
        'done': True, 'reward': env.collision_reward})

    # etalon solutions:
    etalon_relabeled_traj = []
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.0, 5.0]),
        'done': False, 'reward': env.step_reward})
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.3, 4.7]),
        'done': False, 'reward': env.step_reward})
    etalon_relabeled_traj.append({'workspace': etalon_workspace, 'position': np.array([21.6, 4.4]), 
        'done': True, 'reward': env.goal_reward})

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
    test_shifting()
    test_goal_setting()
    test_erease_relabeling()
    print('All tests have run successfully!')

