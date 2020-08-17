import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def random_workspace(grid_size, num_obj_max, obj_size_avg):
    """generates a workspace of size: grid_size x grid_size with obstacles indicated with ones.
        Free space are indicated with a 0. 
        The # of objects, the origin and height & width of the objects are generated from uniform and normal distributions.
    """  

    if num_obj_max >= 1:
        #Throw number of objects from a uniform distribution.
        num_objects= np.random.randint(low=num_obj_max-1, high=num_obj_max+1, size=None)

        #Generate an origin from a uniform distribution for each object
        origin = np.random.randint(low=0, high=grid_size, size=(num_objects,2))
        origin = np.asarray(origin, dtype=None, order=None)

        #Generate a width and height from a Gaussian distribution for each object
        width = np.random.normal(loc=obj_size_avg, scale=1, size=(num_objects,1))
        width= np.asarray(width, dtype=int, order=None)
        
        height = np.random.normal(loc=obj_size_avg, scale=1, size=(num_objects,1))
        height = np.asarray(height, dtype=int, order=None)
    
        #Initialize workspace
        workspace=np.zeros((grid_size, grid_size), dtype=int)

        #Assign each entry with an object a 1. 
        for i in range(num_objects):
            if origin[i,1]+width[i] > grid_size:
                right_bound=grid_size+1
            else: right_bound = (origin[i,1]+width[i]).item()

            if origin[i,0]+height[i] > grid_size:
                upper_bound=grid_size+1
            else: upper_bound = (origin[i,0]+height[i]).item()
            workspace[origin[i,0]:upper_bound, origin[i,1]:right_bound]=1
    else:
        workspace = np.zeros((grid_size, grid_size))

    return workspace

def hard_level_workspace(workspace, grid_size, obj_size_avg):
    start, goal = get_start_goal_for_workspace(workspace)

    distance_start_goal = np.linalg.norm(goal-start)
    straight_step = (goal-start) / distance_start_goal
    vertical_step = np.zeros_like(straight_step)
    vertical_step[0] = straight_step[1] 
    vertical_step[1] = -straight_step[0]
    step_size = np.random.rand()
    step_size *= distance_start_goal

    #Generate a width and height from a Gaussian distribution for each object
    width = int(np.random.normal(loc=obj_size_avg, scale=1))
    height = int(np.random.normal(loc=obj_size_avg, scale=1))
    diagonal = np.sqrt(np.square(0.5*width) + np.square(0.5*height))

    #placing 2 obstacles in a vertical distance of "the diagonal of the object" to the straight line between start and goal
    
    #Creating the first obstacle
    origin = (start + step_size * straight_step  + (diagonal + 1.5)* vertical_step).astype(np.int)
    #assigning bounds for x direction
    if (origin[0] + 0.5 * width) > grid_size:
        right_bound = grid_size
    else: right_bound = int(origin[0] + 0.5 * width)

    if (origin[0] - 0.5 * width) < 0:
        left_bound = 0
    else: left_bound = int(origin[0] - 0.5 * width)

    #assigning bounds for y direction
    if (origin[1] + 0.5 * height) > grid_size:
        upper_bound = grid_size
    else: upper_bound = int(origin[1] + 0.5 * height)

    if (origin[1] - 0.5 *height) < 0:
        lower_bound=0
    else: lower_bound = int(origin[1] - 0.5 * height)

    #Assign each entry with an object a 1.
    workspace[lower_bound:upper_bound, left_bound:right_bound] = 1

    #Creating the second obstacle
    origin = start + step_size * straight_step  - (diagonal + 1.5)* vertical_step
    #assigning bounds for x direction
    if (origin[0] + 0.5 * width) > grid_size:
        right_bound = grid_size
    else: right_bound = int(origin[0] + 0.5 * width)

    if (origin[0] - 0.5 * width) < 0:
        left_bound = 0
    else: left_bound = int(origin[0] - 0.5 * width)

    #assigning bounds for y direction
    if (origin[1] + 0.5 * height) > grid_size:
        upper_bound = grid_size
    else: upper_bound = int(origin[1] + 0.5 * height)

    if (origin[1] - 0.5 *height) < 0:
        lower_bound = 0
    else: lower_bound = int(origin[1] - 0.5 * height)

    #Assign each entry with an object a 1.
    workspace[lower_bound:upper_bound, left_bound:right_bound] = 1

    #Check whether goal feasible
    x_goal = int(goal[0])
    y_goal = int(goal[1])
    goal_blocked = workspace[y_goal-2: y_goal+3, x_goal-2: x_goal+3].any()
                        
    if goal_blocked:
        workspace[y_goal-2: y_goal+3, x_goal-2: x_goal+3] = 0
        
    #Check whether start feasible
    x_start = int(goal[0])
    y_start = int(goal[1])
    start_blocked = workspace[y_start-2: y_start+3, x_start-2: x_start+3].any()
                        
    if start_blocked:
        workspace[y_start-2: y_start+3, x_start-2: x_start+3] = 0

    return workspace, start, goal


def mid_level_workspace(grid_size, num_obj_max, obj_size_avg):
    """generates a workspace of size: grid_size x grid_size with obstacles indicated with ones.
        Free space are indicated with a 0. 
        The # of objects, the origin and height & width of the objects are generated from uniform and normal distributions. Theobstacles do NOT overlap!
    """  

    if num_obj_max >= 1:

        #Throw number of objects from a uniform distribution.
        num_objects= np.random.randint(low=num_obj_max-1, high=num_obj_max+1, size=None)   

        #Initialize workspace
        workspace=np.zeros((grid_size, grid_size), dtype=int)

        for i in range(num_objects): 

            obsj_already_occupied = 1
            while obsj_already_occupied:
                #Generate an origin from a uniform distribution for each object
                origin = np.random.randint(low=0, high=grid_size, size=(1,2))
                origin = np.asarray(origin, dtype=None, order=None)

                #Generate a width and height from a Gaussian distribution for each object
                width = int(np.random.normal(loc=obj_size_avg, scale=1))
                height = int(np.random.normal(loc=obj_size_avg, scale=1))

                #Assign each entry with an object a 1. 
                if origin[0,1]+width > grid_size:
                    right_bound=grid_size+1
                else: right_bound = (origin[0,1]+width).item()

                if origin[0,0]+height > grid_size:
                    upper_bound=grid_size+1
                else: upper_bound = (origin[0,0]+height).item()
                
                if (workspace[origin[0,0]:upper_bound, origin[0,1]:right_bound].any() == 1): 
                    continue
                else:
                    obsj_already_occupied = 0
                    workspace[origin[0,0]:upper_bound, origin[0,1]:right_bound] = 1
    
    else:
        workspace = np.zeros((grid_size, grid_size))

    return workspace


def get_start_goal_for_workspace(workspace, max_goal_dist=None):
    """generates a discrete start and goal point for a given workspace. 
    It throws in randomply points from a uniform distribution until the points are in free space.
    """

    grid_size = workspace.shape[0]

    # Generate start point (repeat until point is found where no object ist placed)
    start_blocked = True
    while start_blocked:
        
        start = np.random.uniform(low=1.0, high=float(grid_size-2), size=(2,)).astype(int)
        x = int(start[0])
        y = int(start[1])

        #Check if there is an obsrtacle near the start position
        y_min = y - 3 if y - 3 >= 0 else 0
        y_max = y + 4 if y + 4 <= (grid_size - 1) else grid_size - 1
        x_min = x - 3 if x - 3 >= 0 else 0
        x_max = x + 4 if x + 4 <= (grid_size - 1) else grid_size - 1
        start_blocked = workspace[y_min: y_max, x_min: x_max].any()
        

    # Generate goal point (repeat until point is found where no object ist placed) and assign goal point with a 1
    # if a maximum goal distance is specified, Points are only generated within that range.
    goal_blocked = True
    while goal_blocked:
        
        if max_goal_dist is None:
            low = 1.
            high = float(grid_size-2)
        else:
            low = np.clip(start - max_goal_dist, a_min=1., a_max=float(grid_size-2))
            high = np.clip(start + max_goal_dist, a_min=1., a_max=float(grid_size-2))    
        
        goal = np.random.uniform(low=low, high=high, size=(2,))

        x = int(goal[0])
        y = int(goal[1])
        y_min = y - 3 if y - 3 >= 0 else 0
        y_max = y + 4 if y + 4 <= (grid_size - 1) else grid_size - 1
        x_min = x - 3 if x - 3 >= 0 else 0
        x_max = x + 4 if x + 4 <= (grid_size - 1) else grid_size - 1
        goal_blocked = workspace[y_min: y_max, x_min: x_max].any()
        
    return start, goal


def visualize_workspace(workspace, fignum=1):
    """for nicely visualizing a workspace."""

    fig = plt.figure(fignum)
    ax = fig.add_subplot(221)
    cax = ax.matshow(workspace)
    fig.colorbar(cax)

    return ax

def visualize_distance_field(workspace, fignum=1):
    """for nicely visualizing the distance field to the obstacles."""
    dist_img = ndimage.distance_transform_edt(-workspace + 1)  # Excpects blocks as 0 and free space as 1
    
    fig = plt.figure(fignum)
    
    dm = fig.add_subplot(222)
    dm.imshow(dist_img, cmap='Reds')
    dm.imshow(workspace, cmap='Blues', alpha=0.3)
    
    return fig

def visualize_robot(current_position, robot_radius, color='m'):
    """for nicely visualizing the distance field to the obstacles. The robot's position is described in the workspace matrix (with indices). Therefore, the x-coordinate is the second element of the array. The y-coordinate is the first element"""
    robot = plt.Circle((current_position[0], current_position[1]), robot_radius, color=color)
    ax = plt.gca()
    ax.add_artist(robot)

    return robot

def image_interpolation(*, img, pixel_size=1, order=1, mode='nearest'):

    factor = 1 / pixel_size

    def interp_fun(x):
        x2 = x.copy()

        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]
        # Transform physical coordinates to image coordinates 
        x2 *= factor
        x2 += 1

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun



    
############# TEST SAMPLE WORKSPACE #########################

if __name__ == '__main__':
    workspace = mid_level_workspace(32, 8, 5)
    workspace_sample, start, goal = hard_level_workspace(workspace, 32, 5)
    #start, goal = get_start_goal_for_workspace(workspace_sample)

    workspace_sample[int(start[1]), int(start[0])] = 2
    workspace_sample[int(goal[1]), int(goal[0])] = 3

    fig1 = visualize_workspace(workspace_sample)
    robot = visualize_robot(start,1)

    fig2 = visualize_distance_field(workspace_sample)
    
    
    plt.show()

    
    #Calculates the distance from a point "x" to the nearest obstacle
    pixel_size=1 #10/32
    dist_img = ndimage.distance_transform_edt(-workspace_sample + 1)  # Excpects blocks as 0 and free space as 1
    dist_fun = image_interpolation(img=dist_img, pixel_size=pixel_size)
    x=np.array([float(start[0]),float(start[1])])
    dist= dist_fun(x=x)
    
    print("Distance to the nearest obstacle at the center: ",
       dist )
