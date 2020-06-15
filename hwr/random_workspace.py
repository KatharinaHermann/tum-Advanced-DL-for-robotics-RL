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
        num_objects= np.random.randint(low=1, high=num_obj_max, size=None)
        #num_objects = tf.random.uniform(shape=[], minval=0, maxval=num_obj_max, dtype=tf.int32, seed=None, name=None)

        #Generate an origin from a uniform distribution for each object
        origin= np.random.randint(low=0, high=grid_size, size=(num_objects,2))
        origin=np.asarray(origin, dtype=None, order=None)
        #origin = tf.random.uniform( 
        #shape=[num_objects,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)

        #Generate a width and height from a Gaussian distribution for each object
        width =np.random.normal(loc=obj_size_avg, scale=2, size=(num_objects,1))
        width=np.asarray(width, dtype=int, order=None)
        #width = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)

        height =np.random.normal(loc=obj_size_avg, scale=2, size=(num_objects,1))
        height =np.asarray(height, dtype=int, order=None)
        #height = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)

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


def get_start_goal_for_workspace(workspace):
    """generates a discrete start and goal point for a given workspace. 
    It throws in randomply points from a uniform distribution until the points are in free space.
    """

    grid_size = workspace.shape[0]

    #Generate start point (repeat until point is found where no object ist placed)
    start_blocked = True
    while start_blocked:
        
        start = np.random.uniform(low=0.0, high=float(grid_size), size=(2,))
        y = int(start[0])
        x = int(start[1])
        start_blocked = (workspace[y, x] == 1)

    #Generate goal point (repeat until point is found where no object ist placed) and assign goal point with a 1
    goal_blocked = True
    while goal_blocked:
        goal = np.random.uniform(low=0.0, high=float(grid_size), size=(2,))
        y = int(goal[0])
        x = int(goal[1])
        goal_blocked = (workspace[y, x] == 1)

    return start, goal


def visualize_workspace(workspace, fignum=1):
    """for nicely visualizing a workspace."""

    fig = plt.figure(fignum)
    ax = fig.add_subplot(221)
    cax = ax.matshow(workspace)
    fig.colorbar(cax)

    return fig

def visualize_distance_field(workspace, fignum=1):
    """for nicely visualizing the distance field to the obstacles."""
    dist_img = ndimage.distance_transform_edt(-workspace + 1)  # Excpects blocks as 0 and free space as 1
    
    fig = plt.figure(fignum)
    
    dm = fig.add_subplot(222)
    dm.imshow(dist_img, cmap='Reds')
    dm.imshow(workspace, cmap='Blues', alpha=0.3)
    
    return fig

def visualize_robot(current_position, color='#d347a8'):
    """for nicely visualizing the distance field to the obstacles. The robot's position is described in the workspace matrix (with indices). Therefore, the x-coordinate is the second element of the array. The y-coordinate is the first element"""
    robot = plt.Circle((current_position[1], current_position[0]), 1, color=color)
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
        x2 -= 0.5

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun



    
############# TEST SAMPLE WORKSPACE #########################

if __name__ == '__main__':
    workspace_sample = random_workspace(32, 10, 5)
    start, goal = get_start_goal_for_workspace(workspace_sample)

    workspace_sample[int(start[0]), int(start[1])] = 2
    workspace_sample[int(goal[0]), int(goal[1])] = 3

    fig1 = visualize_workspace(workspace_sample)
    robot = visualize_robot([3.55555,2.63333333333])

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
