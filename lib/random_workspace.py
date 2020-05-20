import numpy as np
import tensorflow as tf



def random_workspace(grid_size, num_obj_max, obj_size_avg):
    
    #Define number of objects
    num_objects= np.random.randint(low=1, high=num_obj_max, size=None)
    #num_objects = tf.random.uniform( 
    #shape=[], minval=0, maxval=num_obj_max, dtype=tf.int32, seed=None, name=None)

    #Generate an origin from a uniform distribution for each object

    origin= np.random.randint(low=0, high=grid_size, size=(num_objects,2))
    origin=np.asarray(origin, dtype=None, order=None)
    #origin = tf.random.uniform( 
    #shape=[num_objects,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)

    #Generate a width and height from a Gaussian distribution for each object
    width =np.random.normal(loc=obj_size_avg, scale=2, size=(num_objects,1))
    width=np.asarray(width, dtype=int, order=None)

    height =np.random.normal(loc=obj_size_avg, scale=2, size=(num_objects,1))
    height =np.asarray(height, dtype=int, order=None)

    #width = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)
    #height = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)

    #Initialize workspace

    workspace=np.zeros((grid_size,grid_size), dtype=int)

    #Assign each entry with an object a -1. 
    
    for i in range(num_objects):
        if origin[i,0]+width[i]+1 > grid_size:
            right_bound=grid_size+1
        else: right_bound = np.asscalar(origin[i,0]+width[i]+1)

        if origin[i,1]+height[i]+1 > grid_size:
            upper_bound=grid_size+1
        else: upper_bound = np.asscalar(origin[i,1]+height[i]+1)
        
        workspace[origin[i,0]:right_bound, origin[i,1]:upper_bound]=-1

    #Generate start point (repeat until point is found where no object ist placed)

    start_blocked=(-1)
    while (start_blocked==(-1)):
        start= np.random.randint(low=0, high=grid_size, size=(1,2))
        start= np.asarray(start, dtype=None, order=None)
        x=start[0,0]
        y=start[0,1]
        start_blocked=workspace[x,y]
    
    #label start as 1
    workspace[x,y]=1
        #start = tf.random.uniform( 
    #shape=[1,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)

    #Generate goal point (repeat until point is found where no object ist placed) and assign goal point with a 1

    goal_blocked=-1
    while (goal_blocked==-1):
        goal= np.random.randint(low=0, high=grid_size, size=(1,2))
        goal= np.asarray(goal, dtype=None, order=None)
        x=goal[0,0]
        y=goal[0,1]
        goal_blocked=workspace[x,y]
       #goal = tf.random.uniform( 
    #shape=[1,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)
    
    #label goal as 10
    workspace[x,y]=10

    return workspace

workspace_sample=random_workspace(32, 3, 4)
print(workspace_sample)