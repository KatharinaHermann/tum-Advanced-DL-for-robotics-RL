import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage



def random_workspace(grid_size, num_obj_max, obj_size_avg):
    
    #Define number of objects
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

    height =np.random.normal(loc=obj_size_avg, scale=2, size=(num_objects,1))
    height =np.asarray(height, dtype=int, order=None)

    #width = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)
    #height = tf.random.normal(shape=[num_objects,1], mean=obj_size_avg, stddev=2, dtype=tf.int32)

    #Initialize workspace

    workspace=np.zeros((grid_size,grid_size), dtype=int)

    #Assign each entry with an object a 1. 
    
    for i in range(num_objects):
        if origin[i,1]+width[i] > grid_size:
            right_bound=grid_size+1
        else: right_bound = np.asscalar(origin[i,1]+width[i])

        if origin[i,0]+height[i] > grid_size:
            upper_bound=grid_size+1
        else: upper_bound = np.asscalar(origin[i,0]+height[i])
        workspace[origin[i,0]:upper_bound, origin[i,1]:right_bound]=1
        workspace

    #Generate start point (repeat until point is found where no object ist placed)

    start_blocked=1
    while (start_blocked==1):
        start= np.random.randint(low=0, high=grid_size, size=(1,2))
        start= np.asarray(start, dtype=None, order=None)
        y=start[0,0]
        x=start[0,1]
        start_blocked=workspace[y,x]
    
        #start = tf.random.uniform( 
    #shape=[1,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)

    #Generate goal point (repeat until point is found where no object ist placed) and assign goal point with a 1

    goal_blocked=1
    while (goal_blocked==1):
        goal= np.random.randint(low=0, high=grid_size, size=(1,2))
        goal= np.asarray(goal, dtype=None, order=None)
        y=goal[0,0]
        x=goal[0,1]
        goal_blocked=workspace[y,x]
       #goal = tf.random.uniform( 
    #shape=[1,2], minval=0, maxval=grid_size, dtype=tf.int32, seed=None, name=None)
   

    return start, goal, workspace




#################### Random workspace generation TEST#############
start,goal,workspace_sample=random_workspace(32, 10, 5)

workspace_sample[start[0,0],start[0,1]]=2
workspace_sample[goal[0,0],goal[0,1]]=3

fig = plt.figure()
ws = fig.add_subplot(221)
cws = ws.matshow(workspace_sample)
fig.colorbar(cws)


dist_img = ndimage.distance_transform_edt(-workspace_sample + 1)  # Excpects blocks as 0 and free space as 1

dm = fig.add_subplot(222)
cdm=dm.imshow(dist_img, cmap='Reds')
cdm=dm.imshow(workspace_sample, cmap='Blues', alpha=0.3)
plt.show()

def image_interpolation(*, img, pixel_size=1, order=1, mode='nearest'):

    factor = 1 / pixel_size

    def interp_fun(x):
        x2 = x.copy()

        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]
        # Transform physical coordinates to image coordinates 
        #x2 *= factor
        #x2 -= 0.5

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun

pixel_size=1 #10/32

dist_fun = image_interpolation(img=dist_img, pixel_size=pixel_size)
print("Distance to the nearest obstacle at the center: ",
      dist_fun(x=start))
print("Distance to the nearest obstacle for 10 random points: ",
      dist_fun(x=np.random.uniform(low=1, high=9, size=(10, 2))))