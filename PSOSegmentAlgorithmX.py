

#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math

#--- COST FUNCTION ------------------------------------------------------------+

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print ('FINAL:')
        print (pos_best_g)
        print (err_best_g)

if __name__ == "__PSO__":
    main()


# Import opencv and numpy modules
import cv2
import numpy as np
# Open an image using opencv
image = cv2.imread('magpie.jpg')
# Convert the image into grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Reshape the image into a one dimensional vector
vector = np.reshape(gray_image, (-1,))

# Print the shape of the vector
print(vector.shape)

    
# Import numpy for array operations
import numpy as np

# Define a function to calculate the euclidean distance between two points
def euclidean_distance(p1, p2):
  return np.sqrt(np.sum((p1 - p2) ** 2))

# Define a function to calculate the fitness value for image segmentation using PSO
def fitnessSegment(x):
  # Convert x to a numpy array
  x = np.array(x)
  
  # Initialize the sum of distances to zero
  sum = 0
  
  # Loop through the vector variable
  for pixel in vector:
    # Find the index of the closest cluster center for the pixel
    c = np.argmin([euclidean_distance(pixel, m) for m in x])
    # Add the distance between the pixel and the closest cluster center to the sum
    sum += euclidean_distance(pixel, x[c])
  
  # Normalize the sum by the number of pixels and the number of clusters
  sum = sum / (len(vector) * len(x))
  
  # Return the normalized sum as the fitness value
  print("sum=",sum)
  return sum



initial=[50,50,50,50,50,50,50,50,50]               # initial starting location [x1,x2...]
bounds=[(0,255),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(fitnessSegment,initial,bounds,num_particles=15,maxiter=30)














#--- END ----------------------------------------------------------------------+