# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:32:32 2016
@author: rossp
Diamond-Square algorithm for terrain generation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform

def diamond_square(n, H=1, rand_range=1, initialize='flat', plot=True):
    """
    Arguments: 
    n = length of large square (must be power of 2**n + 1, e.g. 5, 9, 17, etc)         
        
    H = reduction of random error added to each point (smoothing)
        note that 0 <= H <= 1, smaller H results in more jagged terrain. Default 1
    
    rand_range = positive number to determine the range for the random uniform distribution
        Range is set to +/- rand_range. Default 1.
    
    initialize = 'flat' (default), 'random'
        If 'flat', entire grid starts with an elevation of 0.
        If 'random', corners of grid initialized to random values between +/- rand_range.
    
    plot = True, False
        Controls whether or not to display plot of surface
    """
    #Check to see square length is a power of 2**n + 1
    if 2**int(np.log2(n-1)) != n-1:
        return 'Length of square must be 2**n + 1.'

    total_length = n-1 #Side length of the starting square (largest overall)
    passes = int(np.log2(n-1)) + 1 #Number of passes of full algorithm
    
    #Initialize grid
    grid = np.empty((n,n), dtype=float)*np.nan
    
    #Set corners to 0 for initialize = 'flat'
    if initialize == 'flat':
        grid[0, 0] =                       0
        grid[0, total_length] =            0
        grid[total_length, 0] =            0
        grid[total_length, total_length] = 0
    #Set corners to a random value in range for initialize = 'random'
    elif initialize == 'random':
        grid[0, 0] =                       uniform(-rand_range, rand_range)
        grid[0, total_length] =            uniform(-rand_range, rand_range)
        grid[total_length, 0] =            uniform(-rand_range, rand_range)
        grid[total_length, total_length] = uniform(-rand_range, rand_range)
        
    else:
        return 'initialize argument only accepts \'flat\' or \'random\''
    
    #Diamond-Square algorithm loop
    for i in range(passes):
        #Determine side length of squares and total number of squares/diamonds
        length = total_length//(2**i) #Integer division for indexing
        #If length = 1, all points have been assigned
        if length == 1: 
            break

        #Initialize array to store index values for diamond centers
        #Row number stored in first column, column number stored in second column
        num_diamonds = 2**(2*(i+1))
        d_ind = np.zeros((num_diamonds, 2), dtype=int)
        count = 0 #counter for inside diamond/square loop for diamond indices
        
        #Set up range for random numbers, account for smoothing parameter
        #rr = random range
        rr = np.array([-rand_range*2**(-H*i), rand_range*2**(-H*i)])   

        #First, perform diamond step for each square
        #x0 and y0 become upper left corner values
        #Note that x = column value, y = row value
        for y0 in range(0, total_length, length):
            for x0 in range(0, total_length, length):
                #Obtain other corner indices
                x1 = x0 + length #Extend in column-direction
                y1 = y0 + length #Extent in row-direction
                
                #Calculate middle points, store to diamond center index array
                x_mid = (x0 + x1)//2
                y_mid = (y0 + y1)//2
                d_ind[count, 0] = y_mid
                d_ind[count, 1] = x_mid
                count += 1
                
                #Assign corner points
                ul = grid[y0, x0]
                ur = grid[y0, x1]
                ll = grid[y1, x0]
                lr = grid[y1, x1]
                
                grid[x_mid, y_mid] = np.mean((ul,ur,ll,lr)) + uniform(rr[0], rr[1])
        
        #Halve rr before diamond step - may decide to remove this line later
        rr = np.array([-rand_range*2**(-H*(i+1)), rand_range*2**(-H*(i+1))])
              
        #Next, perform square step for each diamond
        #Includes wrapping if node is on an edge
        for k in range(num_diamonds):
            y0 = d_ind[k,0]  #Row index of center point
            x0 = d_ind[k,1]  #Column index of center point
            diff = length//2 #distance (number of rows/columns) from center to corner
            
            #Identify corner points (these points are about to be calculated)
            bottom =  (y0 + diff, x0)
            top =     (y0 - diff, x0)
            left =    (y0, x0 - diff)
            right =   (y0, x0 + diff)
            corners = (bottom, top, left, right)
            
            #Get points for averaging (bottom, top, left, right; as compared to corners)
            #and calculate new elevation at the corner point.
            #This is skipped if the point already has a value from a previous diamond
            for corner in corners:
                if np.isnan(grid[corner]):
                    #Bottom
                    if corner[0] + diff > total_length:
                        b = grid[corner[0]+diff-total_length, corner[1]]
                    else:
                        b = grid[corner[0]+diff, corner[1]]
                    
                    #Top
                    if corner[0] - diff < 0:
                        t = grid[corner[0]-diff+total_length, corner[1]]
                    else:
                        t = grid[corner[0]-diff, corner[1]]
                    
                    #Left
                    if corner[1] - diff < 0:
                        l = grid[corner[0], corner[1]-diff+total_length]
                    else:
                        l = grid[corner[0], corner[1]-diff]
                    
                    #Right
                    if corner[1] + diff > total_length:
                        r = grid[corner[0], corner[1]+diff-total_length]
                    else:
                        r = grid[corner[0], corner[1]+diff]              

                    grid[corner] = np.mean((b,t,l,r)) + uniform(rr[0], rr[1])
        
    
    #Plot surface if plot=True
    if plot:
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        ax.plot_surface(x, y, grid, rstride=1, cstride=1, cmap=plt.cm.rainbow)
    
    
    return grid


#Testing the algorithm
n = 129

test = diamond_square(129, rand_range=4, initialize='random', plot=True)

fig1 = plt.figure(figsize=(20,10))
ax1 = fig1.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(np.arange(n), np.arange(n))
#z = 0.5*np.sin(0.1*xx) + np.sin(0.05*yy)
z = 0.5*np.sin(0.05*xx + 0.05*yy) + 0.25*np.sin(0.15*xx)
ax1.plot_surface(xx, yy, z + test, rstride=1, cstride=1, cmap=plt.cm.rainbow)






