#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:23:09 2019

Monte Carlo Methods: 
  Calculating Pi

@author: Conrad Salinas
@email: deephomebrewed@gmail.com
@company: deephomebrew
@project: Project_1_S
"""

# Import libs
import numpy
import torch
import math
import matplotlib.pyplot as plt

# N sample points
n_point_square = [10**3, 10**4, 10**5, 10**6]

# Loop Thru 
for num_pts_sq in n_point_square:
    # Create 1,000-1,000,000 random points for square -1<=x<=1 and -1<=y<=1
    points_square = torch.rand((num_pts_sq, 2)) * 2 -1
    
    # Init points inside circle and list
    n_point_cicrle = 0
    points_circle = []
    
    # For each points from square landing inside circle of radius 1
    # save point to list and increment points inside circle
    for points in points_square:
      r = torch.sqrt(points[0]**2 + points[1]**2)
      if r <= 1.0:
        points_circle.append(points)
        n_point_cicrle += 1
    
    # Circle points to tensor
    points_circle = torch.stack(points_circle)
    
    # Plot the random points inside and outside circle
    plt.plot(points_square[:, 0].numpy(), points_square[:, 1].numpy(), 'y.')
    plt.plot(points_circle[:, 0].numpy(), points_circle[:, 1].numpy(), 'g.')
    
    # Draw circle and square
    i = torch.linspace(0, 2 * math.pi)
    plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy())
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 'r')
    plt.axes().set_aspect('equal')
    plt.title('Monte Carlo π\nN = %d' % num_pts_sq)
    plt.show()
    
    # Calc value of pi
    pi_est = 4 * (n_point_cicrle / num_pts_sq)
    print('Est. val of pi is: %1.8f @ N = %d' % (pi_est, num_pts_sq))
    
# Estimation Vs Iteration up to 10,000
n_point_cicrle = 0
pi_iteration = []
for i in range(1, n_point_square[1] + 1):
    point = torch.rand(2) * 2 -1
    r = torch.sqrt(point[0]**2 + point[1]**2)
    if r <= 1.0:
        n_point_cicrle += 1
    pi_iteration.append(4 * (n_point_cicrle / i))
plt.plot(pi_iteration)
plt.plot([math.pi] * n_point_square[1], '--')
plt.xlabel('Iteration')
plt.ylabel('Estimated π')
plt.title('Estimation History')
plt.show()

