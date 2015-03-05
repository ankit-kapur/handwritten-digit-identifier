# -*- coding: utf-8 -*-
import numpy as np
import csv
    
x = np.array([[0,1,2,5],[0,5,2,8], [0,5,2,1]])
y = np.array([[5,3,7,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])
z = np.array([[8,1,8,5],[0,5,2,8],[0,5,4,9], [0,5,2,1]])

w2 = None

lambda_val = 0.02
lambda_increment = 0.1

if 3 < 5:
    w2 = np.array([[0,1,2,5],[0,5,2,8], [0,5,2,1]])

print '\nw2',w2