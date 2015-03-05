# -*- coding: utf-8 -*-
import numpy as np
    
    
train_data = np.array([[0,1,2,5],[0,5,2,8], [0,5,2,1]])
validation_data = np.array([[5,3,7,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])
test_data = np.array([[8,1,8,5],[0,5,2,8],[0,5,4,9], [0,5,2,1]])

print '\nM', train_data 

x1 = '\nx1', np.arange(9.0).reshape((3, 3))
print x1
x2 = np.arange(3.0)
print '\nx2', x2

print np.multiply(x1, x2)