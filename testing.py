# -*- coding: utf-8 -*-
import numpy as np
    
matrix = np.array([[0,1,2,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])
vector = np.array([0,1,2,5,3])
print '\nM', matrix

ones = np.repeat(np.array([[1]]), matrix.shape[0], 0)
matrix = np.append(matrix, ones, 1)

print '\nOnes', ones
print '\nM', matrix
