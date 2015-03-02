# -*- coding: utf-8 -*-
import numpy as np

def doFeatureSelection(matrix):
    # Also, convert each value to the double data-type here
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    is_first_run = True
    newmatrix = np.empty([2,2])
    
    for i in range(n_cols):
         
        col_flag = False
        temp = matrix[0][i]
        
        for j in range(1, n_rows):
            if matrix[j][i] != temp:
                col_flag = True
                break
        if col_flag is True:
            if is_first_run is True:
                newmatrix = np.array([matrix[:, i]]) #create matrix 
                newmatrix = np.reshape(newmatrix, (n_rows, -1))
                
                is_first_run = False;
            else:
                tempmatrix = np.reshape(np.array([matrix[:, i].T]), (n_rows,-1))
                newmatrix = np.append(newmatrix, tempmatrix, 1)
    
    return newmatrix

matrix = np.array([[0,1,2,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])

print matrix, "\n"

print "Result:\n", doFeatureSelection(matrix)
