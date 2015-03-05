# -*- coding: utf-8 -*-
import numpy as np
    
def doFeatureSelection(train_data, validation_data, test_data):

    n_rows = train_data.shape[0]
    n_cols = train_data.shape[1]
    is_first_run = True
    
    new_train_data = train_data
    new_validation_data = validation_data
    new_test_data = test_data
    
    if train_data.shape[0]!=0:
        for i in range(n_cols):
            
            col_flag = False
            temp = train_data[0][i]
            
            for j in range(1, n_rows):
                if train_data[j][i] != temp:
                    col_flag = True
                    break
            if col_flag is True:
                if is_first_run is True:
                    new_train_data = np.array([train_data[:, i]]) # create matrix 
                    new_train_data = np.reshape(new_train_data, (n_rows, -1))
                    
                    new_validation_data = np.array([validation_data[:, i]]) # create matrix 
                    new_validation_data = np.reshape(new_validation_data, (validation_data.shape[0], -1))
                    
                    new_test_data = np.array([test_data[:, i]]) # create matrix 
                    new_test_data = np.reshape(new_test_data, (test_data.shape[0], -1))
                    
                    is_first_run = False;
                else:
                    tempmatrix = np.reshape(np.array([train_data[:, i].T]), (train_data.shape[0],-1))
                    new_train_data = np.append(new_train_data, tempmatrix, 1)
                    
                    tempmatrix = np.reshape(np.array([validation_data[:, i].T]), (validation_data.shape[0],-1))
                    new_validation_data = np.append(new_validation_data, tempmatrix, 1)
                    
                    tempmatrix = np.reshape(np.array([test_data[:, i].T]), (test_data.shape[0],-1))
                    new_test_data = np.append(new_test_data, tempmatrix, 1)
        
    return new_train_data, new_validation_data, new_test_data

    
train_data = np.array([[0,1,2,5],[0,5,2,8], [0,5,2,1]])
validation_data = np.array([[5,3,7,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])
test_data = np.array([[8,1,8,5],[0,5,2,8],[0,5,4,9], [0,5,2,1]])

print '\nM', train_data 

new_train_data, new_validation_data, new_test_data = doFeatureSelection(train_data, validation_data, test_data)

print '\nnew_train_data', new_train_data
print '\nnew_validation_data', new_validation_data
print '\nnew_test_data', new_test_data