# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def preprocess():
    
    print "..:: Execution begins ::.."
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
    
     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection """
    
    
    # Pick a reasonable size for validation data
    # ==== Percentage of data that will make the validation part === #
    validation_data_percentage = 99
    
    # Initialize the arrays to be returned
    #train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    
    # Load the MAT object as a Dictionary
    mat = loadmat('/home/ankitkap/machinelearning/basecode/mnist_all.mat')
    training_data_size = 0
    is_first_run = True
        
    # For each digit
    for i in range(0,10):
        # digit_matrix - is a matrix in which each row is a training example,
        # and each column is a pixel of an example. So its dimensions are 
        # N x 784 where N is the number of training examples for that digit, 
        # and 784 because the number of pixels is 28x28
        digit_matrix = mat.get('train'+str(i))
        
        # How many examples are present in the data for this digit?
        num_of_examples = digit_matrix.shape[0]
        # How many of the given examples will be used for validation
        validation_size = round(num_of_examples * float(validation_data_percentage / 100))
        training_size = num_of_examples - validation_size
        
        # Split the data for this digit into a validation and a training part
        random_range = range(digit_matrix.shape[0])
        perm = np.random.permutation(random_range)
        
        validation_part = digit_matrix[perm[0:validation_size],:]
        training_part = digit_matrix[perm[validation_size:],:]
        
        # Make a array of repeated labels like [9,9,9,9,...]
        repeated_labels_train = np.repeat(np.array([i]), training_size, 0)
        repeated_labels_valid = np.repeat(np.array([i]), validation_size, 0)
        
        if is_first_run:
            train_data = np.array(training_part)
            train_label = repeated_labels_train
            validation_label = repeated_labels_valid
            is_first_run = False
        else:
            train_data = np.append(train_data, training_part, 0)
            train_label = np.append(train_label, repeated_labels_train, 0)
            validation_label = np.append(train_label, repeated_labels_valid, 0)
                
        # ---- What does 'shape' do? The shape attribute for numpy arrays returns the dimensions of the array. 
        # ---- If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
#        print "\nm.shape[0]: %s" % digit_matrix.shape[0]
#       print "m.shape[1]: %s" % digit_matrix.shape[1]
        training_data_size += digit_matrix.shape[0]
    
    print train_data
    print train_data.shape
    print train_label
    print train_label.shape
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  #your code here
    
        

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

# ====== Parameters to be experimented with ======
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
# set the regularization hyper-parameter
lambdaval = 0;

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();



# ====== Train Neural Network ======

# set the number of nodes in the input layer (not including bias unit)
n_input = train_data.shape[1];
# ---- What does 'shape' do? The shape attribute for numpy arrays returns the dimensions of the array. 
# ---- If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
				   
# Number of nodes in the output layer are fixed as 10, because we've got 10 digits
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# Combine the 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)




# ===== Train Neural Network using fmin_cg or minimize from scipy, optimize module. Check documentation for a working example

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
opts = {'maxiter' : 50}    # Max-iterations: preferred value

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

# In case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)



#====== We now have the trained weights ======
# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))




#====== Test the computed parameters ======

# Find the accuracy on the TRAINING Dataset
predicted_label = nnPredict(w1,w2,train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on the VALIDATION Dataset
predicted_label = nnPredict(w1,w2,validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#find the accuracy on the TEST Dataset
predicted_label = nnPredict(w1,w2,test_data)
print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
