# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time

# added nnObjFunction code
# TODO matrix dimensions fix

#commented featureSelection code
#uncomment later

#changed sigmoid
#simpler now
#check if works

# ============ Configurable parameters ============ #

# Percentage of training-data that we'll use for validation
validation_data_percentage = 10

# No. of nodes in the HIDDEN layer (not including bias unit)
n_hidden = 50;

# Lambda (the regularization hyper-parameter)
lambdaval = 0.3;

run_count = 0

def preprocess():
    
    
    start_time = time.time()
    print("\n--------------------START-Preprocess------------------")
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
     - normalize the data to [0, 1]   ??????????????????
     - feature selection """
    
    
    # Load the MAT object as a Dictionary
    mat = loadmat('/home/harsh/canopy/ML/mnist_all.mat')
    is_first_run = True

    for i in range(0,10):
        
        # training_matrix - each row is a training example, each column is a 
        # pixel. Size: N x 784 where N is the number of training examples
        training_matrix = np.divide(mat.get('train'+str(i)),255.0)
        # test_matrix - size T x 784, where T is number of test data elements
        test_matrix = np.divide(mat.get('test'+str(i)),255.0)
          
        
        # =========== Create test-data matrix =========== #
        
        # How many test data elements are present in the data for this digit?
        test_data_count = test_matrix.shape[0]
        
        # Make an array of repeated labels like [9,9,9,9,...]
        label_vector = generateLabelVector(i)
        repeated_labels_testdata = np.tile(label_vector, (test_data_count,1))
                
        if is_first_run:
            # Create test-data matrix and label vector
            test_data = np.array(test_matrix)
            test_label = repeated_labels_testdata
        else:
            # Append to the test-data matrix and label vector
            test_data = np.append(test_data, test_matrix, 0)
            test_label = np.append(test_label, repeated_labels_testdata, 0) 
        
        
        
        # =========== Create training & validation matrices =========== #
        
        # How many training examples are present in the data for this digit?
        num_of_examples = training_matrix.shape[0]
        # How many of the given examples will be used for validation
        validation_size = round(num_of_examples * float(validation_data_percentage / 100))
        training_size = num_of_examples - validation_size
        
        # Randomly split the data into a validation and a training part
        random_range = range(num_of_examples)
        perm = np.random.permutation(random_range)
        validation_part = training_matrix[perm[0:validation_size],:]
        training_part = training_matrix[perm[validation_size:],:]
        
        # Make an array of repeated labels like [9,9,9,9,...]
        label_vector = generateLabelVector(i)
        repeated_labels_train = np.tile(label_vector, (training_size,1))
        repeated_labels_valid = np.tile(label_vector, (validation_size,1))
        
        if is_first_run:
            # Create training-data matrix and label vector
            train_data = np.array(training_part)
            train_label = repeated_labels_train
            # Create validation-data matrix and label vector
            validation_data = np.array(validation_part)
            validation_label = repeated_labels_valid
        else:
            # Append to the training-data matrix and label vector
            train_data = np.append(train_data, training_part, 0)
            train_label = np.append(train_label, repeated_labels_train, 0)
            # Append to the validation-data matrix and label vector
            validation_data = np.append(validation_data, validation_part, 0)
            validation_label = np.append(validation_label, repeated_labels_valid, 0)
         
        # Not the first run anymore
        is_first_run = False        
                
    print "train_data.shape", train_data.shape
    print"train_label.shape", train_label.shape          
    print"validation_data.shape", validation_data.shape
    print"validation_label.shape", validation_label.shape
    print"test_data.shape", test_data.shape
    print"test_label.shape", test_label.shape
    
    
    #print("--------------------START-FeatureSelection------------------")
    # #Perform feature-selection on all 3 of the matrics
    train_data, validation_data, test_data = doFeatureSelection(train_data, validation_data, test_data)
    
    print "train_data.shape", train_data.shape
    #print"train_label.shape", train_label.shape          
    print"validation_data.shape", validation_data.shape
    #print"validation_label.shape", validation_label.shape
    print"test_data.shape", test_data.shape
    #print"test_label.shape", test_label.shape
    #print("--------------------END-FeatureSelection------------------")
    
    print("Time for preprocessing: ",time.time() - start_time)
    print("--------------------END-Preprocess------------------")
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
def generateLabelVector(x):
    vector = np.repeat(np.array([0]), 10, 0)
    vector[x] = 1
    return vector

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

    #Added by harsh the sigmoid function itself handles the input whether its scalar, a vector or a matrix
    #no need for the for loop
def sigmoid(z):
    return (1 / (1 + np.exp(-1 * z)))    

def nnObjFunction(params, *args):
    print("\n--------------------START - nnObjFunction------------------")
    obj_start_time = time.time()
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
    
    print "training_data,shape ", training_data.shape
    print "training_label.shape ", training_label.shape
    #print n_hidden
    #print n_input
    #Added by Harsh there was a mismatch in the number of hidden nodes.
    w1 = params[0:(n_hidden) * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[((n_hidden) * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    print "w1.shape", w1.shape
    print "w2.shape", w2.shape
    obj_val = 0  
    
    #Your code here
    
    w1_trans = np.transpose(w1)
    w2_trans = np.transpose(w2)
    print "w1_trans.shape", w1_trans.shape
    print "w2_trans.shape", w2_trans.shape
    
    n_examples = training_data.shape[0]
    print n_examples 
    
    #grad_w2 = output x hidden
    #grad_w1 = output x hidden 
    grad_w1 = np.zeros((n_hidden + 1, n_input + 1)) #initialize to 0  
    grad_w2 = np.zeros((n_class, n_hidden + 1)) #initialize to 0
    print "grad_w1.shape",  grad_w1.shape
    print "grad_w2.shape",  grad_w2.shape
    
    # === Add the (d+1)th bias attribute to training data as a column
    ones = np.repeat(np.array([[1]]), training_data.shape[0], 0)
    training_data = np.append(training_data, ones, 1)
     
    for i in range(n_examples):
        #x = 1 x input
        
        ###### DO THIS OUT OF FOR-LOOP
        x = training_data[i]
        #x = np.append(x, 1)
        
        # append bias = 1
       # x_trans=x.reshape(x.size,1)
        #print "x_bias.shape", x_bias.shape
        #a = 1 x hidden
        #print "a.shape", a.shape
        #z = 1 x hidden
        z = sigmoid(np.dot(x, w1_trans))
        z = np.append(z, 1)
        #append bias = 1
        z_trans=z.reshape(1,z.size)
        #print "z_bias.shape", z_bias.shape
        #b = 1 x output
        #print "b.shape", b.shape
        #o = 1 x output
        #print z.shape
        #print w2_trans.shape
        o = sigmoid(np.dot(z_trans, w2_trans))
        #print "o.shape", o.shape
        #y = 1 x output
        y = training_label[i]
        #print "y.shape", y.shape
        #delta = 1 x output
        
        #-----calculation for obj_grad-----
        delta = np.subtract(o, y)
        #delta_trans = np.transpose(delta)
        delta_trans=delta.reshape(delta.size,1)
        #print "delta_trans.shape", delta_trans.shape
        #^just for representation.
        #transpose does nothing to a (1 x something) array
        #so we have to use np.outer later, not np.dot
        
        #grad_w2 = output x hidden
        grad_w2 = np.add(grad_w2, (np.dot(delta_trans, z_trans)))
       # grad_w2 = np.add(grad_w2, (np.dot(delta_trans, z)))
        #calculating [(1-z)*z * summation(delta*w2)] dot [x]
        #prodz = 1 x hidden
        
        prodzXsummation = (np.dot(delta, w2))*(z*(np.subtract(1.0, z)))
        prodzXsummation_trans=prodzXsummation.reshape(prodzXsummation.size,1)
        #prodzXsummation_trans = np.transpose(prodzXsummation) 
        #^this also does nothing
        #grad_w1 = hidden x input
        x_trans_new=x.reshape(1,x.size)
        grad_w1 = np.add(grad_w1,(np.dot(prodzXsummation_trans, x_trans_new)))
             
        #-----calculation for obj_value-----
        
        #logo = np.log(o)
        #minuso = np.subtract(1.0, o)
        #logminuso = np.log(minuso)
        #minusy = np.subtract(1.0, y)
        #print "o.shape", o.shape
        #print "y.shape", y.shape
        #print "minusy.shape", minusy.shape
        #print "logo.shape", logo.shape
        #print "minuso.shape", minuso.shape
        #print "logminuso.shape", logminuso.shape
        
        j = y*(np.log(o)) + ((np.subtract(1.0, y))*(np.log(np.subtract(1.0, o))))
        jsum = np.sum(j)
        
        obj_val += jsum
        
        
                 
    # Make sure you reshape the gradient matrices to a 1D array. for instance 
    # if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
  
    grad_w1 = grad_w1 / n_examples
    # Remove the last row from grad_w1 (to match the dimensions)
    grad_w1=grad_w1[:-1,:]
    
    grad_w2 = grad_w2 / n_examples
    
    #For Regularization in the Neural Network,calculating terms required for regularizing obj_val
    refact_w1=np.square(w1)
    refact_w1_sum =np.sum(refact_w1)
    refact_w2=np.square(w2)
    refact_w2_sum =np.sum(refact_w2)
    final_reg_term =(lambdaval)/(2*n_examples)*(refact_w1_sum+refact_w2_sum)
    #calculating the terns required for regularizing obj_grad
    lambdaw1=w1*lambdaval
    grad_w1=(grad_w1+lambdaw1)
    grad_w1 =  np.divide(grad_w1, n_examples)
    lambdaw2=w2*lambdaval
    grad_w2=(grad_w2+lambdaw2)
    grad_w2 =  np.divide(grad_w2, n_examples)
    ##end of regularization 
    obj_val = (obj_val/n_examples)*-1
    obj_val=obj_val+final_reg_term  
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    print "obj_grad", obj_grad
    print "obj_val",  obj_val
    
    global run_count 
    run_count += 1
    print "run_count", run_count
    
    
    
    print("Time for nnObjFunction: ",time.time() - obj_start_time)
    print("\n--------------------END - nnObjFunction------------------")
    
              
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
        
    w1_trans = np.transpose(w1)
    w2_trans = np.transpose(w2)
    
    n_examples = data.shape[0]
    
    # === Add the (d+1)th bias attribute to training data as a column
    ones = np.repeat(np.array([[1]]), data.shape[0], 0)
    data = np.append(data, ones, 1)
        
    for i in range(n_examples):
        #x = 1 x input
        x = data[i]
        #x = np.append(x, 1)
        
        z = sigmoid(np.dot(x, w1_trans))
        z = np.append(z, 1)         
        
        z_trans=z.reshape(1,z.size)

        # Get the output at the max
        o = sigmoid(np.dot(z_trans, w2_trans))

        # The prediction is the index of the output unit with the max o/p
        prediction = np.argmax(o)
        labels = np.append(labels, prediction)
           
    return labels
    



"""**************Neural Network Script Starts here********************************"""
overall_start_time = time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

# ====== Train Neural Network ======
run_count = 0

# set the number of nodes in the input layer (not including bias unit)
n_input = train_data.shape[1];
				   
# Number of nodes in the output layer are fixed as 10, because we've got 10 digits
n_class = 10;

# initialize the weights into some random matrices
#Added by Harsh there was a mismatch in the number of hidden nodes.
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


#yolo
#====== We now have the trained weights ======
# Reshape nnParams from 1D vector into w1 and w2 matrices
print "1: ", nn_params.x[0:n_hidden * (n_input + 1)].shape
print "2: ", nn_params.x[(n_hidden * (n_input + 1)):].shape
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


# We need to convert the label matrices into column vectors
train_label = train_label.argmax(axis=1)
validation_label = validation_label.argmax(axis=1)
test_label = test_label.argmax(axis=1)
#====== Test the computed parameters ======


# Find the accuracy on the TRAINING Dataset
predicted_label = nnPredict(w1,w2,train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on the VALIDATION Dataset
predicted_label = nnPredict(w1,w2,validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#find the accuracy on the TEST Dataset
predicted_label = nnPredict(w1,w2,test_data)
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')



print("Total time: ",time.time() - overall_start_time)
print("\n--------------------END of code------------------")