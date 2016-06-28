# Assignment 4
# Juliane Reschke (370450)
# Pierre-Henri Mathieu (377099)
# Robert Liebner (368366)

import pylab as pl
import scipy as sp
import numpy as np
from numpy.linalg import inv
from scipy.io import loadmat
import pdb

def load_myo_data(fname):
    ''' Loads EMG data from <fname>                      
    '''
    # load the data
    data = loadmat(fname)
    # extract data and hand positions
    X = data['training_data']
#    X = sp.log(X)
    Y = data['training_labels']
    #Split data into training and test data
    X_train = X[:, :5000]
    X_test = X[:, 5000:]
    Y_train = Y[:, :5000]
    Y_test = Y[:, 5000:]
    return X_train,Y_train,X_test, Y_test
	
def train_ols(X_train, Y_train, llambda = 0):
    ''' Trains ordinary least squares (ols) regression 
    Input:       X_train  -  DxN array of N data points with D features
                 Y        -  D2xN array of length N with D2 multiple labels
                 llambda  -  Regularization parameter
    Output:      W        -  DxD2 array, linear mapping used to estimate labels 
                             with sp.dot(W.T, X)                      
    '''

    W = np.dot(inv(np.dot(X_train, X_train.T) + llambda*np.identity(X_train.shape[0])), np.dot(X_train, Y_train.T))

    return W
    
def apply_ols(W, X_test):
    ''' Applys ordinary least squares (ols) regression 
    Input:       X_test    -  DxN array of N data points with D features
                 W        -  DxD2 array, linear mapping used to estimate labels 
                             trained with train_ols                   
    Output:     Y_test    -  D2xN array
    '''
 
    Y_test = np.dot(W.T, X_test)
    return Y_test
    
def predict_handposition():
    X_train,Y_train,X_test, Y_test = load_myo_data('myo_data.mat')
    # compute weight vector with linear regression
    W = train_ols(X_train, Y_train)
    # predict hand positions
    Y_hat_train = apply_ols(W, X_train)
    Y_hat_test = apply_ols(W, X_test)
        
    pl.figure()
    pl.subplot(2,2,1)
    pl.plot(Y_train[0,:1000],Y_train[1,:1000],'.k',label = 'true')
    pl.plot(Y_hat_train[0,:1000],Y_hat_train[1,:1000],'.r', label = 'predicted')
    pl.title('Training Data')
    pl.xlabel('x position')
    pl.ylabel('y position')
    pl.legend(loc = 0)
    
    pl.subplot(2,2,2)
    pl.plot(Y_test[0,:1000],Y_test[1,:1000],'.k')
    pl.plot(Y_hat_test[0,:1000],Y_hat_test[1,:1000],'.r')
    pl.title('Test Data')
    pl.xlabel('x position')
    pl.ylabel('y position')
    
    pl.subplot(2,2,3)
    pl.plot(Y_train[1,:600], 'k', label = 'true')
    pl.plot(Y_hat_train[1,:600], 'r--', label = 'predicted')
    pl.xlabel('Time')
    pl.ylabel('y position')
    pl.legend(loc = 0)
    
    pl.subplot(2,2,4)
    pl.plot(Y_test[1,:600],'k')
    pl.plot(Y_hat_test[1,:600], 'r--')
    pl.xlabel('Time')
    pl.ylabel('y position')
    
def test_assignment4():
    ##Example without noise
    x_train = sp.array([[ 0,  0,  1 , 1],[ 0,  1,  0, 1]])
    y_train = sp.array([[0, 1, 1, 2]])
    w_est = train_ols(x_train, y_train) 
    w_est_ridge = train_ols(x_train, y_train, llambda = 1)
    assert(sp.all(w_est.T == [[1, 1]])) 
    assert(sp.all(w_est_ridge.T == [[.75, .75]]))
    y_est = apply_ols(w_est,x_train)
    assert(sp.all(y_train == y_est)) 
    print 'No-noise-case tests passed'
	
	##Example with noise
	#Data generation
    w_true = 4
    X_train = sp.arange(10)
    X_train = X_train[None,:]
    Y_train = w_true * X_train + sp.random.normal(0,2,X_train.shape)
    #Regression 
    w_est = train_ols(X_train, Y_train) 
    Y_est = apply_ols(w_est,X_train)
    #Plot result
    pl.figure()
    pl.plot(X_train.T, Y_train.T, '+', label = 'Train Data')
    pl.plot(X_train.T, Y_est.T, label = 'Estimated regression')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(loc = 'lower right')

def test_polynomial_regression():


    x_toy = np.ndarray(shape=(2,11))  #11 datapoints with 2 feature (one is it's x-coordinate, one the y)


    x_toy[0,:] = np.arange(11)       #fill first feature of all datapoint with x-coordinate
    x_toy[1,:]=np.sin(x_toy[0,:])+np.random.normal(0,0.5) #fill second feature of all datapoints with target data


    degree = 4  #test it with some degree
    #llambda = 0

    W = np.polyfit(x_toy[0,:],x_toy[1,:],degree)    #try to estimate correct coefficients for polynomial mapping
    #  (This looks erroneous, because we try to map one feature to the other)
    #  First our toy data was:
    #   a) x_toy: 11 datapoints with one feature which actually the value of the elements being their index
    #   b) y_toy: 11 datapoint with one label and their value being np.sin(x_toy[0,:])+np.random.normal(0,0.5)
    #   but we couldn't feed it to train_ols, because we recvd a "singular matrix"-error (matrix not inversible)


    #build up a polynominal calculated value for all datapoints

    #initialize all elements with w0
    y_estimate = np.full(11,W[0])


    for x in range(11):
        for i in range(1,degree+1):
             y_estimate[x] += W[i] * pow(x,i)   #add up next monome element



    estimate = train_ols(x_toy,y_estimate)

    apply_ols(estimate,x_toy[0,:])


    #apply_ols()

# TO DO: finish this function and questions a and b



#train_data, train_label, test_data, test_label = load_myo_data('myo_data.mat')
#print train_label.shape[0]

#predict_handposition()
#pl.show()
test_polynomial_regression()
