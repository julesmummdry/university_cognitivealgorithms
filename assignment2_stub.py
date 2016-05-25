# Juliane Reschke (370450)
# Pierre-Henri Mathieu (377099)
# Maria Browarska (4970657, FU Berlin)

import scipy as sp
import scipy.io as io
import numpy as np
import pdb
import pylab as pl
import random


def load_usps_data(fname, digit=3):
    ''' Loads USPS (United State Postal Service) data from <fname> 
    Definition:  X, Y = load_usps_data(fname, digit = 3)
    Input:       fname   - string
                 digit   - optional, integer between 0 and 9, default is 3
    Output:      X       -  DxN array with N images with D pixels
                 Y       -  1D array of length N of class labels
                                 1 - where picture contains the <digit>
                                -1 - otherwise                           
    '''
    # load the data
    data = io.loadmat(fname)
    # extract images and labels
    X = data['data_patterns']
    Y = data['data_labels']
    Y = Y[digit, :]
    return X, Y


def train_perceptron(X, Y, iterations=200, eta=.1):
    ''' Trains a linear perceptron
    Definition:  w, b, acc  = train_perceptron(X,Y,iterations=200,eta=.1)
    Input:       X       -  DxN array of N data points with D features
                 Y       -  1D array of length N of class labels {-1, 1}
                 iter    -  optional, number of iterations, default 200
                 eta     -  optional, learning rate, default 0.1
    Output:      w       -  1D array of length D, weight vector 
                 b       -  bias term for linear classification                          
                 acc     -  1D array of length iter, contains classification accuracies 
                            after each iteration  
                            Accuracy = #correctly classified points / N 
    '''
    acc = sp.zeros((iterations))
    # include the bias term by adding a row of ones to X
    X = sp.concatenate((sp.ones((1, X.shape[1])), X))
    # initialize weight vector
    weights = sp.ones((X.shape[0])) / X.shape[0]
    for it in sp.arange(iterations):
        # indices of misclassified data
        wrong = (sp.sign(weights.dot(X)) != Y).nonzero()[0]
        # compute accuracy acc[it]
        # calculate the percentage of correct predictions = accuracy
        acc[it] = float(X.shape[1]-wrong.shape[0])/X.shape[1]
        if wrong.shape[0] > 0:
            # pick a random misclassified data point
            misclass_datapoint = random.choice(wrong)
            miscalss_X = X[:, misclass_datapoint]
            misclass_Y = Y[misclass_datapoint]
            # update weight vector (use variable learning rate (eta/(1.+it)) )
            weights += (eta / (1. + it)) * miscalss_X * misclass_Y
            if it % 20 == 0:
                print "Iteration %d:" % it + "Accuracy %0.2f" % acc[it]
    b = -weights[0]
    w = weights[1:]
    # return weight vector, bias and accuracies
    return w, b, acc


def train_ncc(X, Y):
    ''' Trains a prototype/nearest centroid classifier
    Definition:  w, b   = train_ncc(X,Y)
    Input:       X       -  DxN array of N data points with D features
                 Y       -  1D array of length N of class labels {-1, 1}
    Output:      w       -  1D array of length D, weight vector  
                 b       -  bias term for linear classification                          
    '''
    # get class data w_{-1} and w_{+1}
    w_positive = X.T[(Y == 1)]
    w_negative = X.T[(Y == -1)]
    # compute class mean of class w_{-1} and w_{+1}
    mean_w_pos = sp.mean(w_positive, 0)
    mean_w_neg = sp.mean(w_negative, 0)
    # weight vector: difference of class means w_{+1} - w_{-1}
    weights = mean_w_pos - mean_w_neg
    # bias: 0.5*(w_{+1}.T*w_{+1} - w_{-1}.T*w_{-1})
    bias = 0.5 * (mean_w_pos.dot(mean_w_pos) - mean_w_neg.dot(mean_w_neg))

    return weights, bias



def plot_histogram(X, Y, w, b):
    ''' Plots a histogram of classifier outputs (w^T X) for each class with pl.hist 
    The title of the histogram is the accuracy of the classification
    Accuracy = #correctly classified points / N 
    
    Definition:     plot_histogram(X, Y, w, b)
    Input:          X       -  DxN array of N data points with D features
                    Y       -  1D array of length N of class labels
                    w       -  1D array of length D, weight vector 
                    b       -  bias term for linear classification   
    
    '''
    #calculate the correct classified
    correct = (sp.sign(w.dot(X) - b) == Y).nonzero()[0]

    #class labels 1
    target = w.dot(X[:, (Y == 1)])
    #class balels -1
    non_target = w.dot(X[:, (Y == -1)])

    pl.title("Acc %0.0f%%" % (float(correct.shape[0]) / X.shape[1] * 100,))
    pl.xlabel('w^T X')
    pl.hist(non_target)
    pl.hist(target)
    pl.legend(["non target", "target"], loc=0)
    pl.show()


def compare_classifiers(digit=3):
    ''' Loads usps.mat data, trains the perceptron and the Nearest centroid classifiers, 
    and plots their weight vector and classifier output
    Definition: compare_classifiers(digit = 3)
    '''
    X, Y = load_usps_data('usps.mat', digit)
    w_ncc, b_ncc = train_ncc(X, Y)
    w_per, b_per, _ = train_perceptron(X, Y)

    pl.figure()
    pl.subplot(2, 2, 1)
    plot_img(w_ncc)
    pl.title('NCC')
    pl.subplot(2, 2, 3)
    plot_histogram(X, Y, w_ncc, b_ncc)

    pl.subplot(2, 2, 2)
    plot_img(w_per)
    pl.title('Perceptron')
    pl.subplot(2, 2, 4)
    plot_histogram(X, Y, w_per, b_per)


def analyse_accuracies_perceptron(digit=3):
    ''' Loads usps.mat data and plots digit recognition accuracy in the linear perceptron
    Definition: analyse_perceptron(digit = 3)
    '''
    X, Y = load_usps_data('usps.mat', digit)
    w_per, b_per, acc = train_perceptron(X, Y)

    pl.figure()
    pl.plot(sp.arange(len(acc)), acc)
    pl.title('Digit recognition accuracy')
    pl.xlabel('Iterations')
    pl.ylabel('Accuracy')
    pl.show()


def plot_img(a):
    ''' Plots one image 
    Definition: plot_img(a) 
    Input:      a - 1D array that contains an image 
    '''
    a2 = sp.reshape(a, (sp.sqrt(a.shape[0]), sp.sqrt(a.shape[0])))
    pl.imshow(a2, cmap='gray')
    pl.colorbar()
    pl.setp(pl.gca(), xticks=[], yticks=[])


def plot_imgs(X, Y):
    ''' Plots 3 images from each of the two classes 
    Definition:         plot_imgs(X,Y)
    Input:       X       -  DxN array of N pictures with D pixel
                 Y       -  1D array of length N of class labels {1, -1}                  
    '''
    pl.figure()
    for i in sp.arange(3):
        classpos = (Y == 1).nonzero()[0]
        m = classpos[sp.random.random_integers(0, classpos.shape[0] - 1)]
        pl.subplot(2, 3, 1 + i)
        plot_img(X[:, m])
    for i in sp.arange(3):
        classneg = (Y != 1).nonzero()[0]
        m = classneg[sp.random.random_integers(0, classneg.shape[0] - 1)]
        pl.subplot(2, 3, 4 + i)
        plot_img(X[:, m])
    pl.show()


#image_data, labels = load_usps_data('usps.mat', 6)
#plot_imgs(image_data, labels)
#perc_weights, perc_bias, perc_acc = train_perceptron(image_data, labels)
#analyse_accuracies_perceptron(3)
#ncc_weights, ncc_bias = train_ncc(image_data, labels)
#plot_histogram(image_data, labels, ncc_weights, ncc_bias)
compare_classifiers(9)
