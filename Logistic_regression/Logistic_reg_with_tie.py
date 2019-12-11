# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:57:14 2019

@author: varun
"""

import os, copy, random
import numpy as np
import pylab as py
from sklearn.utils import shuffle


# directory & file setting
main_dir = os.getcwd()+r'\\'
data_name = main_dir+'match_vectors_no_label.csv'

shuffled = True    # are matches shuffled randomly?
shuffled_data_name = main_dir+'match_vectors_no_label_shuffled.csv'
LABEL_LEGEND = []
FEATURE_LABEL = ['h_roster_rating',
                 'h_gk_rating',
                 'h_def_rating',
                 'h_mid_rating',
                 'h_off_rating',
                 'a_roster_rating',
                 'a_gk_rating',
                 'a_def_rating',
                 'a_mid_rating',
                 'a_off_rating',
                 'label']


def add_x0(x):
    """ (np.matrix) -> np.matrix
    Add 1's in the 0th column to the given matrix x.
    """
    return_x = np.zeros((x.shape[0], x.shape[1] + 1))
    return_x[:,0] = 1
    return_x[:,1:] = x
    return return_x

def define_xy(data, L_i, ratio=0.8, draw_is_loss=True):
    """ (np.matrix, list of int) -> np.matrix, np.array, np.matrix, np.array
    Given the imported data matrix, generate input feature matrix x and label
    array y, with the desired input features indicated by index list L_i. Add
    intercepts 1 to x. Split x and y into training and test sets according to
    [training:test = 'ratio':1-'ratio'].
    """
    print(draw_is_loss)
    x = np.copy(add_x0(data[:,L_i]))
    y = np.copy(data[:,-1])
    i_split = int(len(y)*ratio)
    #print(len(y))
    # adjust labels for logistic regression
    
    if draw_is_loss:    # draw is a loss
        print('loss')
        for i in range(i_split):
            if y[i] < 1:    # draw or loss
                y[i] = 0
    else:               # draw is a win
        print('win')
        for i in range(i_split):
            #print(y[i])
            if y[i] >= 0:    # win or draw
                y[i] = 0
            else:
                y[i] = 1   # tagging all the lost matches
    
    x_train, x_test = x[:i_split,:], x[i_split:,:]
    y_train, y_test = y[:i_split], y[i_split:]
    #print(y_train)
    return x_train, y_train, x_test, y_test

def plot_prediction(x_test, y_test, theta_d_is_win, theta_d_is_loss, title=''):
    """ (np.matrix, np.array, np.array) -> None
    Plot a linear plot of the predictions on x_test.
    """
    # make predictions
    h, y_predict = [], []
    for i in range(len(y_test)):
        h_i_win = h_logreg(x_test[i,:], theta_d_is_loss)
        
        h_i_loss=h_logreg(x_test[i,:], theta_d_is_win)
        #print([h_i_loss,h_i_win])
        h.append(h_i_loss)
        if h_i_win >= 0.5: 
            y_predict.append(1)
        elif h_i_loss>=0.5: 
            
            y_predict.append(-1)
            #print('loss')
        else:
            labels=[-1,0,1]
            indexer=np.argmax([h_i_loss,h_i_win,1-h_i_win-h_i_loss])
            y_predict.append(labels[indexer])
    
    h, y_predict = np.array(h), np.array(y_predict)

    # calculate accuracy
    accuracy = np.sum(y_test == y_predict)/len(y_test)

    # linear plots
    f, ax = py.subplots(2,1, sharex=True)
    # ax[0]: plot a linear plot, with x being the test examples and y being the
    # probability of loss/draw/win
    ax[0].plot(h)
    f.suptitle(title)
    ax[0].set_ylabel('h(x)')
    # ax[1]: plot loss/draw/win label predictions and actual labels
    ax[1].plot(y_test, 'ro', markersize=4)
    ax[1].plot(y_predict, 'bo', markersize=2)
    ax[1].legend(['label', 'prediction: '+str(round(accuracy*100,2))+'%'])
    ax[1].set_ylabel('label')
    ax[1].set_xlabel('test set datapoints')
    py.savefig(main_dir+title+'_lin_plot_non_reg.png')
    py.close()

    return None

def plot_decision(x, y, theta, L_i, i1, i2, title=''):
    """ (np.matrix, np.array, list of np.arrays, list, int, int) -> None
    Given the dataset (x, y) and parameters theta, plot the dataset in x[L_i[i2]] vs.
    x[L_i[i1]] along with the decision boundary.
    """
    # plot the dataset
    x1, x2 = x[:,i1+1], x[:,i2+1]   # +1 to take account of adding x0 = 1 intercepts
    py.suptitle(title)
    py.plot(x1[y == 0], x2[y == 0], 'r+')
    py.plot(x1[y == 1], x2[y == 1], 'b+')
    # plot the decision boundaries
    x1_min, x1_max = np.min(x1), np.max(x1)
    x_decision = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
    th0, th1, th2 = theta[0], theta[i1+1], theta[i2+1]
    slope = -th1*1./th2
    intercept = -th0*1./th2
    py.plot(x_decision, slope*x_decision + intercept, 'k:')
    py.xlabel(FEATURE_LABEL[L_i[i1]])
    py.ylabel(FEATURE_LABEL[L_i[i2]])
    py.legend(['loss', 'win'])
    py.savefig(main_dir+title+'_decision_boundary_('+str(i1)+','+str(i2)+').png')
    py.close()

    return None
        

def h_logreg(x_i, theta):
    """ (np.array, np.array) -> np.array
    Given input feature vector x_i and parameter theta, return the
    prediction h = sigmoid.
    """
    return 1. / (1 + np.exp(-x_i.dot(theta)))

def grad_l(x, y, theta, lmda=1e1):
    """ (np.matrix, np.array, list of np.arrays) -> np.array
    Given input feature vectors x, labels y, and matrix of theta_i's, return
    grad_{theta_c}l(theta) = SUM{i=1~n}[x^(i) (1{y^(i) = c}
                                         - P(y^(i) = c|x^(i);\theta))].
    """
    h = 1. / (1 + np.exp(-x.dot(theta)))
    grad = (y - h).dot(x)
    # add regularization term
    return grad - lmda*theta

def logreg(x, y, eps=1e-15, alpha=1e-5, lmda=1e1):
    """ (np.matrix, np.array) -> np.array
    Given the input features x and labels y, perform soft-max regression
    and return the optimized feature parameters theta.
    """
    d = len(x[0])   # dimension of x
    # initialize theta_0 = 0
    theta = np.zeros(d)

    # perform gradient descent
    itr = 0
    while True:
        itr += 1
        # update theta
        prev_theta = theta
        theta = theta + alpha * grad_l(x, y, theta, lmda=lmda)
        # update convergence values
        eps_i = np.linalg.norm(prev_theta - theta)
        if itr % 10000 == 0:
            print(str(itr)+'\t'+str(eps_i))
        if eps_i < eps:
            print(str(itr), 'converged', str(eps_i))
            break
    return theta

def do_logreg(data, L_i, trained=False, ratio=0.8, save_plot=True,
              eps=1e-15, alpha=1e-5, lmda=1e1):
    """ (list) -> None
    Perform soft-max regression with input features specified by index
    list L_i:
    #   i       content
    #   0       h_roster_rating
    #   1       h_gk_rating
    #   2       h_def_rating
    #   3       h_mid_rating
    #   4       h_off_rating
    #   5       a_roster_rating
    #   6       a_gk_rating
    #   7       a_def_rating
    #   8       a_mid_rating
    #   9       a_off_rating
    #   10      label (-1, 0, or 1)
    Save the optimized theta.
    Assess prediction by generating linear plots and x_i vs x_j plots.
    """
    feature_title = str(L_i).replace(' ', '')
    # set the training & test datasets
    x_train_d_isloss, y_train_d_isloss, x_test, y_test = define_xy(data, L_i, ratio=ratio)
    x_train_d_iswin, y_train_d_iswin, x_test, y_test = define_xy(data, L_i, ratio=ratio,draw_is_loss=False)
    print(y_train_d_isloss[500:550])
    print(y_train_d_iswin[500:550])
    # import theta1 from theta1.csv if already trained
    if trained:
        theta_d_is_loss = np.genfromtxt(main_dir+feature_title+'_theta_d_is_loss.csv', delimiter=',')
        theta_d_is_win= np.genfromtxt(main_dir+feature_title+'_theta_d_is_win.csv', delimiter=',')
    else:
        theta_d_is_loss = logreg(x_train_d_isloss, y_train_d_isloss, eps=eps, alpha=alpha, lmda=lmda)
        print(theta_d_is_loss)
        theta_d_is_win = logreg(x_train_d_iswin, y_train_d_iswin, eps=eps, alpha=alpha, lmda=lmda)
        print(theta_d_is_win)
        # save theta
        np.savetxt(main_dir+feature_title+'_theta_d_is_win.csv', theta_d_is_win, delimiter=',')
        np.savetxt(main_dir+feature_title+'_theta_d_is_loss.csv', theta_d_is_loss, delimiter=',')
##    theta = np.array([-0.00378961, 0.39707427, -0.41742261])
    # plot a histogram of predictions
    if save_plot:
        plot_prediction(x_test, y_test, theta_d_is_win, theta_d_is_loss, title=feature_title)
        print('prediction_plotted')
        for i in range(len(L_i)):
            for j in range(len(L_i)):
                if i != j: plot_decision(x_test, y_test, theta_d_is_loss, L_i, i, j, title=feature_title)

    return None


if __name__ == '__main__':
    # shuffle data if not done
    if not shuffled:
        data = np.genfromtxt(data_name, delimiter=',')
        np.savetxt(shuffled_data_name, shuffle(data), delimiter=',')

    # import data
    # data file structure:
    #   i       content
    #   0       h_roster_rating
    #   1       h_gk_rating
    #   2       h_def_rating
    #   3       h_mid_rating
    #   4       h_off_rating
    #   5       a_roster_rating
    #   6       a_gk_rating
    #   7       a_def_rating
    #   8       a_mid_rating
    #   9       a_off_rating
    #   10      label (-1, 0, or 1)
    data = np.genfromtxt(shuffled_data_name, delimiter=',')

    # eliminate draws
    _data = []
    for data_i in data:
        #if data_i[-1] != 0:
        _data.append(data_i)
    data = np.array(_data)

    # 1. using all input features
    do_logreg(data, range(0,10), trained=False, save_plot=True,
              eps=1e-10, alpha=1e-6, lmda=1e1)

    # 2. using 0, 5
    do_logreg(data, [0,5], trained=False, save_plot=True,
              eps=1e-10, alpha=1e-6, lmda=1e1)


