# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:55:44 2019

@author: Varun
"""
import numpy as np
import os, copy, random
import pylab as py
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

main_dir = os.getcwd()+r'\\'
shuffled_data_name = main_dir+'match_vectors_no_label_shuffled.csv'
data_extended_name= main_dir+'match_vectors_extended.csv'
data_name=main_dir+'match_vectors.csv'
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
ratio=0.8

def main(data_file):
    LABEL=[1,-1,0]
    data = shuffle(np.genfromtxt(data_file, delimiter=',',skip_header=1), random_state=0)
    X=data[:,:-1]
    y=data[:,-1]
    #X_quadratic_features=gen_quadratic_features(X)
    #print()
    i_split = int(len(y)*ratio)
    X_training=X[:i_split]
    #X_quadratic_training=X_quadratic_features[:i_split]
    #X_quadratic_test=X_quadratic_features[i_split:]
    y_training=y[:i_split]
    X_test=X[i_split:]
    y_test=y[i_split:]
    for i in range (10,11):
        clf=MLPClassifier(solver='lbfgs', alpha=1e-4, activation='tanh', hidden_layer_sizes=(11), max_iter=20000)
        clf.fit(X_training,y_training)
        #print(clf.out_activation_)
            
        predictions = clf.predict(X_test)
        accuracy=np.sum(predictions==y_test)/len(y_test)
        print('NN accuracy for '+str(i)+' nodes ='+str(accuracy))
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    
        
    '''
    
    clf2=MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', hidden_layer_sizes=(10,10,10,5,5), max_iter=1000000)
    clf2.fit(X_quadratic_training,y_training)
    
    predictions_quadratic=clf2.predict(X_quadratic_test)
    accuracy_q=np.sum(predictions_quadratic==y_test)/len(y_test)
    
    
    print(confusion_matrix(y_test,predictions_quadratic))
    print(classification_report(y_test,predictions_quadratic))
    print('NN accuracy on quadratic features='+str(accuracy_q))
    '''
    
    
    
    
    
def gen_quadratic_features(X):
    new_features=[]
    for i in range(len(X)):
        x=X[i]
        #if i ==1:
        #print(x)
        y=np.outer(x,x)
        y.flatten()
        #print(y)
        x=np.append(x,y)
        new_features.append(x)
    print(np.shape(new_features))
        
    return np.array(new_features) 

if __name__ == '__main__':
    main(data_extended_name)  
    