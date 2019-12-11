# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:36:41 2019

@author: varun
"""
from mpl_toolkits.mplot3d import Axes3D
import os, copy, random
import numpy as np
import pylab as py
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression


# directory & file setting
main_dir = os.getcwd()+r'\\'
data_name = main_dir+'match_vectors.csv'
data_extended_name=main_dir+'match_vectors_extended.csv'

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
ratio=0.8


def main(data_file, plot=False):
    LABEL=[1,-1,0]
    data = shuffle(np.genfromtxt(data_file, delimiter=',',skip_header=1), random_state=0)
    
    X=data[:,:-1]
    y=data[:,-1]
    
    X_quadratic_features=gen_quadratic_features(X)
    #print()
    i_split = int(len(y)*ratio)
    X_training=X[:i_split]
    X_quadratic_training=X_quadratic_features[:i_split]
    X_quadratic_test=X_quadratic_features[i_split:]
    y_training=y[:i_split]
    X_test=X[i_split:]
    y_test=y[i_split:]
    y_t_is_L=np.zeros(i_split) #tie is loss
    y_t_is_W=np.zeros(i_split) #tie is win
    
    for i in range(i_split):
        if y[i] < 1:    # draw or loss
            y_t_is_L[i] = 0
        else:
            y_t_is_L[i] = 1
        
        if y[i] > -1:    # draw or win
            y_t_is_W[i] = 0
        else:
            y_t_is_W[i] = 1
            
        
    
    t_is_L_clf=LogisticRegression(solver='lbfgs', multi_class='ovr',C=1e4,tol=1e-6).fit(X_training, y_t_is_L)
    t_is_W_clf=LogisticRegression(multi_class='ovr',C=1e4,tol=1e-6).fit(X_training, y_t_is_W)
    
    '''
    This Block here does multinomial on linear features
    '''
    
    clf_multi=LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1e4,tol=1e-6, max_iter=10000).fit(X_training, y_training)
    prediction=clf_multi.predict(X_test)
    prediction_train=clf_multi.predict(X_training)
    model_multi_probs=clf_multi.predict_proba(X_training)
    model_test_probs=clf_multi.predict_proba(X_test)
    accuracy=np.sum(prediction==y_test)/len(y_test)
    accuracy_training=np.sum(prediction_train==y_training)/len(y_training)
    print('multinomial accuracy='+str(accuracy))
    print('training accuracy for multinomial='+str(accuracy_training))
    #print(prediction)
    print('the confusion matrix on the test set')
    if plot:
        
        plot_prediction(X_test,y_test, prediction, clf_multi.predict_proba(X_test), title='multinomial_avg_roster')
        plot_prediction(X_training,y_training, prediction_train, clf_multi.predict_proba(X_training), title='multinomial_avg_roster_training')
    #print(clf_multi.coef_[:])
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test,prediction))
    print(classification_report(y_test,prediction))
    
    print('now the confusion matrix for training')
    print(confusion_matrix(y_training,prediction_train)) #this prints the confusion matrix
    print(classification_report(y_training,prediction_train))
    
    p_win=t_is_L_clf.predict_proba(X_training)
    test_layer1_win=t_is_L_clf.predict_proba(X_test)
    p_loss=t_is_W_clf.predict_proba(X_training)
    test_layer1_loss=t_is_W_clf.predict_proba(X_test)
    #print(p_win)
    #print(p_loss)
    #print(clf_multi.classes_)
    
    clf_multi_prob_fts=LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1e4,tol=1e-6, max_iter=10000).fit(model_multi_probs, y_training)
    prediction=clf_multi_prob_fts.predict(model_test_probs)
    accuracy=np.sum(prediction==y_test)/len(y_test)
    
    
    print('probability_feature_accuracy='+str(accuracy))
    if plot:
        
        plot_prediction(X_test,y_test, prediction, clf_multi_prob_fts.predict_proba(model_test_probs), title='multinomial_probabiility_avg_roster')
    
    #from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test,prediction))
    print(classification_report(y_test,prediction))
    '''
    This Block here does logistic refression on quadratic features
    '''
    
    
    clf_multi_quadratic=LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1e4,tol=1e-6, max_iter=10000)\
    .fit(X_quadratic_training, y_training)
    #print(clf_multi_quadratic.coef_)
    prediction=clf_multi_quadratic.predict(X_quadratic_test)
    
    accuracy=np.sum(prediction==y_test)/len(y_test)
    #print(prediction)
    print('multinomial quadratic accuracy='+str(accuracy))
    print(confusion_matrix(y_test,prediction))
    print(classification_report(y_test,prediction))
    
    
    prediction_train=clf_multi_quadratic.predict(X_quadratic_training)
    accuracy_train=np.sum(prediction_train==y_training)/len(y_training)
    print('trainng accuracy for quadratic multinomial='+str(accuracy_train))
    print(confusion_matrix(y_training,prediction_train)) #this prints the confusion matrix
    print(classification_report(y_training,prediction_train))
   
   
    
    #plot_prediction(X_test,y_test, prediction, clf_multi_quadratic.predict_proba(X_quadratic_test), title='multinomial_quadratic')
   
    
    
    '''
    model_prob_features=[]
    for i in range(i_split):
        model_prob_features.append([p_win[i,0],p_loss[i,0],np.max([0,1-p_win[i,0]-p_loss[i,0]])])
        
    model_test_features=[]
    
    for i in range (len(test_layer1_loss)):
        model_test_features.append([test_layer1_win[i,0],test_layer1_loss[i,0],np.max([0,1-test_layer1_loss[i,0]-test_layer1_win[i,0]])])
    
    model_prob_features=np.array(model_prob_features)
    model_test_features=np.array(model_test_features)
    final_clf=LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1e4,tol=1e-6, max_iter=10000).fit(model_prob_features, y_training)
    
    #second_training_loss_clf=LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1e4,tol=1e-6).fit(model_prob_features, y_t_is_W)
    
    prediction=final_clf.predict(model_test_features)
    #print(final_clf.coef_[0])
    
    plot_prediction(X_test,y_test, prediction, final_clf.predict_proba(model_test_features), title='one-layer-reductive')
    
    accuracy=np.sum(prediction==y_test)/len(y_test)
    print(accuracy)
    
    final_win_clf=LogisticRegression(solver='lbfgs', multi_class='ovr',C=1e4,tol=1e-6, max_iter=10000).fit(model_prob_features, y_t_is_L)
    final_loss_clf=LogisticRegression(solver='lbfgs', multi_class='ovr',C=1e4,tol=1e-6, max_iter=10000).fit(model_prob_features, y_t_is_W)
    predict_win=final_win_clf.predict_proba(model_test_features)
    predict_loss=final_loss_clf.predict_proba(model_test_features)
    print(predict_win[0])
    print(predict_loss[0])
    
     
    #fig=plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    prediction=[]
    #print(model_prob_features[3000,1])
    colors=['red','green','blue']
    #print(colors[-1])
    
    for i in range(len(y_test)):
        
        #ax.scatter(model_test_features[i,0],model_test_features[i,1], model_test_features[i,2], color=colors[int(y_test[i])])
        
        #print(np.argmax([predict_win[i,0],predict_loss[i,0],1-predict_win[i,0]-predict_loss[i,0]]))
        if (model_test_features[i,0]>=0.5 ) :
            prediction.append(1)
        elif (model_test_features[i,0]>=0.5) :
            prediction.append(-1)
        else:
            prediction.append(LABEL[np.argmax([model_test_features[i,0],model_test_features[i,1], np.max([0,1-model_test_features[i,0]-model_test_features[i,1]])])])
    #plt.show()        
        
    accuracy=np.sum(prediction==y_test)/len(y_test)
    print(accuracy)
    
    plot_prediction(X_test,y_test, prediction, predict_win[:,0],title='two-layer-reductive')
    '''       
    
    
    
def plot_prediction(x_test, y_test, y_predict, predict_prob, title=''):
    accuracy = np.sum(y_test == y_predict)/len(y_test)
    h=predict_prob
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
    py.savefig(main_dir+title+'_scikit.png')
    py.close()

    return None
    
    
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
    main(data_extended_name, plot=False)
    