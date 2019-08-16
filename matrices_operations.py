import numpy as np
from statistics import mean
import math
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score


def training(cv,k,FS_method):
    accuracy=[]
    featureranking=[]
    correct=0
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if FS_method==reliefF:
            idx=relief_FS(X_train,y_train)
        elif FS_method==MIFS: ##ERROR
            idx=MIFS_FS(k,X_train,y_train)
        elif FS_method==lap_score:
            idx=lap_score_FS()
        elif FS_method==ll_l21:
            idx=ll_l21_FS(X_train,y,train_index)
        elif FS_method==UDFS:
            idx=UDFS_FS()
        elif FS_method==fisher_score:
            idx=fisher_score_FS(X_train,y_train)
        elif FS_method==chi_square:
            idx=chi_square_FS(X_train,y_train)
        elif FS_method==gini_index:
            idx=gini_index_FS(X_train,y_train)
        selected_features = X[:, idx[0:k]]
        featureranking.extend([idx])

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train_index], y[train_index])  # predict the class labels of test data
        y_predict = clf.predict(selected_features[test_index])
        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test_index], y_predict)
        correct = correct + acc

        # output the average classification accuracy over all folds
    accuracy=float(correct)/cv.get_n_splits(X)
    return(np.array(featureranking),accuracy)

def intersection(a,b):
    sum_=0
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            sum_+=np.sum((a[x,y]==b[x,y]))
    percentage=(sum_*100)/a.size
    return(percentage)

def FS_to_FS_similarity(FS_k):
    n=len(FS_k)
    m=np.ones((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            m[i,j]=intersection(FS_k[i],FS_k[j])
            m[j,i]=m[i,j]
    for i in range(n):
        m[i,i]=intersection(FS_k[i],FS_k[i])/100
    return(m)

def avg_similarity(pool_FS,num_fea,cv):
    '''Input:
    
       pool_FS: pool of Feature selection methods
       num_fea: a list of different fs numbers in this case the threshold varies from 10 to 100 with step 10
       cv: Cross validation method (5-fold, 10_fold, LOO)
       
       Output: 
       
       average_similarity: average similarity matrix'''
    S_k={}
    similarities=[]
    for k in num_fea:
        FS_k=[]
    
        for FS in pool_FS:
            _feature_ranking,_acc=training(cv,k,FS)
            #FS_k_ranking.append(_feature_ranking)
            FS_k.append(_feature_ranking)
        S_k[k]=FS_to_FS_similarity(FS_k)
    for k in num_fea:
        similarities.append(S_k[k])
    average_similarity=sum(similarities)/len(num_fea)
    return(average_similarity)

def cost(ai,aj):
    sigma=10
    return(math.exp(-abs(ai-aj)/sigma))

def matrix_acc(list_):
    n=len(list_)
    m=np.ones((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            m[i,j]=cost(list_[i],list_[j])
            m[j,i]=m[i,j]
    for i in range(n):
        m[i,i]=cost(list_[i],list_[i])
    
    return(m)

def accuracy_similarity_matrix(pool_FS,num_fea,cv):
    '''Input:
    
       pool_FS: pool of Feature selection methods
       num_fea: a list of different fs numbers in this case the threshold varies from 10 to 100 with step 10
       cv: Cross validation method (5-fold, 10_fold, LOO)
       
       Output:
       FS-to-FS Accuracy similarity matrix
       
    '''
    #avg_fs_acc_over_k={}
    avg_fs_acc_over_k=[]
    #FS_columns=['reliefF','lap_score','ll_l21','UDFS','fisher_score','chi_square','gini_index']
    for FS in pool_FS:
        acc_k=[]
        for k in num_fea:
            _feature_ranking,_acc=training(cv,k,FS)
            #avg_fs_acc_over_k[k]=
            acc_k.append(_acc)
        #avg_fs_acc_over_k[str(FS.__name__.split('.')[3])]=mean(acc_k)
        avg_fs_acc_over_k.append(mean(acc_k))
    result=matrix_acc(avg_fs_acc_over_k)
    return(result)
