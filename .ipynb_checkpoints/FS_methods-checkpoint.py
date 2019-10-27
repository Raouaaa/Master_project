
from skfeature.function.similarity_based import reliefF
#MutInfFS
from skfeature.function.information_theoretical_based import MIFS
#laplacian
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
# norm
from skfeature.function.sparse_learning_based import ll_l21
from skfeature.utility.sparse_learning import *
#UDFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import gini_index
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.similarity_based import SPEC
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.streaming import alpha_investing
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.sparse_learning_based import ls_l21
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W

from boruta_py.boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def relief_FS(X_train,y_train):
    score=reliefF.reliefF(X_train,y_train)
    idx=reliefF.feature_ranking(score)
    return(idx,score)

def MIFS_FS(k,X_train,y_train):
    idx = MIFS.mifs(X_train, y_train, n_selected_features=k)
    #print(idx)
    return(idx)

def lap_score_FS(X):
    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)
    # obtain the scores of features
    score = lap_score.lap_score(X, W=W)
    idx=lap_score.feature_ranking(score)
    return(idx,score)

def ll_l21_FS(X_train,y,train_index):
    Y = construct_label_matrix_pan(y)
    Y_train=Y[train_index]
    Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X_train, Y[train_index], 0.1, verbose=False)
    #print("weight ",Weight)
    idx = feature_ranking(Weight)
    return(idx,Weight)

def UDFS_FS(X):
    Weight = UDFS.udfs(X)
    idx=feature_ranking(Weight)
    return(idx,Weight)
    
    
def fisher_score_FS(X_train,y_train):
    score = fisher_score.fisher_score(X_train, y_train)
    idx = fisher_score.feature_ranking(score)
    return(idx,score)

def chi_square_FS(X,y):
    score = chi_square.chi_square(X, y)
    idx = chi_square.feature_ranking(score)
    return(idx,score)

def gini_index_FS(X_train,y_train):
    score = gini_index.gini_index(X_train, y_train)
    # rank features in descending order according to score
    idx = gini_index.feature_ranking(score)
    return(idx,score)

def FCBF_FS(X_train,y_train,k):
    #idx = FCBF.fcbf(X_train, y_train, n_selected_features=k)
    #return(idx)
    return(fcbf(X_train, y_train, n_selected_features=k))
def trace_ratio_FS(X,train_index,y_train):
    _,k=X.shape
    feature_idx, feature_score, subset_score=trace_ratio.trace_ratio(X[train_index], y_train, k, style='fisher')
    return(feature_idx,feature_score)
def spec_FS(X_train):
    
    kwargs = {'style': 0}

    # obtain the scores of features
    score = SPEC.spec(X_train, **kwargs)

    # sort the feature scores in an descending order according to the feature scores
    idx = SPEC.feature_ranking(score, **kwargs)
    return(idx,score)

def CIFE_FS(X_train,y_train):
    F, J_CMI, MIfy=CIFE.cife(X_train, y_train, n_selected_features=num_fea)
    return(F)

def alpha_investing_FS(X_train,y_train):
    idx = alpha_investing.alpha_investing(X_train, y_train, 0.05, 0.05)
    return(idx)

def CMIM_FS(X_train,y_train,num_fea):
    idx,_,_ = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
    #F, J_CMIM,MIfy= CMIM.cmim(X_train, y_train, n_selected_features=k)
    return(idx)

def ls_l21_FS(X_train,y,train_index):
    Y = construct_label_matrix_pan(y)
    Y_train=Y[train_index]
    W, obj, value_gamma=ls_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
    idx = feature_ranking(W)

    return(idx,W)

def MCFS_FS(X_train,k):
    # construct affinity matrix
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X_train, **kwargs)
    num_fea_ = k    # specify the number of selected features
    num_cluster = 20    # specify the number of clusters, it is usually set as the number of classes in the ground truth

    # obtain the feature weight matrix
    Weight = MCFS.mcfs(X_train, n_selected_features=num_fea_, W=W, n_clusters=20)

    # sort the feature scores in an ascending order according to the feature scores
    idx = MCFS.feature_ranking(Weight)
    return(idx,Weight)


def boruta_FS(X_train,y_train):
    rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
   # Define Boruta feature selection method 
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
    feat_selector.fit(X_train, y_train)
    idx=feat_selector.ranking_
    
    return(idx)