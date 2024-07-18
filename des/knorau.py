import math 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt  
from sklearn.metrics import accuracy_score
from .util.utils import *
from .util.diversity import *  
from .base import *

from IPython.display import display 
import warnings
warnings.filterwarnings('ignore')

import shap 
import seaborn as sns 
sns.set_style("whitegrid") 


class KNORAU(BaseDES): 
    """
    Late Fusion version of k-Nearest Oracles Union (KNORA-U). 
    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers. 
    DFP : Boolean (Default = False)
        Determines if the dynamic frienemy pruning is applied.     
    knn_metric: str or callable, default='minkowski': {'minkowski', 'cosine', 'manhattan', 'euclidean'} 
    dimensionality_reduction : Boolean (Default = False) 
    reduction_technique : {'pca', 'kernel_pca'}
    n_components : int (Default = 5) 
    cbr_features : list of features for Cased-based reasoning XAI 
    colors : dictionary of assigning colors for each class 
    
    """
    
    def __init__(self, pool_classifiers=None, feature_subsets=None, k=7, DFP=False, knn_metric='minkowski',
                 dimensionality_reduction=False, reduction_technique='pca', n_components = 5, cbr_features = None, 
                 colors=None): 
        
        super(KNORAU, self).__init__(pool_classifiers=pool_classifiers, 
                                     feature_subsets=feature_subsets, 
                                     k=k, 
                                     DFP=DFP, 
                                     knn_metric=knn_metric,
                                     dimensionality_reduction=dimensionality_reduction, 
                                     reduction_technique=reduction_technique, 
                                     n_components = n_components, 
                                     cbr_features = cbr_features, 
                                     colors=colors) 
        
    
    def fit(self, X_dsel=None, y_dsel=None): 
        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel  
        
        self.plot = False   
        self.prepare_explainers() 
        
        
    """
    Select the base classifiers for the classification of the query sample. 
     
    """ 
    def select(self, competences):    
        selected_models_indices = [] 

        while(len(selected_models_indices) <= 0): 
            for i in range(len(competences)): 
                if competences[i] > 0: 
                    selected_models_indices.append(i) 

            if len(selected_models_indices) == 0: 
                for i in range(len(competences)):
                    selected_models_indices.append(i)

        # building contribution table of selected models for explainability (XAI)*
        if self.plot: 
            model_names  = []
            weights_list = []
            for i in selected_models_indices: 
                name = self.pool_classifiers[i].__class__.__name__ + "_" + str(i)
                model_names.append(name)
                weights_list.append(competences[i])
            
            model_dict = {"Model": model_names, "Competence": weights_list} 
            contributionDF = pd.DataFrame(model_dict) 
            
            self.contributionDF = contributionDF
            
        self.selected_models_indices = selected_models_indices   
        self.selected_models_weights = competences  
    
    
