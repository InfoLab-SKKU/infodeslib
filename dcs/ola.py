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

from IPython.display import display 
import warnings
warnings.filterwarnings('ignore')

import shap 
import seaborn as sns 
sns.set_style("whitegrid") 


class OLA: 
    """
    Late Fusion version of Overall Local Accuracy (OLA). 
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
        
        self.pool_classifiers         = pool_classifiers 
        self.feature_subsets          = feature_subsets
        self.k                        = k 
        self.DFP                      = DFP
        self.knn_metric               = knn_metric 
        self.dimensionality_reduction = dimensionality_reduction 
        self.reduction_technique      = reduction_technique
        self.n_components             = n_components
        self.cbr_features             = cbr_features
        self.colors                   = colors 
     
    
    def fit(self, X_dsel=None, y_dsel=None): 
        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel  
        
        self.plot = False   
        self.prepare_explainers() 
        
        
    """
    Prepare explainers based on SHAP for all models   
    """ 
    def prepare_explainers(self): 
        explainer_list = [] 
        
        for i in range(len(self.pool_classifiers)):
            explainer = shap.KernelExplainer(self.pool_classifiers[i].predict, self.X_dsel[self.feature_subsets[i]], silent=True)
            explainer_list.append(explainer)
        
        self.explainer_list = explainer_list 
    
    
    """
    Rareness score
    """
    def get_rareness_score(self, n_clusters=20, query=None): 
        # train a KMeans model with N clusters 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(self.X_dsel) 
        
        # predict the cluster for the new sample
        new_sample_cluster = kmeans.predict(query) 
        
        fig, axes = plt.subplots(1, 2, sharex=False, figsize=(16,6)) 
        fig.suptitle('The cluster label of given sample is {}'.format(new_sample_cluster)) 
        
        clusters = kmeans.predict(self.X_dsel) 
        sns.histplot(ax=axes[0], data=clusters, bins=n_clusters, color="#eb801c") 
        axes[0].set_xticks(np.arange(0, n_clusters, 1)) 
        
        pca = PCA(n_components = 2) 
        pca.fit(self.X_dsel) 
        
        dsel_pca =  pca.transform(self.X_dsel)
        query_pca = pca.transform(query)  
        
        scatter = axes[1].scatter(dsel_pca[:, 0], dsel_pca[:, 1], c=clusters, cmap='tab20', s=15) 
        axes[1].scatter(query_pca[:, 0], query_pca[:, 1], c=new_sample_cluster, marker='X', cmap='tab20', s=70)  
        
        # create a legend with unique cluster values and their colors
        handles, labels = scatter.legend_elements()
        axes[1].legend(handles, [f"Cluster {i}" for i in set(clusters)], bbox_to_anchor=(0.5, -0.2), ncol=5)       
        
        plt.show()
        
        # get the cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # compute distances between the new sample and each cluster center
        distances = np.linalg.norm(query.values - cluster_centers, axis=1)
        
        # identify the nearest neighbor cluster
        nearest_neighbor_cluster = np.argmin(distances)
        
        # get the feature values for the nearest neighbor cluster center
        relevant_features = cluster_centers[nearest_neighbor_cluster]
        
        df = pd.DataFrame() 
        df['Features']  = self.X_dsel.columns
        df['Distances'] = relevant_features 
        df = df.sort_values(by='Distances') 
        
        display(df[:10].style.pipe(make_table_highlights, "The important features that contributes to push the sample to the group."))         
    
    """
    Average diversity of models on the pool on the validation data  
    """
    def get_diversity(self, source_model_idx, target_model_idx, diversity_func):
        diversity_func_map = {"Q": Q_statistic, "CC": correlation_coefficient, "DM": disagreement_measure, 
                              "DF": double_fault, "NDF": negative_double_fault, "RE": ratio_errors}
        
        y_source = self.pool_classifiers[source_model_idx].predict(self.X_dsel[self.feature_subsets[source_model_idx]])
        y_target = self.pool_classifiers[target_model_idx].predict(self.X_dsel[self.feature_subsets[target_model_idx]])
        
        diversity = diversity_func_map[diversity_func](self.y_dsel.values, y_source, y_target) 

        return diversity 
        
        
    """
    Average diversity of models on the pool on the validation data  
    """ 
    def get_pool_diversity(self, diversity_func):
        num_models = len(self.pool_classifiers)
        diversity_names_map = {"Q": "Q-statistic", "CC": "Correlation Coefficient", "DM": "Disagreement Measure", 
                               "DF": "Double Fault", "NDF": "Negative Double Fault", "RE": "Ratio Errors"} 
        
        model_names      = [] 
        diversity_scores = [] 
        
        for i in range(num_models): 
            diversity_list = []
            for j in range(num_models):
                diversity_score = self.get_diversity(i, j, diversity_func)
                diversity_list.append(diversity_score) 
            
            avg_diversity = np.mean(np.array(diversity_list)) 
            std_diversity = np.std(np.array(diversity_list))  
            
            
            model_names.append(self.pool_classifiers[i].__class__.__name__ + " [{}]".format(i))
            diversity_scores.append("{}±{}".format(round(avg_diversity, 3), round(std_diversity, 3))) 
        
        resultsDF = pd.DataFrame({"Model": model_names, diversity_names_map[diversity_func]: diversity_scores}) 
        print("[DSEL] Diversity Measurement: ")
        display(resultsDF)
    
    
    """
    Average accuracy of models on the pool on the validation data  
    """   
    def get_average_accuracy(self): 
        num_modalities = len(self.feature_subsets)         
        scores = [] 
        
        for i in range(num_modalities): 
            score = self.pool_classifiers[i].score(self.X_dsel[self.feature_subsets[i]], self.y_dsel)
            scores.append(score)
        
        scores = np.array(scores)
        scores_mean = round(np.mean(scores), 3) 
        scores_std  = round(np.std(scores), 3) 
        print("[DSEL] Average Accuracy: {} ± {}".format(scores_mean, scores_std))   
    
    
    """
    Counting how many samples in DSEL data can be pridicted correctly by any model of the given pool  
    """   
    def get_coverage_score(self):
        num_modalities = len(self.feature_subsets) 
        coverage_table = pd.DataFrame()
        
        for i in range(num_modalities): 
            preds = self.pool_classifiers[i].predict(self.X_dsel[self.feature_subsets[i]])
            coverage_table["Model {}".format(i)] = preds 
        
        coverage_table['True'] = self.y_dsel 
        
        # coverage_table.to_csv("coverage_table.csv", index=False)
        
        for j in range(num_modalities): 
            coverage_table["Model {}".format(j)] = coverage_table["Model {}".format(j)] == coverage_table['True'] 
        
        coverage_table['coverage'] = coverage_table.iloc[:, :].sum(axis=1) 
        
        score = 1 - coverage_table['coverage'].tolist().count(0) / coverage_table.shape[0]  
        score = round(score * 100, 2)
        
        print("Coverage score: {}%".format(score))        
        print("Number of models: {}".format(num_modalities))
        
        names_list = [] 
        count_list = [] 
        percentages_list = [] 
        
        for k in range(num_modalities + 1): 
            count = coverage_table['coverage'].tolist().count(k) 
            percentage = round(count/self.X_dsel.shape[0]*100, 3)  
            name = "Contributors {}/{}".format(k, num_modalities)
            
            names_list.append(name)
            count_list.append(count)
            percentages_list.append(percentage)
            
        
        coverage_dict = {"Contribution": names_list, "Count": count_list, "Percentage": percentages_list} 
        coverageDF = pd.DataFrame(coverage_dict) 
        
        display(coverageDF.style.pipe(make_table_highlights, "Coverage Scores")) 
        
    
    
    """
    Counting how many different classes in region of competence (RoC)
    If the number of classes in RoC, this RoC is indecision region 
    """    
    def get_indecision_region(self, X, y):   
        num_classes = len(set(y.tolist()))  
        if self.plot: 
            print("Number of classes in RoC: {}".format(num_classes))

        if num_classes > 1: 
            return True 
        else: 
            return False
    
    
    """
    For the given new test example, the funtion returns k neighbors
    
    """
    def get_table_CBR(self): 
        df = self.roc[:] 
        df['target'] = self.roc_labels 
        
        if self.cbr_features: 
            self.cbr_features.append('target')
            display(df[self.cbr_features])      
            self.cbr_features.remove('target') 
        else: 
            display(df)
    
    
    """
    For the given new test example, the funtion returns k neighbors
    
    """
    def plot_region(self, X, y, query):
        pca = PCA(n_components = 2) 
        pca.fit(self.X_dsel) 
        
        dsel_pca =  pca.transform(self.X_dsel) 
        neighbors_pca = pca.transform(X) 
        query_pca = pca.transform(query) 
        
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16,6), sharey=True, sharex=True) 
 
        ax1.scatter(dsel_pca[:, 0], dsel_pca[:, 1], c = self.y_dsel.map(self.colors), marker='.', label=self.colors)  
        ax1.scatter(neighbors_pca[:, 0], neighbors_pca[:, 1], c = y.map(self.colors), marker='o', s=50) 
        ax1.scatter(query_pca[:, 0], query_pca[:, 1], marker='X', s=70) 
        
        ax2.scatter(neighbors_pca[:, 0], neighbors_pca[:, 1], c = y.map(self.colors), marker='o', s=50) 
        ax2.scatter(query_pca[:, 0], query_pca[:, 1], marker='X', s=70) 
        
        plt.show() 
        
        ### show table 
        self.get_table_CBR() 
    
    
    """
    For the given new test example, the funtion returns k neighbors
    
    """
    def get_region_of_competence(self, query): 
        if self.dimensionality_reduction: 
            if self.reduction_technique == 'pca': 
                pca = PCA(n_components=self.n_components)  
            elif self.reduction_technique == 'kernel_pca': 
                pca = KernelPCA(n_components=self.n_components, kernel="rbf", 
                                       gamma=10, fit_inverse_transform=True, alpha=0.1)
            
            pca.fit(self.X_dsel)
            dsel_pca =  pca.transform(self.X_dsel)  
            query_pca = pca.transform(query)  
            
            nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(dsel_pca) 
            indices = nbrs.kneighbors(query_pca, return_distance=False)  
            
        else:  
            nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(self.X_dsel)         
            indices = nbrs.kneighbors(query.values.reshape(1, -1), return_distance=False)   
        
        indecision_region = self.get_indecision_region(self.X_dsel.iloc[indices[0]], self.y_dsel.iloc[indices[0]]) 
        
        self.roc = self.X_dsel.iloc[indices[0]] 
        self.roc_labels = self.y_dsel.iloc[indices[0]] 
        self.indecision_region = indecision_region 
    
    
    """
    The competence of the base classifiers is simply estimated as the
    number of samples in the region of competence that it
    correctly classified.
    """ 
    def estimate_competence(self):  
        num_modalities = len(self.feature_subsets)

        competence_acc_list = [] 
        average_correct_probs_list = [] 

        for i in range(num_modalities): 
            probability = self.pool_classifiers[i].predict_proba(self.roc[self.feature_subsets[i]]) 
            preds = np.argmax(probability, axis=1)
            probs = np.max(probability, axis=1)
            average_correct_probs = 0 

            corrects = 0 

            for j in range(len(self.roc_labels)): 
                if preds[j] == self.roc_labels.iloc[j]: 
                    average_correct_probs += probs[j]
                    corrects = corrects + 1 

            accuracy = round(accuracy_score(self.roc_labels, preds), 3)
            
            if corrects > 0: 
                average_correct_probs = round(average_correct_probs / corrects, 3) 

            competence_acc_list.append(accuracy)
            average_correct_probs_list.append(average_correct_probs) 
                
        self.correct_probs = average_correct_probs_list 
        
        return competence_acc_list 
    
    
    """
    If the query is in indecision region, we estimate the competence differently
    If there 3 classes in indecision region, the classifiers that classify at least one sample 
    in each class correctly will be selected 
    """    
    def estimate_indecision_competence(self): 
        num_modalities = len(self.feature_subsets) 
        classes = set(self.roc_labels.tolist())  

        competence_acc_list = []  
        average_correct_probs_list = []  

        rocs_by_classes = [] 
        roc['target'] = self.roc_labels     

        for c in classes: 
            rocs_by_classes.append(self.roc[self.roc['target'] == c][:])         

        for i in range(num_modalities):        
            corrects_list = [] 
            for iroc in rocs_by_classes:
                corrects = 0 
                probability = self.pool_classifiers[i].predict_proba(iroc[self.feature_subsets[i]])  
                preds = np.argmax(probability, axis=1)
                probs = np.max(probability, axis=1)
                average_correct_probs = 0 

                for j in range(len(iroc['target'])): 
                    if preds[j] == iroc['target'].iloc[j]: 
                        average_correct_probs += probs[j]
                        corrects = corrects + 1 

                corrects_list.append(corrects) 

            if 0 not in corrects_list: 
                preds = self.pool_classifiers[i].predict(self.roc[self.feature_subsets[i]])
                probability = self.pool_classifiers[i].predict_proba(self.roc[self.feature_subsets[i]])   
                probs = np.max(probability, axis=1) 

                accuracy = round(accuracy_score(self.roc_labels, preds), 3) 
                competence_acc_list.append(accuracy)

                average_correct_probs = round(sum(probs) / sum(probs), 3)  
                average_correct_probs_list.append(average_correct_probs) 

            else: 
                competence_acc_list.append(0) 
                average_correct_probs_list.append(0) 
                
            
            self.correct_probs = average_correct_probs_list
            
        return competence_acc_list
    
    
    """
    Select the base classifiers for the classification of the query sample. 
     
    """ 
    def select(self, competences):    
        selected_model_index = competences.index(max(competences))  
        n_models = len(self.pool_classifiers)

        # building contribution table of selected models for explainability (XAI)*
        if self.plot: 
            model_names  = []
            weights_list = [] 
            for i in range(n_models): 
                if i == selected_model_index: 
                    name = "*(Selected) " + self.pool_classifiers[i].__class__.__name__ + " [{}]".format(i) 
                else: 
                    name = self.pool_classifiers[i].__class__.__name__ + " [{}]".format(i)
                model_names.append(name)
                weights_list.append(competences[i])
            
            model_dict = {"Model": model_names, "Competence": weights_list} 
            contributionDF = pd.DataFrame(model_dict) 
                        
            self.contributionDF = contributionDF

        self.selected_model_index = selected_model_index   
    
    
    def get_local_feature_importance(self, query):
        label = self.pool_classifiers[self.selected_model_index].__class__.__name__
        shap_values = self.explainer_list[self.selected_model_index].shap_values(query[self.feature_subsets[self.selected_model_index]]) 
            
        df = pd.DataFrame() 
        df['Features']    = self.feature_subsets[self.selected_model_index]
        df['Shap_values'] = shap_values[0]
            
        df = df.sort_values(by='Shap_values', ascending=False) 
        
        sns.barplot(data=df[:10], y="Features", x='Shap_values', color='Orange') 
        
        print("Feature importance of selected model: {}".format(label))
        plt.show()
                
    
    """
    Gives final prediction.  
     
    """  
    def predict_single_sample(self, query, y_true=None): 
        predictions = [] 
        
        # 1) define region of competence 
        self.get_region_of_competence(query)
        
        # plot option
        if self.plot:  
            self.plot_region(self.roc, self.roc_labels, query) 
        
        # 2) estimate competence 
        if self.DFP: 
            if indecision: 
                competences = self.estimate_indecision_competence()  
            else: 
                competences = self.estimate_competence() 
        else: 
            competences = self.estimate_competence()
        
        # 3) select models 
        self.select(competences) 
        
        # 4) predict 
        individual_model_preds = [] 
        individual_model_confs = [] 
        
        if self.plot: 
            for j in range(len(self.pool_classifiers)): 
                pred = self.pool_classifiers[j].predict_proba(query[self.feature_subsets[j]])
                prediction = [value * competences[j] for value in pred[0]] 

                predictions.append(prediction)
                individual_model_preds.append(prediction.index(max(prediction)))
                individual_model_confs.append(max(prediction))

            self.contributionDF['Prediction'] = individual_model_preds 
            self.contributionDF['Confidence'] = individual_model_confs          
            display(self.contributionDF.style.pipe(make_table_highlights, "Contribution of each selected models."))
        
        if self.plot: 
            self.get_local_feature_importance(query)
        
        prediction = self.pool_classifiers[self.selected_model_index].predict(query[self.feature_subsets[self.selected_model_index]]) 
        conf_list  = self.pool_classifiers[self.selected_model_index].predict_proba(query[self.feature_subsets[self.selected_model_index]]) 

 
        if self.plot: 
            if y_true != None: 
                print("[True Label]: {}".format(y_true)) 
            print("========================================================") 
        
        return prediction[0], conf_list[0] 
    
    
    def predict(self, X, plot=False): 
        self.plot = plot 
        preds = [] 
        for i in range(X.shape[0]):
            query = X.iloc[[i]] 

            pred, _ = self.predict_single_sample(query)
            preds.append(pred) 
        
        return preds 
    
    
    def predict_proba(self, X): 
        probas = [] 
        for i in range(X.shape[0]):
            query = X.iloc[[i]] 

            _, proba = self.predict_single_sample(query)
            probas.append(proba) 
        
        return probas 
    
    
    def score(self, X, y): 
        preds = self.predict(X) 
        
        return accuracy_score(y, preds)