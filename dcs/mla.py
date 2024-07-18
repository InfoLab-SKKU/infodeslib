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


class MLA: 
    """
    Late Fusion version of Modified Local Accuracy (MLA). 
    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.   
    knn_metric: str or callable, default='minkowski': {'minkowski', 'cosine', 'manhattan', 'euclidean'} 
    dimensionality_reduction : Boolean (Default = False) 
    reduction_technique : {'pca', 'kernel_pca'}
    n_components : int (Default = 5) 
    cbr_features : list of features for Cased-based reasoning XAI 
    colors : dictionary of assigning colors for each class 
    
    """
    
    def __init__(self, pool_classifiers=None, feature_subsets=None, k=7, knn_metric='minkowski',
                 dimensionality_reduction=False, reduction_technique='pca', n_components = 5, cbr_features = None, 
                 colors=None): 
        
        self.pool_classifiers         = pool_classifiers 
        self.feature_subsets          = feature_subsets
        self.k                        = k 
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
    For the given new test example, the funtion returns k neighbors
    
    """
    def get_region_of_competence(self, query, predicted_class): 
        validation_w = self.X_dsel[:] 
        validation_w['target'] = self.y_dsel
        validation_w = validation_w[validation_w.target == predicted_class][:] 
        X = validation_w.drop(['target'], axis=1) 
        y = validation_w['target'] 
        
        if self.dimensionality_reduction: 
            if self.reduction_technique == 'pca': 
                pca = PCA(n_components=self.n_components)  
            elif self.reduction_technique == 'kernel_pca': 
                pca = KernelPCA(n_components=self.n_components, kernel="rbf", 
                                       gamma=10, fit_inverse_transform=True, alpha=0.1)
            
            pca.fit(X)
            dsel_pca =  pca.transform(X)  
            query_pca = pca.transform(query)  
            
            nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(dsel_pca) 
            distances, indices = nbrs.kneighbors(query_pca, return_distance=True)  
            
        else:  
            nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(X)         
            distances, indices = nbrs.kneighbors(query.values.reshape(1, -1), return_distance=True)   
        
        
        roc = X.iloc[indices[0]] 
        roc_labels = y.iloc[indices[0]]   
        
        return roc, roc_labels, distances[0] 
    
    
    # Define the distance weighting function
    def distance_weighting(self, distance):
        return 1 / (distance + 0.1) 
    
    """
    The competence of the base classifiers is simply estimated as the
    number of samples in the region of competence that it
    correctly classified.
    """ 
    def estimate_competence(self, query):  
        num_modalities = len(self.feature_subsets)

        competence_acc_list = [] 
        average_correct_probs_list = [] 

        for i in range(num_modalities): 
            t_pred = self.pool_classifiers[i].predict(query[self.feature_subsets[i]]) 
            
            roc, roc_labels, distances = self.get_region_of_competence(query, t_pred[0]) 
            
            probability = self.pool_classifiers[i].predict_proba(roc[self.feature_subsets[i]]) 
            preds = np.argmax(probability, axis=1)
            probs = np.max(probability, axis=1)
            average_correct_probs = 0 

            corrects = 0 
            distance_weight_sum = 0 
            model_weight = 0

            for j in range(len(roc_labels)): 
                distance_weight_sum += self.distance_weighting(distances[j]) 
                
                if preds[j] == roc_labels.iloc[j]: 
                    average_correct_probs += probs[j]
                    corrects = corrects + 1 
                    model_weight += self.distance_weighting(distances[j]) 

            competence = round(model_weight/distance_weight_sum, 3)
            
            if corrects > 0: 
                average_correct_probs = round(average_correct_probs / corrects, 3) 

            competence_acc_list.append(competence)
            average_correct_probs_list.append(average_correct_probs) 
                
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
        
        # 1) estimate competence          
        competences = self.estimate_competence(query)
        
        # 2) select models 
        self.select(competences) 
        
        # 3) predict 
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