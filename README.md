## Infodeslib: Python Library for Dynamic Ensemble Learning using Late Fusion of Multimodal Data
There has been a notable increase in research focusing on dynamic selection (DS) techniques within the field of ensemble learning. This leads to the development of various techniques for ensembling multiple classifiers for a specific instance or set of instances during the prediction phase. Despite this progress, the design and development of DS approaches with late fusion settings and their explainability remain unexplored. This work proposes an open-source Python library, Infodeslib, to address this gap. The library provides an implementation of several DS techniques, including four dynamic classifier selections and seven dynamic ensemble selection techniques, all of which are integrated with late data fusion settings and novel explainability features. Infodeslib offers flexibility and customization options, making it a versatile tool for various complex applications that require the fusion of multimodal data and various explainability features. Multimodal data, which integrates information from diverse sources or sensor modalities, is a common and essential setting for real-world problems, enhancing the robustness and depth of data analysis. These data can be fused in two main ways: early fusion, where different modalities are combined at the feature level before model training, and late fusion, where each modality is processed separately and the results are combined at the decision level. 

For more details, please check [our paper](https://openreview.net/forum?id=WtM2HEkxwo). 

### Documentation   
https://infodeslib.readthedocs.io/en/latest/ 

### Installation 

```bash
pip install infodeslib
```

###  Requirement 
- install SHAP (0.41.0)


### Example 

Loading necessary libraries and dataset:  

```python
import warnings
warnings.filterwarnings('ignore') 

import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import accuracy_score 

## Load simple open dataset 
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target 

```

Split the dataset into training, validation for DES (DSEL), and testing. 

```python
X = df.drop(['target'], axis=1) 
y = df.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_pool, X_dsel, y_pool, y_dsel   = train_test_split(X_train, y_train, test_size=0.30, random_state=42) 

```

1. Models and Feature sets Generation 

```python
model1 = SVC(probability=True, random_state=42)
model2 = RandomForestClassifier(random_state=42) 
model3 = KNeighborsClassifier() 

feature_set1 = data.feature_names[:10] 
feature_set2 = data.feature_names[10:20]
feature_set3 = data.feature_names[20:]

model_pool = [model1, 
              model2, 
              model3]

feature_sets = [feature_set1, 
                feature_set2, 
                feature_set3] 
```

2. Train the models (pool): 

```python 
for i in range(len(model_pool)): 
    model_pool[i].fit(X_pool[feature_sets[i]], y_pool)
    
    acc = round(model_pool[i].score(X_dsel[feature_sets[i]], y_dsel), 3) 
    print("[DSEL] Model {} acc: {}".format(i, acc)) 

    acc = round(model_pool[i].score(X_test[feature_sets[i]], y_test), 3)  
    print("[Test] Model {} acc: {}".format(i, acc))  
```

3. Usage of our library: 

```python
import shap 
from infodeslib.des.knorau import KNORAU 

# initializing 
knorau = KNORAU(model_pool, feature_sets, k=7)
knorau.fit(X_dsel, y_dsel)
``` 

4. Testing 

```python 
preds =  knorau.predict(X_test)  

acc = round(accuracy_score(y_test, preds), 3) 
print("[Test] acc: {}".format(acc))
```

5. Explainability 

```python 
colors = {0: 'red', 1: 'green'}  

knorau = KNORAU(model_pool, feature_sets, k=7, colors=colors)
knorau.fit(X_dsel, y_dsel) 
```

```python 
index = 18
query = X_test.iloc[[index]]

## Make plot=True 
knorau.predict(query, plot=True)
```
