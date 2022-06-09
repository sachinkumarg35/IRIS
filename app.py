import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Classification Algorithms')

st.write("""
Which one is best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Breast Cancer', 'Wine')
)

st.write(f"### {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data =pd.read_csv('iris.csv')
        X = data.iloc[:,0:4]
        y = data['Species']
        
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    return X, y
                          
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))
                        
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1,15)
        params['K'] = K
    else:
        max_depth =st.sidebar.number_input('Max Depth', 1, 100)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.number_input('Number of Estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
  
params = add_parameter_ui(classifier_name)
               
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = KNeighborsClassifier(n_estimators=params['n_estimators'],
              max_depth=params['max_depth'], random_state=1234)
    return clf
               
clf = get_classifier(classifier_name, params)

#### CLASSIFICATION #####

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
               
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
               
acc = accuracy_score(y_test, y_pred)
               
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ', acc)
               
st.write("""
## Predict for new data
""")
if dataset_name == 'Iris':
    sepal_length = petal_length = sepal_width = petal_width = 0.0
               
    sepal_length = st.number_input('Sepal.Length', None,None)
    petal_length = st.number_input('Petal.Length', None,None)
    sepal_width = st.number_input('Sepal.Width', None,None)
    petal_width = st.number_input('Petal.Width', None,None)
    predictors = [[sepal_length, sepal_width, petal_length, petal_width]]
               
if st.button('Predict New'):
    pred_species = clf.predict(predictors)
    st.write(f'Species = {pred_species[0]}')
               
               
               
               