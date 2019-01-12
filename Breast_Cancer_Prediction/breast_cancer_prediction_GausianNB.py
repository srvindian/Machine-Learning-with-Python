# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:33:00 2019

@author: Sourav
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
#print(label_names)
#%%
# splitting the data into training set(60%) and test set(40%)
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)
#%%
gnb = GaussianNB() # initialize the model
model = gnb.fit(train, train_labels)
#%%
preds = gnb.predict(test) #applying the trained model over the test data
#print(preds)
[label_names[i] for i in preds]
#%%
print('Training Accuracy: ',accuracy_score(train_labels, gnb.predict(train)))
print('Test Accuracy: ',accuracy_score(test_labels, preds))

#%% plotting using PCA
x = StandardScaler().fit_transform(train)
pca = PCA(n_components = 2)
pcomps = pca.fit_transform(x)
#%%
col=['red' if i==1 else 'green' for i in train_labels]
fig, ax = plt.subplots()
ax.scatter(pcomps[:, 0], pcomps[:, 1], c = col)