# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:33:53 2019

@author: Sourav
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
raw_data = open('heart.dat', 'rt');
data = np.loadtxt(raw_data, delimiter = " ")
col_names = ['age','gender','chest_pain_type','resting_blood_pressure',\
             'serum_cholestoral_in_mg/dl','fasting_blood_sugar>120',\
             'resting_electrocardiographic_results','maximum_heart_rate_achieved',\
             'exercise_induced_angina','oldpeak','the_slope_of_the_peak',\
             'number_of_major_vessels','thal','heart_disease']
df=pd.DataFrame(data = data[0:,0:],columns=col_names)
#print("dimension of heart data: {}".format(df.shape))
#print(df.groupby('heart_disease').size())
sns.countplot(df['heart_disease'],label='count') 
#%%

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'heart_disease'], df['heart_disease'], stratify=df['heart_disease'], random_state=50)
#%%
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
#%%Accuracy is 9 as per the model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

#%% Evaluation on the given test data
with open('testdata.dat') as f:
    freader = csv.reader(f, delimiter=' ')
    testdata = [[col1, col2, col3, col4, col5, col6, col7, col8, col9,\
                 col10, col11, col12, col13,'0.0']
                for col1, col2, col3, col4, col5, col6, col7, col8, col9,\
                col10, col11, col12, col13, _ in freader]
testdata = [list(map(float,item)) for item in testdata]
length_of_testset = len(testdata)
result = [knn.predict([testdata[i][0:13]]) for i in range(1,length_of_testset)];
result = list(map(int, np.reshape(result,[length_of_testset - 1])))
#%%
print(result)
#%% plotting using PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
x = StandardScaler().fit_transform(X_train)
pca = PCA(n_components = 3)
pcomps = pca.fit_transform(x)
#%%
col=['red' if i==1 else 'green' for i in y_train]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pcomps[:, 0], pcomps[:, 1], pcomps[:, 1], c = col)
plt.show()