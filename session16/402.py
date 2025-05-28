import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')           # Turn off the warnings. 


df = sns.load_dataset('iris')
X = df.drop(columns=['species'])                       
Y = df['species']                                      
header_X = X.columns                                  
print(df.head())

kmeans = KMeans(n_clusters=3, random_state=123)         
kmeans.fit(X)                                           
res = pd.Series(kmeans.labels_)  

case0 = Y[res==0]
print(case0.value_counts())

case1 = Y[res==1]
print(case1.value_counts())

case2 = Y[res==2]
print(case2.value_counts())

# A list that contains the learned labels.
learnedLabels = ['Virginica','Setosa','Versicolor']  

# Print out the cluster centers (centroids).
np.round(pd.DataFrame(kmeans.cluster_centers_,columns=header_X,index=['Cluster 0','Cluster 1','Cluster 2']),2)

type(case0)

# Visualize the labeling content of the cluster 0. 
sns.countplot(x=case0).set_title('Cluster 0')
plt.savefig('session16/fig/402/cluster0.png')

# Visualize the labeling content of the cluster 1. 
sns.countplot(x=case1).set_title('Cluster 1')
plt.savefig('session16/fig/402/cluster1.png')

# Visualize the labeling content of the cluster 2. 
sns.countplot(x=case2).set_title('Cluster 2')
plt.savefig('session16/fig/402/cluster2.png')
# For a given observation of X, predict the species from what we have learned. 
# Case #1.
X_test = {'sepal_length': [7.0] ,'sepal_width': [3.0] , 'petal_length': [5.0]  ,'petal_width': [1.5] }   # Only X is given.
X_test = pd.DataFrame(X_test)
predCluster = kmeans.predict(X_test)[0]
print("Predicted cluster {} with the most probable label '{}'".format(predCluster,learnedLabels[predCluster]))

# Case #2.
X_test = {'sepal_length': [4.5] ,'sepal_width': [3.0] , 'petal_length': [1.0]  ,'petal_width': [1.0] }   # Only X is given.
X_test = pd.DataFrame(X_test)
predCluster = kmeans.predict(X_test)[0]
print("Predicted cluster {} with the most probable label '{}'".format(predCluster,learnedLabels[predCluster]))

# Case #3.
X_test = {'sepal_length': [6.0] ,'sepal_width': [3.0] , 'petal_length': [4.0]  ,'petal_width': [1.0] }   # Only X is given.
X_test = pd.DataFrame(X_test)
predCluster = kmeans.predict(X_test)[0]
print("Predicted cluster {} with the most probable label '{}'".format(predCluster,learnedLabels[predCluster]))