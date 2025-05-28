from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
# Load the data

df= pd.read_csv("session14/make_blobs_x.csv")

KMeans = KMeans(n_clusters=2, random_state=42,n_init="auto")
KMeans.fit(df)
# Predict the labels
labels = KMeans.predict(df)
# Add the labels to the dataframe
df['labels'] = labels
# Plot the data with the labels
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['labels'], cmap='viridis')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("session14/kmeans_clustering.png")
plt.clf()

