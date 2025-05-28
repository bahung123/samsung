from sklearn.datasets import make_blobs , make_moons
import matplotlib.pyplot as plt
import pandas as pd

x1,labels1 = make_blobs(n_samples=200, cluster_std=5.0 ,n_features=2, centers=2, random_state=42)

# save data
pd.DataFrame(x1).to_csv("session14/make_blobs_x.csv", index=False)


plt.scatter(x1[:,0], x1[:,1], c=labels1)
plt.title("make_blobs")
plt.savefig("session14/make_blobs.png")
plt.clf()

# x2,labels2 = make_moons(n_samples=200, noise=0.1, random_state=42) 
# plt.scatter(x2[:,0], x2[:,1], c=labels2)
# plt.title("make_moons")
# plt.savefig("session14/make_moons.png")
# plt.clf()
