from sklearn.datasets import make_blobs , make_moons
import matplotlib.pyplot as plt

x1,labels1 = make_blobs(n_samples=100, cluster_std=1.0 , centers=3, random_state=42)
plt.scatter(x1[:,0], x1[:,1], c=labels1)
plt.title("make_blobs")
plt.savefig("session14/make_blobs.png")
plt.clf()

x2,labels2 = make_moons(n_samples=100, noise=0.1)
plt.scatter(x2[:,0], x2[:,1], c=labels2)
plt.title("make_moons")
plt.savefig("session14/make_moons.png")
plt.clf()
