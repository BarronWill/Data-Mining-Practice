import numpy as np
import matplotlib.pyplot as plt
from utils import *

def display(arr, labels, centroid):
    color = ['red', 'blue', 'orange', 'green', 'violet', 'yellow']
    markers = ['o', 's', 'D', '^', 'v', '*']
    for i in range(arr.shape[0]):
        plt.scatter(arr[i, 0], arr[i, 1], s=60, c=color[labels[i]], alpha=0.6)
    for i in range(centroid.shape[0]):
        plt.scatter(centroid[i, 0], centroid[i, 1], c='Black', marker=markers[i], edgecolor='face', linewidth=1, s=60, label=f'Centroid {i+1}')
    plt.legend()
    plt.show()


arr = np.random.randint(100, size=(1000,2))

kmeans = Kmeans(k_clusters=5, max_iter=100)
kmeans.fit(arr)
print(kmeans.centroids)
display(arr, kmeans.labels, kmeans.centroids)




