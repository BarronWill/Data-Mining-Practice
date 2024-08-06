import numpy as np
import pandas as pd


# mat = np.random.randint(0,100, size=(3,3))
# print(mat)
#
# dis_mat = np.round(np.sqrt(((mat[:, np.newaxis] - mat) ** 2).sum(axis=2)), 3)
# print(dis_mat)
# print(np.argmax(mat))


import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def find_closest_clusters(clusters):
    min_distance = float('inf')
    closest_clusters = None
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distance = euclidean_distance(clusters[i]['centroid'], clusters[j]['centroid'])

            if distance < min_distance:
                min_distance = distance
                closest_clusters = (i, j)
    return closest_clusters

def merge_clusters(clusters, i, j):
    merged_cluster = {
        'points': clusters[i]['points'] + clusters[j]['points'],
        'centroid': np.mean(clusters[i]['points'] + clusters[j]['points'], axis=0)
    }
    del clusters[j]
    del clusters[i]
    clusters.append(merged_cluster)

def agglomerative_clustering(X, n_clusters):
    clusters = [{'points': [x], 'centroid': x} for x in X]
    while len(clusters) > n_clusters:
        i, j = find_closest_clusters(clusters)
        merge_clusters(clusters, i, j)
    return clusters

# Ví dụ sử dụng
X = [[5, 3],
     [10, 15],
     [15, 12],
     [24, 10],
     [30, 30],
     [85, 70],
     [71, 80],
     [60, 78],
     [70, 55],
     [80, 91]]

clusters = [{'points': [x], 'centroid': x} for x in X]
clusters = agglomerative_clustering(X, 2)
print(clusters)
