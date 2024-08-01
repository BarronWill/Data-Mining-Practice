import numpy as np
import pandas as pd


class Kmeans:
    def __init__(self, k_clusters=2, max_iter=100):
        self.k_clusters = k_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Chuyển dữ liệu sang dạng ma trận
        # X = mat.reshape((-1,3))
        # Khởi tạo centroids dùng k-means++
        self.centroids = self.__kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            # Gán nhãn cho mỗi điểm dữ liệu dựa trên centroid gần nhất (dùng broadcasting)
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=1)

            # Cập nhật centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.k_clusters)], dtype=np.int64)

            # Kiểm tra điều kiện dừng (so sánh centroids mới và cũ)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        # Gán nhãn cho dữ liệu mới (tương tự như trong fit)
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def __kmeans_plus_plus(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.k_clusters):
            distances = np.min(np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2)), axis=1)
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(X.shape[0], p=probs)])
        return np.array(centroids)