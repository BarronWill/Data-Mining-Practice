import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Tạo dữ liệu mẫu
np.random.seed(42)
data = np.random.rand(10, 2)  # 10 điểm dữ liệu trong không gian 2 chiều
print(data)
# Thực hiện phân cụm gộp
linked = linkage(data, method='single')  # Sử dụng phương pháp 'single' (link đơn)
print(linked)
# Vẽ dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Phân cụm gộp')
plt.xlabel('Chỉ số')
plt.ylabel('Khoảng cách')
plt.show()