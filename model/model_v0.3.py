import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import h5py

# ----------------------------------------
# 1. Load images and preprocess
# ----------------------------------------
data_dir = "/Users/dexterj/Desktop/cropping/merge"
img_size = (224, 224)

# Load all image paths
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load and preprocess images
images = []
for img_path in image_files:
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
flattened_images = images.reshape(len(images), -1)

# ----------------------------------------
# 2. Perform K-Means and PCA
# ----------------------------------------
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(flattened_images)

pca = PCA(n_components=2)
reduced_images = pca.fit_transform(flattened_images)

# ----------------------------------------
# 3. Save Models in h5 Format
# ----------------------------------------
# Save K-Means model
kmeans_save_path = "kmeans_model.h5"
with h5py.File(kmeans_save_path, 'w') as f:
    f.create_dataset("cluster_centers", data=kmeans.cluster_centers_)
    f.create_dataset("labels", data=kmeans.labels_)
    f.attrs["n_clusters"] = kmeans.n_clusters
print(f"K-Means model saved to {kmeans_save_path}")

# Save PCA model
pca_save_path = "pca_model.h5"
with h5py.File(pca_save_path, 'w') as f:
    f.create_dataset("components", data=pca.components_)
    f.create_dataset("explained_variance", data=pca.explained_variance_)
    f.attrs["n_components"] = pca.n_components
print(f"PCA model saved to {pca_save_path}")

# ----------------------------------------
# 4. Load Models from h5 Format
# ----------------------------------------
# Load K-Means model
with h5py.File(kmeans_save_path, 'r') as f:
    cluster_centers = f["cluster_centers"][:]
    labels = f["labels"][:]
    n_clusters = f.attrs["n_clusters"]

loaded_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
loaded_kmeans.cluster_centers_ = cluster_centers
loaded_kmeans.labels_ = labels
print("K-Means model loaded.")

# Load PCA model
with h5py.File(pca_save_path, 'r') as f:
    components = f["components"][:]
    explained_variance = f["explained_variance"][:]
    n_components = f.attrs["n_components"]

loaded_pca = PCA(n_components=n_components)
loaded_pca.components_ = components
loaded_pca.explained_variance_ = explained_variance
print("PCA model loaded.")