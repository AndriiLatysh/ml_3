import pandas as pd
import sklearn.cluster as sk_cluster
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

object_sizes = pd.read_csv("data/object_sizes.csv")

X = object_sizes[["width", "height"]]

ax1.set_title("K-means++")
print("K-means++")

kmeans_pp_clustering_model = sk_cluster.KMeans(n_clusters=5, init="k-means++", n_init=10)

kmeans_pp_clustering_model.fit(X)

kmeans_pp_classes = kmeans_pp_clustering_model.predict(X)

ax1.scatter(x=object_sizes["width"], y=object_sizes["height"], c=kmeans_pp_classes, cmap="prism")

kmeans_pp_centroids = kmeans_pp_clustering_model.cluster_centers_
# print(kmeans_pp_centroids)

ax1.scatter(x=kmeans_pp_centroids[:, 0], y=kmeans_pp_centroids[:, 1], marker="X", color="k", s=100)

kmeans_pp_db_score = sk_metrics.davies_bouldin_score(X, kmeans_pp_classes)
print("Davies-Bouldin score: {:.5g} (less is better).".format(kmeans_pp_db_score))

kmeans_pp_s_score = sk_metrics.silhouette_score(X, kmeans_pp_classes)
print("Silhouette score: {:.5g} (more is better).".format(kmeans_pp_s_score))

ax2.set_title("K-means")
print("K-means")

kmeans_clustering_model = sk_cluster.KMeans(n_clusters=5, init="random", n_init=1)

kmeans_clustering_model.fit(X)

kmeans_classes = kmeans_clustering_model.predict(X)

ax2.scatter(x=object_sizes["width"], y=object_sizes["height"], c=kmeans_classes, cmap="prism")

kmeans_centroids = kmeans_clustering_model.cluster_centers_
# print(kmeans_centroids)

ax2.scatter(x=kmeans_centroids[:, 0], y=kmeans_centroids[:, 1], marker="X", color="k", s=100)

kmeans_db_score = sk_metrics.davies_bouldin_score(X, kmeans_classes)
print("Davies-Bouldin score: {:.5g} (less is better).".format(kmeans_db_score))

kmeans_s_score = sk_metrics.silhouette_score(X, kmeans_classes)
print("Silhouette score: {:.5g} (more is better).".format(kmeans_s_score))

plt.show()
