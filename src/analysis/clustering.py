import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
import leidenalg as la
import igraph as ig
from tensorflow.keras import layers, models


def graph_based_clustering_leiden(data, resolution=1.0, n_iterations=2):
    """
    Perform graph-based clustering using the Leiden algorithm.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - resolution (float): Resolution parameter for the Leiden algorithm.
    - n_iterations (int): Number of iterations for the Leiden algorithm.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    # Construct k-nearest neighbors graph
    nn = NearestNeighbors(n_neighbors=15, metric='euclidean')
    nn.fit(data)
    adj_matrix = nn.kneighbors_graph(data, mode='distance')

    # Convert to igraph
    i_graph = ig.Graph.Weighted_Adjacency(adj_matrix.toarray().tolist())

    # Apply Leiden clustering
    partition = la.find_partition(i_graph, la.RBConfigurationVertexPartition,
                                  resolution_parameter=resolution,
                                  n_iterations=n_iterations)

    # Convert partition to labels
    labels = np.array(partition.membership)

    return pd.Series(labels, index=data.index)


def density_based_clustering(data, eps=0.5, min_samples=5):
    """
    Perform density-based clustering using DBSCAN.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return pd.Series(labels, index=data.index)


def distance_based_clustering(data, n_clusters=10):
    """
    Perform k-means clustering.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return pd.Series(labels, index=data.index)


def hierarchical_clustering(data, n_clusters=10):
    """
    Perform hierarchical clustering.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(data)
    return pd.Series(labels, index=data.index)


def deep_learning_clustering(data, n_clusters=10, encoding_dim=32):
    """
    Perform deep learning-based clustering using an autoencoder followed by KMeans.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_clusters (int): Number of clusters for KMeans.
    - encoding_dim (int): Dimension of the encoded representation.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    # Define the autoencoder model
    input_layer = layers.Input(shape=(data.shape[1],))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(data.shape[1], activation='sigmoid')(encoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True)

    # Encode the data
    encoder = models.Model(input_layer, encoded)
    encoded_data = encoder.predict(data)

    # Perform KMeans clustering on the encoded data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(encoded_data)

    return pd.Series(labels, index=data.index)


def affinity_propagation_clustering(data):
    """
    Perform affinity propagation clustering.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    affinity_propagation = AffinityPropagation(random_state=42)
    labels = affinity_propagation.fit_predict(data)

    return pd.Series(labels, index=data.index)


def mixture_model_clustering(data, n_components=10):
    """
    Perform clustering using Gaussian Mixture Models.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_components (int): Number of mixture components (clusters).

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(data)

    return pd.Series(labels, index=data.index)


def ensemble_clustering(data, n_clusters=10, eps=0.5, min_samples=5, n_components=10):
    """
    Perform ensemble clustering by combining results from multiple algorithms (KMeans, DBSCAN and GMM).

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.

    Returns:
    - pd.Series: Consensus cluster labels for each cell.
    """

    # Run multiple clustering algorithms
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data)
    dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
    gmm = GaussianMixture(n_components=n_components, random_state=42).fit(data)
    gmm_labels = gmm.predict(data)

    # Combine results using majority voting
    combined_labels = np.array([kmeans_labels, dbscan_labels, gmm_labels])

    # Use a simple voting mechanism to determine final labels
    final_labels = []

    for i in range(combined_labels.shape[1]):
        label_counts = np.bincount(combined_labels[:, i].astype(int) + 1)  # +1 to handle noise label (-1)
        final_label = np.argmax(label_counts) - 1  # Adjust back to original label range
        final_labels.append(final_label)

    return pd.Series(final_labels, index=data.index)


"""
# Create a DataFrame with barcodes and their corresponding clusters
clustering_results_df = pd.DataFrame({
    'Barcode': cluster_labels.index,
    'Cluster': cluster_labels.values
})

# Set Barcode as index for easier access later
clustering_results_df.set_index('Barcode', inplace=True)

# Display the clustering results
print(clustering_results_df)
"""