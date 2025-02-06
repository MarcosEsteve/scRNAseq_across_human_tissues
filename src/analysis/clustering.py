import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
import leidenalg as la
import igraph as ig
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt


def graph_based_clustering_leiden(data,  n_clusters, n_neighbors_range=(5, 100), resolution_range=(0.1, 5), step=0.1):
    """
    Perform graph-based clustering using the Leiden algorithm.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data. Each row represents a sample, and
      each column represents a feature.
    - n_clusters (int): Desired number of clusters. The algorithm will attempt to find the
      closest match to this number.
    - n_neighbors_range (tuple, optional): Range of values for the number of neighbors used
      in the k-nearest neighbors (k-NN) graph construction. Default is (10, 50).
    - resolution_range (tuple, optional): Range of resolution parameter values for the Leiden
      algorithm. Higher resolutions tend to produce more clusters. Default is (0.2, 2.0).
    - step (float, optional): Step size for iterating over the resolution parameter. Smaller
      step sizes provide finer resolution adjustments but may increase computation time.
      Default is 0.2.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    # Variables for searching the exact n_clusters
    closest_diff = float('inf')
    best_result = None

    # Try different n_neighbors for NN
    for n_neighbors in range(n_neighbors_range[0], n_neighbors_range[1] + 1, 5):
        # Construct k-nearest neighbors graph
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(data)
        adj_matrix = nn.kneighbors_graph(data, mode='distance')

        # Convert to igraph directly from the sparse matrix
        sources, targets = adj_matrix.nonzero()  # Extract edges
        weights = adj_matrix.data  # Edge weights
        i_graph = ig.Graph(edges=list(zip(sources, targets)), edge_attrs={"weight": weights})

        # Try different resolution for Leiden
        for resolution in np.arange(resolution_range[0], resolution_range[1]+step, step):
            # Apply Leiden clustering
            partition = la.find_partition(
                i_graph, la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                n_iterations=2
            )

            # Get number of clusters
            labels = np.array(partition.membership)
            num_clusters = len(set(labels))

            # Stop early if exact match is found
            if num_clusters == n_clusters:
                best_result = pd.Series(labels, index=data.index)
                return best_result

            # Check how close this result is to the target
            diff = abs(num_clusters - n_clusters)
            if diff < closest_diff:
                closest_diff = diff
                best_result = pd.Series(labels, index=data.index)

            # Skip the rest of resolution values if clusters are too high
            if num_clusters > n_clusters:
                break

    return best_result


def density_based_clustering(data, n_clusters, eps_range=(0.5, 5), min_samples_range=(3, 15), step=0.1):
    """
    Perform density-based clustering using DBSCAN.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data. Each row represents a sample, and
      each column represents a feature.
    - n_clusters (int): Desired number of clusters. The algorithm will attempt to find the
      closest match to this number.
    - eps_range (tuple, optional): Range of values for the eps parameter of DBSCAN. Default is (0.1, 1.0).
    - min_samples_range (tuple, optional): Range of values for the min_samples parameter of DBSCAN.
      Default is (3, 20).
    - step (float, optional): Step size for iterating over the eps parameter. Smaller step sizes
      provide finer resolution adjustments but may increase computation time. Default is 0.1.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    # Variables for tracking the best result
    closest_diff = float('inf')
    best_result = None

    # Iterate over the range of eps values
    for eps in np.arange(eps_range[0], eps_range[1] + step, step):
        # Iterate over the range of min_samples values
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Calculate the number of clusters (excluding noise points labeled as -1)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Stop early if an exact match is found
            if num_clusters == n_clusters:
                best_result = pd.Series(labels, index=data.index)
                return best_result

            # Update the best result if the current is closer to the target
            diff = abs(num_clusters - n_clusters)
            if diff < closest_diff:
                closest_diff = diff
                best_result = pd.Series(labels, index=data.index)

            # Skip the rest of min_samples values if clusters are too low
            if num_clusters < n_clusters:
                break

    return best_result


def distance_based_clustering(data, n_clusters=10):
    """
    Perform k-means clustering.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - pd.Series: Cluster labels for each cell.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return pd.Series(labels, index=data.index)


def hierarchical_clustering(data, n_clusters=10, sample_fraction=0.1, random_state=6):
    """
    Perform hierarchical clustering. Agglomerative Clustering on a subsample and assign remaining points using k-NN.

    Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data (cells as rows, features as columns).
    - n_clusters (int): Number of clusters to form.
    - sample_fraction (float): Fraction of cells to use for clustering.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pd.Series: Cluster labels for all cells.
    """
    # Step 1: Subsampling
    np.random.seed(random_state)
    sampled_indices = np.random.choice(data.index, size=int(len(data) * sample_fraction), replace=False)
    sampled_data = data.loc[sampled_indices]

    # Step 2: Agglomerative Clustering on the subsample
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    sampled_labels = agglomerative.fit_predict(sampled_data)

    # Step 3: Assign remaining cells using k-NN
    remaining_indices = data.index.difference(sampled_indices)
    remaining_data = data.loc[remaining_indices]

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(sampled_data)
    distances, neighbors = knn.kneighbors(remaining_data)

    # Assign each point to the most common cluster among its neighbors
    remaining_labels = np.array([np.bincount(sampled_labels[neighbors[i]]).argmax() for i in range(len(remaining_data))])

    # Combine labels
    labels = pd.Series(index=data.index, dtype=int)
    labels.loc[sampled_indices] = sampled_labels
    labels.loc[remaining_indices] = remaining_labels

    labels = labels.astype(int)

    return labels


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
    encoded = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    decoded = layers.Dense(data.shape[1], activation='linear')(encoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True)

    # Encode the data
    encoder = models.Model(input_layer, encoded)
    encoded_data = encoder.predict(data)

    # Perform KMeans clustering on the encoded data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(encoded_data)

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


def ensemble_clustering(data, n_clusters=10):
    """
    Perform ensemble clustering by combining results from multiple algorithms:
    - Mixture Model (Gaussian Mixture)
    - Distance-based (KMeans)
    - Graph-based (Leiden Algorithm)

     Parameters:
    - data (pd.DataFrame): Dimensionality-reduced data.
    - n_clusters (int): Number of clusters.

    Returns:
    - pd.Series: Consensus cluster labels for each cell.
    """

    # Run custom clustering algorithms
    kmeans_labels = distance_based_clustering(data, n_clusters)
    gmm_labels = mixture_model_clustering(data, n_clusters)
    leiden_labels = graph_based_clustering_leiden(data, n_clusters)  # Only use labels

    # Combine results using majority voting
    combined_labels = np.array([kmeans_labels, gmm_labels, leiden_labels])
    final_labels = []

    for i in range(combined_labels.shape[1]):
        label_counts = np.bincount(combined_labels[:, i].astype(int))
        final_label = np.argmax(label_counts)
        final_labels.append(final_label)

    return pd.Series(final_labels, index=data.index)


def visualize_clusters(reduced_data, title, labels):
    """
    Visualize the results of clustering.

    Parameters:
    - reduced_data (pd.DataFrame): Dimensionality reduced data.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Check if label is not -1 (outliers, in case of DBSCAN)
        if label != -1:
            plt.scatter(reduced_data[labels == label].iloc[:, 0],
                        reduced_data[labels == label].iloc[:, 1],
                        label=f'Cluster {label}', alpha=0.5, s=10)
    plt.title(title)
    plt.legend(title='Clusters', loc='upper right')
    plt.xlabel(reduced_data.columns[0])
    plt.ylabel(reduced_data.columns[1])
    plt.grid(True)
    plt.show()

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