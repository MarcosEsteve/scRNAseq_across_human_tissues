import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def apply_pca(expression_matrix, n_components=50):
    """
    Apply PCA to the expression matrix.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_components (int): Number of principal components to retain.

    Returns:
    - pd.DataFrame: PCA-transformed data with cells as rows and PCs as columns.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(expression_matrix.T.sparse.to_dense())
    return pd.DataFrame(pca_result, index=expression_matrix.columns,
                        columns=[f'PC_{i+1}' for i in range(n_components)])


def apply_umap(expression_matrix, n_components=2):
    """
    Apply UMAP to the expression matrix.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_components (int): Number of dimensions for UMAP output.

    Returns:
    - pd.DataFrame: UMAP-transformed data with cells as rows and UMAP dimensions as columns.
    """
    reducer = umap.UMAP(n_components=n_components)
    umap_result = reducer.fit_transform(expression_matrix.T.sparse.to_dense())
    return pd.DataFrame(umap_result, index=expression_matrix.columns,
                        columns=[f'UMAP_{i+1}' for i in range(n_components)])


def apply_tsne(expression_matrix, n_components=2):
    """
    Apply t-SNE to the expression matrix.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_components (int): Number of dimensions for t-SNE output.

    Returns:
    - pd.DataFrame: t-SNE-transformed data with cells as rows and t-SNE dimensions as columns.
    """
    tsne = TSNE(n_components=n_components)
    tsne_result = tsne.fit_transform(expression_matrix.T.sparse.to_dense())
    return pd.DataFrame(tsne_result, index=expression_matrix.columns,
                        columns=[f'tSNE_{i+1}' for i in range(n_components)])


def visualize_dim_reduction(reduced_data, title):
    """
    Visualize the results of dimensionality reduction in 2D.

    Parameters:
    - reduced_data (pd.DataFrame): Dimensionality reduced data.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel(reduced_data.columns[0])
    plt.ylabel(reduced_data.columns[1])
    plt.show()


def elbow_plot(pca_result):
    """
    Create an elbow plot for PCA results.

    Parameters:
    - pca_result (pd.DataFrame): PCA transformed data.
    """
    explained_variance = np.var(pca_result, axis=0)
    cumulative_variance_ratio = np.cumsum(explained_variance) / np.sum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Elbow Plot')
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.show()


def select_pca_components(pca_result, threshold=0.90):
    """
    Select the optimal number of PCA components based on cumulative explained variance.

    Parameters:
    - pca_result (pd.DataFrame): PCA transformed data.
    - threshold (float): Desired cumulative explained variance threshold.

    Returns:
    - int: Optimal number of components.
    """
    explained_variance_ratio = np.var(pca_result, axis=0) / np.sum(np.var(pca_result, axis=0))
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    return n_components

"""
# Example usage:
# Assuming you have already performed PCA, UMAP, and t-SNE

# Visualize results
visualize_dim_reduction(pca_result.iloc[:, :2], 'PCA')
visualize_dim_reduction(umap_result.iloc[:, :2], 'UMAP')
visualize_dim_reduction(tsne_result.iloc[:, :2], 't-SNE')

# Create elbow plot for PCA
elbow_plot(pca_result)

# Select optimal number of PCA components
optimal_pca_components = select_pca_components(pca_result)
print(f"Optimal number of PCA components: {optimal_pca_components}")

# Use this to determine dimensions for UMAP and t-SNE
n_dims_umap_tsne = min(optimal_pca_components, 10)
print(f"Number of dimensions for UMAP and t-SNE: {n_dims_umap_tsne}")
"""