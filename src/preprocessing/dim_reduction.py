import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def apply_pca_no_optim(expression_matrix, n_components=50):
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
    pca_df = pd.DataFrame(
        pca_result,
        index=expression_matrix.columns,
        columns=[f'PC_{i + 1}' for i in range(n_components)]
    )
    return pca, pca_df


def apply_pca(expression_matrix, threshold=0.5):
    """
    Apply PCA to the expression matrix, then select the optimal number of components
    based on cumulative explained variance, and return only the selected components.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - threshold (float): Desired cumulative explained variance threshold to select the optimal number of components.

    Returns:
    - pd.DataFrame: PCA-transformed data with cells as rows and selected PCs as columns.
    """
    # Apply PCA with all components
    pca = PCA()
    pca_result = pca.fit_transform(expression_matrix.T.sparse.to_dense())

    n_components = select_pca_components(pca, std_threshold=threshold)
    pca_result_selected = pca_result[:, :n_components]

    # Return the PCA-transformed data with the optimal number of components
    pca_df = pd.DataFrame(
        pca_result_selected,
        index=expression_matrix.columns,
        columns=[f'PC_{i + 1}' for i in range(n_components)]
    )
    return pca, pca_df


def apply_umap(expression_matrix, n_components=2):
    """
    Apply UMAP to the expression matrix.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_components (int): Number of dimensions for UMAP output.

    Returns:
    - pd.DataFrame: UMAP-transformed data with cells as rows and UMAP dimensions as columns.
    """
    # Limit the number of components to 20 to avoid computational running increase
    n_components = min(n_components, 20)

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
    # Limit the number of components to 20 to avoid computational running increase
    n_components = min(n_components, 20)

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
    plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], alpha=0.5, s=10)
    plt.title(title)
    plt.xlabel(reduced_data.columns[0])
    plt.ylabel(reduced_data.columns[1])
    plt.grid(True)
    plt.show()


def elbow_plot(pca_result, threshold=0.5):
    """
    Create an elbow plot using the standard deviation of each principal component.

    Parameters:
    - pca (PCA object): The fitted PCA object.
    - threshold (float): A threshold line to help decide the number of components to retain.

    Returns:
    - None: Displays the elbow plot.
    """
    # Compute standard deviation of each principal component
    std_dev = np.sqrt(pca_result.explained_variance_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(std_dev) + 1), std_dev, 'bo-', label='Standard Deviation')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
    plt.xlabel('Principal Component')
    plt.ylabel('Standard Deviation')
    plt.title('PCA Elbow Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


def select_pca_components(pca, std_threshold=0.5):
    """
    Select the optimal number of PCA components based on a standard deviation threshold.

    Parameters:
    - pca (PCA object): The fitted PCA object.
    - std_threshold (float): Minimum standard deviation to retain a principal component.

    Returns:
    - int: Optimal number of components.
    """
    # Compute standard deviation of each component
    std_dev = np.sqrt(pca.explained_variance_)

    # Count components with standard deviation above the threshold
    n_components = np.sum(std_dev >= std_threshold)

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