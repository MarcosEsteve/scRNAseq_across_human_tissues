import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def select_highly_variable_genes(expression_matrix, n_top_genes=2000):
    """
    Select highly variable genes using the Seurat method.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_top_genes (int): Number of top variable genes to select.

    Returns:
    - pd.DataFrame: A SparseDataFrame containing only the highly variable genes.
    """
    # Convert the SparseDataFrame to a sparse CSR matrix
    sparse_matrix = csr_matrix(expression_matrix.sparse.to_coo())

    # Calculate mean and variance for each gene (row) using sparse matrix methods
    gene_means = sparse_matrix.mean(axis=1).A1  # Convert to dense array (1D)
    gene_vars = sparse_matrix.multiply(sparse_matrix).mean(axis=1).A1 - gene_means ** 2  # Variance formula

    # Calculate the variance to mean ratio (dispersion)
    dispersion = gene_vars / gene_means

    # Sort genes by dispersion and select top n_top_genes
    top_genes = np.argsort(dispersion)[::-1][:n_top_genes]

    # Return as a SparseDataFrame, ensuring we maintain the sparse format
    return pd.DataFrame.sparse.from_spmatrix(sparse_matrix[top_genes],
                                             index=expression_matrix.index[top_genes],
                                             columns=expression_matrix.columns)


def select_genes_by_variance(expression_matrix, percentile=90):
    """
    Select genes based on their variance across cells.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - percentile (float): Percentile of gene variances to use as the threshold (default: 90).

    Returns:
    - pd.DataFrame: A SparseDataFrame containing only the selected genes.
    """
    # Convert the SparseDataFrame to a sparse CSR matrix
    sparse_matrix = csr_matrix(expression_matrix.sparse.to_coo())

    # Calculate variance for each gene (row) using sparse matrix methods
    gene_vars = sparse_matrix.multiply(sparse_matrix).mean(axis=1).A1 - sparse_matrix.mean(axis=1).A1 ** 2

    # Calculate the variance threshold using the given percentile
    var_threshold = np.percentile(gene_vars, percentile)

    # Select genes that have variance above the threshold
    selected_genes = np.where(gene_vars > var_threshold)[0]

    # Return as a SparseDataFrame, ensuring we maintain the sparse format
    return pd.DataFrame.sparse.from_spmatrix(sparse_matrix[selected_genes],
                                             index=expression_matrix.index[selected_genes],
                                             columns=expression_matrix.columns)
