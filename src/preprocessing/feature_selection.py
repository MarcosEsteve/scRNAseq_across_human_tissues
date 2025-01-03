import pandas as pd
import numpy as np


def select_highly_variable_genes(expression_matrix, n_top_genes=2000):
    """
    Select highly variable genes using the Seurat method.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - n_top_genes (int): Number of top variable genes to select.

    Returns:
    - pd.DataFrame: A SparseDataFrame containing only the highly variable genes.
    """
    # Convert the sparse matrix to dense for calculations
    dense_matrix = expression_matrix.sparse.to_dense()

    # Calculate mean and variance for each gene
    gene_means = dense_matrix.mean(axis=1)
    gene_vars = dense_matrix.var(axis=1)

    # Calculate the variance to mean ratio
    dispersion = gene_vars / gene_means

    # Sort genes by dispersion and select top n_top_genes
    top_genes = dispersion.sort_values(ascending=False).head(n_top_genes).index

    return expression_matrix.loc[top_genes]


def select_genes_by_variance(expression_matrix, var_threshold=0.1):
    """
    Select genes based on their variance across cells.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows are genes and columns are cells.
    - var_threshold (float): Minimum variance threshold for gene selection.

    Returns:
    - pd.DataFrame: A SparseDataFrame containing only the selected genes.
    """
    # Convert the sparse matrix to dense for calculations
    dense_matrix = expression_matrix.sparse.to_dense()
    gene_vars = dense_matrix.var(axis=1)
    selected_genes = gene_vars[gene_vars > var_threshold].index
    return expression_matrix.loc[selected_genes]
