import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def remove_duplicated_genes(expression_matrix):
    # Remove duplicate genes by averaging their expression
    # Identify duplicated genes in the expression matrix
    duplicated_genes = expression_matrix.index[expression_matrix.index.duplicated(keep=False)]

    # Compute the average expression for the duplicated genes
    averaged_duplicates = expression_matrix.loc[duplicated_genes].groupby(
        expression_matrix.loc[duplicated_genes].index).mean()

    # Remove the duplicated genes from the original matrix
    unique_matrix = expression_matrix.drop(index=duplicated_genes)

    # Combine the unique genes with the averaged duplicates into a single matrix
    deduplicated_matrix = pd.concat([unique_matrix, averaged_duplicates])

    return deduplicated_matrix


def filter_lowly_expressed_genes(expression_matrix, min_cells=3):
    """
        Filter Lowly Expressed Genes

        This function filters out genes that are expressed in fewer than a specified number of cells.
        It is part of the "Filtering by Gene Expression Counts" step in scRNA-seq data cleaning.

        Parameters:
        - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
        - min_cells (int): The minimum number of cells in which a gene must be expressed to be retained. Default is 3.

        Returns:
        - pd.DataFrame: A filtered SparseDataFrame with only genes expressed in at least 'min_cells' cells.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix for column-wise operations
    sparse_matrix = csr_matrix(expression_matrix.sparse.to_coo())

    # Count the number of cells in which each gene is expressed (non-zero entries)
    expressed_cells = (sparse_matrix > 0).sum(axis=1).A1  # .A1 converts the result to a 1D array

    # Find the indices of genes expressed in at least 'min_cells' cells
    valid_genes = expressed_cells >= min_cells

    # Filter the matrix to keep only the valid genes (rows)
    filtered_sparse_matrix = sparse_matrix[valid_genes, :]

    # Convert back to pandas SparseDataFrame
    filtered_expression_matrix = pd.DataFrame.sparse.from_spmatrix(filtered_sparse_matrix)
    filtered_expression_matrix.index = expression_matrix.index[valid_genes]
    filtered_expression_matrix.columns = expression_matrix.columns

    return filtered_expression_matrix


def filter_high_mitochondrial_content(expression_matrix, max_mito_pct=0.1):
    """
    Filter High Mitochondrial Content

    This function filters out cells with a high proportion of mitochondrial gene expression.
    It is part of the "Mitochondrial Gene Filtering" step in scRNA-seq data cleaning.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
      Gene names should be the row indices.
    - max_mito_pct (float): The maximum allowable percentage of mitochondrial gene expression per cell. Default is 0.1 (10%).

    Returns:
    - pd.DataFrame: A filtered SparseDataFrame with cells having mitochondrial content below the specified threshold.
    """
    # Identify mitochondrial genes (assuming gene symbols start with 'MT-')
    mito_genes = expression_matrix.index.str.startswith('MT-')

    # Calculate mitochondrial expression and total expression per cell
    mito_expression = expression_matrix.loc[mito_genes].sum(axis=0)
    total_expression = expression_matrix.sum(axis=0)

    # Calculate the mitochondrial percentage per cell
    mito_pct = mito_expression / total_expression

    # Filter cells with mitochondrial content above the threshold
    filtered_matrix = expression_matrix.loc[:, mito_pct <= max_mito_pct]

    return filtered_matrix


def filter_doublets_cxds(expression_matrix, threshold=0.9, block_size=10000):
    """
    Detect Doublets using Co-expression Based Doublet Scoring (cxds)

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
    - threshold (float): Threshold for doublet classification. Default is 0.9.
    - block_size (int): Number of cells to process in each block to reduce memory usage.

    Returns:
    - pd.DataFrame: A filtered SparseDataFrame with potential doublets removed.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix
    binarized_matrix_sparse = csc_matrix((expression_matrix > 0).sparse.to_coo())

    n_cells = binarized_matrix_sparse.shape[1]
    diagonal = np.zeros(n_cells)  # To store co-expression with itself
    gene_pair_sums = np.zeros(n_cells)  # To store row-wise sums

    # Process in blocks to reduce memory usage
    for start_idx in range(0, n_cells, block_size):
        end_idx = min(start_idx + block_size, n_cells)
        block = binarized_matrix_sparse[:, start_idx:end_idx]  # Select block of cells

        # Co-expression scores for the block
        block_gene_pairs = binarized_matrix_sparse.T.dot(block)

        # Update diagonal (self-coexpression for cells in the block)
        diagonal[start_idx:end_idx] = block_gene_pairs.diagonal()

        # Update row sums
        gene_pair_sums[start_idx:end_idx] = block_gene_pairs.sum(axis=0).A1

    # Compute doublet scores
    doublet_scores = (gene_pair_sums - diagonal) / (
        binarized_matrix_sparse.shape[1] * (binarized_matrix_sparse.shape[1] - 1) / 2
    )

    # Identify non-doublets (thresholding the scores)
    non_doublets = doublet_scores <= threshold

    # Filter the original expression matrix to keep only non-doublets
    filtered_matrix = expression_matrix.loc[:, non_doublets]

    return filtered_matrix


def combined_cleaning(expression_matrix, min_cells=3, max_mito_pct=0.1, doublet_threshold=0.9):
    """
       Filter cells based on multiple criteria: low gene expression, high mitochondrial expression, and doublet scores.

       Parameters:
       - expression_matrix (pd.DataFrame): A pandas SparseDataFrame with cells as columns and genes as rows.
       - low_expression_threshold (float): Threshold for low expression genes (default is 0.1).
       - mitochondrial_threshold (float): Threshold for high mitochondrial gene expression (default is 0.05).
       - doublet_threshold (float): Threshold for doublet detection (default is 0.9).

       Returns:
       - pd.DataFrame: A filtered SparseDataFrame with only the valid cells remaining.
       """
    # 1. Filter low expression genes
    lowly_expressed_genes = filter_lowly_expressed_genes(expression_matrix, min_cells=min_cells)

    # 2. Filter cells with high mitochondrial expression
    non_mitochondrial_cells = filter_high_mitochondrial_content(expression_matrix, max_mito_pct=max_mito_pct)

    # 3. Filter doublets based on co-expression scores
    non_doublets = filter_doublets_cxds(expression_matrix, threshold=doublet_threshold)

    # Combine all filters (only keep cells and genes that pass all three filters)
    valid_genes = lowly_expressed_genes.index
    valid_cells = non_mitochondrial_cells.columns.intersection(non_doublets.columns)

    # Filter the expression matrix to keep only the valid cells
    filtered_matrix = expression_matrix.loc[valid_genes, valid_cells]

    # Remove duplicate genes by averaging their expression
    filtered_matrix = remove_duplicated_genes(filtered_matrix)

    return filtered_matrix

"""
# Step 1: Filter lowly expressed genes
expression_matrix_filtered = filter_lowly_expressed_genes(expression_matrix)

# Step 2: Filter cells with high mitochondrial content
expression_matrix_filtered = filter_high_mitochondrial_content(expression_matrix_filtered, genes)

# Step 3: Detect and remove doublets
expression_matrix_cleaned = filter_doublets_cxds(expression_matrix_filtered)
"""
