import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags


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
    # Remove duplicate genes by averaging their expression
    deduplicated_matrix = remove_duplicated_genes(expression_matrix)

    # Count the number of cells in which each gene is expressed
    expressed_cells = (deduplicated_matrix > 0).sum(axis=1)

    # Filter genes that are expressed in at least 'min_cells' cells
    filtered_matrix = deduplicated_matrix.loc[expressed_cells >= min_cells]

    return filtered_matrix


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
    # Remove duplicate genes by averaging their expression
    deduplicated_matrix = remove_duplicated_genes(expression_matrix)

    # Identify mitochondrial genes (assuming gene symbols start with 'MT-')
    mito_genes = deduplicated_matrix.index.str.startswith('MT-')

    # Calculate mitochondrial expression and total expression per cell
    mito_expression = deduplicated_matrix.loc[mito_genes].sum(axis=0)
    total_expression = deduplicated_matrix.sum(axis=0)

    # Calculate the mitochondrial percentage per cell
    mito_pct = mito_expression / total_expression

    # Filter cells with mitochondrial content above the threshold
    filtered_matrix = deduplicated_matrix.loc[:, mito_pct <= max_mito_pct]

    return filtered_matrix


def filter_doublets_cxds(expression_matrix, threshold=0.9):
    """
    Detect Doublets using Co-expression Based Doublet Scoring (cxds)

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
    - threshold (float): Threshold for doublet classification. Default is 0.9.

    Returns:
    - pd.DataFrame: A filtered SparseDataFrame with potential doublets removed.
    """
    # Remove duplicate genes by averaging their expression
    deduplicated_matrix = remove_duplicated_genes(expression_matrix)

    # Binarize the expression matrix: Convert the sparse DataFrame to binary (0 or 1)
    binarized_matrix = (deduplicated_matrix > 0).astype(int)

    # Convert the binarized matrix to a scipy sparse matrix (csc_matrix)
    binarized_matrix_sparse = csc_matrix(binarized_matrix)

    # Calculate co-expression scores (this will be a square matrix of size #cells x #cells)
    gene_pairs = binarized_matrix_sparse.T.dot(binarized_matrix_sparse)

    # Get the diagonal elements (co-expression of each cell with itself)
    diagonal = gene_pairs.diagonal()

    # Sum the co-expression matrix along the rows (axis=0)
    gene_pair_sums = gene_pairs.sum(axis=0)

    # Compute doublet scores
    doublet_scores = (gene_pair_sums - diagonal) / (
                binarized_matrix_sparse.shape[1] * (binarized_matrix_sparse.shape[1] - 1) / 2)

    # Identify non-doublets (thresholding the scores)
    non_doublets = doublet_scores.A1 <= threshold  # .A1 flattens the sparse matrix to a 1D array

    # Filter the original expression matrix to keep only non-doublets
    filtered_matrix = deduplicated_matrix.loc[:, non_doublets]

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
