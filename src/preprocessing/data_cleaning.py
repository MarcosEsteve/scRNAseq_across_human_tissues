import numpy as np


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
    # Count the number of cells in which each gene is expressed
    expressed_cells = (expression_matrix > 0).sum(axis=1)
    # Filter genes that are expressed in at least 'min_cells' cells
    filtered_matrix = expression_matrix.loc[expressed_cells >= min_cells]
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


def filter_doublets_cxds(expression_matrix, threshold=0.9):
    """
    Detect Doublets using Co-expression Based Doublet Scoring (cxds)

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
    - threshold (float): Threshold for doublet classification. Default is 0.9.

    Returns:
    - pd.DataFrame: A filtered SparseDataFrame with potential doublets removed.
    """
    # Binarize the expression matrix
    binarized_matrix = (expression_matrix > 0).astype(int)

    # Calculate co-expression scores
    gene_pairs = binarized_matrix.T.sparse.dot(binarized_matrix)

    # Compute doublet scores
    doublet_scores = (gene_pairs.sum(axis=0) - np.diag(gene_pairs)) / (
                len(binarized_matrix) * (len(binarized_matrix) - 1) / 2)

    # Identify non-doublets
    non_doublets = doublet_scores <= threshold

    # Filter the original matrix
    filtered_matrix = expression_matrix.loc[:, non_doublets]

    return filtered_matrix


def combined_cleaning(expression_matrix, min_cells=3, max_mito_pct=0.1, doublet_threshold=0.9):
    """
       Combines all three cleaning methods in sequence.
    """
    # Step 1: Filter lowly expressed genes
    matrix_step1 = filter_lowly_expressed_genes(expression_matrix, min_cells)
    # Step 2: Filter high mitochondrial content
    matrix_step2 = filter_high_mitochondrial_content(matrix_step1, max_mito_pct)
    # Step 3: Filter doublets
    matrix_step3 = filter_doublets_cxds(matrix_step2, threshold=doublet_threshold)
    return matrix_step3

"""
# Step 1: Filter lowly expressed genes
expression_matrix_filtered = filter_lowly_expressed_genes(expression_matrix)

# Step 2: Filter cells with high mitochondrial content
expression_matrix_filtered = filter_high_mitochondrial_content(expression_matrix_filtered, genes)

# Step 3: Detect and remove doublets
expression_matrix_cleaned = filter_doublets_cxds(expression_matrix_filtered)
"""
