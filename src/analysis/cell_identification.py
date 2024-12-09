import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def generate_expression_profiles(expression_matrix, metadata_path, celltype_column='celltype_minor', sep=','):
    """
    Generate reference expression profiles (medians) from metadata and expression matrix.

    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        Gene expression matrix (genes x cells), where columns are cell barcodes.
    metadata_path : str
        Path to the metadata CSV file containing cell annotations.
    celltype_column : str
        Column in the metadata file indicating the cell type to group by.
    sep : str, optional
        Delimiter for the metadata file (default is ',' for CSV files, use '\t' for TSV files).

    Returns:
    --------
    pd.DataFrame
        A dataframe with genes as rows and cell types as columns, containing
        the mean expression for each gene in each cell type.
    """
    # Load the metadata
    metadata = pd.read_csv(metadata_path, sep=sep)

    # Ensure the metadata and expression matrix align by barcode
    metadata.set_index(metadata.columns[0], inplace=True)  # First column is assumed to be barcodes
    aligned_expression = expression_matrix.loc[:, metadata.index]

    # Add cell type information to expression matrix
    aligned_expression = aligned_expression.T  # Transpose for easier grouping (cells x genes)
    aligned_expression[celltype_column] = metadata[celltype_column]

    # Calculate mean expression for each cell type
    medians = aligned_expression.groupby(celltype_column).mean().T  # Transpose back to genes x cell types

    return medians


def reference_based_assignment_from_metadata(expression_matrix, metadata_path, celltype_column='celltype_minor', sep=','):
    """
    Assign cells to known cell types using reference profiles generated from metadata.

    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        Gene expression matrix (genes x cells), normalized and processed.
    metadata_path : str
        Path to the metadata CSV file.
    celltype_column : str
        Column in the metadata indicating the cell type to group by.
    sep : str, optional
        Delimiter for the metadata file (default is ',' for CSV files, use '\t' for TSV files).

    Returns:
    --------
    pd.Series
        A series with cell barcodes as index and assigned cell types as values.
    """
    # Generate expression profiles
    medians = generate_expression_profiles(expression_matrix, metadata_path, celltype_column, sep)

    # Calculate Euclidean distance between cells and reference profiles
    distances = cdist(expression_matrix.T, medians.T, metric='euclidean')  # Transpose for cell alignment
    closest_indices = np.argmin(distances, axis=1)
    cell_types = medians.columns

    return pd.Series([cell_types[i] for i in closest_indices], index=expression_matrix.columns)


def correlation_based_assignment_from_metadata(expression_matrix, metadata_path, celltype_column='celltype_minor', sep=','):
    """
    Assign cells to cell types using correlation with profiles generated from metadata.

    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        Gene expression matrix (genes x cells), normalized and processed.
    metadata_path : str
        Path to the metadata CSV file.
    celltype_column : str
        Column in the metadata indicating the cell type to group by.
    sep : str, optional
        Delimiter for the metadata file (default is ',' for CSV files, use '\t' for TSV files).

    Returns:
    --------
    pd.Series
        A series with cell barcodes as index and assigned cell types as values.
    """
    # Generate expression profiles
    medians = generate_expression_profiles(expression_matrix, metadata_path, celltype_column, sep)

    # Calculate correlation between cells and reference profiles
    correlation_matrix = expression_matrix.T.corrwith(medians.T, method='pearson')
    assigned_cell_types = correlation_matrix.idxmax(axis=1)

    return assigned_cell_types


def marker_based_assignment(expression_df, marker_genes):  # to be adjusted, mapeo jerárquico¿?
    # Calculate marker gene scores
    cell_type_scores = pd.DataFrame(index=expression_df.index, columns=marker_genes.keys())
    for cell_type, genes in marker_genes.items():
        cell_type_scores[cell_type] = expression_df[genes].mean(axis=1)

    # Assign cell types based on highest marker score
    cell_type_assignments = cell_type_scores.idxmax(axis=1)
    return cell_type_assignments



