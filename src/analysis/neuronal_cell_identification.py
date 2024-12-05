import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def reference_based_assignment(df_expression, medians_path):
    """
    Assign cells in df_expression to known cell types based on comparison with reference expression profiles (medians).

    Parameters:
    - df_expression: pandas dataframe of target dataset (genes x cells), with clustering result as an additional column.
    - medians_path: str, path to the CSV file containing reference expression profiles (genes x cell_types), where columns are cell types and rows are genes.

    Returns:
    - cell_assignments: pandas Series with cell barcode as index and assigned cell type as values.
    """
    # Load the reference medians data
    medians = pd.read_csv(medians_path, index_col=0)  # Assumes genes are in the first column

    # Extract the gene expression matrix from df_expression (excluding clustering column)
    gene_expression = df_expression.drop(columns=['clustering'])  # Adjust 'clustering' if needed

    # Calculate Euclidean distance between each cell's expression profile and each reference cell type profile
    distances = cdist(gene_expression.T, medians.T, metric='euclidean')  # Transpose to align cells with columns

    # Find the closest reference profile for each cell
    closest_indices = np.argmin(distances, axis=1)

    # Get the cell type names corresponding to the closest reference profiles
    cell_types = medians.columns
    cell_assignments = pd.Series([cell_types[i] for i in closest_indices], index=df_expression.columns)

    return cell_assignments


def correlation_based_assignment(df_expression, medians_path):
    """
    Assign cells in df_expression to cell types based on correlation with reference cell types (medians).

    Parameters:
    - df_expression: pandas dataframe of target dataset (genes x cells), with clustering result as an additional column.
    - medians_path: str, path to the CSV file containing reference expression profiles (genes x cell_types), where columns are cell types and rows are genes.

    Returns:
    - cell_assignments: pandas Series with cell barcode as index and assigned cell type as values.
    """
    # Load the reference medians data
    medians = pd.read_csv(medians_path, index_col=0)  # Assumes genes are in the first column

    # Extract the gene expression matrix from df_expression (excluding clustering column)
    gene_expression = df_expression.drop(columns=['clustering'])  # Adjust 'clustering' if needed

    # Calculate the correlation between each cell's expression profile and each reference cell type's profile
    correlation_matrix = gene_expression.T.corrwith(medians.T,
                                                    method='pearson')  # Transpose to align cells with columns

    # Assign each cell to the reference cell type with the highest correlation
    cell_assignments = correlation_matrix.idxmax(axis=1)

    return cell_assignments


def marker_based_assignment(df_expression, markers_path):  # to be adjusted, falta cuadrar el markers file
    """
    Assign cells in df_expression to cell types based on marker gene expression.

    Parameters:
    - df_expression: pandas dataframe of target dataset (genes x cells), with clustering result as an additional column.
    - markers_path: str, path to the CSV file containing marker genes for each cell type. The file should have
                    columns 'cell_type' and 'marker_genes' where 'marker_genes' is a list of genes for each cell type.

    Returns:
    - cell_assignments: pandas Series with cell barcode as index and assigned cell type as values.
    """
    # Load the marker genes data
    markers_df = pd.read_csv(markers_path)

    # Initialize a dictionary to store marker genes for each cell type
    cell_type_markers = {}

    # Process marker genes
    for _, row in markers_df.iterrows():
        cell_type = row['cell_type']
        markers = row['marker_genes'].split(',')  # Assume marker genes are comma-separated
        cell_type_markers[cell_type] = markers

    # Calculate expression of marker genes in df_expression
    cell_assignments = []
    for cell in df_expression.columns:
        cell_expression = df_expression[cell]

        # For each cell, check if the expression of marker genes matches the expected pattern
        assigned_cell_type = None
        for cell_type, markers in cell_type_markers.items():
            marker_expression = cell_expression[markers].mean()  # Use mean expression of marker genes
            if marker_expression > 0:  # If expression of marker genes is above a threshold
                assigned_cell_type = cell_type
                break

        # If no cell type matches, assign as 'Unknown' (or implement other logic)
        if not assigned_cell_type:
            assigned_cell_type = 'Unknown'

        cell_assignments.append(assigned_cell_type)

    return pd.Series(cell_assignments, index=df_expression.columns)


"""
Ejemplo de uso
assigned_types_reference_based = reference_based_assignment(expression_matrix, "path_to_medians.csv")
"""
