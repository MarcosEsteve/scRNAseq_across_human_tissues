import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# Function to generate expression profiles
def generate_expression_profiles(expression_matrix_raw, metadata_path, celltype_column='celltype_minor', sep=','):
    """
    Generate reference expression profiles (medians) from metadata and expression matrix.

    Parameters:
    -----------
    expression_matrix_raw : pd.DataFrame
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
        the median expression for each gene in each cell type.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path, sep=sep)
    metadata.set_index('barcodes', inplace=True)  # Ensure barcodes are the index

    # Align metadata with the expression matrix
    filtered_metadata = metadata.loc[metadata.index.intersection(expression_matrix_raw.columns)]

    # Align expression matrix with metadata
    aligned_expression_matrix = expression_matrix_raw.sparse.to_dense().loc[:, filtered_metadata.index]

    # Transpose to facilitate grouping by cell type
    aligned_expression_matrix = aligned_expression_matrix.T
    aligned_expression_matrix[celltype_column] = filtered_metadata[celltype_column]

    # Compute median expression for each gene grouped by cell type
    medians = aligned_expression_matrix.groupby(celltype_column).median().T

    return medians


# Function to generate marker reference
def generate_marker_reference(expression_matrix_raw, metadata_path, celltype_column='celltype_minor', top_n_genes=5,
                              sep=','):
    """
    Generate a marker gene reference file based on the most expressed genes per cell type.

    Parameters:
    -----------
    expression_matrix_raw : pd.DataFrame
        Gene expression matrix (genes x cells), where columns are cell barcodes.
    metadata_path : str
        Path to the metadata file containing cell annotations.
    celltype_column : str
        Column in the metadata file indicating the cell type to group by.
    top_n_genes : int, optional
        Number of top expressed genes to select as markers for each cell type (default is 5).
    sep : str, optional
        Delimiter for the metadata file (default is ',' for CSV files, use '\t' for TSV files).

    Returns:
    --------
    dict
        A dictionary where keys are cell types and values are lists of top marker genes.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path, sep=sep)
    metadata.set_index('barcodes', inplace=True)  # Ensure barcodes are the index

    # Align metadata with the expression matrix
    filtered_metadata = metadata.loc[metadata.index.intersection(expression_matrix_raw.columns)]

    # Align expression matrix with filtered metadata
    aligned_expression_matrix = expression_matrix_raw.sparse.to_dense().loc[:, filtered_metadata.index]

    # Transpose to facilitate grouping by cell type
    aligned_expression_matrix = aligned_expression_matrix.T
    aligned_expression_matrix[celltype_column] = filtered_metadata[celltype_column]

    # Generate marker genes
    marker_genes = {}
    for cell_type, group in aligned_expression_matrix.groupby(celltype_column):
        mean_expression = group.drop(columns=[celltype_column]).mean(axis=0)
        top_genes = mean_expression.nlargest(top_n_genes).index.tolist()
        marker_genes[cell_type] = top_genes

    return marker_genes


# Reference-based assignment function
def reference_based_assignment(expression_matrix_processed, cluster_results, expression_profile):
    """
    Assign cell types to clusters using a reference expression profile.

    Parameters:
    -----------
    expression_matrix_processed : pd.DataFrame
        Gene expression matrix (genes x cells), where columns are cell barcodes.
    cluster_results : pd.Series
        Series with barcodes as index and cluster labels as values.
    expression_profile : pd.DataFrame
        Gene expression reference profile with genes as rows and cell types as columns.

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'barcode', 'cluster', and 'celltype' columns.
    """
    # Align the processed expression matrix with cluster_results (barcodes as index)
    aligned_expression_matrix = expression_matrix_processed.loc[:, cluster_results.index]

    # Align the genes of expression_profile with the processed expression matrix
    common_genes = expression_profile.index.intersection(expression_matrix_processed.index)

    # Filter both expression_profile and expression_matrix_processed to include only common genes
    expression_profile_filtered = expression_profile.loc[common_genes]
    expression_matrix_processed_filtered = expression_matrix_processed.loc[common_genes, :]

    # Calculate the average expression profile for each cluster using processed data
    cluster_profiles = cluster_results.groupby(cluster_results).apply(
        lambda group: expression_matrix_processed_filtered.loc[:, group.index].mean(axis=1)
    ).unstack()  # unstack to make it (genes x clusters)

    # Step 4: Calculate the distance between each cluster profile and each reference cell type profile
    distances = cdist(cluster_profiles.values, expression_profile_filtered.T.values, metric="correlation")

    # Assign cell types to each cluster based on the closest profile
    assigned_cell_types = expression_profile_filtered.columns[np.argmin(distances, axis=1)]

    # Create the results DataFrame with barcodes, cluster assignments, and assigned cell types
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,  # Use the barcodes (index of cluster_results)
        'cluster': cluster_results.values,  # Use the cluster labels (values of cluster_results)
        'celltype': [assigned_cell_types[cluster] for cluster in cluster_results.values] # Use the assigned celltypes for each cluster
    })

    return results_df


# Correlation-Based assignment function
def correlation_based_assignment(cluster_results, expression_profile):
    """
    Assign cell types to clusters based on correlation with a reference profile.

    Parameters:
    -----------
    cluster_results : pd.Series
        Series with barcodes as index and cluster labels as values.
    expression_profile : pd.DataFrame
        Gene expression reference profile with genes as rows and cell types as columns.

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'barcode', 'cluster', and 'celltype' columns.
    """
    # Calculate the average expression profile for each cluster
    cluster_profiles = cluster_results.groupby(cluster_results).apply(
        lambda indices: expression_profile.loc[:, indices.index].mean(axis=1)
    )

    # Assign cell types based on maximum correlation
    cell_type_assignments = []
    for cluster in cluster_profiles.columns:
        correlations = expression_profile.corrwith(cluster_profiles[cluster])
        cell_type_assignments.append(correlations.idxmax())

    # Create the results DataFrame, ensuring the alignment with barcodes
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,  # Align with barcode index
        'cluster': cluster_results.values,
        'celltype': [cell_type_assignments[cluster - 1] for cluster in cluster_results.values]
    })

    return results_df


# Marker-based assignment function
def marker_based_assignment(cluster_results, marker_reference):
    """
    Assign cell types to clusters based on marker gene expression using the marker reference.

    Parameters:
    -----------
    cluster_results : pd.Series
        Series with barcodes as index and cluster labels as values.
    marker_reference : dict
        Dictionary of cell types and their corresponding marker genes,
        with cell types as keys and lists of marker genes as values.

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'barcode', 'cluster', and 'celltype' columns.
    """
    cluster_scores = {}

    for cluster, indices in cluster_results.groupby(cluster_results):
        scores = {}

        # Iterate over cell types and their markers in the marker_reference
        for cell_type, markers in marker_reference.items():
            valid_markers = [gene for gene in markers if gene in indices.index]

            if valid_markers:
                scores[cell_type] = len(valid_markers)

        # Assign the cell type with the highest score
        if scores:
            cluster_scores[cluster] = max(scores, key=scores.get)
        else:
            cluster_scores[cluster] = None

    # Create the results DataFrame, ensuring the alignment with barcodes
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,  # Align with barcode index
        'cluster': cluster_results.values,
        'celltype': [cluster_scores[cluster] for cluster in cluster_results.values]
    })

    return results_df


