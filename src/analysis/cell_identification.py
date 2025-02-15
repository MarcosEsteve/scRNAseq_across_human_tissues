import pandas as pd
import numpy as np
import scipy.sparse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# Function to generate expression profiles
def generate_expression_profiles(expression_matrix_raw, metadata_path, celltype_column='celltype_major', barcode_column='barcodes', sep=','):
    """
    Generate reference expression profiles (medians) from metadata and expression matrix.

    Parameters:
    -----------
    expression_matrix_raw : pd.SparseDataFrame
        Gene expression matrix (genes x cells), where columns are cell barcodes.
    metadata_path : str
        Path to the metadata CSV file containing cell annotations.
    celltype_column : str
        Column in the metadata file indicating the cell type to group by.
    barcode_column : str
        Name of the column containing the unique cell barcodes.
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
    metadata.set_index(barcode_column, inplace=True)  # Ensure barcodes are the index

    # Filter metadata to match the expression matrix
    filtered_metadata = metadata.loc[metadata.index.intersection(expression_matrix_raw.columns)]

    # Align expression matrix with metadata without converting to dense
    aligned_expression_matrix = expression_matrix_raw.loc[:, filtered_metadata.index]

    # Convert sparse dataframe to COO matrix for efficient operations
    sparse_matrix = aligned_expression_matrix.sparse.to_coo()

    # Create a DataFrame mapping cell barcodes to cell types
    cell_types = filtered_metadata[celltype_column].reindex(aligned_expression_matrix.columns)

    # Convert COO matrix to CSR for fast row-based operations
    sparse_matrix = sparse_matrix.tocsr()

    # Compute median per gene per cell type using sparse operations
    unique_celltypes = cell_types.unique()
    medians_dict = {}

    for cell_type in unique_celltypes:
        # Select columns corresponding to the current cell type
        mask = cell_types == cell_type
        selected_columns = sparse_matrix[:, mask.to_numpy()]

        if selected_columns.shape[1] > 0:
            # Compute median without converting to dense
            median_values = np.array([np.median(selected_columns[i, :].data) if selected_columns[i, :].nnz > 0 else 0
                                      for i in range(selected_columns.shape[0])])

            # Store as a sparse matrix
            medians_dict[cell_type] = scipy.sparse.csr_matrix(median_values.reshape(-1, 1))

    # Convert the dictionary of sparse vectors to a DataFrame
    medians_sparse = scipy.sparse.hstack(list(medians_dict.values()))

    # Convert to a Pandas sparse DataFrame
    medians_df = pd.DataFrame.sparse.from_spmatrix(medians_sparse, index=aligned_expression_matrix.index,
                                                   columns=unique_celltypes)

    return medians_df


def load_expression_profiles(medians_path, metadata_path):
    """
        Load medians expression profile and aggregate cell types in subclasses

        Parameters:
        -----------
        medians_path : str
            Path to the CSV file containing median gene expression values (genes x cell types).
        metadata_path : str
            Path to the metadata CSV file containing cell type annotations.

        Returns:
        --------
        pd.DataFrame
            Aggregated gene expression profile where columns represent subclass labels.
        """
    # Load CSV
    medians_df = pd.read_csv(medians_path, index_col=0, header=0)  # Genes como índice
    metadata_df = pd.read_csv(metadata_path, header=0)

    # Map cluster_label to subclass_label
    cluster_to_subclass = metadata_df.set_index("cluster_label")["subclass_label"].to_dict()

    # Rename medians_df columns using the dict
    medians_df = medians_df.rename(columns=cluster_to_subclass)

    # Calculate median for each subclass
    aggregated_df = medians_df.groupby(level=0, axis=1).median()

    return aggregated_df


# Function to generate marker reference
def generate_marker_reference(expression_matrix_raw, metadata_path, celltype_column='celltype_minor', barcode_column='barcodes', top_n_genes=5,
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
    metadata.set_index(barcode_column, inplace=True)  # Ensure barcodes are the index

    # Filter metadata to match the expression matrix
    filtered_metadata = metadata.loc[metadata.index.intersection(expression_matrix_raw.columns)]

    # Align expression matrix with filtered metadata without converting to dense
    aligned_expression_matrix = expression_matrix_raw.loc[:, filtered_metadata.index]

    # Convert to COO format for efficient operations
    sparse_matrix = aligned_expression_matrix.sparse.to_coo()

    # Create a DataFrame mapping cell barcodes to cell types
    cell_types = filtered_metadata[celltype_column].reindex(aligned_expression_matrix.columns)

    # Convert to CSR format for fast row-based operations
    sparse_matrix = sparse_matrix.tocsr()

    # Generate marker genes
    marker_genes = {}
    unique_celltypes = cell_types.unique()

    for cell_type in unique_celltypes:
        # Select columns corresponding to the current cell type
        mask = cell_types == cell_type
        selected_columns = sparse_matrix[:, mask.to_numpy()]  # Sparse matrix slice

        if selected_columns.shape[1] > 0:
            # Sum non-zero elements in each row
            row_sums = selected_columns.sum(axis=1).A1  # .A1 flattens the result to a 1D array

            # Count non-zero elements in each row
            row_counts = np.diff(selected_columns.indptr)  # This gives the number of non-zero elements per row

            # Avoid division by zero (if no non-zero elements in a row, set mean to 0)
            mean_expression = np.divide(row_sums, row_counts, where=row_counts > 0)

            # Get top N genes with the highest mean expression
            top_genes_indices = np.argsort(mean_expression)[::-1][:top_n_genes]
            top_genes = aligned_expression_matrix.index[top_genes_indices].tolist()
            marker_genes[cell_type] = top_genes

    return marker_genes


def generate_marker_reference_from_medians(expression_profile, top_n_genes=5):
    """
    Retrieve the top N most expressed genes for each cell type.

    Parameters:
    -----------
    expression_profile : pd.DataFrame
        Aggregated gene expression matrix (genes x subclass labels).
    top_n_genes : int, optional
        Number of top expressed genes to select as markers for each cell type (default is 5).

    Returns:
    --------
    dict
        A dictionary where keys are subclass labels and values are lists of top marker genes.
    """
    top_genes = {}
    for cell_type in expression_profile.columns:
        top_genes[cell_type] = expression_profile[cell_type].nlargest(top_n_genes).index.tolist()
    return top_genes


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
    common_genes = expression_profile.index.intersection(aligned_expression_matrix.index)

    # Filter both expression_profile and expression_matrix_processed to include only common genes
    expression_profile_filtered = expression_profile.loc[common_genes]
    expression_matrix_processed_filtered = aligned_expression_matrix.loc[common_genes, :]

    # Calculate the average expression profile for each cluster using processed data
    cluster_profiles = cluster_results.groupby(cluster_results).apply(
        lambda group: expression_matrix_processed_filtered.loc[:, group.index].mean(axis=1)
    ).unstack()  # unstack to make it (genes x clusters)

    # Calculate the distance between each cluster profile and each reference cell type profile
    distances = cdist(cluster_profiles.values, expression_profile_filtered.T.values, metric="euclidean")

    # Get the available cell types (columns of the reference expression profile)
    available_cell_types = list(expression_profile_filtered.columns)

    # Create an array to store the final assignments
    final_assignments = [None] * len(cluster_results)

    # Loop through all clusters and assign a unique cell type
    for cluster_idx in range(distances.shape[0]):
        # Get the index of the closest cell type (minimum distance)
        closest_cell_type_idx = np.argmin(distances[cluster_idx])

        # Assign the closest cell type to this cluster
        final_assignments[cluster_idx] = available_cell_types[closest_cell_type_idx]

        # Set the distance of this cell type to infinity for the remaining clusters
        distances[:, closest_cell_type_idx] = np.inf

    # Create the results DataFrame with barcodes, cluster assignments, and assigned cell types
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,  # Use the barcodes (index of cluster_results)
        'cluster': cluster_results.values,  # Use the cluster labels (values of cluster_results)
        'celltype': [final_assignments[cluster] for cluster in cluster_results.values]  # Use the assigned celltypes for each cluster
    })

    return results_df


# Correlation-Based assignment function
def correlation_based_assignment(expression_matrix_processed, cluster_results, expression_profile):
    """
    Assign cell types to clusters based on correlation with a reference profile.

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
    # Align the processed expression matrix with the cluster results
    aligned_expression_matrix = expression_matrix_processed.loc[:, cluster_results.index]

    # Align the genes in the expression matrix and reference profile
    common_genes = expression_profile.index.intersection(aligned_expression_matrix.index)
    expression_profile_filtered = expression_profile.loc[common_genes]
    expression_matrix_processed_filtered = aligned_expression_matrix.loc[common_genes, :]

    # Calculate the average expression profile for each cluster
    cluster_profiles = cluster_results.groupby(cluster_results).apply(
        lambda group: expression_matrix_processed_filtered.loc[:, group.index].mean(axis=1)
    ).unstack().T

    # Calculate and store correlations for each cluster with each celltype
    correlations_dict = {}
    for cluster in cluster_profiles.columns:
        # Compute correlation with each cell type in the reference profile
        correlations = expression_profile_filtered.corrwith(cluster_profiles[cluster])
        correlations_dict[cluster] = correlations

    # Assign cell types and clusters ensuring no duplicates
    cell_type_assignments = []
    assigned_cell_types = set()  # To keep track of assigned cell types
    assigned_clusters = set()  # To keep track of assigned clusters

    # Flatten the correlations and sort by value to prioritize the maximum correlation
    all_correlations = [
        (cluster, cell_type, correlations_dict[cluster][cell_type])
        for cluster in correlations_dict
        for cell_type in correlations_dict[cluster].index
    ]

    # Sort correlations in descending order (highest correlation first)
    sorted_correlations = sorted(all_correlations, key=lambda x: x[2], reverse=True)

    # Iterate over the sorted correlations and assign cell types
    for cluster, cell_type, correlation in sorted_correlations:
        if cluster not in assigned_clusters and cell_type not in assigned_cell_types:
            cell_type_assignments.append((cluster, cell_type))  # Assign this cell type
            assigned_cell_types.add(cell_type)  # Mark it as assigned
            assigned_clusters.add(cluster)  # Mark this cluster as assigned

    # Now, create a results DataFrame aligning barcodes with their assigned cell types
    # Create a mapping from clusters to cell types
    cluster_to_cell_type = dict(cell_type_assignments)

    # Assign cell types to the clusters in the result DataFrame
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,
        'cluster': cluster_results.values,
        'celltype': [cluster_to_cell_type[cluster] for cluster in cluster_results.values]
    })

    return results_df


# Marker-based assignment function
def marker_based_assignment(expression_matrix_processed, cluster_results, marker_reference):
    """
    Assign cell types to clusters based on marker gene expression using the marker reference.

    Parameters:
    -----------
    expression_matrix_processed : pd.DataFrame
        Processed gene expression matrix (genes x cells).
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
    # Initialize sets to track assigned clusters and cell types
    assigned_clusters = set()
    assigned_cell_types = set()

    # Store scores for each cluster and cell type pair
    scores_dict = {}

    # Iterate over clusters in the cluster_results
    for cluster, barcodes in cluster_results.groupby(cluster_results):
        cluster_expression = expression_matrix_processed.loc[:, barcodes.index]

        # Iterate over cell types and their marker genes
        for cell_type, markers in marker_reference.items():
            valid_markers = [gene for gene in markers if gene in cluster_expression.index]

            if valid_markers:
                # Calculate the mean expression of the marker genes for the cluster
                mean_expression = cluster_expression.loc[valid_markers].mean(axis=1).sum()
                scores_dict[(cluster, cell_type)] = mean_expression

    # Sort all the scores in descending order (highest score first)
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    # Now assign cell types to clusters ensuring no duplicates
    cell_type_assignments = []

    # Iterate over sorted scores and assign cell types
    for (cluster, cell_type), score in sorted_scores:
        if cluster not in assigned_clusters and cell_type not in assigned_cell_types:
            cell_type_assignments.append((cluster, cell_type))
            assigned_clusters.add(cluster)  # Mark the cluster as assigned
            assigned_cell_types.add(cell_type)  # Mark the cell type as assigned

    # Create the results DataFrame, ensuring alignment with barcodes
    cluster_to_cell_type = dict(cell_type_assignments)

    # Construct results DataFrame
    results_df = pd.DataFrame({
        'barcode': cluster_results.index,
        'cluster': cluster_results.values,
        'celltype': [cluster_to_cell_type[cluster] for cluster in cluster_results.values]
    })

    return results_df


def visualize_cells(reduced_data, title, cell_assignment_results):
    """
    Visualize the results of clustering with cell type labels.

    Parameters:
    - reduced_data (pd.DataFrame): Dimensionality reduced data (e.g., PCA or t-SNE).
    - title (str): Title for the plot.
    - cell_assignment_results (pd.DataFrame): DataFrame containing 'barcode', 'cluster', and 'celltype'.
    """
    plt.figure(figsize=(12, 10))

    # Extract labels (cell types) from cell assignment results
    cell_assignment_results = cell_assignment_results.set_index('barcode')
    labels = cell_assignment_results['celltype']

    unique_labels = labels.unique()
    for label in unique_labels:
        # Filter reduced data points based on the barcode and label
        subset = reduced_data.loc[cell_assignment_results.index[labels == label]]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1],
                    label=label, alpha=0.6, s=15)

    plt.title(title)
    plt.legend(title='Cell Types', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.xlabel(reduced_data.columns[0])
    plt.ylabel(reduced_data.columns[1])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


