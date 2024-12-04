# 4 metodos: reference-based, marker-based, correlation-based y ontology.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import networkx as nx


def merge_expression_csv_and_metadata(expression_file, metadata_file):
    """
    Merge gene expression data with metadata for the Neuronal data which comes in a csv.

    Parameters:
    - expression_file: str, path to the gene expression data file (CSV).
    - metadata_file: str, path to the metadata file (CSV).

    Returns:
    - merged_data: DataFrame containing merged expression and metadata.
    """
    # Load gene expression data
    expression_data = pd.read_csv(expression_file)  # Adjust file path if necessary

    # Load metadata
    metadata = pd.read_csv(metadata_file)  # Adjust file path if necessary

    # Assuming 'sample_name' is a common identifier in both datasets
    merged_data = expression_data.merge(metadata, left_on='sample_name', right_on='sample_name')

    return merged_data


# Example usage:
# merged_data = merge_expression_and_metadata('path_to_expression_data.csv', 'path_to_metadata.csv')


def create_neuronal_reference_profiles(merged_data):
    """
    Create a reference profile DataFrame for neuronal data.

    Parameters:
    - merged_data: DataFrame containing merged expression and metadata.

    Returns:
    - DataFrame in the format: Gene, CellType, Expression1, Expression2, ...
    """
    # Melt the DataFrame to long format
    melted_data = merged_data.melt(id_vars=['sample_name', 'cluster_label', 'cell_type_accession_label'],
                                   var_name='Gene',
                                   value_name='Expression')

    # Create a unique identifier for each sample and cluster combination
    melted_data['Sample'] = melted_data['sample_name'] + '_' + melted_data['cluster_label']

    # Pivot to get genes as rows and samples as columns
    reference_profiles = melted_data.pivot_table(index=['Gene', 'cluster_label'],
                                                 columns='Sample',
                                                 values='Expression',
                                                 aggfunc='first').reset_index()

    # Rename columns to match the expected output format
    reference_profiles.columns.name = None
    reference_profiles.columns = ['Gene', 'CellType'] + [f'Expression{i + 1}' for i in
                                                         range(reference_profiles.shape[1] - 2)]

    return reference_profiles


# Generate reference profiles
# neuronal_reference_profiles = create_neuronal_reference_profiles(merged_data)


def create_expression_profiles_tumor(expression_matrix, metadata_file):
    """
    Create a gene expression profile DataFrame in the specified format for the tumor.

    Parameters:
    - expression_matrix: DataFrame containing gene expression data (genes as index, barcodes as columns).
    - metadata_file: str, path to the metadata file (CSV).

    Returns:
    - DataFrame in the format: Gene, CellType, Expression1, Expression2, ...
    """
    # Load metadata
    metadata = pd.read_csv(metadata_file)  # Adjust file path if necessary

    # Melt the expression matrix to long format
    melted_data = expression_matrix.reset_index().melt(id_vars='gene_symbol',
                                                       var_name='barcode',
                                                       value_name='Expression')

    # Merge melted expression data with metadata on barcode
    combined_data = melted_data.merge(metadata, left_on='barcode', right_on=',orig.ident', how='left')

    # Create a unique identifier for each barcode and cell type combination
    combined_data['Sample'] = combined_data['barcode'] + '_' + combined_data['celltype_major']

    # Pivot to get genes as rows and samples as columns
    reference_profiles = combined_data.pivot_table(index=['gene_symbol', 'celltype_major'],
                                                   columns='Sample',
                                                   values='Expression',
                                                   aggfunc='first').reset_index()

    # Rename columns to match the expected output format
    reference_profiles.columns.name = None
    reference_profiles.columns = ['Gene', 'CellType'] + [f'Expression{i + 1}' for i in
                                                         range(reference_profiles.shape[1] - 2)]

    return reference_profiles


# Example usage:
# reference_profiles = create_expression_profiles(expression_df, 'path_to_metadata.csv')


# Example usage:
# expression_profiles = create_expression_profiles('path_to_expression_data.csv', 'path_to_metadata.csv')
# print(expression_profiles)


def reference_based_assignment(expression_df, reference_profiles):
    """
    Assign cell types based on reference profiles using KNN.

    Parameters:
    - expression_df: DataFrame with normalized gene expression data including cluster labels.
    - reference_profiles: DataFrame in the format: Gene, CellType, Expression1, Expression2, ...

    Returns:
    - Predictions: Series with predicted cell types for each sample.
    """

    # Pivot the reference profiles to create a matrix with genes as rows and cell types as columns
    # Here we will keep all expression columns for predictions
    reference_matrix = reference_profiles.pivot(index='Gene', columns='CellType')

    # Prepare predictions for each cluster
    predictions = []

    # Iterate over unique clusters in the expression_df
    for cluster in expression_df['Cluster'].unique():
        # Filter expression data for current cluster
        cluster_data = expression_df[expression_df['Cluster'] == cluster]

        # Ensure we only use common genes present in both datasets
        common_genes = reference_matrix.index.intersection(cluster_data.columns)

        if len(common_genes) == 0:
            continue  # Skip if there are no common genes

        # Filter both datasets to only include common genes
        cluster_expression_filtered = cluster_data[common_genes]

        # Prepare the data for KNN by flattening the reference matrix
        # We will use all available expressions for prediction
        knn = KNeighborsClassifier(n_neighbors=5)

        # Fit the model using all samples in the reference matrix (flattened)
        knn.fit(reference_matrix.values.T, reference_matrix.columns)

        # Predict cell types for each sample in the current cluster
        predictions_cluster = knn.predict(cluster_expression_filtered)

        # Append predictions for this cluster
        predictions.extend(predictions_cluster)

    return pd.Series(predictions, index=expression_df.index)


def marker_based_assignment(expression_df, marker_genes):
    # Calculate marker gene scores
    cell_type_scores = pd.DataFrame(index=expression_df.index, columns=marker_genes.keys())
    for cell_type, genes in marker_genes.items():
        cell_type_scores[cell_type] = expression_df[genes].mean(axis=1)

    # Assign cell types based on highest marker score
    cell_type_assignments = cell_type_scores.idxmax(axis=1)
    return cell_type_assignments


# Required data:
# - expression_df: pandas DataFrame with preprocessed gene expression data (genes as columns, cells as rows)
# - marker_genes: dictionary with cell types as keys and lists of marker genes as values


def correlation_based_assignment(expression_df, reference_profiles):
    """
    Assign cell types based on correlation with reference profiles.

    Parameters:
    - expression_df: DataFrame with normalized gene expression data (genes as columns, cells as rows).
    - reference_profiles: DataFrame in the format: Gene, CellType, Expression1, Expression2, ...

    Returns:
    - Assignments: Series with assigned cell types for each cell.
    """

    # Pivot the reference profiles to create a matrix with genes as rows and cell types as columns
    reference_matrix = reference_profiles.pivot(index='Gene', columns='CellType')

    assignments = []

    # Iterate over each cell in the expression_df
    for _, cell in expression_df.iterrows():
        # Initialize a dictionary to hold correlation values for each cell type
        correlations = {}

        # Calculate correlations for each cell type
        for cell_type in reference_matrix.columns:
            # Get the expression values for the current cell type across samples
            ref_expressions = reference_matrix[cell_type].dropna().values

            # Ensure we only use common genes present in both datasets
            common_genes = reference_matrix.index.intersection(cell.index)

            if len(ref_expressions) == 0 or len(common_genes) == 0:
                continue  # Skip if there are no expressions or common genes

            # Get the corresponding expression values for the current cell
            cell_expressions = cell[common_genes].values

            # Calculate Pearson correlation
            if len(ref_expressions) > 0 and len(cell_expressions) > 0:
                correlation, _ = pearsonr(cell_expressions, ref_expressions)
                correlations[cell_type] = correlation

        # Assign the best matching cell type based on maximum correlation
        if correlations:
            best_match = max(correlations, key=correlations.get)
            assignments.append(best_match)
        else:
            assignments.append(None)  # Handle cases with no correlations

    return pd.Series(assignments, index=expression_df.index)


# Example usage:
# Assuming expression_df is your normalized and processed DataFrame,
# and you have already created the reference_profiles using create_expression_profiles.
# assignments = correlation_based_assignment(expression_df, reference_profiles)

def ontology_based_assignment(expression_df, ontology, marker_genes):
    # First, use marker-based assignment
    initial_assignments = marker_based_assignment(expression_df, marker_genes)

    # Create a graph from the ontology
    G = nx.DiGraph(ontology)

    # Refine assignments using ontology
    def refine_assignment(assignment):
        if assignment in G:
            ancestors = nx.ancestors(G, assignment)
            return min(ancestors, key=lambda x: len(nx.descendants(G, x)))
        return assignment

    refined_assignments = initial_assignments.apply(refine_assignment)
    return refined_assignments

# Required data:
# - expression_df: pandas DataFrame with preprocessed gene expression data (genes as columns, cells as rows)
# - ontology: dictionary representing the cell type ontology (parent as key, list of children as value)
# - marker_genes: dictionary with cell types as keys and lists of marker genes as values


"""
# Add the cell type assignments to the clustering results DataFrame
clustering_results_df['CellType'] = cell_type_assignments.values

# Display the updated clustering results DataFrame
print(clustering_results_df)
"""
