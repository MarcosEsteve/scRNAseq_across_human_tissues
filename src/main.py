import pandas as pd
import scipy.io
import numpy as np
import scipy.sparse
import os
import src.preprocessing.data_cleaning as data_cleaning
import src.preprocessing.normalization as normalization
import src.preprocessing.feature_selection as feature_selection
import src.preprocessing.dim_reduction as dim_reduction
import src.analysis.clustering as clustering
import src.analysis.cell_identification as cell_identification
import src.evaluation.evaluation as evaluation


def load_expression_data_from_mtx(path, barcodes_labeled=None, n_sample=None, random_state=None):
    """
    Load gene expression data from MTX format into a sparse DataFrame.

    Parameters:
    - path: str, path to the folder with the matrix, genes and barcodes.

    Returns:
    - expression_matrix: Sparse DataFrame containing gene expression data.
    """
    # Load expression matrix
    matrix = scipy.io.mmread(path+"matrix.mtx").tocsc()

    # Load genes
    genes = pd.read_csv(path+"genes.tsv", header=None, sep='\t', names=['gene_id', 'gene_symbol'])

    # Load barcodes
    barcodes = pd.read_csv(path+"barcodes.tsv", header=None, sep='\t', names=['barcode'])

    # Filter by labeled barcodes if barcodes_labeled is provided
    if barcodes_labeled is not None:
        if 'barcode' not in barcodes_labeled.columns:
            raise ValueError("The provided barcodes_labeled DataFrame must contain a 'barcode' column.")
        # Find the intersection of barcodes
        labeled_barcodes = set(barcodes_labeled['barcode'])
        matching_indices = barcodes['barcode'].isin(labeled_barcodes)
        matrix = matrix[:, matching_indices]
        barcodes = barcodes[matching_indices]

    # Apply random sampling to barcodes if n_sample is specified
    if n_sample is not None and n_sample < len(barcodes):
        np.random.seed(random_state)
        sampled_indices = np.random.choice(len(barcodes), size=n_sample, replace=False)
        matrix = matrix[:, sampled_indices]  # Subset columns (barcodes)
        barcodes = barcodes.iloc[sampled_indices]

    # Transform sparse matrix into pandas sparse DataFrame
    expression_matrix = pd.DataFrame.sparse.from_spmatrix(matrix)
    expression_matrix.index = genes['gene_symbol']
    expression_matrix.columns = barcodes['barcode']

    expression_matrix = data_cleaning.remove_duplicated_genes(expression_matrix)

    return expression_matrix


def load_expression_data_from_csv(csv_path, chunk_size=10000):
    """
    Read a large CSV file in chunks, transpose, convert to sparse, and concatenate to avoid RAM overloading.

    Parameters:
    - csv_path: str, path to the CSV file.
    - chunk_size: int, number of rows to read in each chunk.

    Returns:
    - expression_matrix: Sparse DataFrame containing the full gene expression data.
    """
    # Initialize a list to store processed chunks
    processed_chunks = []

    # Read the CSV in chunks
    for chunk in pd.read_csv(csv_path, sep=',', chunksize=chunk_size, header=0, index_col=0):
        # Transpose the chunk so rows are genes and columns are barcodes
        chunk = chunk.T

        # Convert to Sparse DataFrame
        sparse_chunk = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(chunk))

        # Add the processed chunk to the list
        processed_chunks.append(sparse_chunk)

        # Free memory from the current chunk
        del chunk  # Optional, to ensure the original chunk memory is freed

    # Concatenate all processed chunks into a single Sparse DataFrame
    expression_matrix = pd.concat(processed_chunks, axis=1)

    # Free memory from the processed chunks
    del processed_chunks

    # Return the complete Sparse DataFrame
    return expression_matrix


def save_results(results_path, pipeline_id, cell_identification_results, internal_metrics,
                 external_metrics, reduced_matrix, tissue):
    """
        Save the results of the pipeline including clustering, cell identification, and metrics.

        Parameters:
        -----------
        results_path : str
            Folder to save the results in.
        pipeline_id : str
            Unique identifier for the pipeline configuration.
        cell_identification_results : pd.DataFrame
            DataFrame containing 'barcode', 'cluster', and 'celltype' columns.
        internal_metrics : dict
            Dictionary containing internal evaluation metrics such as ARI, Silhouette, etc.
        external_metrics : dict
            Dictionary containing external evaluation metrics such as Accuracy, Precision, etc.
        reduced_matrix : pd.DataFrame
            Reduced dimensionality matrix (cells x dimensions).
        tissue : str
            Name of the tissue analyzed (e.g. PBMC).
        """
    # Create the results folder for the tissue if it doesn't exist
    tissue_results_path = f'{results_path}/{tissue}_results'
    os.makedirs(tissue_results_path, exist_ok=True)

    # Create the reduced matrix folder for the tissue if it doesn't exist
    tissue_reduced_matrix_path = f'{tissue_results_path}/{tissue}_reduced_matrix'
    os.makedirs(tissue_reduced_matrix_path, exist_ok=True)

    # Create a new row with the results of the pipeline
    result_row = {
        'pipeline_id': pipeline_id,  # Unique identifier for the pipeline
        'barcodes': ','.join(cell_identification_results['barcode'].tolist()),  # Convert the barcode list to a comma-separated string
        'clusters': ','.join(map(str, cell_identification_results['cluster'].tolist())),  # Convert clusters to string
        'cell_types': ','.join(cell_identification_results['celltype'].tolist()),  # Convert cell types to string
        'ARI': internal_metrics['ARI'],
        'Silhouette_Score': internal_metrics['Silhouette_Score'],
        'NMI': internal_metrics['NMI'],
        'V_measure': internal_metrics['V_measure'],
        'Accuracy': external_metrics['Accuracy'],
        'Precision': external_metrics['Precision'],
        'Recall': external_metrics['Recall'],
        'F1_score': external_metrics['F1_score']
    }

    # Save the results to a CSV file
    results_df = pd.DataFrame([result_row])

    # Check if the file already exists
    results_file_path = f'{tissue_results_path}/{tissue}_all_results.csv'
    if not os.path.exists(results_file_path):
        # If it doesn't exist, create it with headers
        results_df.to_csv(results_file_path, index=False, mode='w', header=True)
    else:
        # If it exists, append the new row without headers
        results_df.to_csv(results_file_path, index=False, mode='a', header=False)

    # Save the reduced dimensionality matrix in compressed format (CSV gzip) in the tissue-specific folder
    reduced_matrix_file_path = f'{tissue_reduced_matrix_path}/{pipeline_id}_matrix.csv.gz'
    reduced_matrix.to_csv(reduced_matrix_file_path, index=False, compression='gzip')

    print(f"Results for pipeline {pipeline_id} saved successfully.")


def generate_pipeline_id(methods_list):
    """
    Generate a unique pipeline ID based on the methods used.

    Parameters:
    -----------
    methods_list : list of str
        List of method names used in the pipeline.

    Returns:
    --------
    pipeline_id : str
        Unique pipeline identifier.
    """
    return "_".join(methods_list)


def execute_step(step_name, methods_dict, method_name, data, extra_params=None):
    """
    Execute a specific step using the provided method.

    Parameters:
    -----------
    step_name : str
        The name of the pipeline step (e.g., 'data_cleaning', 'normalization', etc.).
    methods_dict : dict
        Dictionary containing methods for the current step.
    method_name : str
        The name of the method to be executed.
    data : pd.DataFrame
        The data to be processed by the method.
    extra_params : dict, optional
        Additional parameters required by the method (e.g., file paths, metadata).

    Returns:
    --------
    result : pd.DataFrame or pd.Series
        The result of the executed method.
    """
    method_entry = methods_dict[method_name]
    # Check if the dictionary contains a direct function or a sub-dictionary
    if callable(method_entry):  # Direct function
        method_func = method_entry
    else:  # Sub-dictionary with 'func' and 'params'
        method_func = method_entry['func']

    print(f"Running {step_name} - {method_name}")

    if extra_params:
        result = method_func(data, **extra_params)  # If the function needs extra_params, call it with them
    else:
        result = method_func(data)  # Else, no extra_params is needed
    print(f"Execution of {step_name} - {method_name} done")
    return result


# Dictionaries to execute methods in pipeline
data_cleaning_methods = {
    'FLEG': data_cleaning.filter_lowly_expressed_genes,
    'FHMC': data_cleaning.filter_high_mitochondrial_content,
    'FD': data_cleaning.filter_doublets_cxds,
    'CC': data_cleaning.combined_cleaning,
}

normalization_methods = {
    "CPM": normalization.normalize_cpm,
    "QR": normalization.normalize_quantile_regression,
    "NB": normalization.normalize_negative_binomial,
}

feature_selection_methods = {
    'SHVG': feature_selection.select_highly_variable_genes,
    'SGbV': feature_selection.select_genes_by_variance,
}

dim_reduction_methods = {
    'PCA': dim_reduction.apply_pca,
    'UMAP': dim_reduction.apply_umap,
    'TSNE': dim_reduction.apply_tsne,
}

# TO DO: adjust params
clustering_methods = {
    'GBC': {'func': clustering.graph_based_clustering_leiden, 'params': {'resolution': 1}},
    'DeBC': {'func': clustering.density_based_clustering, 'params': {'eps': 0.5, 'min_samples': 5}},
    'DiBC': {'func': clustering.distance_based_clustering, 'params': {'n_clusters': 11}},
    'HC': {'func': clustering.hierarchical_clustering, 'params': {'n_clusters': 11}},
    'DLC': {'func': clustering.deep_learning_clustering, 'params': {'n_clusters': 11, 'encoding_dim': 32}},
    'APC': {'func': clustering.affinity_propagation_clustering, 'params': None},
    'MMC': {'func': clustering.mixture_model_clustering, 'params': {'n_components': 11}},  # n_components is n_clusters
    'EC': {'func': clustering.ensemble_clustering, 'params': {'n_clusters': 11, 'eps': 0.5, 'min_samples': 5}},
}

cell_identification_methods = {
    'RBA': cell_identification.reference_based_assignment,
    'CBA': cell_identification.correlation_based_assignment,
    'MBA': cell_identification.marker_based_assignment,
}



### Main Pipeline ###
# Change params of dictionaries if needed
tissue = 'PBMC'
metadata_path = "../data/PBMC/PBMC_68k/hg19/68k_pbmc_barcodes_annotation.tsv"
results_path = "../results"

celltype_column = 'celltype'
true_labels = evaluation.load_true_labels(metadata_path, 'barcodes', celltype_column, "\t")

expression_matrix = load_expression_data_from_mtx("../data/PBMC/PBMC_68k/hg19/", barcodes_labeled=true_labels)

print(expression_matrix.info())

for cleaning_method in data_cleaning_methods.keys():
    cleaned_matrix = execute_step('data_cleaning', data_cleaning_methods, cleaning_method, expression_matrix)

    for norm_method in normalization_methods.keys():
        normalized_matrix = execute_step('normalization', normalization_methods, norm_method, cleaned_matrix)

        for fs_method in feature_selection_methods.keys():
            selected_matrix = execute_step('feature_selection', feature_selection_methods, fs_method, normalized_matrix)

            for dr_method in dim_reduction_methods.keys():
                # If tSNE, execute with predefined 2 dimensions
                if dr_method == 'TSNE':
                    reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, dr_method,
                                 selected_matrix)
                else:  # PCA or UMAP
                    # Execute PCA
                    pca_object, reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, 'PCA',
                                                            selected_matrix)
                    # If dr_method == PCA, continue, this step is already done
                    # else, if dr_method == UMAP, execute umap with the same number of dimensions as PCA
                    if dr_method == 'UMAP':
                        optimal_num_dimensions = reduced_matrix.shape[1]  # Get the number of components (columns)
                        reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, dr_method,
                                                    selected_matrix, optimal_num_dimensions)

                for cluster_method, cluster_config in clustering_methods.items():
                    # Skip DBSCAN for UMAP, bc it is too dense and can't hold it with my hardware
                    if dr_method == 'UMAP' and cluster_method == 'DeBC':
                        continue

                    # Perform clustering
                    clustering_results = execute_step('clustering', clustering_methods, cluster_method, reduced_matrix,
                                                      cluster_config['params'])

                    for cell_id_method in cell_identification_methods.keys():
                        # For marker_based_assignment, marker reference is needed
                        if cell_id_method == 'MBA':
                            reference = cell_identification.generate_marker_reference(expression_matrix, metadata_path, celltype_column=celltype_column, sep='\t')
                            key = 'marker_reference'
                        # For the other 2 methods, expression_profile is needed
                        else:
                            reference = cell_identification.generate_expression_profiles(expression_matrix, metadata_path, celltype_column=celltype_column, sep='\t')
                            key = 'expression_profile'

                        extra_params = {'cluster_results': clustering_results, key: reference}

                        cell_identification_results = execute_step(
                            'cell_identification', cell_identification_methods, cell_id_method, selected_matrix, extra_params
                        )

                        # Internal Evaluation
                        internal_metrics = evaluation.internal_evaluation(reduced_matrix, cell_identification_results)

                        # External Evaluation
                        external_metrics = evaluation.external_evaluation(cell_identification_results, true_labels)

                        # Generate unique pipeline identifier
                        pipeline_id = generate_pipeline_id([
                            cleaning_method, norm_method, fs_method, dr_method, cluster_method, cell_id_method
                        ])

                        # Save results
                        save_results(
                            results_path=results_path,
                            pipeline_id=pipeline_id,
                            cell_identification_results=cell_identification_results,
                            internal_metrics=internal_metrics,
                            external_metrics=external_metrics,
                            reduced_matrix=reduced_matrix,
                            tissue=tissue
                        )

print("Finish!")
