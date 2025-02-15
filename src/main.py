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


def load_expression_data_from_mtx(path, matrix_name, genes_name, barcodes_name, header=None, barcodes_labeled=None, n_sample=None, random_state=None):
    """
    Load gene expression data from MTX format into a sparse DataFrame.

    Parameters:
    - path: str, path to the folder with the matrix, genes and barcodes.

    Returns:
    - expression_matrix: Sparse DataFrame containing gene expression data.
    """
    # Load expression matrix
    matrix = scipy.io.mmread(path+matrix_name).tocsc()

    # Load genes
    genes = pd.read_csv(path+genes_name, header=header, sep='\t', names=['gene_symbol'])

    # Load barcodes
    barcodes = pd.read_csv(path+barcodes_name, header=header, sep='\t', names=['barcode'])

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


def load_expression_data_from_csv(file_path, delimiter=",", dtype="float32", chunksize=10000):
    """
    Reads a large scRNA-seq gene expression matrix from a CSV file efficiently into a Pandas SparseDataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - delimiter (str): CSV delimiter, default is ",".
    - dtype (str): Data type to optimize memory usage (float32 recommended).
    - chunksize (int): Number of rows to read per chunk.

    Returns:
    - pd.DataFrame: SparseDataFrame with genes as index and barcodes as columns.
    """
    # Initialize a list to accumulate the processed chunks
    processed_chunks = []

    # Read the matrix in chunks to optimize memory usage
    data_chunks = pd.read_csv(
        file_path, delimiter=delimiter, index_col=0,
        chunksize=chunksize, low_memory=True
    )
    print("Start loading")
    i = 1
    for chunk in data_chunks:
        print(f"Processing chunk:", i)
        # Convert to SparseDataFrame to optimize memory usage
        chunk_sparse = chunk.astype(pd.SparseDtype(dtype, fill_value=0))

        # Transpose each chunk (Genes in rows, Barcodes in columns)
        chunk_sparse = chunk_sparse.T

        # Add the processed chunk to the list
        processed_chunks.append(chunk_sparse)
        i = i + 1

    # Concatenate all processed chunks
    print("Loop finished")
    expression_matrix_sparse = pd.concat(processed_chunks, axis=1)

    expression_matrix_sparse = data_cleaning.remove_genes_without_expression(expression_matrix_sparse)

    return expression_matrix_sparse


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

    # Remove rows where 'celltype' is None or NaN
    cell_identification_results = cell_identification_results.dropna(subset=['celltype'])

    # Create a new row with the results of the pipeline
    result_row = {
        'pipeline_id': pipeline_id,  # Unique identifier for the pipeline
        'barcodes': ','.join(cell_identification_results['barcode'].tolist()),  # Convert the barcode list to a comma-separated string
        'clusters': ','.join(map(str, cell_identification_results['cluster'].tolist())),  # Convert clusters to string
        'cell_types': ','.join(cell_identification_results['celltype'].tolist()),  # Convert cell types to string
        'Silhouette_Score': internal_metrics['Silhouette_Score'],
        'Davies_Bouldin_Index': internal_metrics['Davies_Bouldin_Index'],
        'Calinski_Harabasz_Score': internal_metrics['Calinski_Harabasz_Score'],
        'ARI': internal_metrics['ARI'],
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
    'CC': data_cleaning.combined_cleaning
}

normalization_methods = {
    "CPM": normalization.normalize_cpm,
    "QR": normalization.normalize_quantile_regression,
    "NB": normalization.normalize_negative_binomial
}

feature_selection_methods = {
    'SHVG': feature_selection.select_highly_variable_genes,
    'SGbV': feature_selection.select_genes_by_variance
}

dim_reduction_methods = {
    'PCA': dim_reduction.apply_pca,
    'UMAP': dim_reduction.apply_umap,
    'TSNE': dim_reduction.apply_tsne
}

clustering_methods = {
    'GBC': {'func': clustering.graph_based_clustering_leiden, 'params': {'n_clusters': 20}},
    'DeBC': {'func': clustering.density_based_clustering, 'params': {'n_clusters': 20}},
    'DiBC': {'func': clustering.distance_based_clustering, 'params': {'n_clusters': 20}},
    'HC': {'func': clustering.hierarchical_clustering, 'params': {'n_clusters': 20}},
    'DLC': {'func': clustering.deep_learning_clustering, 'params': {'n_clusters': 20}},
    'MMC': {'func': clustering.mixture_model_clustering, 'params': {'n_components': 20}},  # n_components is n_clusters
    'EC': {'func': clustering.ensemble_clustering, 'params': {'n_clusters': 20}}
}

cell_identification_methods = {
    'RBA': cell_identification.reference_based_assignment,
    'CBA': cell_identification.correlation_based_assignment,
    'MBA': cell_identification.marker_based_assignment
}


### Main Pipeline ###
# Variables for tissue
tissue = 'Neuronal'
metadata_path = "../data/Neuronal/M1/metadata.csv"
medians_path = "../data/Neuronal/M1/medians.csv"
results_path = "../results"
celltype_column = 'subclass_label'
barcode_column = 'sample_name'
pca_threshold = {'threshold': 2}  # 5 for PBMC, 10 for Tumor, 2 for Neuronal
num_celltypes = 20  # 11 for PBMC, 9 for Tumor, 20 for Neuronal

print(f"Starting analysis for", tissue)

# Load true labels
true_labels = evaluation.load_true_labels(metadata_path, barcode_column, celltype_column, ",")

# Load expression matrix
if tissue == 'Neuronal':
    expression_matrix = load_expression_data_from_csv("../data/Neuronal/M1/")
else:
    expression_matrix = load_expression_data_from_mtx("../data/Tumor/",
                                                      matrix_name="all_matrix.mtx",
                                                      genes_name="all_genes.tsv",
                                                      barcodes_name="all_barcodes.tsv",
                                                      header=0,
                                                      barcodes_labeled=None)

print(expression_matrix.info())

# Generate reference data for cell identification
if tissue == 'Neuronal':
    expression_profile = cell_identification.load_expression_profiles(medians_path, metadata_path)
    marker_genes = cell_identification.generate_marker_reference_from_medians(expression_profile, top_n_genes=10)
else:
    expression_profile = cell_identification.generate_expression_profiles(expression_matrix, metadata_path, celltype_column=celltype_column, barcode_column=barcode_column, sep=',')
    marker_genes = cell_identification.generate_marker_reference(expression_matrix, metadata_path, celltype_column=celltype_column, barcode_column=barcode_column, sep=',')

# Free memory if execution > 1
cleaning_exec = 0
for cleaning_method in data_cleaning_methods.keys():
    cleaning_exec += 1
    if cleaning_exec > 1:
        del cleaned_matrix, selected_matrix, normalized_matrix, reduced_matrix, clustering_results, cell_identification_results, internal_metrics, external_metrics

    cleaned_matrix = execute_step('data_cleaning', data_cleaning_methods, cleaning_method, expression_matrix)

    selecting_exec = 0
    for fs_method in feature_selection_methods.keys():
        selecting_exec += 1
        if selecting_exec > 1:
            del selected_matrix, normalized_matrix, reduced_matrix, clustering_results, cell_identification_results, internal_metrics, external_metrics

        selected_matrix = execute_step('feature_selection', feature_selection_methods, fs_method, cleaned_matrix)

        norm_exec = 0
        for norm_method in normalization_methods.keys():
            norm_exec += 1
            if norm_exec > 1:
                del normalized_matrix, reduced_matrix, clustering_results, cell_identification_results, internal_metrics, external_metrics

            normalized_matrix = execute_step('normalization', normalization_methods, norm_method, selected_matrix)

            dr_exec = 0
            for dr_method in dim_reduction_methods.keys():
                dr_exec += 1
                if dr_exec > 1:
                    del reduced_matrix, clustering_results, cell_identification_results, internal_metrics, external_metrics

                # If tSNE, execute with predefined 2 dimensions
                if dr_method == 'TSNE':
                    reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, dr_method,
                                 normalized_matrix)
                else:  # PCA or UMAP
                    # Execute PCA
                    pca_object, reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, 'PCA',
                                                            normalized_matrix, pca_threshold)
                    # If dr_method == PCA, continue, this step is already done
                    # else, if dr_method == UMAP, execute umap with the same number of dimensions as PCA
                    if dr_method == 'UMAP':
                        optimal_num_dimensions = {'n_components': reduced_matrix.shape[1]}  # Get the number of components (columns) to run UMAP
                        reduced_matrix = execute_step('dim_reduction', dim_reduction_methods, dr_method,
                                                    normalized_matrix, optimal_num_dimensions)

                for cluster_method, cluster_config in clustering_methods.items():
                    # Skip DBSCAN for UMAP, bc it is too dense and can't hold it with my hardware
                    if dr_method == 'UMAP' and cluster_method == 'DeBC':
                        continue

                    # Perform clustering
                    clustering_results = execute_step('clustering', clustering_methods, cluster_method, reduced_matrix,
                                                      cluster_config['params'])

                    # If no clusters, skip and continue with the next clustering method
                    if clustering_results.empty or len(clustering_results.unique()) != num_celltypes:
                        print("Number of clusters not reached, skipping this clustering method")
                        continue

                    # Internal metrics are common between cell_id_methods
                    n_metric = 0
                    for cell_id_method in cell_identification_methods.keys():
                        n_metric += 1
                        # For marker_based_assignment, marker reference is needed
                        if cell_id_method == 'MBA':
                            extra_params = {'cluster_results': clustering_results, 'marker_reference': marker_genes}
                        # For the other 2 methods, expression_profile is needed
                        else:
                            extra_params = {'cluster_results': clustering_results,
                                            'expression_profile': expression_profile}

                        cell_identification_results = execute_step('cell_identification', cell_identification_methods, cell_id_method, normalized_matrix, extra_params)

                        # Just run internal metrics once, as it depends on clustering, not cell_id
                        if n_metric == 1:
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
