import pandas as pd
import scipy.io
import numpy as np
import scipy.sparse
import src.preprocessing.data_cleaning as data_cleaning
import src.preprocessing.normalization as normalization
import src.preprocessing.feature_selection as feature_selection
import src.preprocessing.dim_reduction as dim_reduction
import src.analysis.clustering as clustering
import src.analysis.cell_identification as cell_identification
import src.evaluation.evaluation as evaluation


def load_expression_data_from_mtx(path, n_sample=None, random_state=None):
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


# TO DO: Change this function to adapt to output from cell identification methods
def save_results(barcodes, clustering_results, cell_identification_results, pipeline_id, internal_metrics, external_metrics):
    # Build a dataframe with the results
    results = pd.DataFrame({
        'pipeline_metadata': pipeline_id,  # string indicating the configuration
        'barcode': barcodes,
        'cluster': clustering_results,
        'cell_type': cell_identification_results,
        'ARI': internal_metrics['ARI'],
        'Silhouette': internal_metrics['Silhouette'],
        'NMI': internal_metrics['NMI'],
        'V_measure': internal_metrics['V-measure'],
        'Accuracy': external_metrics['Accuracy'],
        'Precision': external_metrics['Precision'],
        'Recall': external_metrics['Recall'],
        'F1_score': external_metrics['F1-score']
    })
    # Save the dataframe in a csv file
    results.to_csv(f'results/{pipeline_id}_results.csv', index=False)


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
    method_func = methods_dict[method_name]
    print(f"Running {step_name} - {method_name}")
    if extra_params:
        result = method_func(data, **extra_params) # If the function needs extra_params, call it with them
    else:
        result = method_func(data)  # Else, no extra_params is needed
    return result


# Dictionaries to execute methods in pipeline
data_cleaning_methods = {
    'filter_lowly_expressed_genes': data_cleaning.filter_lowly_expressed_genes,
    'filter_high_mitochondrial_content': data_cleaning.filter_high_mitochondrial_content,
    'filter_doublets_cxds': data_cleaning.filter_doublets_cxds,
    'combined_cleaning': data_cleaning.combined_cleaning,
    'no_cleaning': lambda expression_matrix, *args, **kwargs: expression_matrix  # Returns the matrix with no changes
}

normalization_methods = {
    "cpm": normalization.normalize_cpm,
    "quantile_regression": normalization.normalize_quantile_regression,
    "negative_binomial": normalization.normalize_negative_binomial,
}

feature_selection_methods = {
    'select_highly_variable_genes': feature_selection.select_highly_variable_genes,
    'select_genes_by_variance': feature_selection.select_genes_by_variance,
}

dim_reduction_methods = {
    'apply_pca': dim_reduction.apply_pca,
    'apply_umap': dim_reduction.apply_umap,
    'apply_tsne': dim_reduction.apply_tsne,
}

# TO DO: adjust params
clustering_methods = {
    'graph_based_clustering_leiden': {'func': clustering.graph_based_clustering_leiden, 'params': {'resolution': 1}},
    'density_based_clustering': {'func': clustering.density_based_clustering, 'params': {'eps': 0.5, 'min_samples': 5}},
    'distance_based_clustering': {'func': clustering.distance_based_clustering, 'params': {'n_clusters': 10}},
    'hierarchical_clustering': {'func': clustering.hierarchical_clustering, 'params': {'n_clusters': 5}},
    'deep_learning_clustering': {'func': clustering.deep_learning_clustering, 'params': {'n_clusters': 10, 'encoding_dim': 32}},
    'affinity_propagation_clustering': {'func': clustering.affinity_propagation_clustering, 'params': None},
    'mixture_model_clustering': {'func': clustering.mixture_model_clustering, 'params': {'n_components': None}},  # n_components is chosen from previous step
    'ensemble_clustering': {'func': clustering.ensemble_clustering, 'params': {'n_estimators': 10}},
}

metadata_path = "./data/PBMC/PBMC_68k/hg19/68_pbmc_barcodes_annotation.tsv"

cell_identification_methods = {
    'reference_based_assignment': cell_identification.reference_based_assignment,
    'correlation_based_assignment': cell_identification.correlation_based_assignment,
    'marker_based_assignment': cell_identification.marker_based_assignment,
}


### Main Pipeline ###
#this is done for PBMC, I have to replicate it for the other tissues
expression_matrix = load_expression_data_from_mtx("../data/PBMC/PBMC_68k/hg19/")
true_labels = evaluation.load_true_labels(metadata_path, "barcodes", "celltype", "\t")

print(expression_matrix.info())

"""
for cleaning_method in data_cleaning_methods.keys():
    cleaned_data = execute_step('data_cleaning', data_cleaning_methods, cleaning_method, expression_matrix)

    for norm_method in normalization_methods.keys():
        normalized_data = execute_step('normalization', normalization_methods, norm_method, cleaned_data)

        for fs_method in feature_selection_methods.keys():
            selected_features = execute_step('feature_selection', feature_selection_methods, fs_method, normalized_data)

            for dr_method in dim_reduction_methods.keys():
                # Execution of PCA no matter the dr_method selected
                reduced_data = execute_step('dim_reduction', dim_reduction_methods, 'PCA',
                                            selected_features)
                optimal_num_dimensions = reduced_data.shape[1]  # Get the number of components (columns)
                # If dr_method == 'PCA', continue
                # Else run the dr_method using the number of optimal components extracted from PCA
                if dr_method is not 'PCA':
                    # ejecutar el dim reduction method correspondiente con el numero optimo de dimensiones
                    extra_params = {'n_components': optimal_num_dimensions}
                    reduced_data = execute_step('dim_reduction', dim_reduction_methods, dr_method,
                                                selected_features, extra_params)

                for cluster_method, cluster_config in clustering_methods.items():
                    # Mixture model clustering requires num of components (dimensions), we take the optimal from previous step
                    if cluster_method == 'mixture_model_clustering':
                        cluster_config['params']['n_components'] = optimal_num_dimensions

                    clustering_results = execute_step('clustering', clustering_methods, cluster_method, reduced_data,
                                                      cluster_config['params'])

                    for cell_id_method in cell_identification_methods.keys():
                        if cell_id_method == 'marker_based_assignment':
                            reference = cell_identification.generate_marker_reference()
                        else:
                            reference = cell_identification.generate_expression_profiles()
                            
                        
                        cell_identification_results = execute_step(
                            'cell_identification', cell_identification_methods, cell_id_method, clustering_results, reference
                        )

                        # Internal Evaluation
                        internal_metrics = evaluation.internal_evaluation(reduced_data, cell_identification_results)

                        # External Evaluation
                        external_metrics = evaluation.external_evaluation(cell_identification_results, true_labels_df)

                        # Generate unique pipeline identifier
                        pipeline_id = generate_pipeline_id([
                            cleaning_method, norm_method, fs_method, dr_method, cluster_method, cell_id_method
                        ])

                        # Save results
                        save_results(
                            pipeline_id=pipeline_id,
                            barcodes=expression_matrix.columns,
                            clustering_results=clustering_results,
                            cell_identification_results=cell_identification_results,
                            internal_metrics=internal_metrics,
                            external_metrics=external_metrics
                        )
                        print(f"Results saved for pipeline: {pipeline_id}")

print("Finish!")"""

