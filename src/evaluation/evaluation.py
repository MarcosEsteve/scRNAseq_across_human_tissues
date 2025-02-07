import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def internal_evaluation(reduced_matrix, results_df):
    """
    Perform internal evaluation of clustering and cell type assignment.

    Parameters:
    -----------
    reduced_matrix : pd.DataFrame
        The reduced gene expression matrix (barcodes x dimensions).
    results_df : pd.DataFrame
        A dataframe containing:
        - 'barcode': Cell barcodes.
        - 'cluster': Cluster assigned by the analysis.
        - 'celltype': Predicted cell type based on the cluster.

    Returns:
    --------
    dict
        A dictionary containing the calculated internal metrics:
        - Silhouette Score
        - Davies Bouldin Index
        - Calinski Harabasz Score
        - ARI (Adjusted Rand Index)
        - NMI (Normalized Mutual Information)
        - V-measure
    """
    # Remove rows where 'celltype' is None or NaN
    results_df = results_df.dropna(subset=['celltype'])

    # Align the expression matrix with the barcodes in the results_df
    aligned_matrix = reduced_matrix.loc[results_df['barcode']]

    # Extract cluster labels and predicted cell types
    cluster_labels = results_df['cluster']
    cell_type_labels = results_df['celltype']

    # Calculate internal metrics
    silhouette = silhouette_score(aligned_matrix, cluster_labels)
    davies_bouldin = davies_bouldin_score(aligned_matrix, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(aligned_matrix, cluster_labels)
    ari = adjusted_rand_score(cell_type_labels, cluster_labels)
    nmi = normalized_mutual_info_score(cell_type_labels, cluster_labels)
    v_measure = v_measure_score(cell_type_labels, cluster_labels)

    return {
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": davies_bouldin,
        "Calinski_Harabasz_Score": calinski_harabasz,
        "ARI": ari,
        "NMI": nmi,
        "V_measure": v_measure
    }


def external_evaluation(results_df, true_labels_df):
    """
    Perform external evaluation of cell type predictions.

    Parameters:
    -----------
    results_df : pd.DataFrame
        A dataframe containing:
        - 'barcode': Cell barcodes.
        - 'cluster': Cluster assigned by the analysis.
        - 'celltype': Predicted cell type based on the cluster.

    true_labels_df : pd.DataFrame
        A dataframe containing:
        - 'barcode': Cell barcodes.
        - 'true_label': Ground truth cell type labels.

    Returns:
    --------
    dict
        A dictionary containing the calculated external metrics:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1-score (weighted)
    """
    # Remove rows where 'celltype' is None or NaN
    results_df = results_df.dropna(subset=['celltype'])

    # Merge the two dataframes on 'barcode' to align predictions with true labels
    merged_df = pd.merge(results_df, true_labels_df, on='barcode', how='inner')

    # Extract the true and predicted labels
    true_labels = merged_df['true_label']
    predicted_labels = merged_df['celltype']

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1
    }


def load_true_labels(metadata_path, barcode_column, celltype_column, sep='\t'):
    """
    Load ground truth cell type labels from a metadata file.

    Parameters:
    -----------
    metadata_path : str
        Path to the metadata file.
    barcode_column : str
        Name of the column containing the unique cell barcodes.
    celltype_column : str
        Name of the column containing the ground truth cell type labels.

    Returns:
    --------
    pd.DataFrame
        A dataframe containing the ground truth labels with two columns:
        - 'barcode': Cell barcodes.
        - 'true_label': Ground truth cell type labels.
    """
    # Read the metadata CSV file
    metadata = pd.read_csv(metadata_path, sep=sep)

    # Create a DataFrame with relevant data
    true_labels = metadata[[barcode_column, celltype_column]].rename(
        columns={barcode_column: 'barcode', celltype_column: 'true_label'}
    )

    return true_labels
