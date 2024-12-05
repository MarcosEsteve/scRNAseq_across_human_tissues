import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def internal_evaluation(expression_matrix, results_df):
    """
    Perform internal evaluation of clustering and cell type assignment.

    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        The gene expression matrix (genes x barcodes), preprocessed and dimensionally reduced.
    results_df : pd.DataFrame
        A dataframe containing:
        - 'barcode': Cell barcodes.
        - 'cluster': Cluster assigned by the analysis.
        - 'celltype': Predicted cell type based on the cluster.

    Returns:
    --------
    dict
        A dictionary containing the calculated internal metrics:
        - ARI (Adjusted Rand Index)
        - Silhouette Score
        - NMI (Normalized Mutual Information)
        - V-measure
    """
    # Ensure the input DataFrame has the required columns
    if not {'barcode', 'cluster', 'celltype'}.issubset(results_df.columns):
        raise ValueError("The results_df must contain 'barcode', 'cluster', and 'celltype' columns.")

    # Align the expression matrix with the barcodes in the results_df
    aligned_matrix = expression_matrix.loc[:, results_df['barcode']]

    # Extract cluster labels and predicted cell types
    cluster_labels = results_df['cluster']
    cell_type_labels = results_df['celltype']

    # Calculate internal metrics
    ari = adjusted_rand_score(cell_type_labels, cluster_labels)
    silhouette = silhouette_score(aligned_matrix.T, cluster_labels)  # Transpose to make samples x features
    nmi = normalized_mutual_info_score(cell_type_labels, cluster_labels)
    v_measure = v_measure_score(cell_type_labels, cluster_labels)

    return {
        "ARI": ari,
        "Silhouette": silhouette,
        "NMI": nmi,
        "V-measure": v_measure
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
    # Merge the two dataframes on 'barcode' to align predictions with true labels
    merged_df = pd.merge(results_df, true_labels_df, on='barcode', how='inner')

    # Check if the merge resulted in an empty dataframe
    if merged_df.empty:
        raise ValueError("No overlapping barcodes found between the results and the true labels.")

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
        "F1-score": f1
    }


def load_true_labels(metadata_path, barcode_column, celltype_column):
    """
    Load ground truth cell type labels from a metadata CSV file.

    Parameters:
    -----------
    metadata_path : str
        Path to the metadata CSV file.
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
    metadata = pd.read_csv(metadata_path)

    # Validate required columns
    if barcode_column not in metadata.columns or celltype_column not in metadata.columns:
        raise ValueError(f"Columns '{barcode_column}' and '{celltype_column}' must exist in the metadata file.")

    # Create a DataFrame with relevant data
    true_labels = metadata[[barcode_column, celltype_column]].rename(
        columns={barcode_column: 'barcode', celltype_column: 'true_label'}
    )

    return true_labels
