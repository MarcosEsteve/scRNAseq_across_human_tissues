import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def internal_evaluation(expression_matrix, cluster_labels, cell_type_labels):
    """
    Perform internal evaluation of clustering and cell type assignment.

    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        The gene expression matrix (cells x genes).
    cluster_labels : pd.Series
        Series containing cluster assignments for each cell.
    cell_type_labels : pd.Series
        Series containing cell type assignments for each cell.

    Returns:
    --------
    dict
        A dictionary containing the calculated internal metrics:
        - ARI (Adjusted Rand Index)
        - Silhouette Score
        - NMI (Normalized Mutual Information)
        - V-measure
    """
    ari = adjusted_rand_score(cell_type_labels, cluster_labels)
    silhouette = silhouette_score(expression_matrix, cluster_labels)
    nmi = normalized_mutual_info_score(cell_type_labels, cluster_labels)
    v_measure = v_measure_score(cell_type_labels, cluster_labels)

    return {
        "ARI": ari,
        "Silhouette": silhouette,
        "NMI": nmi,
        "V-measure": v_measure
    }


def external_evaluation(true_labels, predicted_labels):
    """
    Perform external evaluation of cell type predictions.

    Parameters:
    -----------
    true_labels : pd.Series
        Series containing the true (reference) cell type labels.
    predicted_labels : pd.Series
        Series containing the predicted cell type labels.

    Returns:
    --------
    dict
        A dictionary containing the calculated external metrics:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1-score (weighted)
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def load_true_labels(metadata_path):
    """
    Load ground truth cell type labels from a metadata CSV file.

    Parameters:
    -----------
    metadata_path : str
        Path to the metadata CSV file.

    Returns:
    --------
    pd.Series
        A series containing the ground truth cell type labels, indexed by cell barcodes.
    """
    # Read the CSV file
    metadata = pd.read_csv(metadata_path)

    # Extract the barcode (first column) and celltype_minor
    ground_truth = pd.Series(metadata['celltype_minor'].values, index=metadata.iloc[:, 0])

    return ground_truth


"""
# Example usage:
# Assuming you have the following data:
# expression_df: your expression matrix (pandas DataFrame)
# df: a DataFrame containing 'cluster', 'predicted_cell_type', and 'true_cell_type' columns

# Internal evaluation
internal_metrics = internal_evaluation(expression_df, df['cluster'], df['predicted_cell_type'])
print("Internal Metrics:")
for metric, value in internal_metrics.items():
    print(f"{metric}: {value}")

# External evaluation (if you have reference data)
external_metrics = external_evaluation(df['true_cell_type'], df['predicted_cell_type'])
print("\nExternal Metrics:")
for metric, value in external_metrics.items():
    print(f"{metric}: {value}")
"""