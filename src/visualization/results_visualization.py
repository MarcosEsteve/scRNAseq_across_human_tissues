import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def filter_by_step(df, step_methods, step_position):
    """
    Filter the results based on a specific step in the pipeline ID.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing pipeline results.
    step_methods : list of str
        List of method codes for the step (e.g., ['FLEG', 'FHMC', 'FD', 'CC'] for data cleaning).
    step_position : int
        The position of the step in the pipeline ID (0-based index).

    Returns:
    --------
    dict
        Dictionary where keys are the step methods and values are the corresponding filtered DataFrames.
    """
    filtered_results = {}

    for method in step_methods:
        filtered_results[method] = df[df['pipeline_id'].apply(lambda x: x.split('_')[step_position] == method)]

    return filtered_results


def plot_metric_by_step(filtered_results, step_name, metric, tissue, plot_type="box"):
    """
    Generate a boxplot to compare a given metric across different methods within a pipeline step.

    This function takes the filtered results from `filter_by_step` and generates a
    seaborn boxplot, which is useful for visualizing the distribution of the metric
    across different methods.

    Parameters:
    -----------
    filtered_results : dict
        Dictionary where keys are method names and values are DataFrames filtered by the method.
    step_name : str
        Name of the step being analyzed (e.g., 'Data Cleaning', 'Feature Selection').
    metric : str
        The metric to visualize (e.g., 'Silhouette_Score', 'ARI', 'Accuracy').
    plot_type : str, optional
        Type of plot to generate ('box' for boxplot, 'violin' for violin plot). Default is 'box'.
   tissue : str
        The tissue type shown.

    Returns:
    --------
    None
    """
    plot_data = []
    for method, df in filtered_results.items():
        for value in df[metric]:
            plot_data.append({step_name: method, metric: value})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 5))
    if plot_type == "violin":
        sns.violinplot(x=step_name, y=metric, hue=step_name, data=plot_df, palette="Set2", split=True)
        plt.title(f"Violin Plot: {metric} Across {step_name} Methods")
    else:
        sns.boxplot(x=step_name, y=metric, hue=step_name, data=plot_df, palette="Set2")
        plt.title(f"Boxplot: {metric} Across {step_name} Methods")

    plt.title(f"Comparison of {metric} Across {step_name} Methods ({tissue})")
    plt.xlabel(f"{step_name} Method")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend([], [], frameon=False)
    plt.show()


def get_top_n_performers(df, metric, n=10):
    """
    Retrieve the top N best-performing pipelines based on a given metric.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results of different pipelines.
    metric : str
        The metric to evaluate (e.g., 'Silhouette_Score', 'ARI', 'Accuracy').
    N : int, optional (default=10)
        The number of top-performing results to return.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the top N best-performing results sorted by the metric.
    """

    # Select the top N rows based on the metric (always sorting in descending order)
    top_df = df.sort_values(by=metric, ascending=False).head(n)

    return top_df


def filter_by_barcode_length(df):
    """
    Filter out pipelines whose barcode length is more than 20000 less than the maximum barcode length.

    Parameters:
    -----------
    df : pd.DataFrame
        The original dataframe containing all pipeline results.

    Returns:
    --------
    tuple (pd.DataFrame, list)
        - The filtered dataframe.
        - List of pipeline IDs that were removed.
    """
    # Convert barcodes column to list of barcodes and calculate their lengths
    df['barcode_length'] = df['barcodes'].apply(lambda x: len(x.split(',')))

    # Compute the maximum barcode length
    max_length = df['barcode_length'].max()

    # Identify pipelines to drop (barcode length < max_length - 20000)
    dropped_pipelines = df[df['barcode_length'] < max_length - 20000]['pipeline_id'].tolist()

    # Filter the dataframe: keep only rows with barcode length >= max_length - 20000
    filtered_df = df[df['barcode_length'] >= max_length - 20000].drop(columns=['barcode_length'])

    print(f"Max barcode length: {max_length}")
    print(f"Original size: {len(df)}, Filtered size: {len(filtered_df)}, Dropped: {len(dropped_pipelines)}")

    return filtered_df, dropped_pipelines


def filter_metrics(df):
    """
    Filter the DataFrame to retain only the columns that are metrics.
    It keeps the 'pipeline_id' and removes non-metric columns (like barcodes, clusters, etc.).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the pipeline results.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing only the 'pipeline_id' and the metric columns.
    """
    metrics_columns = [
        'Silhouette_Score', 'Davies_Bouldin_Index', 'Calinski_Harabasz_Score',
        'ARI', 'NMI', 'V_measure', 'Accuracy', 'Precision', 'Recall', 'F1_score'
    ]

    columns_to_keep = ['pipeline_id'] + metrics_columns
    df_filtered = df[columns_to_keep]

    return df_filtered


def filter_metrics_by_step(filtered_results):
    """
    Filter the DataFrames within the filtered_results dictionary to retain only the columns that are metrics.
    It keeps the 'pipeline_id' and removes non-metric columns (like barcodes, clusters, etc.) for each step's DataFrame.

    Parameters:
    -----------
    filtered_results : dict
        Dictionary where keys are step methods (e.g., 'FLEG', 'FHMC') and values are DataFrames filtered by method.

    Returns:
    --------
    dict
        A dictionary where each key (method) corresponds to a DataFrame with only the 'pipeline_id' and metric columns.
    """
    metrics_columns = [
        'Silhouette_Score', 'Davies_Bouldin_Index', 'Calinski_Harabasz_Score',
        'ARI', 'NMI', 'V_measure', 'Accuracy', 'Precision', 'Recall', 'F1_score'
    ]

    # Nuevo diccionario para los resultados filtrados por métricas
    filtered_metrics_results = {}

    for method, df in filtered_results.items():
        columns_to_keep = ['pipeline_id'] + metrics_columns
        filtered_metrics_results[method] = df[columns_to_keep]

    return filtered_metrics_results


def plot_heatmap_by_step(filtered_results, step_name):
    """
    Generate a heatmap to visualize the correlation between metrics for a given step.

    Parameters:
    -----------
    filtered_results : dict
        Dictionary of DataFrames filtered by method, where each DataFrame contains metrics for a specific pipeline.
    step_name : str
        The step name (e.g., 'Data Cleaning', 'Feature Selection').

    Returns:
    --------
    None
    """
    plot_data = []
    for method, df in filtered_results.items():
        df_metrics = df.drop(columns=['pipeline_id'])

        for metric in df_metrics.columns:
            plot_data.append({step_name: method, 'Metric': metric, 'Value': df_metrics[metric].values[0]})

    plot_df = pd.DataFrame(plot_data)

    plot_df_pivot = plot_df.pivot(index=step_name, columns='Metric', values='Value')

    plt.figure(figsize=(12, 6))
    sns.heatmap(plot_df_pivot.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f"Correlation of Metrics for {step_name}")
    plt.show()


def plot_global_heatmap(df, tissue):
    """
    Generate a global heatmap to visualize the correlation between all metrics
    across different pipelines.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results of all pipelines with only metrics columns.
    tissue : str
        The tissue type shown.

    Returns:
    --------
    None
    """
    # Ensure we're only working with metrics columns (not pipeline_id)
    df_metrics = df.drop(columns=['pipeline_id'])

    # Calculate the correlation matrix for the metrics
    corr_matrix = df_metrics.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    plt.title(f"Global Correlation of Metrics Across {tissue} Pipelines ")
    plt.show()


def plot_global_scatter(df):
    """
    Generate a scatter plot matrix (pair plot) to visualize the relationships
    between all metrics across all pipelines.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results of all pipelines with only metrics columns.

    Returns:
    --------
    None
    """
    # Ensure we're only working with metrics columns (not pipeline_id)
    df_metrics = df.drop(columns=['pipeline_id'])

    # Generate the pair plot (scatter plot matrix)
    sns.pairplot(df_metrics)
    plt.suptitle("Scatter Plot Matrix of Metrics Across Pipelines", y=1.02)
    plt.show()