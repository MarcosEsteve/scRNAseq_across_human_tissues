import numpy as np
import pandas as pd
import gc
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, csc_matrix, vstack
from joblib import Parallel, delayed


def normalize_cpm(expression_matrix):
    """
    Normalize using Counts per Million (CPM)

    This function normalizes gene expression data by scaling each cell's counts to a total of one million.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with CPM normalization applied.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())

    # Calculate total counts per cell (column-wise sum)
    total_counts = sparse_matrix.sum(axis=0).A1  # .A1 to convert to 1D array

    # Create a Dense array of total counts to allow broadcasting
    total_counts_inv = 1 / total_counts  # Inverse of total counts

    # Scale counts to counts per million (element-wise division)
    cpm_matrix = sparse_matrix.multiply(1e6).multiply(total_counts_inv)

    # Apply scaling
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    cpm_matrix = scaler.fit_transform(cpm_matrix)

    # Return the result as a SparseDataFrame
    return pd.DataFrame.sparse.from_spmatrix(cpm_matrix, index=expression_matrix.index,
                                             columns=expression_matrix.columns)


def process_gene(i, sparse_matrix, library_size_normalized):
    #print(i)
    sparse_row = sparse_matrix.getrow(i)
    y_data = sparse_row.data  # Non-zero values
    y_indices = sparse_row.indices  # Column indices of the non-zero values

    if y_data.size > 0:
        y_log = np.log1p(y_data)
        y_mean, y_std = y_log.mean(), y_log.std()

        if y_std > 0:
            y_normalized = (y_log - y_mean) / y_std
            mod = sm.QuantReg(y_normalized, sm.add_constant(library_size_normalized[y_indices]))
            res = mod.fit(q=0.5, max_iter=500)
            predicted = res.predict(sm.add_constant(library_size_normalized[y_indices]))
            final_normalized = y_normalized - predicted
            final_normalized = np.expm1(final_normalized * y_std + y_mean)
            sparse_row_normalized = csr_matrix((final_normalized, (np.zeros_like(y_indices), y_indices)),
                                               shape=(1, sparse_matrix.shape[1]))
        else:
            sparse_row_normalized = sparse_row
    else:
        sparse_row_normalized = sparse_row

    return i, sparse_row_normalized


def normalize_quantile_regression(expression_matrix):
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())
    library_size = sparse_matrix.sum(axis=0).A1
    library_size_normalized = np.log1p(library_size)
    library_size_normalized = (library_size_normalized - library_size_normalized.mean()) / library_size_normalized.std()
    print("Starting parallelization")
    results = Parallel(n_jobs=-1)(
        delayed(process_gene)(i, sparse_matrix, library_size_normalized)
        for i in range(sparse_matrix.shape[0])
    )
    print("Parallelization finished")
    results_sorted = sorted(results, key=lambda x: x[0])  # Sort by index
    normalized_rows = [row for _, row in results_sorted]
    normalized_matrix = csr_matrix(vstack(normalized_rows))

    del sparse_matrix, normalized_rows, results
    gc.collect()

    scaler = StandardScaler(with_mean=False)
    normalized_matrix = scaler.fit_transform(normalized_matrix)

    normalized_data = pd.DataFrame.sparse.from_spmatrix(normalized_matrix,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data


def process_gene_nb(i, sparse_matrix):
    #print(i)
    sparse_row = sparse_matrix.getrow(i)
    y_data = sparse_row.data  # Non-zero values
    y_indices = sparse_row.indices  # Column indices of the non-zero values

    if y_data.size > 0 and np.std(y_data) > 0:  # Only process genes with variability
        X = np.ones_like(y_data)  # Intercept-only model (no covariates)

        # Fit Negative Binomial regression model using statsmodels
        model = sm.GLM(y_data, X, family=sm.families.NegativeBinomial(alpha=1.0))
        result = model.fit()

        # Get predicted values from the model
        predicted = result.predict(X)

        # Perform normalization by adjusting for the predicted values
        normalized_values = y_data / predicted * y_data.mean()

        # Convert the normalized values into a sparse row (CSR format)
        sparse_row_normalized = csr_matrix((normalized_values, (np.zeros_like(y_indices), y_indices)),
                                           shape=(1, sparse_matrix.shape[1]))
    else:
        sparse_row_normalized = sparse_row

    return i, sparse_row_normalized


def normalize_negative_binomial(expression_matrix):
    """
    Normalize using Negative Binomial Regression with parallelization.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with negative binomial normalization applied.
    """
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())

    print("Starting parallelization")
    results = Parallel(n_jobs=-1)(
        delayed(process_gene_nb)(i, sparse_matrix)
        for i in range(sparse_matrix.shape[0])
    )
    print("Parallelization finished")

    results_sorted = sorted(results, key=lambda x: x[0])  # Sort by index
    normalized_rows = [row for _, row in results_sorted]
    normalized_matrix = csr_matrix(vstack(normalized_rows))

    del sparse_matrix, normalized_rows, results, results_sorted
    gc.collect()

    scaler = StandardScaler(with_mean=False)
    normalized_matrix = scaler.fit_transform(normalized_matrix)

    normalized_data = pd.DataFrame.sparse.from_spmatrix(normalized_matrix,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data
