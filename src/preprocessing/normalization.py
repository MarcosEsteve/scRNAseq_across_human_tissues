import numpy as np
import pandas as pd
import gc
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, csc_matrix, vstack


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


def normalize_quantile_regression(expression_matrix):
    """
    Normalize using Quantile Regression

    This function applies quantile regression to adjust for library size and other technical variations.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with quantile regression normalization applied.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix for column-wise operations
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())

    # Calculate library size for each cell
    library_size = sparse_matrix.sum(axis=0).A1  # Sum across rows for each column, returning a 1D array

    # Normalize library_size with log and mean-centering
    library_size_normalized = np.log1p(library_size)
    library_size_normalized = (library_size_normalized - library_size_normalized.mean()) / library_size_normalized.std()

    # Create a list to store normalized rows
    normalized_rows = []

    # Iterate through each gene (row in the sparse matrix)
    for i in range(sparse_matrix.shape[0]):
        # Extract the current row as a sparse matrix
        sparse_row = sparse_matrix.getrow(i)

        # Extract non-zero values and their indices
        y_data = sparse_row.data  # Non-zero values
        y_indices = sparse_row.indices  # Column indices of the non-zero values

        if y_data.size > 0:
            # Log-transform the non-zero values
            y_log = np.log1p(y_data)

            # Calculate mean and standard deviation of the log-transformed values
            y_mean = y_log.mean()
            y_std = y_log.std()

            if y_std > 0:
                # Standardize the log-transformed values
                y_normalized = (y_log - y_mean) / y_std

                # Perform quantile regression to adjust for library size
                mod = sm.QuantReg(y_normalized, sm.add_constant(library_size_normalized[y_indices]))
                res = mod.fit(q=0.5, max_iter=500)  # Median regression

                # Predict the adjusted values and apply normalization
                predicted = res.predict(sm.add_constant(library_size_normalized[y_indices]))

                final_normalized = y_normalized - predicted

                # Convert back to the original scale
                final_normalized = np.expm1(final_normalized * y_std + y_mean)

                # Create a sparse row with the normalized values
                sparse_row_normalized = csr_matrix((final_normalized, (np.zeros_like(y_indices), y_indices)),
                                                   shape=(1, sparse_matrix.shape[1]))

            else:
                # If the standard deviation is zero, keep the original sparse row
                sparse_row_normalized = sparse_row
        else:
            # If no non-zero values, append an empty sparse row
            sparse_row_normalized = sparse_row

        # Append the normalized sparse row to the list
        normalized_rows.append(sparse_row_normalized)

    # Stack all normalized rows into a single sparse matrix
    normalized_matrix = csr_matrix(vstack(normalized_rows))

    # Release memory for large intermediate variables
    del sparse_matrix, normalized_rows
    gc.collect()  # Force garbage collection to free unused memory

    # Apply scaling
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    normalized_matrix = scaler.fit_transform(normalized_matrix)

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(normalized_matrix,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data


def normalize_negative_binomial(expression_matrix):
    """
    Normalize using Negative Binomial Regression

    This function applies negative binomial regression to account for overdispersion in the data.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with negative binomial normalization applied.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix for column-wise operations
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())

    # Create a list to store normalized rows
    normalized_rows = []

    for i in range(sparse_matrix.shape[0]):
        # Extract gene expression values as a sparse array for the current row (gene)
        sparse_row = sparse_matrix.getrow(i)

        # Extract non-zero values and their indices for the sparse row
        y_data = sparse_row.data  # Non-zero values
        y_indices = sparse_row.indices  # Column indices of the non-zero values
        if y_data.size > 0 and np.std(y_data) > 0:  # Only process genes with variability:
            # Create the design matrix X for the regression (intercept-only model)
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

            # Append the normalized row to the list of rows
            normalized_rows.append(sparse_row_normalized)
        else:
            # If gene is not expressed in any cell, append the original sparse row
            normalized_rows.append(sparse_row)

    # Concatenate all rows (as sparse matrices) into a single sparse matrix
    sparse_matrix_normalized = csr_matrix(vstack(normalized_rows))

    # Release memory for large intermediate variables
    del sparse_matrix, normalized_rows
    gc.collect()  # Force garbage collection to free unused memory

    # Apply scaling
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    sparse_matrix_normalized = scaler.fit_transform(sparse_matrix_normalized)

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(sparse_matrix_normalized,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data
