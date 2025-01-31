import numpy as np
import pandas as pd
import gc
import statsmodels.api as sm
from scipy.stats import scoreatpercentile
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
    print(f"Non-zero elements before normalization: {sparse_matrix.nnz}")

    # Calculate total counts per cell (column-wise sum)
    total_counts = sparse_matrix.sum(axis=0).A1  # .A1 to convert to 1D array

    # Create a Dense array of total counts to allow broadcasting
    total_counts_inv = 1 / total_counts  # Inverse of total counts

    # Scale counts to counts per million (element-wise division)
    cpm_matrix = sparse_matrix.multiply(1e6).multiply(total_counts_inv)
    print(f"Non-zero elements after normalization: {cpm_matrix.nnz}")

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
    print(f"Non-zero elements before normalization: {sparse_matrix.nnz}")
    print(sparse_matrix.min(), sparse_matrix.max(), sparse_matrix.mean())

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
        #sparse_row = sparse_matrix[i, :]
        sparse_row = sparse_matrix.getrow(i)

        # Extract non-zero values and their indices
        y_data = sparse_row.data  # Non-zero values
        #print(y_data)
        y_indices = sparse_row.indices  # Column indices of the non-zero values
        #print(y_indices)

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
                res = mod.fit(q=0.5)  # Median regression

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
    print(f"Non-zero elements after normalization: {normalized_matrix.nnz}")
    print(normalized_matrix.min(), normalized_matrix.max(), normalized_matrix.mean())

    # Release memory for large intermediate variables
    del sparse_matrix, normalized_rows
    gc.collect()  # Force garbage collection to free unused memory

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(normalized_matrix,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data


"""
def normalize_quantile_regression(expression_matrix):
    
    Normalize gene expression using Quantile Regression, adjusting for library size, for all genes simultaneously.

    This function uses quantile regression (at the median, q=0.5) to adjust gene expression values based on the
    total library size of each cell. It applies the regression to the entire gene set at once, rather than iterating
    over genes individually.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with quantile regression-based normalization applied.
    

    # Convert the SparseDataFrame to a sparse CSC matrix for efficient column-wise operations
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())
    print("1")

    # Calculate the total counts (library size) per cell (sum of expression counts for each column)
    total_counts = sparse_matrix.sum(axis=0).A1  # Convert to dense array (1D)
    print("2")

    # Normalize the total counts by log transformation and mean-centering
    total_counts_log = np.log1p(total_counts)  # Apply log(1+x) transformation to library size
    total_counts_log = (total_counts_log - np.mean(total_counts_log)) / np.std(
        total_counts_log)  # Mean-centering and standardization
    print("3")

    # Prepare the design matrix with a constant for intercept and the normalized library size as the covariate
    X = sm.add_constant(total_counts_log)  # Add a constant to the design matrix (for the intercept term)
    print("Shape of X:", X.shape)
    print("4")

    # Convert the sparse matrix to a dense matrix for regression analysis
    #Y = sparse_matrix.toarray().T  # Dense representation of the gene expression data
    #print("Shape of Y:", Y.shape)
    print("5")

    # Fit a Quantile Regression model for the entire expression matrix
    model = sm.QuantReg(sparse_matrix.T, X)  # Quantile Regression, with the total counts as the independent variable
    print("6")
    result = model.fit(q=0.5)  # Fit the model at the median quantile (q=0.5)
    print("7")

    # Get the predicted values for all genes
    predicted_values = result.predict(X)  # Predicted normalization factors based on the library size
    print("8")

    # Normalize gene expression by subtracting the predicted values (adjust for library size)
    normalized_expression = sparse_matrix - predicted_values.T  # Remove the library size effect from the expression values
    print("9")

    # Convert the resulting matrix back to a sparse matrix for efficiency
    sparse_matrix = csc_matrix(normalized_expression.T)
    print("10")

    # Return the normalized matrix as a SparseDataFrame
    return pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=expression_matrix.index,
                                             columns=expression_matrix.columns)
"""


def normalize_tmm(expression_matrix, trim_percent=0.2):
    """
    Normalize using Trimmed Mean of M-values (TMM)

    This function normalizes gene expression data using TMM, which adjusts for
    compositional bias between samples by trimming extreme values.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.
    - trim_percent (float): The proportion of genes to trim from the high and low ends of the distribution (default 0.2 for 20%).

    Returns:
    - pd.DataFrame: A SparseDataFrame with TMM normalization applied.
    """
    # Convert the SparseDataFrame to a sparse CSC matrix
    sparse_matrix = csc_matrix(expression_matrix.sparse.to_coo())
    print(1)

    # Calculate the geometric mean of gene expression across cells
    gene_means = sparse_matrix.mean(axis=1).A1  # Use .A1 to get a 1D array directly from the sparse matrix
    print(2)

    # Calculate M-values (log fold changes) for each gene
    log_gene_means = np.log1p(gene_means)
    log_expression = np.log1p(sparse_matrix)  # Log transformation directly on sparse matrix
    print("2.5")

    M_values = log_expression - log_gene_means[:, None]  # Broadcasting to get M-values for each gene
    print(3)

    # Calculate the trimming thresholds (lower and upper percentiles)
    lower_trim = scoreatpercentile(M_values.data, trim_percent * 100)  # Use the data part of the sparse matrix
    upper_trim = scoreatpercentile(M_values.data, (1 - trim_percent) * 100)
    print(4)

    # Normalize M-values based on trimming
    M_trimmed = M_values.copy()
    M_trimmed.data = np.clip(M_values.data, lower_trim, upper_trim)
    print(5)

    # Apply inverse transformation to get normalized expression values
    # Inverse transformation: exp(M_trimmed) - 1
    tmm_normalized = np.expm1(M_trimmed)
    print(6)

    # Convert back to sparse matrix
    sparse_matrix = csc_matrix(tmm_normalized)
    print(7)

    # Return as SparseDataFrame
    return pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=expression_matrix.index,
                                             columns=expression_matrix.columns)


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
    print(f"Non-zero elements before normalization: {sparse_matrix.nnz}")
    print(sparse_matrix.min(), sparse_matrix.max(), sparse_matrix.mean())

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

    print(f"Non-zero elements after normalization: {sparse_matrix_normalized.nnz}")
    print(sparse_matrix_normalized.min(), sparse_matrix_normalized.max(), sparse_matrix_normalized.mean())

    # Release memory for large intermediate variables
    del sparse_matrix, normalized_rows
    gc.collect()  # Force garbage collection to free unused memory

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(sparse_matrix_normalized,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data
