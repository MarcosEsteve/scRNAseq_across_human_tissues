import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csr_matrix, vstack


def normalize_cpm(expression_matrix):
    """
    Normalize using Counts per Million (CPM)

    This function normalizes gene expression data by scaling each cell's counts to a total of one million.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with CPM normalization applied.
    """
    # Calculate total counts per cell
    total_counts = expression_matrix.sum(axis=0)

    # Scale counts to counts per million
    cpm_matrix = expression_matrix.multiply(1e6).divide(total_counts)

    return cpm_matrix


def normalize_quantile_regression(expression_matrix):
    """
    Normalize using Quantile Regression

    This function applies quantile regression to adjust for library size and other technical variations.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with quantile regression normalization applied.
    """
    # Calculate library size
    library_size = expression_matrix.sum(axis=0)

    # Normalizing library_size with log and mean-centering
    library_size_normalized = np.log1p(library_size)
    library_size_normalized = (library_size_normalized - library_size_normalized.mean()) / library_size_normalized.std()

    # Create a list to store normalized rows
    normalized_rows = []

    for gene in expression_matrix.index:
        # Extract gene expression values as a sparse array
        y = expression_matrix.loc[gene].values

        # Ensure values are dense for statsmodels
        y_dense = np.array(y, dtype=float)  # Convert sparse values to dense

        # Check if the gen is expressed in any cell, if not, no normalization required
        if np.any(y_dense > 0):
            # Log-transform and normalize gene expression
            y_dense_normalized = np.log1p(y_dense)
            y_mean = np.mean(y_dense_normalized)
            y_std = np.std(y_dense_normalized)
            if y_std > 0:
                # Normalize and perform quantile regression
                y_dense_normalized = (y_dense_normalized - y_mean) / y_std

                mod = sm.QuantReg(y_dense_normalized, sm.add_constant(library_size_normalized))
                res = mod.fit(q=0.5)  # Median regression

                # Predicted values and normalization
                predicted = res.predict(sm.add_constant(library_size_normalized))
                normalized_values = y_dense_normalized - predicted

                # Convert back to original scale
                normalized_values = np.expm1(normalized_values * y_std + y_mean)
            else:
                # If std is zero, return the original values
                normalized_values = y_dense

            # Convert the normalized values into a sparse array (CSR format)
            sparse_row = csr_matrix(normalized_values)

            # Append the normalized row to the list of rows (each as a sparse array)
            normalized_rows.append(sparse_row)
        else:
            # If gene is not expressed in any cell, append the original values as sparse
            sparse_row = csr_matrix(y_dense)
            normalized_rows.append(sparse_row)

    # Concatenate all rows (as sparse matrices) into a single sparse matrix
    sparse_matrix = csr_matrix(vstack(normalized_rows))

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,
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
    # Create a list to store normalized rows
    normalized_rows = []

    for gene in expression_matrix.index:
        # Extract gene expression values as a sparse array and convert to dense for regression
        y = expression_matrix.loc[gene].sparse.to_dense().values

        # If the gene is expressed in any cell, apply normalization
        if np.any(y > 0):
            # Create the design matrix X for the regression (intercept only model)
            X = np.ones_like(y)  # Intercept-only model (no covariates)

            # Fit Negative Binomial regression model using statsmodels
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))
            result = model.fit()

            # Get predicted values from the model
            predicted = result.predict(X)

            # Perform normalization by adjusting for the predicted values
            normalized_values = y / predicted * y.mean()

            # Convert the normalized values into a sparse array (CSR format)
            sparse_row = csr_matrix(normalized_values)

            # Append the normalized row to the list of rows (as a sparse array)
            normalized_rows.append(sparse_row)
        else:
            # If gene is not expressed in any cell, append the original values as sparse
            sparse_row = csr_matrix(y)
            normalized_rows.append(sparse_row)

    # Concatenate all rows (as sparse matrices) into a single sparse matrix
    sparse_matrix = vstack(normalized_rows)

    # Convert the resulting sparse matrix into a SparseDataFrame
    normalized_data = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,
                                                        index=expression_matrix.index,
                                                        columns=expression_matrix.columns)

    return normalized_data
