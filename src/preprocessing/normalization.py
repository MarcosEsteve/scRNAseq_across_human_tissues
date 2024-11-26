import numpy as np
import statsmodels.api as sm


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

    # Log-transform the library size for regression
    log_library_size = np.log1p(library_size).values.reshape(-1, 1)

    normalized_data = expression_matrix.copy()

    for gene in expression_matrix.index:
        y = expression_matrix.loc[gene].to_dense().values
        mod = sm.QuantReg(y, log_library_size)
        res = mod.fit(q=0.5)  # Median regression
        predicted = res.predict(log_library_size)
        normalized_data.loc[gene] = y - predicted + y.mean()

    return normalized_data.sparse.to_sparse()


def normalize_negative_binomial(expression_matrix):
    """
    Normalize using Negative Binomial Regression

    This function applies negative binomial regression to account for overdispersion in the data.

    Parameters:
    - expression_matrix (pd.DataFrame): A pandas SparseDataFrame where rows represent genes and columns represent cells.

    Returns:
    - pd.DataFrame: A SparseDataFrame with negative binomial normalization applied.
    """
    normalized_data = expression_matrix.copy()

    for gene in expression_matrix.index:
        y = expression_matrix.loc[gene].to_dense().values
        X = np.ones_like(y)  # Intercept only model
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        result = model.fit()
        predicted = result.predict(X)
        normalized_data.loc[gene] = y / predicted * y.mean()

    return normalized_data.sparse.to_sparse()
