import pandas
import numpy as np

def cardinality(dSeries: pandas.Series) -> list:
    """
    Return all of the unique values in the Series.
    """
    return list(dSeries.nunique())

def cardinalityCount(dSeries: pandas.Series) -> int:
    """
    Return the count of unique values in the Series.
    """
    return len(dSeries.unique())

def countApprox(data: list) -> int:
    """
    Use hyperloglog to compute the approximate count of unique values in the data.
    """
    from datasketch import HyperLogLog
    hll = HyperLogLog()
    for item in data:
        hll.update(str(item).encode('utf-8'))
    return hll.count()


def cardinalityDF(dataframe : pandas.DataFrame):
    """
    Compute Cardinality for each column in the dataframe and return a dictionary of column name to cardinality.
    """
    return dataframe.nunique()

def monotonicity(dSeries: pandas.Series) -> bool:
    """
    Return True if the Series is monotonic increasing or decreasing, False otherwise.
    Uisng pandas built in function to check for monotonicity.
    """
    return dSeries.is_monotonic_increasing or dSeries.is_monotonic_decreasing

def degreeMonotonicity(dSeries: pandas.Series) -> float:
    """
    Return the degree of monotonicity of the Series, which is the proportion of values that are in order.
    """
    if len(dSeries) <= 1:
        return 1.0
    return np.diff(dSeries.mean())
