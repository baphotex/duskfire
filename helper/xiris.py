"""xiris
grabs iris from sns
"""

import pandas as pd
import seaborn as sns

def get_iris():
    """grabs iris from sns

    returns
    -------
    DataFrame
        returns the iris df
    """
    iris = sns.load_dataset('iris')
    return iris 