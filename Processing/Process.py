import pandas as pd
from numpy import amin

def getIDXMinMSE(df : pd.DataFrame):

    indexes = list(set(df.index))
    lowerValue = amin(df['mse'].values)

    for i in indexes:
        if df.loc[i, 'mse'] == lowerValue:
            return i

    print("value not found")

