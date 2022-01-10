from typing import Tuple, Union

from pandas import concat, Series, DataFrame, Grouper
from statsmodels.tsa.stattools import adfuller
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import pacf

def _toStationary(series: Series) -> Tuple[Series, Union[Series, DataFrame], Union[Series, DataFrame]]:
    monthlyMean: Union[Series, DataFrame] = series.groupby(Grouper(freq="D")).mean()
    monthlySTD: Union[Series, DataFrame] = series.groupby(Grouper(freq="D")).std()

    for date in monthlySTD.index.date:
        filterSeries: bool = series.index.date == date
        filterMean: bool = monthlyMean.index.date == date
        filterSTD: bool = monthlySTD.index.date == date

        series[filterSeries] = (series[filterSeries] - monthlyMean[filterMean].values[0]) \
                               / monthlySTD[filterSTD].values[0]

    return series, monthlySTD, monthlyMean


def _isStationary(series: Series) -> bool:
    results = adfuller(series)
    return results[1] <= 0.05

def verify_toStationary(serie: Series):

    scalerSTD, scalerMean = None, None

    if not _isStationary(serie):
        print("Aplicando Z-Score")

        serie, scalerSTD, scalerMean = _toStationary(serie)

    return serie, scalerSTD, scalerMean

def featureSelectionCorr(serie):

    data = DataFrame(index=serie.index)

    for i in range(1, 10 + 1):
        data["lag_" + str(i)] = serie.shift(i)

    data = data[10:]

    data["actual"] = serie

    corr = {}
    for column in data.loc[:, data.columns!="actual"]:
        corr[column] = np.abs(data[column].corr(data["actual"], method="spearman"))
    sortedCorr = sorted(corr.items(), key=lambda x: x[1], reverse=True)
    filter = list(dict(sortedCorr).keys())[:5]
    print("\t" + str(filter))
    return filter

def featureSelectionPACF(serie):
    pacf_values, confint = pacf(serie, nlags=30, alpha=0.05)
    lower_bound = confint[1,0] - pacf_values[1]
    upper_bound = confint[1,1] - pacf_values[1]
    print(lower_bound, upper_bound)
    significant_pacf_values = {}
    for i in range(1, len(pacf_values)):
        if pacf_values[i] >= upper_bound or pacf_values[i] <= lower_bound:
            significant_pacf_values[i] = pacf_values[i]
    sorted_significant_pacf_values = sorted(significant_pacf_values.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_significant_pacf_values) >= 10:
        sorted_significant_pacf_values = sorted_significant_pacf_values[:10]
    significant_lags = [a_tuple[0] for a_tuple in sorted_significant_pacf_values]
    significant_lags.sort()
    print("\t" + str(significant_lags))
    return significant_lags

def generate_X(serie, lags):
    data = DataFrame()
    data["actual"] = serie
    for lag in lags:
        data["lag_" + str(lag)] = serie.shift(lag)
    return data[lags[-1]:]

def prepareDataToANN(serie):

    train, valTest = train_test_split(serie, train_size=0.6, shuffle=False)
    validation, test = train_test_split(valTest, test_size=0.5, shuffle=False)

    '''train, _, _ = verify_toStationary(train.copy())
    validation, scalerSTDValidation, scalerMeanValidation = verify_toStationary(validation.copy())
    test, scalerSTDTest, scalerMeanTest = verify_toStationary(train.copy())'''

    lags = featureSelectionPACF(train)

    train = generate_X(train, lags)
    validation = generate_X(validation, lags)
    test = generate_X(test, lags)

    minMaxVal = (validation.to_numpy().min(), validation.to_numpy().max())
    minMaxTest = (test.to_numpy().min(), test.to_numpy().max())

    scalerTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerVal = MinMaxScaler(feature_range=(-1, 1))
    scalerTest = MinMaxScaler(feature_range=(-1, 1))

    processedTrain = scalerTrain.fit_transform(train)
    processedVal = scalerVal.fit_transform(validation)
    processedTest = scalerTest.fit_transform(test)

    dfProcessedTrain = DataFrame(data=processedTrain, index=train.index, columns=train.columns)
    dfProcessedVal = DataFrame(data=processedVal, index=validation.index, columns=validation.columns)
    dfProcessedTest = DataFrame(data=processedTest, index=test.index, columns=test.columns)


    return dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest
