from typing import Tuple, Union

from numpy import arange
from pandas import Series, DataFrame, Grouper
from statsmodels.tsa.stattools import adfuller

from estimators.baseELM import elmPredict
from estimators.baseESN import esnPredict
from estimators.baseMLP import mlpPredict

from Processing.Evaluation import metricError


def _toStationary(series: Series) -> Tuple[Series, Union[Series, DataFrame], Union[Series, DataFrame]]:
    monthlyMean: Union[Series, DataFrame] = series.groupby(Grouper(freq='D')).mean()
    monthlySTD: Union[Series, DataFrame] = series.groupby(Grouper(freq='D')).std()

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

def preProcessSeries(serie: Series):

    scalerSTD, scalerMean = None, None

    if not _isStationary(serie):
        print('Serie is not stationary')

        serie, scalerSTD, scalerMean = _toStationary(serie)

    return serie, scalerSTD, scalerMean

def _usingESN(X, y, n_reservoir: int = 20) -> float:
    trainingRows: int = int(0.7 * len(X))
    y_predicted_test = esnPredict(n_input=len(X.columns), n_reservoir=n_reservoir,
                                      x_training=X[:trainingRows], y_training=y[:trainingRows],
                                      x_test=X[trainingRows:], y_test=y[trainingRows:])
    mse, mae, error = metricError(actualValues=y[trainingRows:], predictedValues=y_predicted_test)

    return mse


def _usingELM(X, y) -> float:
    trainingRows = int(0.7 * len(X))
    y_predicted_test = elmPredict(hidden_dim=1, x_train=X[:trainingRows],
                                  y_train=y[:trainingRows], x_test=X[trainingRows:],
                                  y_test=y[trainingRows:])
    mse, mae, error = metricError(actualValues=y[trainingRows:], predictedValues=y_predicted_test)

    return mse

def _usingMLP(X, y) -> float:
    trainingRows = int(0.7 * len(X))
    y_predicted_test = mlpPredict(hidden_dim=1, x_train=X[:trainingRows],
                                  y_train=y[:trainingRows], x_test=X[trainingRows:],
                                  y_test=y[trainingRows:])
    mse, mae, error = metricError(actualValues=y[trainingRows:], predictedValues=y_predicted_test)
    return mse

def cross_val_score(estimator, X, y) -> list:
    MSE: list = []
    if estimator == 'ESN':
        MSE.append(_usingESN(X, y))
    elif estimator == 'ELM':
        MSE.append(_usingELM(X, y))
    elif estimator == 'MLP':
        MSE.append(_usingMLP(X, y))
    else:
        print("\n\n=============== NO ESTIMATOR WAS SELECTED!!")

    return MSE
