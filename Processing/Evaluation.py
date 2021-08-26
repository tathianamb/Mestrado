from pandas import Series, DataFrame
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def saveFigSeries(serie: Series, name, type):
    serie.plot.line()
    path = "data\\" + name + "_" + type + ".png"
    #plt.legend(serie.columns)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def boxplot(mseTEsts, name):

    fig1, ax1 = plt.subplots()
    ax1.boxplot(mseTEsts)
    path = "data\\" + "boxplot" + name + ".png"
    plt.savefig(path)
    plt.clf()

def metricError(predictedValues: Series, actualValues: Series):

    mse, mae, error = None, None, None

    if ( type(predictedValues) == Series ) or ( type(predictedValues) == DataFrame):
        mse = MSE(y_true=actualValues.values, y_pred=predictedValues.values)
        mae = MAE(y_true=actualValues.values, y_pred=predictedValues.values)
        error = actualValues - predictedValues

    elif type(predictedValues) == np.ndarray:
        predictedValues = predictedValues.reshape(actualValues.shape, )
        mse = MSE(y_true=actualValues, y_pred=predictedValues)
        mae = MAE(y_true=actualValues, y_pred=predictedValues)
        error = actualValues - predictedValues

    else:
        print("\n\n =============== metric error problem!!!")

    return mse, mae, error


def plotACF_PACF(series: Series, title):
    series.plot(title=title)
    plot_acf(series, title=title + ' ACF')
    plot_pacf(series, title=title + ' PACF')
    plt.show()
    plt.clf()

def plotResult(ALLValues):

    plt.plot(ALLValues)
    plt.ylabel('Wind speed (m/s)')
    plt.show()
    plt.clf()

def wilcoxonTest(outputs, actual):

    testSize = len(outputs[0].values)
    actual = actual[-testSize:]

    mseTests = []

    for column in outputs:

        mseTests.append(MSE(y_true=actual.values, y_pred=outputs[column].values))

    idx = mseTests.index(min(mseTests))

    mae = MAE(y_true=actual.values, y_pred=outputs[idx].values)

    print('\tmse: ', mseTests[idx], ', mae:', mae)

    w, p = wilcoxon(outputs.iloc[idx])

    return w, p, outputs[idx], mseTests
