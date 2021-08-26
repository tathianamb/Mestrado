from pandas import Series
from statsmodels.tsa.arima.model import ARIMA
from Processing.Evaluation import metricError
from sklearn.model_selection import train_test_split
from numpy import concatenate

def armaPredict(serie: Series, isHybrid=False, order=None):

    trainS, testS = train_test_split(serie, test_size=0.2, shuffle=False)
    forecastsTest, predictedTest = [], []

    if isHybrid:
        model = ARIMA(trainS, order=order).fit()
        # predict train series
        predictedTrain = model.predict(n_periods=len(trainS))

        errorTrain = trainS - predictedTrain

        # predict test series
        for sample in testS.values:
            predictedTest.append(model.forecast()[0])
            model = model.append([sample])

        errorTest = testS - predictedTest

        errorSeries = Series(concatenate((errorTrain, errorTest), axis=None), index=serie.index)
        predictedSeries = Series(concatenate((predictedTrain, predictedTest), axis=None), index=serie.index)

        return errorSeries, predictedSeries

    forecastsTest = []
    aic = float('inf')

    for p_ in range(1, 10):
        for q_ in range(1, 10):
            model = ARIMA(trainS, order=(p_, 0, q_)).fit()
            aic_ = model.aic
            if aic_ < aic:
                p = p_
                q = q_

    model = ARIMA(trainS, order=(p, 0, q)).fit()
    order=(p, 0, q)
    print('\t' + str((order)))

    # predict test series
    for sample in testS:
        forecastsTest.append(model.forecast()[0])
        model = model.append([sample])

    predictedTest = Series(data=forecastsTest, index=testS.index, name='Predicted')
    mse, mae, errorTest = metricError(predictedTest, testS)

    return mse, mae, predictedTest, order