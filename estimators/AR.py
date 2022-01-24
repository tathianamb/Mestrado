from pandas import Series, concat
from statsmodels.tsa.arima.model import ARIMA
from Processing.Evaluation import metricError
from numpy import concatenate

def arPredict(trainS, testS, minMaxTest_LM, isHybrid=False, order=None):

    forecastsTest, predictedTest = [], []

    if isHybrid:
        model = ARIMA(trainS, order=order).fit(method='yule_walker')
        # predict train series
        predictedTrain = model.predict(n_periods=len(trainS))

        errorTrain = trainS - predictedTrain

        # predict test series
        for sample in testS.values:
            predictedTest.append(model.forecast()[0])
            model = model.append([sample])

        errorTest = testS - predictedTest

        errorSeries = concat([errorTrain, errorTest])
        predictedSeries = concat([predictedTrain, Series(predictedTest)])

        return errorSeries, predictedSeries

    forecastsTest = []
    aic = float('inf')

    for p_ in range(1, 10):
        model = ARIMA(trainS, order=(p_, 0, 0)).fit(method='yule_walker')
        aic_ = model.aic
        if aic_ < aic:
            p = p_


    model = ARIMA(trainS, order=(p, 0, 0)).fit(method='yule_walker')
    order=(p, 0, 0)
    print('\t' + str((order)))

    # predict test series
    for sample in testS.values:
        forecastsTest.append(model.forecast()[0])
        model = model.append([sample])

    predictedTest = Series(data=forecastsTest, index=testS.index, name='Predicted')
    predictedTest = (((predictedTest + 1) / 2) * (max(minMaxTest_LM) - min(minMaxTest_LM))) + min(minMaxTest_LM)
    mse, mae, errorTest = metricError(predictedTest, testS)

    return mse, mae, predictedTest, order