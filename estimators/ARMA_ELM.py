from estimators.ARMA import armaPredict
from estimators.baseELM import elmPredict
from pandas import DataFrame, MultiIndex
from Processing.Evaluation import metricError
from Processing.Process import getIDXMinMSE, prepareDataToANN


def armaElmPredict(serie, order):
    predictionErrorSerie, predictedSeries = armaPredict(serie, isHybrid=True, order=order)

    # --------------- LINEAR MODELS PREDICT ENDING  ---------------
    # --------------- ANN PREDICT BEGINNING ---------------

    X_train, y_train, X_val, y_val, X_test, y_test, scalerTest = prepareDataToANN(predictionErrorSerie, estimator='ELM')

    idx: MultiIndex = MultiIndex.from_product([[i for i in range(5, 30, 2)], [j for j in range(0, 30)]],
                                              names=['nneurons', 'test'])

    # --------------- VALIDATION ---------------

    validationErrorDF: DataFrame = DataFrame(index=idx, columns=['mse', 'mae'])
    validationErrorAverageDF = DataFrame(index=[i for i in range(5, 30, 2)], columns=['mse', 'mae'])

    for n_hidden in range(5, 30, 2):

        for test in range(0, 30):
            predicted = elmPredict(hidden_dim=n_hidden,
                                   x_train=X_train,
                                   y_train=y_train, x_test=X_test,
                                   y_test=y_test)

            validationErrorMSE, validationErrorMAE, _ = metricError(predictedValues=predicted, actualValues=y_test)

            validationErrorDF.loc[(n_hidden, test), 'mse'] = validationErrorMSE
            validationErrorDF.loc[(n_hidden, test), 'mae'] = validationErrorMAE

        validationErrorAverageDF.loc[n_hidden, 'mse'] = validationErrorDF.loc[n_hidden, 'mse'].mean()
        validationErrorAverageDF.loc[n_hidden, 'mae'] = validationErrorDF.loc[n_hidden, 'mae'].mean()

    n_hidden = (getIDXMinMSE(validationErrorAverageDF))

    # --------------- TEST ---------------

    testDF: DataFrame = DataFrame(index=y_test.index, columns=[i for i in range(0, 30)])

    for test in range(0, 30):
        predicted = elmPredict(hidden_dim=n_hidden,
                               x_train=X_train,
                               y_train=y_train, x_test=X_test,
                               y_test=y_test)
        testDF[test] = ((((predicted + 1) / 2) * (max(scalerTest) - min(scalerTest))) + min(scalerTest)).values.reshape(1, -1)[0] + predictedSeries[-len(y_test):]
    return n_hidden, validationErrorAverageDF.loc[n_hidden], testDF