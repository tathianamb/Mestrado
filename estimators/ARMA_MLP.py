from estimators.ARMA import armaPredict
from estimators.baseMLP import mlpPredict
from pandas import DataFrame, MultiIndex, concat
from Processing.Evaluation import metricError
from Processing.Process import getIDXMinMSE

def armaMlpPredict(dfProcessedTrain,dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest, order):
    predictionErrorSerie, predictedSeries = armaPredict(concat(dfProcessedTrain["actual"], dfProcessedVal["actual"], dfProcessedTest["actual"]), isHybrid=True, order=order)

    # --------------- LINEAR MODELS PREDICT ENDING  ---------------
    # --------------- ANN PREDICT BEGINNING ---------------
    X_train = dfProcessedTrain.loc[:, dfProcessedTrain.columns != "actual"]
    y_train = dfProcessedTrain["actual"]
    X_val = dfProcessedVal.loc[:, dfProcessedVal.columns != "actual"]
    y_val = dfProcessedVal["actual"]
    X_test = dfProcessedTest.loc[:, dfProcessedTest.columns != "actual"]
    y_test = dfProcessedTest["actual"]

    idx: MultiIndex = MultiIndex.from_product([[i for i in range(10, 51, 10)], [j for j in range(0, 30)]],
                                              names=['nneurons', 'test'])

    # --------------- VALIDATION ---------------

    validationErrorDF: DataFrame = DataFrame(index=idx, columns=['mse', 'mae'])
    validationErrorAverageDF = DataFrame(index=[i for i in range(10, 51, 10)], columns=['mse', 'mae'])

    for n_hidden in range(10, 51, 10):

        for test in range(0, 30):
            predicted = mlpPredict(hidden_dim=n_hidden,
                                   x_train=X_train,
                                   y_train=y_train, x_test=X_val,
                                   y_test=y_val)

            validationErrorMSE, validationErrorMAE, _ = metricError(predictedValues=predicted, actualValues=y_val)

            validationErrorDF.loc[(n_hidden, test), 'mse'] = validationErrorMSE
            validationErrorDF.loc[(n_hidden, test), 'mae'] = validationErrorMAE

        validationErrorAverageDF.loc[n_hidden, 'mse'] = validationErrorDF.loc[n_hidden, 'mse'].mean()
        validationErrorAverageDF.loc[n_hidden, 'mae'] = validationErrorDF.loc[n_hidden, 'mae'].mean()

    n_hidden = (getIDXMinMSE(validationErrorAverageDF))

    # --------------- TEST ---------------

    testDF: DataFrame = DataFrame(index=y_test.index)

    for test in range(0, 30):
        predicted = mlpPredict(hidden_dim=n_hidden,
                               x_train=X_train,
                               y_train=y_train, x_test=X_test,
                               y_test=y_test)

        testDF[test] = ((((predicted + 1) / 2) * (max(scalerTest) - min(scalerTest))) + min(scalerTest)).reshape(1,-1)[0] + predictedSeries[-len(y_test):]

    return n_hidden, validationErrorAverageDF.loc[n_hidden], testDF