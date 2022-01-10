from Processing.Evaluation import metricError
from estimators.baseRBF import rbfPredict
from Processing.Process import getIDXMinMSE
from pandas import DataFrame, MultiIndex


def RbfPredict(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):

    X_train = dfProcessedTrain.loc[:, dfProcessedTrain.columns != "actual"]
    y_train = dfProcessedTrain["actual"]
    X_val = dfProcessedVal.loc[:, dfProcessedVal.columns != "actual"]
    y_val = dfProcessedVal["actual"]
    X_test = dfProcessedTest.loc[:, dfProcessedTest.columns != "actual"]
    y_test = dfProcessedTest["actual"]

    idx: MultiIndex = MultiIndex.from_product([[i for i in range(10, 51, 10)], [j for j in range(0, 30)]], names=['nneurons', 'test'])

    # ----------------- VALIDATION -----------------

    validationErrorDF: DataFrame = DataFrame(index=idx, columns=['mse', 'mae'])
    validationErrorAverageDF = DataFrame(index=[i for i in range(10, 51, 10)], columns=['mse', 'mae'])

    for n_hidden in range(10, 51, 10):

        for test in range(0, 30):
            predicted = rbfPredict(hidden_dim=n_hidden,
                                   x_train=X_train,
                                   y_train=y_train,
                                   x_test=X_val,
                                   y_test=y_val)

            validationErrorMSE, validationErrorMAE, _ = metricError(predicted, actualValues=y_val)

            validationErrorDF.loc[(n_hidden, test), 'mse'] = validationErrorMSE
            validationErrorDF.loc[(n_hidden, test), 'mae'] = validationErrorMAE

        validationErrorAverageDF.loc[n_hidden, 'mse'] = validationErrorDF.loc[n_hidden, 'mse'].mean()
        validationErrorAverageDF.loc[n_hidden, 'mae'] = validationErrorDF.loc[n_hidden, 'mae'].mean()

    n_hidden = (getIDXMinMSE(validationErrorAverageDF))

    # ----------------- TEST -----------------

    testDF: DataFrame = DataFrame(index = y_test.index, columns=[i for i in range(0, 30)])

    for test in range(0, 30):
        predicted = rbfPredict(hidden_dim=n_hidden,
                               x_train=X_train,
                               y_train=y_train,
                               x_test=X_test,
                               y_test=y_test)

        testDF[test] = predicted
        testDF[test] = (((testDF[test] + 1) / 2) * (max(scalerTest) - min(scalerTest))) + min(scalerTest)

    return n_hidden, validationErrorAverageDF.loc[n_hidden], testDF