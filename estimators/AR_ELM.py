from estimators.AR import arPredict
from estimators.baseELM import elmPredict
from pandas import DataFrame, MultiIndex, concat
from Processing.Evaluation import metricError
from Processing.Process import getIDXMinMSE
from preProcessing.PreProcessing import prepareDataToANN

def arElmPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    errorSeries, predictedSeries = arPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM,
                                             isHybrid=True, order=order)
    dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest = prepareDataToANN(errorSeries)

    # --------------- LINEAR MODELS PREDICT ENDING  ---------------
    # --------------- ANN PREDICT BEGINNING ---------------

    X_train = dfProcessedTrain.loc[:, dfProcessedTrain.columns != "actual"]
    y_train = dfProcessedTrain["actual"]
    X_val = dfProcessedVal.loc[:, dfProcessedVal.columns != "actual"]
    y_val = dfProcessedVal["actual"]
    X_test = dfProcessedTest.loc[:, dfProcessedTest.columns != "actual"]
    y_test = dfProcessedTest["actual"]

    idx: MultiIndex = MultiIndex.from_product([[i for i in range(70, 150, 10)], [j for j in range(0, 30)]],
                                              names=['nneurons', 'test'])

    # --------------- VALIDATION ---------------

    validationErrorDF: DataFrame = DataFrame(index=idx, columns=['mse', 'mae'])
    validationErrorAverageDF = DataFrame(index=[i for i in range(70, 150, 10)], columns=['mse', 'mae'])

    for n_hidden in range(70, 150, 10):

        for test in range(0, 30):
            predicted = elmPredict(hidden_dim=n_hidden,
                                   x_train=X_train,
                                   y_train=y_train, x_test=X_test,
                                   y_test=y_test)

            predicted = (((predicted + 1) / 2) * (max(minMaxVal) - min(minMaxVal))) + min(minMaxVal)

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
        testDF[test] = ((((predicted + 1) / 2) * (max(minMaxTest) - min(minMaxTest))) + min(minMaxTest)).values.reshape(1, -1)[0] + predictedSeries[-len(y_test):].values
    return n_hidden, validationErrorAverageDF.loc[n_hidden], testDF
