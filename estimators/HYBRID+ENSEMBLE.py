from estimators.ARMA import armaPredict
from estimators.ELM import elmPredict
from estimators.baseMLP import mlpPredict
from preProcessing.FeatureSelection import featureSelection
from pandas import Series, DataFrame, MultiIndex
from Processing.Evaluation import metricError
from sklearn.preprocessing import MinMaxScaler
from numpy import amin


def getIDXMinMSE(df : DataFrame):

    indexes = list(set(df.index))
    lowerValue = amin(df['mse'].values)

    for i in indexes:
        if df.loc[i, 'mse'] == lowerValue:
            return i

    print("value not found")


def prepareDataToANN(mask, trainSet, testSet, window = 10):

    X_train, X_test, y_train, y_test = DataFrame(), DataFrame(), DataFrame(), DataFrame()

    for i in range(1, window + 1):
        X_train['lag_' + str(i)] = trainSet.shift(i)
        X_test['lag_' + str(i)] = testSet.shift(i)

    X_train = X_train[window:]
    y_train = trainSet[window:]

    X_test = X_test[window:]
    y_test = testSet[window:]

    scalerXTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerYTrain = MinMaxScaler(feature_range=(-1, 1))

    scalerXTest = MinMaxScaler(feature_range=(-1, 1))
    scalerYTest = MinMaxScaler(feature_range=(-1, 1))

    processedXTrain = scalerXTrain.fit_transform(X_train)
    processedYTrain = scalerYTrain.fit_transform(y_train.values.reshape(-1, 1))

    processedXTest = scalerXTest.fit_transform(X_test.values.reshape(-1, 1))
    processedYTest = scalerYTest.fit_transform(y_test.values.reshape(-1, 1))

    Xtrain = DataFrame(data=processedXTrain.reshape(-1, window), index=trainSet[window:].index)
    Ytrain = DataFrame(data=processedYTrain.reshape(1, -1)[0], index=trainSet[window:].index)

    Xtest = DataFrame(data=processedXTest.reshape(-1, window), index=testSet[window:].index)
    Ytest = DataFrame(data=processedYTest.reshape(1, -1)[0], index=testSet[window:].index)

    return Xtrain.loc[:, mask], Ytrain, Xtest.loc[:, mask], Ytest, scalerYTest

def armaANNPredict(trainS: Series, testS: Series, estimator):

    errorTrainARMA, errorTestARMA, y_predictedTestARMA = armaPredict(trainS, testS, isHybrid=True)

    #--------------- LINEAR MODELS PREDICT ENDING  ---------------
    #--------------- ANN PREDICT BEGINNING ---------------

    X_train, y_train, X_test, y_test, scalerTest = prepareDataToANN(mask=featureSelection(train=errorTrainARMA, typeEstimator=estimator), trainSet=errorTrainARMA, testSet=errorTestARMA)

    idx: MultiIndex = MultiIndex.from_product([[i for i in range(5, 30, 5)], [j for j in range(0, 30)]], names=['nneurons', 'test'])

    #--------------- VALIDATION ---------------

    columnsLen = int(len(y_test) / 2)

    validationErrorDF: DataFrame = DataFrame(index=idx, columns=['mse', 'mae'])
    validationErrorAverageDF = DataFrame(index=[i for i in range(5, 30, 2) ], columns=['mse', 'mae'])

    for n_hidden in range(5, 30, 2):

        for test in range(0,30):

            if estimator == 'ELM':
                predicted = elmPredict(hidden_dim=n_hidden,
                                       x_train=X_train,
                                       y_train=y_train, x_test=X_test[:columnsLen],
                                       y_test=y_test[:columnsLen])
            elif estimator == 'MLP':
                predicted = mlpPredict(hidden_dim=n_hidden,
                                       x_train=X_train,
                                       y_train=y_train, x_test=X_test[:columnsLen],
                                       y_test=y_test[:columnsLen])
            else:
                print("NO ESTIMATOR WAS SELECTED!!")
                return 0, DataFrame(), DataFrame()


            validationErrorMSE, validationErrorMAE, _ = metricError(predictedValues=scalerTest.inverse_transform(predicted), actualValues=y_test[:columnsLen])

            validationErrorDF.loc[(n_hidden, test), 'mse'] = validationErrorMSE
            validationErrorDF.loc[(n_hidden, test), 'mae'] = validationErrorMAE

        validationErrorAverageDF.loc[n_hidden, 'mse'] = validationErrorDF.loc[n_hidden, 'mse'].mean()
        validationErrorAverageDF.loc[n_hidden, 'mae'] = validationErrorDF.loc[n_hidden, 'mae'].mean()

    n_hidden = (getIDXMinMSE(validationErrorAverageDF))

    # --------------- TEST ---------------

    testDF: DataFrame = DataFrame(index = testS[-columnsLen:].index, columns=[i for i in range(0, 30)])

    for test in range(0, 30):

        if estimator == 'ELM':
            predicted = elmPredict(hidden_dim=n_hidden,
                                                                   x_train=X_train,
                                                                   y_train=y_train, x_test=X_test[-columnsLen:],
                                                                   y_test=y_test[-columnsLen:])
        elif estimator == 'MLP':
            predicted = mlpPredict(hidden_dim=n_hidden,
                                   x_train=X_train,
                                   y_train=y_train, x_test=X_test[-columnsLen:],
                                   y_test=y_test[-columnsLen:])

        predicted = scalerTest.inverse_transform(predicted)

        list = predicted.reshape(1,-1)[0] + y_predictedTestARMA[-columnsLen:]

        testDF[test] = list

    return n_hidden, validationErrorAverageDF.loc[n_hidden], testDF
