from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import amin
from sklearn.model_selection import train_test_split
from preProcessing.FeatureSelection import featureSelectionWrapper, featureSelectionCorr

def prepareDataToANN(serie, estimator, window = 15):

    data = pd.DataFrame(index=serie.index)

    for i in range(1, window + 1):
        data['lag_' + str(i)] = serie.shift(i)

    data = data[window:]

    data['actual'] = serie

    train, valTest = train_test_split(data, train_size=0.6, shuffle=False)
    validation, test = train_test_split(valTest, test_size=0.5, shuffle=False)

    #Xtrain = featureSelectionWrapper(X=train.iloc[:, :-1], y=train['actual'], estimator=estimator)
    Xtrain = featureSelectionCorr(X=train.iloc[:, :-1], y=train['actual'])
    columns = Xtrain.columns
    train = pd.concat([Xtrain, train['actual']], axis=1)
    validation = pd.concat([validation[columns], validation['actual']], axis=1)
    test = pd.concat([test[columns], test['actual']], axis=1)

    minMaxValue = (test.to_numpy().min(), test.to_numpy().max())

    scalerTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerVal = MinMaxScaler(feature_range=(-1, 1))
    scalerTest = MinMaxScaler(feature_range=(-1, 1))

    processedTrain = scalerTrain.fit_transform(train)
    processedVal = scalerVal.fit_transform(validation)
    processedTest = scalerTest.fit_transform(test)

    dfTrainProcessed = pd.DataFrame(data=processedTrain, index=train.index, columns=train.columns)
    dfValProcessed = pd.DataFrame(data=processedVal, index=validation.index, columns=validation.columns)
    dfTestProcessed = pd.DataFrame(data=processedTest, index=test.index, columns=test.columns)

    X_train = dfTrainProcessed[columns]
    y_train = dfTrainProcessed['actual']

    X_val = dfValProcessed[columns]
    y_val = dfValProcessed['actual']

    X_test = dfTestProcessed[columns]
    y_test = dfTestProcessed['actual']

    return X_train, y_train, X_val, y_val, X_test, y_test, minMaxValue

def getIDXMinMSE(df : pd.DataFrame):

    indexes = list(set(df.index))
    lowerValue = amin(df['mse'].values)

    for i in indexes:
        if df.loc[i, 'mse'] == lowerValue:
            return i

    print("value not found")

