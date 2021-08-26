from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import amin
from sklearn.model_selection import train_test_split
from preProcessing.FeatureSelection import featureSelectionWrapper, featureSelectionCorr

def prepareDataToANN(serie, estimator, window = 30):

    data = pd.DataFrame(index=serie.index)

    for i in range(1, window + 1):
        data['lag_' + str(i)] = serie.shift(i)

    data = data[window:]

    data['actual'] = serie

    train, valTest = train_test_split(data, train_size=0.6, shuffle=False)
    validation, test = train_test_split(valTest, test_size=0.5, shuffle=False)

    minMaxValue = (test.to_numpy().min(), test.to_numpy().max())

    scalerTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerVal = MinMaxScaler(feature_range=(-1, 1))
    scalerTest = MinMaxScaler(feature_range=(-1, 1))

    processedTrain = scalerTrain.fit_transform(train)
    processedVal = scalerVal.fit_transform(validation)
    processedTest = scalerTest.fit_transform(test)

    dfTrainProcessed = pd.DataFrame(data=processedTrain, index=train.index, columns=data.columns)
    dfValProcessed = pd.DataFrame(data=processedVal, index=validation.index, columns=data.columns)
    dfTestProcessed = pd.DataFrame(data=processedTest, index=test.index, columns=data.columns)

    X_train = dfTrainProcessed.iloc[:, :window]
    y_train = dfTrainProcessed['actual']

    X_val = dfValProcessed.iloc[:, :window]
    y_val = dfValProcessed['actual']

    X_test = dfTestProcessed.iloc[:, :window]
    y_test = dfTestProcessed['actual']

    # X_train = featureSelectionWrapper(X=X_train, y=y_train, estimator=estimator)
    X_train = featureSelectionCorr(X=X_train, y=y_train)

    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    return X_train, y_train, X_val, y_val, X_test, y_test, minMaxValue

def getIDXMinMSE(df : pd.DataFrame):

    indexes = list(set(df.index))
    lowerValue = amin(df['mse'].values)

    for i in indexes:
        if df.loc[i, 'mse'] == lowerValue:
            return i

    print("value not found")

