from typing import Tuple, Union

import numpy as np
from pandas import Series, DataFrame, Grouper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf

'''
    1) Para todo o código é interessante lembrarmos
    do operar "del" do python, pode servir muito
    bem a você, principalmente quando não precisa
    usar uma variável mais, que ocupe bastante espaço
    em memória.
    
    2) Deixo também a dica de colocar o tipo da variável
    que está mexendo, isso ajudará a IDE a te mostrar as
    funções que o tipo específico tem.
'''


def _toStationary(series: Series) -> Tuple[Series, Union[Series, DataFrame], Union[Series, DataFrame]]:
    # talvez o retorno aqui seja "SeriesGroupBy" e não "Union", verificar...
    monthlyMean: Union[Series, DataFrame] = series.groupby(Grouper(freq="D")).mean()
    monthlySTD: Union[Series, DataFrame] = series.groupby(Grouper(freq="D")).std()

    # Aqui diz que não existe o atributo DATE pro tipo INDEX, vale a pena verificar
    for date in monthlySTD.index.date:
        filterSeries: bool = series.index.date == date
        filterMean: bool = monthlyMean.index.date == date
        filterSTD: bool = monthlySTD.index.date == date

        series[filterSeries] = (series[filterSeries] - monthlyMean[filterMean].values[0]) \
                               / monthlySTD[filterSTD].values[0]

    return series, monthlySTD, monthlyMean


def _isStationary(series: Series) -> bool:
    results = adfuller(series)
    return results[1] <= 0.05


def verify_toStationary(serie: Series):
    scalerSTD, scalerMean = None, None

    if not _isStationary(serie):
        print("Aplicando Z-Score")
        serie, scalerSTD, scalerMean = _toStationary(serie)

    return serie, scalerSTD, scalerMean


# Alterar o parâmetro para ter o tipo específico (serie: Series)
# Colocar o tipo do retorno é interessante também
def featureSelectionCorr(serie):
    data = DataFrame(index=serie.index)

    for i in range(1, 10 + 1):
        data["lag_" + str(i)] = serie.shift(i)

    # Não entendi muito bem o que a linha
    # abaixo tem como objetivo...
    data = data[10:]

    data["actual"] = serie

    # talvez trocar a linha abaixo para "corr: list"
    corr = {}
    for column in data.loc[:, data.columns != "actual"]:
        corr[column] = np.abs(data[column].corr(data["actual"], method="spearman"))

    sortedCorr = sorted(corr.items(), key=lambda x: x[1], reverse=True)

    filter = list(dict(sortedCorr).keys())[:5]
    print("\t" + str(filter))

    return filter


# Alterar o parâmetro para ter o tipo específico (serie: Series)
def featureSelectionPACF(serie):
    pacf_values, confint = pacf(serie, nlags=30, alpha=0.05)
    lower_bound = confint[1, 0] - pacf_values[1]
    upper_bound = confint[1, 1] - pacf_values[1]

    significant_pacf_values = {}
    for i in range(1, len(pacf_values)):
        if pacf_values[i] >= upper_bound or pacf_values[i] <= lower_bound:
            significant_pacf_values[i] = pacf_values[i]

    sorted_significant_pacf_values = sorted(significant_pacf_values.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_significant_pacf_values) >= 10:
        sorted_significant_pacf_values = sorted_significant_pacf_values[:10]

    significant_lags = [a_tuple[0] for a_tuple in sorted_significant_pacf_values]
    significant_lags.sort()
    print("Significant lags: ")
    print("\t" + str(significant_lags))

    return significant_lags


# Alterar o parâmetro para ter o tipo específico (serie: Series, ...)
def generate_X(serie, lags):
    data = DataFrame()
    data["actual"] = serie
    for lag in lags:
        data["lag_" + str(lag)] = serie.shift(lag)
    return data[lags[-1]:]


# Alterar o parâmetro para ter o tipo específico (serie: Series)
# Talvez renomear para prepareToANN seja suficiente
def prepareDataToANN(serie):
    train, valTest = train_test_split(serie, train_size=0.6, shuffle=False)
    validation, test = train_test_split(valTest, test_size=0.5, shuffle=False)

    lags = featureSelectionPACF(train)

    train = generate_X(train, lags)
    validation = generate_X(validation, lags)
    test = generate_X(test, lags)

    # Tentar tirar essas conversões em memória (to_numpy()) verificando
    # se o tipo retornado já não tem uma função pra isso. Lendo a documentação
    # acredito que consegui pegar uma dica...
    minMaxVal = (validation.values.min(), validation.values.max())
    minMaxTest = (test.values.min(), test.values.max())

    # Não daria pra usar somente um "MinMaxScaler(feature_range=(-1, 1))"???
    scalerTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerVal = MinMaxScaler(feature_range=(-1, 1))
    scalerTest = MinMaxScaler(feature_range=(-1, 1))

    columns = train.columns != "actual"

    processedTrain = scalerTrain.fit_transform(train)
    processedVal = scalerVal.fit_transform(validation.loc[:, columns])
    processedTest = scalerTest.fit_transform(test.loc[:, columns])

    dfProcessedTrain = DataFrame(data=processedTrain, index=train.index, columns=train.columns)
    dfProcessedVal = DataFrame(data=processedVal, index=validation.index, columns=validation.loc[:, columns].columns)
    dfProcessedTest = DataFrame(data=processedTest, index=test.index, columns=test.loc[:, columns].columns)

    dfProcessedVal["actual"] = validation["actual"]
    dfProcessedTest["actual"] = test["actual"]

    return dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest


# Alterar o parâmetro para ter o tipo específico (serie: Series)
# Talvez trocar o nome da função, tentando reduzí-la (ex: prepareToLM)
#     - A omissão de DATA pois você já estará passando por parâmetro
#     - A alteração de LinearModels por LM por convenção do código
def prepareDataToLinearModels(serie: Series):
    train, test = train_test_split(serie, train_size=0.6, shuffle=False)

    minMaxTest = (test.values.min(), test.values.max())

    scalerTrain = MinMaxScaler(feature_range=(-1, 1))
    scalerTest = MinMaxScaler(feature_range=(-1, 1))

    processedTrain = scalerTrain.fit_transform(train.values.reshape(-1, 1))
    processedTest = scalerTest.fit_transform(test.values.reshape(-1, 1))

    dfProcessedTrain = Series(data=processedTrain.reshape(1,-1)[0], index=train.index)
    dfProcessedTest = Series(data=processedTest.reshape(1,-1)[0], index=test.index)

    return dfProcessedTrain, dfProcessedTest, minMaxTest
