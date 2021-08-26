from numpy import concatenate
from pandas import Series, DataFrame
from sklearn.preprocessing import MinMaxScaler

def posProcessing(output, monthlySTD, monthlyMean):

    if type(output) == Series:
        if monthlySTD is not None:

            for date in monthlySTD.index.date:

                filterSeries = output.index.date == date
                filterMean = monthlyMean.index.date == date
                filterSTD = monthlySTD.index.date == date

                output[filterSeries] = (output[filterSeries] * monthlySTD[filterSTD].values[0]) + monthlyMean[filterMean].values[0]

    elif type(output) == DataFrame:

        for column in output:
            output[column] = posProcessing(output[column], monthlySTD=monthlySTD, monthlyMean=monthlyMean)

    return output