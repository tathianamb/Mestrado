from pandas import read_csv as csv, to_datetime as dateTime, Series


def __convertTime(minutes):
    minutes = minutes % (24 * 60)
    hour = minutes // 60
    minutes %= 60
    return "%02d:%02d" % (hour, minutes)


def __convertDate(days):
    if days <= 31:
        date = "%02d/01" % (days)
    else:
        days = days - 31
        date = "%02d/02" % (days)
    return date


def fileToSerie(name: str):
    dataframe = csv('data/time-series-input/' + name, names=['year', 'day', 'min', 'ws_10m'], delimiter=';', keep_default_na=False)
    dataframe['min'] = dataframe['min'].apply(__convertTime)
    dataframe['day'] = dataframe['day'].apply(__convertDate)

    dataframe['date'] = dataframe['day'] + '/' + dataframe['year'].astype(str) + '-' + dataframe['min']
    dataframe['date'] = dateTime(dataframe['date'], format='%d/%m/%Y-%H:%M')
    return Series(data=list(dataframe['ws_10m'].astype(float)), index=list(dataframe['date']), name='Actual').asfreq('min')