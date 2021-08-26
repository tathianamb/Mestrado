from pandas import DataFrame
from preProcessing.PreProcessing import preProcessSeries
from posProcessing.PosProcessing import posProcessing
from readfile.ReadFile import fileToSerie
from Processing.Evaluation import wilcoxonTest
from estimators.ARMA import armaPredict
from estimators.ARMA_MLP import armaMlpPredict
from estimators.ARMA_ELM import armaElmPredict
from estimators.MLP import MlpPredict
from estimators.ELM import ElmPredict
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

from tensorflow import autograph
autograph.set_verbosity(0)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys
sys.stdout = open('outputNA\\output.txt', 'a', encoding='utf-8')

from statsmodels.tsa.stattools import pacf
'''
        ####################
        #       MAIN       #
        ####################
'''

serie = fileToSerie('Natal-Jan-Fev.csv')


serie, scalerSTDTest, scalerMeanTest = preProcessSeries(serie.copy())

serie = serie[:-1]

allMSE = DataFrame(index=[i for i in range(0,30)], columns=['ELM', 'ARMA+ELM', 'MLP', 'ARMA+MLP'])

#plotACF_PACF(series=serie, title='Natal')

#------------------------ ARMA ------------------------

print('ARMA:')
mse, mae, testDF_arma, order = armaPredict(serie)
print('\tmse: ', mse, ', mae:', mae)
armaPredicted = posProcessing(testDF_arma, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

#------------------------ ELM ------------------------
print('ELM:')
n_hidden_ELM, validationErrorAverageDF_ELM, testDF_ELM = ElmPredict(serie)

print('\tHidden Neurons: ' + str(n_hidden_ELM))

output_elm = posProcessing(testDF_ELM, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, ElmPredicted, mseTests = wilcoxonTest(outputs=output_elm, actual=serie)

allMSE['ELM'] = mseTests

print('\tP-value _ELM: ', p)

#------------------------ ARMA ELM ------------------------

print('ARMA+ELM')
n_hidden_armaELM, validationErrorAverageDF_armaELM, testDF_armaELM = armaElmPredict(serie, order)

print('\tHidden Neurons : ' + str(n_hidden_armaELM))

output_armaELM = posProcessing(testDF_armaELM, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, armaElmPredicted, mseTests = wilcoxonTest(outputs=output_armaELM, actual=serie)

allMSE['ARMA+ELM'] = mseTests

print('\tP-value _armaELM: ', p)

#------------------------ MLP ------------------------
print('MLP:')

n_hidden_MLP, validationErrorAverageDF_MLP, testDF_MLP = MlpPredict(serie)

print('\tHidden Neurons: ' + str(n_hidden_MLP))

output_MLP = posProcessing(testDF_MLP, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, MlpPredicted, mseTests = wilcoxonTest(outputs=output_MLP, actual=serie)

allMSE['MLP'] = mseTests

print('\tP-value _MLP: ', p)

#------------------------ ARMA MLP ------------------------
print('ARMA+MLP')

n_hidden_armaMLP, validationErrorAverageDF_armaMLP, testDF_armaMLP = armaMlpPredict(serie, order)

print('\tHidden Neurons : ' + str(n_hidden_armaMLP))

output_armaMLP = posProcessing(testDF_armaMLP, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, armaMlpPredicted, mseTests = wilcoxonTest(outputs=output_armaMLP, actual=serie)

allMSE['ARMA+MLP'] = mseTests

print('\tP-value _armaMLP: ', p)

#------------------------------------------------

outputFinal = DataFrame(index=ElmPredicted.index, columns=['ARMA', 'ELM', 'ARMA+ELM', 'MLP', 'ARMA+MLP', 'ACTUAL'])
outputFinal['ARMA'] = armaPredicted[-len(ElmPredicted):].values
outputFinal['ELM'] = ElmPredicted.values
outputFinal['ARMA+ELM'] = armaElmPredicted.values
outputFinal['MLP'] = MlpPredicted.values
outputFinal['ARMA+MLP'] = armaMlpPredicted.values
outputFinal['ACTUAL'] = serie[-len(ElmPredicted):].values
outputFinal.to_csv('outputNA\\saidaFinalNatal.csv', header=True, index=True)
#plotResult(outputFinal)
allMSE.boxplot()
plt.savefig('outputNA\\boxplotNatal.png', format='png')

'''
    Seu namorado gostaria de lembrá-la
    que te ama muito e que sempre que
    precisar pode chamá-lo e contar com ele!!!
    
    TE AMOOOOOOO!!!!
    
    PS: CTRL+ALT+L(impa)  && CTRL+ALT+O(rganiza)
'''