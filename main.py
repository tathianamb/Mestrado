from warnings import filterwarnings

from datetime import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame

from Processing.Evaluation import wilcoxonTest
from estimators.ARMA import armaPredict
from estimators.ARMA_ELM import armaElmPredict
from estimators.ARMA_ESN import armaEsnPredict
from estimators.ARMA_MLP import armaMlpPredict
from estimators.ARMA_RBF import armaRbfPredict
from estimators.ELM import ElmPredict
from estimators.ESN import EsnPredict
from estimators.MLP import MlpPredict
from estimators.RBF import RbfPredict
from posProcessing.PosProcessing import posProcessing
from preProcessing.PreProcessing import preProcessSeries
from readfile.ReadFile import fileToSerie

filterwarnings("ignore")

from tensorflow import autograph

autograph.set_verbosity(0)

import sys

####################
#       MAIN       #
####################

base = "CampoGrande"

sys.stdout = open("IA/data/time-series-output/output" + base + "/log.txt", "a", encoding="utf-8")

print("Started: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

serie = fileToSerie(base + ".csv")

serie, scalerSTDTest, scalerMeanTest = preProcessSeries(serie.copy())

serie = serie[:-1]

allMSE = DataFrame(index=[i for i in range(0, 30)], columns=["ELM", "ARMA+ELM", "MLP", "ARMA+MLP"])

# ------------------------ ARMA ------------------------

print("ARMA:")
mse, mae, testDF_arma, order = armaPredict(serie)
print("\tmse: ", mse, ", mae:", mae)
output_arma = posProcessing(testDF_arma, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

# ------------------------ ELM ------------------------
print("ELM:")
n_hidden_ELM, validationErrorAverageDF_ELM, testDF_ELM = ElmPredict(serie)

print("\tHidden Neurons: " + str(n_hidden_ELM))

output_elm = posProcessing(testDF_ELM, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, ELMPredict, mseTests = wilcoxonTest(outputs=output_elm, actual=serie)

allMSE["ELM"] = mseTests

print("\tP-value _ELM: ", p)

# ------------------------ ARMA ELM ------------------------

print("ARMA+ELM")
n_hidden_armaELM, validationErrorAverageDF_armaELM, testDF_armaELM = armaElmPredict(serie, order)

print("\tHidden Neurons : " + str(n_hidden_armaELM))

output_armaELM = posProcessing(testDF_armaELM, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=output_armaELM, actual=serie)

allMSE["ARMA+ELM"] = mseTests

print("\tP-value _armaELM: ", p)

# ------------------------ MLP ------------------------
print("MLP:")

n_hidden_MLP, validationErrorAverageDF_MLP, testDF_MLP = MlpPredict(serie)

print("\tHidden Neurons: " + str(n_hidden_MLP))

output_MLP = posProcessing(testDF_MLP, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, MLPPredict, mseTests = wilcoxonTest(outputs=output_MLP, actual=serie)

allMSE["MLP"] = mseTests

print("\tP-value _MLP: ", p)

# ------------------------ ARMA MLP ------------------------
print("ARMA+MLP")

n_hidden_armaMLP, validationErrorAverageDF_armaMLP, testDF_armaMLP = armaMlpPredict(serie, order)

print("\tHidden Neurons : " + str(n_hidden_armaMLP))

output_armaMLP = posProcessing(testDF_armaMLP, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=output_armaMLP, actual=serie)

allMSE["ARMA+MLP"] = mseTests

print("\tP-value _armaMLP: ", p)

# ------------------------------------------------

print("ESN:")

n_hidden_ESN, validationErrorAverageDF_ESN, testDF_ESN = EsnPredict(serie)

print("\tHidden Neurons: " + str(n_hidden_ESN))

output_ESN = posProcessing(testDF_ESN, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, EsnPredicted, mseTests = wilcoxonTest(outputs=output_ESN, actual=serie)

allMSE["ESN"] = mseTests

print("\tP-value _ESN: ", p)

# ------------------------------------------------


print("ARMA+ESN")

n_hidden_armaESN, validationErrorAverageDF_armaESN, testDF_armaESN = armaEsnPredict(serie, order)

print("\tHidden Neurons : " + str(n_hidden_armaESN))

output_armaESN = posProcessing(testDF_armaESN, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, armaEsnPredicted, mseTests = wilcoxonTest(outputs=output_armaESN, actual=serie)

allMSE["ARMA+ESN"] = mseTests

print("\tP-value _armaESN: ", p)

# ------------------------------------------------

print("RBF:")

n_hidden_RBF, validationErrorAverageDF_RBF, testDF_RBF = RbfPredict(serie)

print("\tHidden Neurons: " + str(n_hidden_RBF))

output_RBF = posProcessing(testDF_RBF, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, RbfPredicted, mseTests = wilcoxonTest(outputs=output_RBF, actual=serie)

allMSE["RBF"] = mseTests

print("\tP-value _RBF: ", p)

# ------------------------------------------------


print("ARMA+RBF")

n_hidden_armaRBF, validationErrorAverageDF_armaRBF, testDF_armaRBF = armaRbfPredict(serie, order)

print("\tHidden Neurons : " + str(n_hidden_armaRBF))

output_armaRBF = posProcessing(testDF_armaRBF, monthlySTD=scalerSTDTest, monthlyMean=scalerMeanTest)

w, p, armaRbfPredicted, mseTests = wilcoxonTest(outputs=output_armaRBF, actual=serie)

allMSE["ARMA+RBF"] = mseTests

print("\tP-value _armaRBF: ", p)

outputFinal = DataFrame(index=hybridSystemPredict.index,
                        columns=["ARMA", "ELM", "ARMA+ELM", "MLP", "ARMA+MLP", "ESN", "ARMA+ESN", "RBF", "ARMA+RBF",
                                 "ACTUAL"])
outputFinal["ARMA"] = output_arma
outputFinal["ELM"] = output_elm
outputFinal["ARMA+ELM"] = output_armaELM
outputFinal["MLP"] = output_MLP
outputFinal["ARMA+MLP"] = output_armaMLP
outputFinal["ESN"] = output_ESN
outputFinal["ARMA+ESN"] = output_armaESN
outputFinal["RBF"] = output_RBF
outputFinal["ARMA+RBF"] = output_armaRBF
outputFinal["ACTUAL"] = serie[-len(output_armaELM):]
outputFinal.to_csv("IA/data/time-series-output/output" + base + "/saida_teste_mse.csv", header=True, index=True)
allMSE.boxplot()
plt.savefig("IA/data/time-series-output/output" + base + "/boxplot.png", format="png")

print("Ended: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

'''
    Seu namorado gostaria de lembrá-la
    que te ama muito e que sempre que
    precisar pode chamá-lo e contar com ele!!!

    TE AMOOOOOOO!!!!

    PS: CTRL+ALT+L(impa)  && CTRL+ALT+O(rganiza)
'''

'''
Esse aqui é um exemplo!!!

'''
