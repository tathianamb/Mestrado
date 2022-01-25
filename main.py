from tensorflow import autograph

from warnings import filterwarnings

from statsmodels.graphics.tsaplots import plot_pacf

from datetime import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame, concat

from Processing.Evaluation import wilcoxonTest
from estimators.AR import arPredict
from estimators.AR_ELM import arElmPredict
from estimators.AR_MLP import arMlpPredict
from estimators.AR_RBF import arRbfPredict
from estimators.AR_ESN import arEsnPredict
from estimators.ARMA import armaPredict
from estimators.ARMA_ELM import armaElmPredict
from estimators.ARMA_ESN import armaEsnPredict
from estimators.ARMA_MLP import armaMlpPredict
from estimators.ARMA_RBF import armaRbfPredict
from estimators.ELM import ElmPredict
from estimators.ESN import EsnPredict
from estimators.MLP import MlpPredict
from estimators.RBF import RbfPredict
from preProcessing.PreProcessing import prepareDataToANN, prepareDataToLinearModels
from readfile.ReadFile import fileToSerie

filterwarnings("ignore")

autograph.set_verbosity(0)

import sys

base = "Brasilia"
base_dir = "./data/time-series-output/output" + base


sys.stdout = open(base_dir + "/log.txt", "a", encoding="utf-8")

def baseData():
    serie = fileToSerie(base + ".csv")
    plot_pacf(serie)
    plt.savefig(base_dir + "/pacf.png", format="png")
    return serie


def apply_preProcess(serie, toLinearModels = False):
    if toLinearModels:
        return prepareDataToLinearModels(serie)
    else:
        return prepareDataToANN(serie)


def AR(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM):
    print("AR:")
    mse, mae, testDF_ar, order = arPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)
    print("\tmse: ", mse, ", mae:", mae)
    return testDF_ar, order


def ARMA(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM):
    print("ARMA:")
    mse, mae, testDF_arma, order = armaPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)
    print("\tmse: ", mse, ", mae:", mae)
    return testDF_arma, order


def ELM(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("ELM:")
    n_hidden_ELM, validationErrorAverageDF_ELM, testDF_ELM = ElmPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_ELM))
    w, p, ELMPredict, mseTests = wilcoxonTest(outputs=testDF_ELM, actual=dfProcessedTest["actual"])
    print("\tP-value _ELM: ", p)
    return testDF_ELM, ELMPredict, mseTests


def ARandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("AR+ELM")
    n_hidden_arELM, validationErrorAverageDF_arELM, testDF_arELM = arElmPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_arELM))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_arELM, actual=dfProcessedTest_LM[-len(testDF_arELM):])
    print("\tP-value _arELM: ", p)
    return testDF_arELM, hybridSystemPredict, mseTests


def ARMAandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+ELM")
    n_hidden_armaELM, validationErrorAverageDF_armaELM, testDF_armaELM = armaElmPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaELM))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_armaELM, actual=dfProcessedTest_LM[-len(testDF_armaELM):])
    print("\tP-value _armaELM: ", p)
    return testDF_armaELM, hybridSystemPredict, mseTests


def MLP(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("MLP:")
    n_hidden_MLP, validationErrorAverageDF_MLP, testDF_MLP = MlpPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_MLP))
    w, p, MLPPredict, mseTests = wilcoxonTest(outputs=testDF_MLP, actual=dfProcessedTest["actual"])
    print("\tP-value _MLP: ", p)
    return testDF_MLP, MLPPredict, mseTests


def ARandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("AR+MLP")
    n_hidden_arMLP, validationErrorAverageDF_arMLP, testDF_arMLP = arMlpPredict(dfProcessedTrain_LM, dfProcessedTest_LM,
                                                                                minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_arMLP))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_arMLP,
                                                       actual=dfProcessedTest_LM[-len(testDF_arMLP):])
    print("\tP-value _arMLP: ", p)
    return testDF_arMLP, hybridSystemPredict, mseTests


def ARMAandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+MLP")
    n_hidden_armaMLP, validationErrorAverageDF_armaMLP, testDF_armaMLP = armaMlpPredict(dfProcessedTrain_LM,
                                                                                        dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaMLP))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_armaMLP,
                                                       actual=dfProcessedTest_LM[-len(testDF_armaMLP):])
    print("\tP-value _armaMLP: ", p)
    return testDF_armaMLP, hybridSystemPredict, mseTests


def ESN(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("ESN:")
    n_hidden_ESN, validationErrorAverageDF_ESN, testDF_ESN = EsnPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_ESN))
    w, p, EsnPredicted, mseTests = wilcoxonTest(outputs=testDF_ESN, actual=dfProcessedTest["actual"])
    print("\tP-value _ESN: ", p)
    return testDF_ESN, EsnPredicted, mseTests


def ARandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("AR+ESN")
    n_hidden_armaESN, validationErrorAverageDF_armaESN, testDF_armaESN = arEsnPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaESN))
    w, p, armaEsnPredicted, mseTests = wilcoxonTest(outputs=testDF_armaESN, actual=dfProcessedTest_LM[-len(testDF_armaESN):])
    print("\tP-value _arESN: ", p)
    return testDF_armaESN, armaEsnPredicted, mseTests


def ARMAandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+ESN")
    n_hidden_armaESN, validationErrorAverageDF_armaESN, testDF_armaESN = armaEsnPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaESN))
    w, p, armaEsnPredicted, mseTests = wilcoxonTest(outputs=testDF_armaESN, actual=dfProcessedTest_LM[-len(testDF_armaESN):])
    print("\tP-value _armaESN: ", p)
    return testDF_armaESN, armaEsnPredicted, mseTests


def RBF(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("RBF:")
    n_hidden_RBF, validationErrorAverageDF_RBF, testDF_RBF = RbfPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_RBF))
    w, p, RbfPredicted, mseTests = wilcoxonTest(outputs=testDF_RBF, actual=dfProcessedTest["actual"])
    print("\tP-value _RBF: ", p)
    return testDF_RBF, RbfPredicted, mseTests


def ARandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("AR+RBF")
    n_hidden_arRBF, validationErrorAverageDF_arRBF, testDF_arRBF = arRbfPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_arRBF))
    w, p, arRbfPredicted, mseTests = wilcoxonTest(outputs=testDF_arRBF, actual=dfProcessedTest_LM[-len(testDF_arRBF):])
    print("\tP-value _arRBF: ", p)
    return testDF_arRBF, arRbfPredicted, mseTests


def ARMAandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+RBF")
    n_hidden_armaRBF, validationErrorAverageDF_armaRBF, testDF_armaRBF = armaRbfPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaRBF))
    w, p, armaRbfPredicted, mseTests = wilcoxonTest(outputs=testDF_armaRBF, actual=dfProcessedTest_LM[-len(testDF_armaRBF):])
    print("\tP-value _armaRBF: ", p)
    return testDF_armaRBF, armaRbfPredicted, mseTests


####################
#       MAIN       #
####################
def main():
    print("Started: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    serie = baseData()
    dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest = apply_preProcess(serie[-300:], toLinearModels=False)
    dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM = apply_preProcess(serie[-300:], toLinearModels=True)
    allMSE = DataFrame(index=[i for i in range(0, 30)])
    outputFinal = DataFrame(index=dfProcessedTest.index)
    outputFinal["ACTUAL"] = serie
    # ------------------------ AR ------------------------

    output_ar, order_ar = AR(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)
    outputFinal["AR"] = output_ar
    del output_ar

    # ------------------------ ARMA ------------------------

    output_arma, order_arma = ARMA(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)
    outputFinal["ARMA"] = output_arma
    del output_arma

    # ------------------------ ELM ------------------------

    tests, output_elm, mseTests = ELM(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    allMSE["ELM"] = mseTests
    outputFinal["ELM"] = output_elm
    del output_elm

    # ------------------------ AR ELM ------------------------

    tests, output_arELM, mseTests = ARandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_ar)
    allMSE["AR+ELM"] = mseTests
    outputFinal["AR+ELM"] = output_arELM
    del output_arELM

    # ------------------------ ARMA ELM ------------------------

    tests, output_armaELM, mseTests = ARMAandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_arma)
    allMSE["ARMA+ELM"] = mseTests
    outputFinal["ARMA+ELM"] = output_armaELM
    del output_armaELM

    # ------------------------ MLP ------------------------

    tests, output_MLP, mseTests = MLP(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    allMSE["MLP"] = mseTests
    outputFinal["MLP"] = output_MLP
    del output_MLP

    # ------------------------ AR MLP ------------------------

    tests, output_arMLP, mseTests,  = ARandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_ar)
    allMSE["AR+MLP"] = mseTests
    outputFinal["AR+MLP"] = output_arMLP
    del output_arMLP

    # ------------------------ ARMA MLP ------------------------

    tests, output_armaMLP, mseTests = ARMAandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_arma)
    allMSE["ARMA+MLP"] = mseTests
    outputFinal["ARMA+MLP"] = output_armaMLP
    del output_armaMLP

    # ----------------------- ESN ---------------------

    tests, output_ESN, mseTests = ESN(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    allMSE["ESN"] = mseTests
    outputFinal["ESN"] = output_ESN
    del output_ESN

    # -------------------- AR ESN ----------------------

    tests, output_arESN, mseTests = ARandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_ar)
    allMSE["AR+ESN"] = mseTests
    outputFinal["AR+ESN"] = output_arESN
    del output_arESN

    # -------------------- ARMA ESN ----------------------

    tests, output_armaESN, mseTests = ARMAandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_arma)
    allMSE["ARMA+ESN"] = mseTests
    outputFinal["ARMA+ESN"] = output_armaESN
    del output_armaESN

    # -------------------- RBF ----------------------

    tests, output_RBF, mseTests = RBF(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    allMSE["RBF"] = mseTests
    outputFinal["RBF"] = output_RBF
    del output_RBF

    # --------------------- AR RBF ------------------------

    tests, output_arRBF, mseTests = ARandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_ar)
    allMSE["AR+RBF"] = mseTests
    outputFinal["AR+RBF"] = output_arRBF
    del output_arRBF

    # --------------------- ARMA RBF ------------------------

    tests, output_armaRBF, mseTests = ARMAandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order_arma)
    allMSE["ARMA+RBF"] = mseTests
    outputFinal["ARMA+RBF"] = output_armaRBF
    del output_armaRBF

    # ------------------------------------------------

    outputFinal.to_csv(base_dir + "/saida_teste_mse.csv", header=True, index=True)
    allMSE.to_csv(base_dir + "/all_mse.csv", header=True, index=True)
    plt.savefig(base_dir + "/boxplot.png", format="png")

    print("Ended: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

main()

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
