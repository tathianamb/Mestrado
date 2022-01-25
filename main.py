from tensorflow import autograph

from warnings import filterwarnings

from statsmodels.graphics.tsaplots import plot_pacf

from datetime import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame, concat

from Processing.Evaluation import wilcoxonTest
from estimators.AR import arPredict
from estimators.AR_ELM import arElmPredict
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


# sys.stdout = open(base_dir + "/log.txt", "a", encoding="utf-8")

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

def ARandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("AR+ELM")
    n_hidden_arELM, validationErrorAverageDF_arELM, testDF_arELM = arElmPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_arELM))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_arELM, actual=dfProcessedTest_LM[-len(testDF_arELM):])
    print("\tP-value _arELM: ", p)
    return testDF_arELM, mseTests


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
    return testDF_ELM, mseTests


def ARMAandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+ELM")
    n_hidden_armaELM, validationErrorAverageDF_armaELM, testDF_armaELM = armaElmPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaELM))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_armaELM, actual=dfProcessedTest_LM[-len(testDF_armaELM):])
    print("\tP-value _armaELM: ", p)
    return testDF_armaELM, mseTests


def MLP(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("MLP:")
    n_hidden_MLP, validationErrorAverageDF_MLP, testDF_MLP = MlpPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_MLP))
    w, p, MLPPredict, mseTests = wilcoxonTest(outputs=testDF_MLP, actual=dfProcessedTest["actual"])
    print("\tP-value _MLP: ", p)
    return testDF_MLP, mseTests


def ARMAandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+MLP")
    n_hidden_armaMLP, validationErrorAverageDF_armaMLP, testDF_armaMLP = armaMlpPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaMLP))
    w, p, hybridSystemPredict, mseTests = wilcoxonTest(outputs=testDF_armaMLP, actual=dfProcessedTest_LM[-len(testDF_armaMLP):])
    print("\tP-value _armaMLP: ", p)
    return testDF_armaMLP, mseTests, hybridSystemPredict


def ESN(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("ESN:")
    n_hidden_ESN, validationErrorAverageDF_ESN, testDF_ESN = EsnPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_ESN))
    w, p, EsnPredicted, mseTests = wilcoxonTest(outputs=testDF_ESN, actual=dfProcessedTest["actual"])
    print("\tP-value _ESN: ", p)
    return testDF_ESN, mseTests


def ARMAandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+ESN")
    n_hidden_armaESN, validationErrorAverageDF_armaESN, testDF_armaESN = armaEsnPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaESN))
    w, p, armaEsnPredicted, mseTests = wilcoxonTest(outputs=testDF_armaESN, actual=dfProcessedTest_LM[-len(testDF_armaESN):])
    print("\tP-value _armaESN: ", p)
    return testDF_armaESN, mseTests


def RBF(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest):
    print("RBF:")
    n_hidden_RBF, validationErrorAverageDF_RBF, testDF_RBF = RbfPredict(dfProcessedTrain, dfProcessedVal,
                                                                        dfProcessedTest, minMaxVal, minMaxTest)
    print("\tHidden Neurons: " + str(n_hidden_RBF))
    w, p, RbfPredicted, mseTests = wilcoxonTest(outputs=testDF_RBF, actual=dfProcessedTest["actual"])
    print("\tP-value _RBF: ", p)
    return testDF_RBF, mseTests


def ARMAandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order):
    print("ARMA+RBF")
    n_hidden_armaRBF, validationErrorAverageDF_armaRBF, testDF_armaRBF = armaRbfPredict(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    print("\tHidden Neurons: " + str(n_hidden_armaRBF))
    w, p, armaRbfPredicted, mseTests = wilcoxonTest(outputs=testDF_armaRBF, actual=dfProcessedTest_LM[-len(testDF_armaRBF):])
    print("\tP-value _armaRBF: ", p)
    return testDF_armaRBF, mseTests


####################
#       MAIN       #
####################
def main():
    print("Started: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    serie = baseData()
    dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest = apply_preProcess(serie[-300:], toLinearModels=False)
    dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM = apply_preProcess(serie[-300:], toLinearModels=True)
    allMSE = DataFrame(index=[i for i in range(0, 30)])

    # ------------------------ AR ------------------------

    #output_ar, order = AR(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)

    # ------------------------ ARMA ------------------------

    output_arma, order = ARMA(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM)

    # ------------------------ ELM ------------------------

    #output_elm, mseTests = ELM(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    #allMSE["ELM"] = mseTests

    # ------------------------ AR ELM ------------------------

    #output_arELM, mseTests = ARandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    #allMSE["AR+ELM"] = mseTests

    # ------------------------ ARMA ELM ------------------------

    output_armaELM, mseTests = ARMAandELM(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    allMSE["ARMA+ELM"] = mseTests

    # ------------------------ MLP ------------------------

    #output_MLP, mseTests = MLP(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    #allMSE["MLP"] = mseTests

    # ------------------------ ARMA MLP ------------------------

    #output_armaMLP, mseTests, hybridSystemPredict = ARMAandMLP(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    #allMSE["ARMA+MLP"] = mseTests

    # ------------------------------------------------

    #output_ESN, mseTests = ESN(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    #allMSE["ESN"] = mseTests

    # ------------------------------------------------

    output_armaESN, mseTests = ARMAandESN(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    allMSE["ARMA+ESN"] = mseTests

    # ------------------------------------------------

    output_RBF, mseTests = RBF(dfProcessedTrain, dfProcessedVal, dfProcessedTest, minMaxVal, minMaxTest)
    allMSE["RBF"] = mseTests

    # ------------------------------------------------

    output_armaRBF, mseTests = ARMAandRBF(dfProcessedTrain_LM, dfProcessedTest_LM, minMaxTest_LM, order)
    allMSE["ARMA+RBF"] = mseTests

    # ------------------------------------------------

    outputFinal = DataFrame(index=hybridSystemPredict.index)

    '''Aqui poderia ser feito o preenchimento das variáveis e depois a alteração do index??
        
        -- trocar isso --
            outputFinal = DataFrame(index=hybridSystemPredict.index)
        
        -- por isso --
            outputFinal: DataFrame = DataFrame()
            ....
            outputFinal["AR"] = output_ar
            ....
            outputFinal.set_index(hybridSystemPredict.index)
    '''

    outputFinal["AR"] = output_ar
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
