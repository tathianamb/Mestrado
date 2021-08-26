from typing import Tuple, Any

from External.pyESN import ESN
from Processing.Evaluation import metricError, plotResult


def esnPredict(n_input, n_reservoir, x_training, y_training, x_test, y_test) -> Tuple[float, Any, Any]:
    esn = ESN(n_inputs=n_input, n_outputs=1, n_reservoir=n_reservoir)
    esn.fit(x_training, y_training)
    y_predicted = esn.predict(x_test)

    plotResult(predictedValues=y_predicted, actualValues=y_test, title="Actual vs Predicted")

    return metricError(y_predicted, y_test)
