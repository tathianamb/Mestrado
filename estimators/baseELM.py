import numpy as np

def _tanh(x):
    return np.tanh(x)

class ELM(object):
    #num_input_nodes: number of columns
    def __init__(self, num_input_nodes, num_hidden_units, num_out_units, beta_init=None, w_init=None, bias_init=None):
        self._num_input_nodes = num_input_nodes
        self._num_hidden_units = num_hidden_units
        self._num_out_units = num_out_units

        self._activation = _tanh

        if isinstance(beta_init, np.ndarray):
            self._beta = beta_init
        else:
            self._beta = np.random.uniform(-1., 1., size=(self._num_hidden_units, self._num_out_units))

        if isinstance(w_init, np.ndarray):
            self._w = w_init
        else:
            self._w = np.random.uniform(-1, 1, size=(self._num_input_nodes, self._num_hidden_units))

        if isinstance(bias_init, np.ndarray):
            self._bias = bias_init
        else:
            self._bias = np.zeros(shape=(self._num_hidden_units,))

    def fit(self, X, Y):
        H = self._activation(X.dot(self._w) + self._bias)

        H_pinv = np.linalg.pinv(H)

        self._beta = H_pinv.dot(Y)

    def __call__(self, X):
        H = self._activation(X.dot(self._w) + self._bias)
        return H.dot(self._beta)

def elmPredict(hidden_dim, x_train, y_train, x_test, y_test):

    elm = ELM(x_train.shape[1], hidden_dim, 1)

    elm.fit(x_train, y_train.values.reshape(-1,1))

    y_predicted_test = elm(x_test)

    return y_predicted_test