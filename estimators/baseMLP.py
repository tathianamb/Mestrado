from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras.activations import tanh


class MLP(object):
    def __init__(self, num_input_nodes, num_hidden_nodes, num_out_nodes, epochs=10):

        self._num_input_nodes = num_input_nodes
        self._num_hidden_nodes = num_hidden_nodes
        self._num_out_nodes = num_out_nodes
        self._optimizer = optimizers.SGD()
        self._activation = tanh
        self._epochs = epochs
        self._buildModel()

    def _buildModel(self):

        self._model = Sequential()
        self._model.add(Dense(self._num_hidden_nodes, input_dim=self._num_input_nodes, activation=self._activation))
        self._model.add(Dense(1))
        self._model.compile(loss='mean_squared_error', optimizer=self._optimizer)

    def fit(self, X, y):
        self._model.fit(X, y, verbose=0, epochs=self._epochs, batch_size=10, shuffle=False)

    def __call__(self, X):
        predicted = self._model.predict(X)
        return predicted

def mlpPredict(hidden_dim, x_train, y_train, x_test, y_test):

    mlp = MLP(x_train.shape[1], hidden_dim, 1)
    mlp.fit(x_train, y_train.values.reshape(-1, 1))

    y_predicted_test = mlp(x_test)

    return y_predicted_test