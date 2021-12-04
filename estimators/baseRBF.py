from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
import pandas as pd

class InitCentersRandom(Initializer):

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        # type checking to access elements of data correctly
        if type(self.X) == np.ndarray:
                return self.X[idx, :]
        elif type(self.X) == pd.core.frame.DataFrame:
                return self.X.iloc[idx, :]

class RBFLayer(Layer):

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):

        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RBF(object):

    def __init__(self, num_input_nodes, num_hidden_nodes, epochs=100):

        self._num_input_nodes = num_input_nodes
        self._num_hidden_nodes = num_hidden_nodes
        self._optimizer = optimizers.SGD()
        self._epochs = epochs

    def fit(self, X, y):

        self._model = Sequential()
        self._model.add(RBFLayer(output_dim=self._num_hidden_nodes,
                      initializer=InitCentersRandom(X),
                      betas=2.0,
                      input_shape=(self._num_input_nodes,)))
        self._model.add(Dense(1))
        self._model.compile(loss='mean_squared_error', optimizer=self._optimizer)

        self._model.fit(X, y, verbose=0, epochs=self._epochs, batch_size=1, shuffle=False)

    def __call__(self, X):
        predicted = self._model.predict(X)
        return predicted

def rbfPredict(hidden_dim, x_train, y_train, x_test, y_test):

    rbf = RBF(x_train.shape[1], hidden_dim, 1)
    rbf.fit(x_train, y_train.values.reshape(1, -1)[0])
    y_predicted_test = rbf(x_test)

    return y_predicted_test