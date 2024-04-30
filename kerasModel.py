from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as K
from keras.layers.core import Dense, Activation, Dropout, Masking

batch_size = 50

inputShape=[256, 128, 100]
sequence = np.random.randn(*inputShape)
n_in = 128
# optimizer = tf.keras.optimizers.Adam(lr=0.001, clipvalue=0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-08)
# define model
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(n_in, 100)))
model.add(LSTM(100, activation='tanh', batch_input_shape=(batch_size, n_in, 100)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(n_in, 100)))
model.add(TimeDistributed(Dense(100)))
model.compile(optimizer=optimizer, loss='mse')
# fit model
model.fit(sequence, sequence, epochs=5000, verbose=0, callbacks=[TensorBoard(log_dir='mytensorboard')])
outputs = [layer.output for layer in model.layers]
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat)
print(np.shape(yhat))
# print(model.layers[0].output)
# outputs
# print(outputs[0])
