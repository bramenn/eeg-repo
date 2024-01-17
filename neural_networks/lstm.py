import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

Model = tf.keras.Sequential()
Model.add(LSTM(128, return_sequences=True, input_shape=(100, 512)))
Model.add(Dropout(0.3))  # Ajuste de la tasa de Dropout
Model.add(LSTM(64, return_sequences=True))
Model.add(LSTM(32, return_sequences=True)) # Capa adicional
Model.add(Dense(32, activation='relu', activity_regularizer=l2(0.001)))
Model.add(Dense(512, activation='linear'))
Model.compile(loss='mean_squared_error', optimizer=Adam())