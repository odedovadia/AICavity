import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Dropout, BatchNormalization, MaxPool1D, LeakyReLU, Add, Lambda
from tensorflow.keras.models import Model

from constants import NUM_SIGNALS, MAX_SIG_LEN


def conv_architecture(fourier: bool = False, add_fully_connected: bool = False):
    input_tensor = Input(shape=(NUM_SIGNALS, MAX_SIG_LEN))

    if fourier:
        x = Lambda(lambda v: tf.abs(tf.signal.fft(tf.cast(v, tf.complex64))))(input_tensor)
        x = Conv1D(16, 8, data_format='channels_first')(x)
    else:
        x = Conv1D(16, 8, data_format='channels_first')(input_tensor)

    x = BatchNormalization()(x)
    x = LeakyReLU(0.0)(x)
    x = MaxPool1D(16, 16, data_format='channels_first')(x)

    x = Conv1D(64, 3, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.0)(x)

    for _ in range(10):
        in_x = x
        x = Conv1D(64, 3, data_format='channels_first', padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.0)(x)
        x = Add()([x, in_x])

    x = Conv1D(1, 1, data_format='channels_first')(x)

    x = Flatten()(x)

    if add_fully_connected:
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)

    y = Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, y)
    return model


def fully_connected_architecture(fourier: bool = False):
    input_tensor = Input(shape=(NUM_SIGNALS, MAX_SIG_LEN))
    x = Flatten()(input_tensor)

    if fourier:
        x = Lambda(lambda v: tf.abs(tf.signal.fft(tf.cast(v, tf.complex64))))(x)
    else:
        x = Dense(64, activation='relu')(x)

    for _ in range(5):
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    y = Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, y)
    return model
