import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, Dropout, BatchNormalization, MaxPool1D, MaxPool2D, Reshape, LeakyReLU, Add, GRU, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from architectures import conv_architecture, fully_connected_architecture
from data_generator import generate_data_end2end
from constants import TEST_DATA_PATH, TRAIN_DATA_PATH, MODEL_NAME

tf.random.set_seed(123)


def run_model():
    model = fully_connected_architecture(fourier=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = []
    checkpoint = ModelCheckpoint(os.path.join('models', MODEL_NAME + '.h5'), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    logger = CSVLogger(os.path.join('history', MODEL_NAME + '_history.CSV'))
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, verbose=2, factor=0.5)
    callbacks += [checkpoint, logger, reduce_lr]

    ds_train, ds_val = generate_data_end2end(file_path=TRAIN_DATA_PATH)
    model.fit(ds_train, validation_data=ds_val, epochs=100, verbose=2, callbacks=callbacks)


def run_inference():
    model = load_model(os.path.join('models', MODEL_NAME + '.h5'))
    ds_test = generate_data_end2end(file_path=TEST_DATA_PATH)
    model.evaluate(ds_test, verbose=2)


if __name__ == '__main__':
    run_model()
    run_inference()
