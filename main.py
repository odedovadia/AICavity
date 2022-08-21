import os
from tabnanny import verbose

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, Dropout, BatchNormalization, MaxPool1D, MaxPool2D, Reshape, LeakyReLU, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from data_generator import generate_data_end2end
from constants import NUM_SIGNALS, MAX_SIG_LEN, TEST_DATA_PATH, TRAIN_DATA_PATH


def build_model():
    input_tensor = Input(shape=(NUM_SIGNALS, MAX_SIG_LEN))
    x = Conv1D(8, 128, data_format='channels_first')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(16, 64, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(32, 32, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(64, 16, data_format='channels_first')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(128, 8, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(1, 1, data_format='channels_first')(x)

    x = Flatten()(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, y)
    return model


def run_model():
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = []
    checkpoint = ModelCheckpoint(os.path.join('models', 'model.h5'), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    logger = CSVLogger(os.path.join('history', 'history.CSV'))
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, verbose=2)
    callbacks += [checkpoint, logger, reduce_lr]

    ds_train, ds_val = generate_data_end2end(file_path=TRAIN_DATA_PATH)
    model.fit(ds_train, validation_data=ds_val, epochs=500, verbose=2, callbacks=callbacks)


def run_inference():
    model = load_model(os.path.join('models', 'model.h5'))
    ds_test = generate_data_end2end(file_path=TEST_DATA_PATH)
    model.evaluate(ds_test, verbose=2)


if __name__ == '__main__':
    run_model()
    run_inference()
