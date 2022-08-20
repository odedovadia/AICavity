from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, Dropout, BatchNormalization, MaxPool1D, MaxPool2D, Reshape, LeakyReLU
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2.adam import Adam


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
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(128, 8, data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool1D(2, 2, data_format='channels_first')(x)

    x = Conv1D(1, 1, data_format='channels_first')(x)

    x = Flatten()(x)
    # x = Dense(300, activation='relu')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, y)
    return model


def run_model():
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    ds_train, ds_val = generate_data_end2end(file_path=TRAIN_DATA_PATH)
    model.fit(ds_train, validation_data=ds_val, epochs=500, verbose=2)

if __name__ == '__main__':
    run_model()
