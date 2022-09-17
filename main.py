import os

import tensorflow as tf
import scipy.io
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from architectures import conv_architecture, fully_connected_architecture
from metrics import calculate_metrics, save_roc_curve
from data_generator import generate_data_end2end
from constants import TEST_DATA_PATH, TRAIN_DATA_PATH, MODEL_NAME

tf.random.set_seed(42)


def run_model():
    #model = conv_architecture(fourier=True, add_fully_connected=True,to_concat=True)
    model = conv_architecture(fourier=True, add_fully_connected=False,to_concat=True)
    #model = fully_connected_architecture(fourier=True, to_concat=True)
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
    pred = model.predict(ds_test, verbose=2)
    ts = ds_test.unbatch()
    y = list(ts.map(lambda x, y: y))
    ynp = [x.numpy() for x in y]
    metrics = calculate_metrics(ynp, pred)
    print(metrics)
    # save_roc_curve(ynp, pred, MODEL_NAME)
    scipy.io.savemat(os.path.join('test_pred', MODEL_NAME + '_pred_on_test.mat'), {'prediction': pred, 'labels': ynp})


if __name__ == '__main__':
    run_model()
    run_inference()
