import math
from tqdm import tqdm

import tensorflow as tf

import numpy as np
import h5py

from constants import (TEST_DATA_PATH, TRAIN_DATA_PATH, BATCH_SIZE, MAX_SIG_LEN, VALIDATION_SIZE) 

def declare_variables(file_path):
    global f, data, size, max_sig_length
    f = h5py.File(file_path)
    data = f['data']
    size = len(data)
    max_sig_length = MAX_SIG_LEN   


def gen():
    np.random.seed(42)
    for i in range(size):
        if np.random.rand() < 0.5:
            x = f[data[i, 0]]['scat_full_hole']
            y = np.array(1.)
        else:
            x = f[data[i, 0]]['scat_full_whole']
            y = np.array(0.)

        signals_size = x.shape[1]
        pad_size = max_sig_length - signals_size
        x = np.pad(x, ((0,0), (0, pad_size))) * 1e24
        yield x, y


def build_tf_dataset():
    dataset = tf.data.Dataset.from_generator(gen, output_signature=(
         tf.TensorSpec(shape=(9, max_sig_length), dtype=tf.float32),
         tf.TensorSpec(shape=(), dtype=tf.int32)))
    return dataset


def generate_data_end2end(file_path: str = TEST_DATA_PATH, batch_size: int = BATCH_SIZE):
    declare_variables(file_path)
    dataset = build_tf_dataset()

    if file_path == TEST_DATA_PATH:
        test_ds = dataset
        test_ds = unroll_ds(test_ds)
        test_ds = test_ds.batch(batch_size)
        return test_ds
    else:
        train_size = math.ceil(size / batch_size * (1. - VALIDATION_SIZE))

        train_ds = dataset.skip(train_size)
        val_ds = dataset.take(train_size)

        train_ds = unroll_ds(train_ds)
        val_ds = unroll_ds(val_ds)

        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        return train_ds, val_ds


def unroll_ds(ds):
    x = []
    y = []
    for _, j in tqdm(enumerate(ds)):
        x += [j[0]]
        y += [j[1]]
    new_ds = tf.data.Dataset.from_tensor_slices((x, y))
    return new_ds


if __name__ == '__main__':
    t_ds, v_ds = generate_data_end2end()
