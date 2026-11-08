"""TensorFlow reference training implementation; imported framework only when fitting."""
import os

import numpy as np
import pandas as pd

from .dl_contract import HISTORY_COLUMNS, TrainingFailure, integer
from .supervised import metrics

MODEL_CONFIG = {
    'layers': [32, 16, 8], 'activation': 'relu', 'output_activation': 'softmax',
    'optimizer': 'Adam', 'learning_rate': .001,
    'loss': 'sparse_categorical_crossentropy', 'deterministic_ops': True,
    'shuffle': 'tf.data full training buffer; reshuffle each epoch',
}


def fit_seed(prepared, training_seed=42, epochs=100, batch_size=16):
    training_seed = integer(training_seed, 'training_seed', maximum=2**32-1)
    epochs = integer(epochs, 'epochs', minimum=1)
    batch_size = integer(batch_size, 'batch_size', minimum=1)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(training_seed)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([
        tf.keras.layers.Input((prepared.scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(prepared.target)), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def dataset(indices, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (prepared.scaled[indices].astype('float32'), prepared.target[indices]))
        if shuffle:
            ds = ds.shuffle(len(indices), seed=training_seed, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return ds.batch(batch_size).with_options(options)

    captured = []

    class Capture(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            captured.append({**(logs or {}), 'Epoch': epoch+1})
            if not all(np.isfinite(float(v)) for v in (logs or {}).values()):
                raise FloatingPointError('Non-finite training history')

    try:
        history = model.fit(dataset(prepared.train, True),
                            validation_data=dataset(prepared.validation), epochs=epochs,
                            verbose=0, shuffle=False, callbacks=[Capture()]).history
        pred = np.argmax(model.predict(dataset(prepared.test), verbose=0), axis=1)
    except Exception as error:
        partial = pd.DataFrame(captured, columns=HISTORY_COLUMNS)
        raise TrainingFailure(error, partial, model.get_weights()) from error
    row = {'Model': 'MLP', 'Layers': '32-16-8',
           **metrics(prepared.target[prepared.test], pred, np.unique(prepared.target)),
           'Epoch': epochs, 'Batch Size': batch_size,
           'Split Seed': prepared.split_seed, 'Training Seed': training_seed}
    arrays = {**prepared.arrays(), 'predictions': pred}
    arrays.update({f'weight_{i}': w for i, w in enumerate(model.get_weights())})
    return (pd.DataFrame([row]),
            pd.DataFrame(history).assign(Epoch=np.arange(1, epochs+1)), arrays)
