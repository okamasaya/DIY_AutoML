"""Fixed-split MLP training shared by single and repeated workflows."""
from dataclasses import dataclass
from numbers import Integral
import os

import numpy as np
import pandas as pd

from .data import dl_indices, scale_train
from .supervised import metrics

HISTORY_COLUMNS = ['accuracy', 'loss', 'val_accuracy', 'val_loss', 'Epoch']
MODEL_CONFIG = {
    'layers': [32, 16, 8], 'activation': 'relu', 'output_activation': 'softmax',
    'optimizer': 'Adam', 'learning_rate': .001,
    'loss': 'sparse_categorical_crossentropy', 'deterministic_ops': True,
    'shuffle': 'tf.data full training buffer; reshuffle each epoch',
}


def integer(value, name, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be an integer in the supported range')
    return int(value)


@dataclass(frozen=True)
class PreparedDL:
    scaled: np.ndarray
    target: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    split_seed: int

    def arrays(self):
        return {'train': self.train, 'validation': self.validation, 'test': self.test,
                'scaler_mean': self.mean, 'scaler_scale': self.scale}


def prepare_data(X, y, split_seed=42):
    split_seed = integer(split_seed, 'split_seed', maximum=2**32-1)
    train, validation, test = dl_indices(y, split_seed)
    scaled, scaler = scale_train(X, train)
    values = [scaled, np.array(y, copy=True), train, validation, test,
              scaler.mean_, scaler.scale_]
    for value in values:
        value.setflags(write=False)
    return PreparedDL(*values, split_seed)


class TrainingFailure(RuntimeError):
    """Available epoch history and weights from an interrupted training attempt."""
    def __init__(self, error, history, weights):
        super().__init__(f'{type(error).__name__}: {error}')
        self.history = history
        self.weights = weights


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


def run(X, y, seed=42, split_seed=None, training_seed=None, epochs=100, batch_size=16):
    # Legacy seed supplies both defaults; explicit training_seed never changes indices.
    split_seed = seed if split_seed is None else split_seed
    training_seed = seed if training_seed is None else training_seed
    return fit_seed(prepare_data(X, y, split_seed), training_seed, epochs, batch_size)
