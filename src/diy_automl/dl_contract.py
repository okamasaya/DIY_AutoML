"""Framework-independent inputs and result checks for the reference dense MLP."""
from dataclasses import dataclass
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd

from .supervised import metrics

HISTORY_COLUMNS = ['accuracy', 'loss', 'val_accuracy', 'val_loss', 'Epoch']
HIDDEN_LAYERS = (32, 16, 8)
RESULT_CONTRACT = 'dense-32-16-8-v1'

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


class TrainingFailure(RuntimeError):
    """Available epoch history and weights from an interrupted training attempt."""
    def __init__(self, error, history, weights):
        super().__init__(f'{type(error).__name__}: {error}')
        self.history = history
        self.weights = weights


def preparation_fingerprint(prepared):
    """Detect accidental writes through array-sharing framework adapters."""
    fields=('scaled','target','train','validation','test','mean','scale')
    return (prepared.split_seed,tuple(
        (name,value.shape,value.dtype.str,value.flags.writeable,
         hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        for name in fields for value in (getattr(prepared,name),)))


def require_unchanged_preparation(prepared, fingerprint):
    if preparation_fingerprint(prepared)!=fingerprint:
        raise ValueError('DL result contract: shared prepared data was modified')


def validate_result(prepared, result, training_seed, epochs, batch_size):
    """Check the common boundary; native framework weights must be converted by adapters."""
    def require(condition, message):
        if not condition:
            raise ValueError('DL result contract: '+message)
    require(isinstance(result, tuple) and len(result)==3, 'expected result, history, arrays tuple')
    row_frame, history, arrays = result
    require(isinstance(row_frame,pd.DataFrame) and len(row_frame)==1 and row_frame.columns.is_unique,
            'expected one result row')
    required={'Model','Layers','Accuracy','Precision','Recall','F1','Epoch','Batch Size','Split Seed','Training Seed'}
    require(required.issubset(row_frame.columns),'missing result columns')
    row=row_frame.iloc[0]
    require(row['Model']=='MLP' and row['Layers']=='32-16-8','model identity differs')
    require(row['Epoch']==epochs and row['Batch Size']==batch_size
            and row['Split Seed']==prepared.split_seed and row['Training Seed']==training_seed,
            'training configuration differs')
    require(isinstance(history,pd.DataFrame) and history.columns.is_unique
            and set(history.columns)==set(HISTORY_COLUMNS),'history columns differ')
    require(len(history)==epochs and np.array_equal(history.Epoch,np.arange(1,epochs+1)),
            'history epochs differ')
    require(np.isfinite(history[HISTORY_COLUMNS].to_numpy(dtype=float)).all(),'non-finite history')
    require(isinstance(arrays,dict),'expected array dictionary')
    fixed=prepared.arrays()
    keys=set(fixed)|{'predictions'}|{f'weight_{i}' for i in range(8)}
    require(set(arrays)==keys,'array names differ')
    for key,value in fixed.items():
        require(np.array_equal(arrays[key],value),f'fixed preparation changed: {key}')
    pred=np.asarray(arrays['predictions'])
    classes=np.unique(prepared.target)
    require(pred.shape==(len(prepared.test),) and np.issubdtype(pred.dtype,np.integer)
            and np.isin(pred,classes).all(),'invalid predictions')
    for metric,value in metrics(prepared.target[prepared.test],pred,classes).items():
        require(np.isclose(float(row[metric]),value,atol=1e-12,rtol=0),f'{metric} differs from predictions')
    dimensions=[prepared.scaled.shape[1],*HIDDEN_LAYERS,len(classes)]
    for layer,(left,right) in enumerate(zip(dimensions,dimensions[1:])):
        for key,shape in [(f'weight_{2*layer}',(left,right)),(f'weight_{2*layer+1}',(right,))]:
            value=np.asarray(arrays[key])
            require(value.shape==shape and np.issubdtype(value.dtype,np.floating)
                    and np.isfinite(value).all(),f'invalid canonical dense weight: {key}')
