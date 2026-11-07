import os
import numpy as np
import pandas as pd
from .data import dl_indices, scale_train
from .supervised import metrics

def run(X,y,seed=42,split_seed=None,training_seed=None,epochs=100,batch_size=16):
    # The legacy seed supplies both defaults; explicit training_seed never changes indices.
    split_seed=seed if split_seed is None else split_seed
    training_seed=seed if training_seed is None else training_seed
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(training_seed)
    tf.config.experimental.enable_op_determinism()
    train,validation,test=dl_indices(y,split_seed)
    Z,scaler=scale_train(X,train)
    model=tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],)),
        tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(8,activation="relu"),tf.keras.layers.Dense(len(np.unique(y)),activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    def dataset(indices,shuffle=False):
        ds=tf.data.Dataset.from_tensor_slices((Z[indices].astype("float32"),y[indices]))
        if shuffle: ds=ds.shuffle(len(indices),seed=training_seed,reshuffle_each_iteration=True)
        options=tf.data.Options(); options.threading.private_threadpool_size=1
        options.threading.max_intra_op_parallelism=1
        return ds.batch(batch_size).with_options(options)
    history=model.fit(dataset(train,True),validation_data=dataset(validation),epochs=epochs,verbose=0,shuffle=False).history
    pred=np.argmax(model.predict(dataset(test),verbose=0),axis=1)
    row={"Model":"MLP","Layers":"32-16-8",**metrics(y[test],pred,np.unique(y)),
         "Epoch":epochs,"Batch Size":batch_size,"Split Seed":split_seed,"Training Seed":training_seed}
    arrays={"train":train,"validation":validation,"test":test,"predictions":pred,
            "scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_}
    for i,w in enumerate(model.get_weights()): arrays[f"weight_{i}"]=w
    return pd.DataFrame([row]),pd.DataFrame(history).assign(Epoch=np.arange(1,epochs+1)),arrays
