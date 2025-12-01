"""Sequence modeling skeleton (LSTM/GRU) for future temporal experiments.

This file contains a minimal Keras example that demonstrates how one could
structure a sequence model using per-customer monthly time-series. It is
intentionally lightweight and is not executed by default; it documents steps
to prepare data and a small model for prototyping.
"""
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Masking, LSTM, GRU, Dense
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False


def build_sequence_model(input_shape, rnn_type='LSTM', units=64):
    if not _TF_AVAILABLE:
        raise ImportError('TensorFlow is not installed. Install tensorflow to run sequence modeling experiments.')

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    if rnn_type == 'LSTM':
        model.add(LSTM(units))
    else:
        model.add(GRU(units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


def prepare_sequence_data(df, customer_id_col='customer_id', time_col='month', feature_cols=None, seq_len=6):
    """Convert a transactional/monthly dataframe into (N, seq_len, F) arrays.

    - `df` should contain per-customer monthly rows.
    - `feature_cols` is a list of numeric features to include per timestep.
    """
    if feature_cols is None:
        feature_cols = ['spend', 'payment_amt', 'utilization']

    # pivot to fixed-length sequences (pad with zeros if shorter)
    groups = df.groupby(customer_id_col)
    X_list = []
    ids = []
    for cid, g in groups:
        g = g.sort_values(time_col).tail(seq_len)
        arr = g[feature_cols].to_numpy()
        if arr.shape[0] < seq_len:
            pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]))
            arr = np.vstack([pad, arr])
        X_list.append(arr)
        ids.append(cid)

    X = np.stack(X_list)
    return X, ids
