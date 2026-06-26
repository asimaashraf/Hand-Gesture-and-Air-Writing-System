#!/usr/bin/env python
import csv
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model' / 'keypoint_classifier'
CSV_PATH = MODEL_DIR / 'keypoint.csv'
LABEL_PATH = MODEL_DIR / 'keypoint_classifier_label.csv'
HDF5_PATH = MODEL_DIR / 'keypoint_classifier.hdf5'
TFLITE_PATH = MODEL_DIR / 'keypoint_classifier.tflite'


def load_rows():
    labels = []
    features = []
    with CSV_PATH.open(newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            labels.append(int(float(row[0])))
            features.append([float(value) for value in row[1:]])
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_label_count():
    with LABEL_PATH.open(encoding='utf-8-sig') as f:
        return sum(1 for row in csv.reader(f) if row)


def make_model(class_count):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(42,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(class_count, activation='softmax'),
        ]
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def backup_existing():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in (HDF5_PATH, TFLITE_PATH):
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + f'.bak_{stamp}'))


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)


def main():
    x, y = load_rows()
    class_count = load_label_count()
    if x.shape[1] != 42:
        raise ValueError(f'Expected 42 keypoint features, got {x.shape[1]}')
    if int(y.max()) >= class_count:
        raise ValueError(f'CSV contains label {int(y.max())}, but only {class_count} labels exist.')

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    x = x[indices]
    y = y[indices]
    split = int(len(x) * 0.85)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = make_model(class_count)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=60,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    backup_existing()
    model.save(HDF5_PATH, include_optimizer=False)
    export_tflite(model)
    best_val = max(history.history.get('val_accuracy', [0.0]))
    print(f'Retrained keypoint classifier with {class_count} classes.')
    print(f'Best validation accuracy: {best_val:.4f}')
    print(f'Saved: {HDF5_PATH}')
    print(f'Saved: {TFLITE_PATH}')


if __name__ == '__main__':
    main()
a