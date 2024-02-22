"""Tuner module
"""
from typing import NamedTuple, Dict, Text, Any
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner

from modules.transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    transformed_name,
)
from modules.trainer import (
    input_fn,
)

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])


def model_builder(hp):
    """Build machine learning model"""
    num_hidden_layers = hp.Choice(
        "num_hidden_layers", values=[1, 2, 3]
    )
    dense_units = hp.Int(
        "dense_units", min_value=32, max_value=256, step=32
    )
    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4]
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )

    #
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    x = tf.keras.layers.concatenate(input_features)
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics='binary_accuracy'
    )

    return model


def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.

    Returns:
    A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(
        fn_args.train_files[0], tf_transform_output)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.RandomSearch(hypermodel=model_builder,
                            objective='val_binary_accuracy',
                            max_trials=100,
                            directory=fn_args.working_dir,
                            project_name="kt_randomsearch",
                            )
    # Define the early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
