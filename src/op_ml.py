"""
Script that carries out Bayesian optimisation of NN hyperparameters.

WARNING: THIS SCRIPT IS CURRENTLY INCOMPATIBLE WITH THE REST OF THE CODEBASE
"""

from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf
import optuna
import pickle

from src.data_gen_new.utils import normalise
from utils import calibrate_images

data_path = "data/images-optimal-uncalibrated.pickle"
best_model_path = "output/best_model.keras"
study_db_path = "output/op_ml.db"
study_name = "study-16-03-24"
timeout = 22 * 60 * 60


def machine_learn(trial: optuna.Trial, images: np.ndarray, labels: np.ndarray) -> Tuple[float, Any]:
    """Hyperparameter optimisation of a Keras model."""
    # Suggesting hyperparameters
    conv_1_weights = trial.suggest_categorical("conv_1_weights", [16, 32, 64, 128])
    conv_1_kernel = trial.suggest_categorical("conv_1_kernel", [1, 3, 5])
    pool_1_kernel = trial.suggest_categorical("pool_1_kernel", [2, 3])
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)

    n_dense_layers = trial.suggest_int("n_dense_layers", 1, 3)
    try:
        model = tf.keras.Sequential()
        # Keeping the model architecture similar but adjusting the input shape
        n_rows, n_columns = images[0].shape
        model.add(
            tf.keras.layers.Conv2D(
                conv_1_weights,
                (conv_1_kernel, conv_1_kernel),
                activation="relu",
                input_shape=(n_rows, n_columns, 1),
            )
        )
        model.add(
            tf.keras.layers.MaxPooling2D(pool_size=(pool_1_kernel, pool_1_kernel))
        )
        for i in range(1, n_conv_layers):
            n_weights = trial.suggest_categorical(f"conv_{i+1}_weights", [16, 32, 64, 128])
            conv_kernel_size = trial.suggest_categorical(
                f"conv_{i+1}_kernel", [1, 3, 5]
            )
            pool_kernel_size = trial.suggest_categorical(f"pool_{i+1}_kernel", [2, 3])

            model.add(
                tf.keras.layers.Conv2D(
                    n_weights,
                    (conv_kernel_size, conv_kernel_size),
                    activation="relu",
                    input_shape=(n_rows, n_columns, 1),
                )
            )
            model.add(
                tf.keras.layers.MaxPooling2D(
                    pool_size=(pool_kernel_size, pool_kernel_size)
                )
            )

        model.add(tf.keras.layers.Flatten())

        for i in range(0, n_dense_layers - 1):
            n_weights = trial.suggest_categorical(f"dense_{i+1}_weights", [32, 64, 128, 256])
            model.add(tf.keras.layers.Dense(n_weights, activation="relu"))

        # Output layer with 3 neurons for the 3 outputs
        model.add(tf.keras.layers.Dense(3, activation="linear"))

        model.compile(optimizer="adam", loss="mean_squared_error")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,  # Number of epochs with no improvement after which training will be stopped
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        info = model.fit(
            images,
            labels,
            validation_split=0.25,
            shuffle=True,
            batch_size=32,
            epochs=1000,
            verbose=True,
            callbacks=[early_stopping],
        )

        loss = info.history["val_loss"][-1]
        return loss, model

    except Exception as e:
        # Handle the failure case or log the error
        print(f"An error occurred: {e}")
        return 1000, None


def objective(trial: optuna.Trial, images: np.ndarray, labels: np.ndarray, best_attributes: Dict[str, Any]) -> float:
    # Tracking best trial attributes
    best_loss = None

    # Training Model
    loss, model = machine_learn(trial, images, labels)

    # Updating if this is a new best trial
    if best_loss is None or loss < best_loss:
        best_loss = loss
        best_attributes["model"] = model

    return loss


def main() -> None:
    # Reading and normalising the data
    with open(data_path, "rb") as file:
        images_and_labels = pickle.load(file)

    unnormalised_images = calibrate_images(images_and_labels["images"], 4096)
    unnormalised_labels = images_and_labels["labels"]

    images, labels, denormaliser = normalise(
        np.array(unnormalised_images), np.array(unnormalised_labels)
    )

    # Creating an Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{study_db_path}",
    )
    best_attributes = {"images_and_labels": None, "model": None}
    study.optimize(
        lambda trial: objective(trial, images, labels, best_attributes), timeout=timeout
    )

    # Saving the best model
    best_attributes["model"].save(best_model_path)


if __name__ == "__main__":
    main()