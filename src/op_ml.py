"""
Script that carries out Bayesian optimisation of NN hyperparameters.
"""

import os
from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf
import optuna
import pickle

from machine_learning.keras_trial import ml_trial
from utils import calibrate_images, create_output_dirs, DynamicMinMaxScaler


data_path = "data/images-optimal-uncalibrated.pickle"
best_model_path = "output/nn_models/hyperparam_opt_model.keras"
study_db_path = "output/optuna_studies/hyperparam_opt.db"
study_name = "study-16-03-24"
timeout = 22 * 60 * 60

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray]:
    # Reading and normalising the data
    with open(data_path, "rb") as file:
        images_and_labels = pickle.load(file)

    images = calibrate_images(images_and_labels["images"], 4096)
    labels = images_and_labels["labels"]

    images_scaler = DynamicMinMaxScaler()
    labels_scaler = DynamicMinMaxScaler()

    norm_images = images_scaler.fit_transform(images)
    norm_labels = labels_scaler.fit_transform(labels)

    return norm_images, norm_labels


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "n_conv_layers": trial.suggest_int("n_conv_layers", 1, 3),
        "n_dense_layers": trial.suggest_int("n_dense_layers", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }

    for i in range(params["n_conv_layers"]):
        params[f"conv_{i+1}_filters"] = trial.suggest_categorical(
            f"conv_{i+1}_filters", [16, 32, 64, 128]
        )
        params[f"conv_{i+1}_kernel"] = trial.suggest_categorical(
            f"conv_{i+1}_kernel", [1, 3, 5]
        )
        params[f"pool_{i+1}_kernel"] = trial.suggest_categorical(
            f"pool_{i+1}_kernel", [2, 3]
        )

    for i in range(params["n_dense_layers"] - 1):
        params[f"dense_{i+1}_units"] = trial.suggest_categorical(
            f"dense_{i+1}_units", [32, 64, 128, 256]
        )

    return params


def objective(trial: optuna.Trial, data) -> float:
    images, labels = data

    n_train = (len(images) * 3) // 4
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    val_images = images[n_train:]
    val_labels = labels[n_train:]

    params = suggest_hyperparameters(trial)

    loss, model = ml_trial(
        train_images,
        train_labels,
        val_images,
        val_labels,
        batch_size=params["batch_size"],
        max_epochs=1000,
        patience=10,
        hyperparams=params,
    )

    trial.set_user_attr("params", params)

    return loss


def main() -> None:
    create_output_dirs()

    if os.path.exists(study_db_path):
        os.remove(study_db_path)

    images, labels = load_and_prepare_data()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{study_db_path}",
    )

    study.optimize(
        lambda trial: objective(trial, (images, labels)),
        timeout=timeout,
    )

    best_trial = study.best_trial

    print("Best loss:", best_trial.value)
    print("Best params:", best_trial.params)

    _, best_model = ml_trial(
        images,
        labels,
        images,  
        labels,
        batch_size=best_trial.params["batch_size"],
        max_epochs=1000,
        patience=10,
        hyperparams=best_trial.params,
    )

    best_model.save(best_model_path)

    with open("output/best_hyperparameters.pickle", "wb") as f:
        pickle.dump(best_trial.params, f)



if __name__ == "__main__":
    main()