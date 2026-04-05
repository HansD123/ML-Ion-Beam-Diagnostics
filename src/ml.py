"""
Trains a Keras model on a dataset of images and labels. Outputs the true labels,
predicted labels, and the trained network.
"""

import numpy as np
import pickle

import utils
from machine_learning.keras_trial import ml_trial

# Input data path
input_path = "../data/50Kelectrons.pickle"

# Output data paths
output_model_dir = "../output/nn_models"
model_filename = "model.keras"
model_path = f"{output_model_dir}/{model_filename}"

output_labels_dir = "../output/labels"

true_labels_filename = "true_labels.pickle"
true_labels_path = f"{output_labels_dir}/{true_labels_filename}"

predicted_labels_filename = "predicted_labels.pickle"
predicted_labels_path = f"{output_labels_dir}/{predicted_labels_filename}"


def main() -> None:
    utils.create_output_dirs()

    # Reading images and labels
    # Splitting into training and validation data
    with open(input_path, "rb") as file:
        images_and_labels = pickle.load(file)

    images = np.array(images_and_labels["images"])
    labels = np.array(images_and_labels["labels"])[:, :3]  # First 3 are protons params

    # Taking the logarithm of N0
    labels[:, 2] = np.log(labels[:, 2])

    # Normalising images and labels
    images_scaler = utils.DynamicMinMaxScaler()
    labels_scaler = utils.DynamicMinMaxScaler()

    images = images_scaler.fit_transform(images)
    labels = labels_scaler.fit_transform(labels)

    # Splitting into training and validation datasets
    FRAC_TRAIN = 3 / 4
    n_train = int(FRAC_TRAIN * len(images))
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]

    # Training Model
    BATCH_SIZE = 32
    MAX_EPOCHS = 10
    PATIENCE = 15
    val_loss, model = ml_trial(
        train_images,
        train_labels,
        test_images,
        test_labels,
        BATCH_SIZE,
        MAX_EPOCHS,
        patience=PATIENCE,
    )

    # Outputting datasets of true vs predicted labels
    norm_predictions = model.predict(test_images)

    # Rescaling labels to original size
    true_labels = labels_scaler.inverse_transform(labels)
    true_labels[:, 2] = 10 ** true_labels[:, 2]
    predicted_labels = labels_scaler.inverse_transform(norm_predictions)
    predicted_labels[:, 2] = 10 ** predicted_labels[:, 2]

    # Outputting as pickle files
    with open(true_labels_path, "wb") as f:
        pickle.dump(true_labels, f)

    with open(predicted_labels_path, "wb") as f:
        pickle.dump(predicted_labels, f)

    # Saving the model
    model.save(model_path)


if __name__ == "__main__":
    main()
