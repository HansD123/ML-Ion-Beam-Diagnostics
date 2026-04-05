from typing import Any, Optional, Tuple
import numpy as np
import keras


def ml_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    max_epochs: int,
    patience: Optional[int] = None,
    loss: str = "mean_squared_error",
    out_activation: str = "linear",
) -> Tuple[float, Any]:
    """Uses Keras to train on batches of images."""

    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)

    model = keras.Sequential()
    # Keeping the model architecture similar but adjusting the input shape
    model.add(
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=x_train.shape[1:]
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the conv layers to feed into the dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="sigmoid"))
    model.add(keras.layers.Dense(y_train.shape[1], activation=out_activation))
    model.compile(optimizer="adam", loss=loss)

    callbacks = []
    if patience:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)

    info = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )
    val_loss = info.history["val_loss"][-1]

    return val_loss, model
