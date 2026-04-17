from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Analyser:
    def __init__(
        self,
        images_and_labels: dict,
        label_titles: Tuple[str],
    ) -> None:
        """
        Initialises the Analyser instance with true_labels and predicted_labels.

        images_and_labels dictionary contain the following keys:
        - images
        - labels
        - predictions
        """
        images = images_and_labels["images"]
        true_labels = images_and_labels["labels"]
        predicted_labels = images_and_labels["predictions"]

        if not (len(images) == len(true_labels) == len(predicted_labels)):
            raise ValueError(
                "The size of each element list in the dictionary must be the same but"
                f" are: {len(images)}, {len(true_labels)}, {len(predicted_labels)}"
            )

        if not (len(true_labels[0]) == len(predicted_labels[0]) == len(label_titles)):
            raise ValueError(
                "The size of each element in labels and predictions should be"
                " equal to the size of label_titles."
            )

        self.images: list = images
        self.true_labels: np.ndarray = np.array(true_labels)
        self.predicted_labels: np.ndarray = np.array(predicted_labels)
        self.label_titles: np.ndarray = np.array(label_titles)

    def histogram_2d(self) -> None:
        """Plots 2d histograms of the true labels vs predicted labels."""

        for index, title in enumerate(self.label_titles):
            plt.figure(figsize=(14, 6))
            plt.subplot(131)
            plt.title(title)
            plt.hist2d(
                self.true_labels[:, index],
                self.predicted_labels[:, index],
                bins=75,
                cmap="hot",
            )
            plt.plot(
                self.true_labels[:, index],
                self.true_labels[:, index],
                label="Perfect fit",
            )
            plt.xlabel("Actual value")
            plt.ylabel("Predicted value")
            plt.legend()

    def categorise_by_threshold(self, rae_maxes: List[float]) -> dict:
        """
        Calculates the RAEs and creates new images and binary labels. For each image
        the corresponding binary label is determined by its RAE in relation to the RAE
        maxes:

        1) If all RAEs are equal to or less than the RAE maxes -> label = 0
        2) Else -> label = 1
        """
        raes_list = abs(self.true_labels - self.predicted_labels) / abs(
            self.true_labels
        )

        images_bin_labels = {"images": self.images, "labels": []}
        for raes in raes_list:
            res = 0
            for i, max_rae in enumerate(rae_maxes):
                if raes[i] > max_rae:
                    res = 1
            images_bin_labels["labels"].append(res)
        images_bin_labels["labels"] = np.array(images_bin_labels["labels"]).reshape(
            -1, 1
        )
        return images_bin_labels

    def mraes(self) -> Dict[str, float]:
        """Determines the mean relative absolute errors between lists."""
        errors = {}
        for i, title in enumerate(self.label_titles):
            relative_abs_errors = abs(
                self.true_labels[:, i] - self.predicted_labels[:, i]
            ) / abs(self.true_labels[:, i])
            errors[title] = np.mean(relative_abs_errors)

        return errors


def print_error_rates(labels: np.ndarray, predictions: np.ndarray) -> None:
    """
    Prints the false positive and false negative rates of binary classified
    labels and predictions.
    """
    labels.reshape(-1)
    predictions.reshape(-1)

    titles = ["False Positive", "False Negative"]
    for i in range(2):
        mask = labels == i
        mask = mask.reshape(-1)
        res = predictions[mask]

        n_wrong = 0
        for elem in res:
            if elem == (not i):
                n_wrong += 1

        print(f"{titles[i]}: {round(n_wrong/len(res)*100,2)} %")
