from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy.typing as npt
from matplotlib.axes import Axes
from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay

import util
import tensorflow as tf
from pathlib import Path


class Label(Enum):
    ANNUALCROP = 0
    FOREST = 1
    HERBACEOUSVEGETATION = 2
    HIGHWAY = 3
    INDUSTRIAL = 4
    PASTURE = 5
    PERMANENTCROP = 6
    RESIDENTIAL = 7
    RIVER = 8
    SEALAKE = 9

    @classmethod
    def label_mappings(cls, encoded=False):
        decoded_mappings = {
            "AnnualCrop": 0,
            "Forest": 1,
            "HerbaceousVegetation": 2,
            "Highway": 3,
            "Industrial": 4,
            "Pasture": 5,
            "PermanentCrop": 6,
            "Residential": 7,
            "River": 8,
            "SeaLake": 9
        }
        if encoded:
            return {value: key for key, value in decoded_mappings.items()}
        return decoded_mappings


class Sample:
    __img_name: str
    __img_array: npt.NDArray[np.float64]
    __img_label: Label

    @property
    def img_name(self):
        return self.__img_name

    @property
    def img_array(self):
        return self.__img_array

    @property
    def img_label(self):
        return self.__img_label

    def __init__(self, img_name: str, img_array: npt.NDArray[np.float64], img_label: Label) -> None:
        self.__img_name = img_name
        self.__img_array = img_array
        self.__img_label = img_label

    @classmethod
    def reading_list(cls, df: pd.DataFrame) -> list:
        return list(map(lambda x: Sample(img_name=x[0], img_array=x[2], img_label=x[1]), df.values.tolist()))

    def __str__(self) -> str:
        return f"Image name: {self.__img_name} | Label: {self.__img_label}"

    def __repr__(self) -> str:
        return self.__str__()


class SatClassModel(tf.keras.Model):
    def __init__(self, units, input_shape):
        super(SatClassModel, self).__init__()
        self.base_model: tf.keras.Model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                                            input_shape=input_shape)
        self.base_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=units)
        self.activation = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.activation(x)


def plot_train_vs_validation_metric(axis: Axes, train_metrics: List[float], validation_metrics: List[float],
                                    metric: str, epoch: int) -> None:
    axis.set_title(f"Epoch vs {metric}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.plot(train_metrics)
    axis.plot(validation_metrics)
    axis.set_xticks(np.arange(0, epoch, 1))
    axis.legend(["Training", "Validation"])


def test_model(model: tf.keras.Model, x_test_data: tf.Tensor, y_test_data: npt.NDArray) -> Tuple[float, float, float]:
    prediction = model.predict(x=x_test_data, verbose=2)
    y_prediction = np.argmax(prediction, axis=1)

    correct = y_test_data == y_prediction
    test_accuracy: float = np.sum(correct) / len(y_test_data)
    p_score: float = precision_score(y_true=y_test_data, y_pred=y_prediction, average="micro")
    r_score: float = recall_score(y_true=y_test_data, y_pred=y_prediction, average="micro")

    return test_accuracy, p_score, r_score


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    INPUT_SHAPE = (64, 64, 3)
    UNITS = 10
    EPOCHS = 6
    BATCH_SIZE = 16
    LR = 0.001
    MODEL_NAME = "SatClass"
    RESULTS_DIRECTORY = "./pipeline_result"

    TRAIN_DATA = "./data/EuroSatData/train.csv"
    TEST_DATA = "./data/EuroSatData/test.csv"
    VALIDATION_DATA = "./data/EuroSatData/validation.csv"
    BLANK_MODEL = "./data/blank_vgg16.h5"
    MAIN_MODEL = "./data/trained_vgg16.h5"

    train_df = pd.read_csv(TRAIN_DATA)
    train_df["Filename"] = "./data/EuroSatData/" + train_df["Filename"].astype(str)
    test_df = pd.read_csv(TEST_DATA)
    test_df["Filename"] = "./data/EuroSatData/" + test_df["Filename"].astype(str)
    validation_df = pd.read_csv(VALIDATION_DATA)
    validation_df["Filename"] = "./data/EuroSatData/" + validation_df["Filename"].astype(str)

    train_df["Image"] = train_df["Filename"].apply(util.read_img)
    test_df["Image"] = test_df["Filename"].apply(util.read_img)
    validation_df["Image"] = validation_df["Filename"].apply(util.read_img)

    x_train, y_train = util.pd_series_to_tensor(train_df["Image"]), train_df["Label"].values
    x_test, y_test = util.pd_series_to_tensor(test_df["Image"]), test_df["Label"].values
    x_validation, y_validation = util.pd_series_to_tensor(validation_df["Image"]), validation_df["Label"].values

    my_model = SatClassModel(UNITS, INPUT_SHAPE)
    my_model.build((None,) + INPUT_SHAPE)
    my_model.summary()

    my_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    train_history = my_model.fit(x=x_train, y=y_train,
                                 validation_data=(x_validation, y_validation),
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS, verbose=2)
    my_model.save(MODEL_NAME)

    train_losses = train_history.history["loss"]
    train_accuracy = train_history.history["accuracy"]
    validation_losses = train_history.history["val_loss"]
    validation_accuracy = train_history.history["val_accuracy"]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    plot_train_vs_validation_metric(ax[0][0], train_losses, validation_losses, "Loss", EPOCHS)
    plot_train_vs_validation_metric(ax[0][1], train_accuracy, train_accuracy, "Accuracy", EPOCHS)

    Path(RESULTS_DIRECTORY).mkdir(parents=False, exist_ok=True)
    fig.saveFig(f"{RESULTS_DIRECTORY}/loss_accuracy_graph.png")

    accuracy, p_score, r_score = test_model(my_model, x_test, y_test)
