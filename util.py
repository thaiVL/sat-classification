import skimage.util as skutil
import skimage.io as skio
import tensorflow as tf
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt


def read_img(path: str) -> npt.NDArray[np.float64]:
    return tf.convert_to_tensor(skutil.img_as_float64(skio.imread(path)), dtype=tf.float64)


def pd_series_to_np_img(series: pd.Series) -> npt.NDArray[np.float64]:
    vanilla_list = series.to_list()
    return np.array(vanilla_list, dtype=np.float64)


def pd_series_to_tensor(series: pd.Series) -> tf.Tensor:
    vanilla_list = series.to_list()
    return tf.convert_to_tensor(vanilla_list, dtype=tf.float64)


def tf_read_img(path):
    return tf.convert_to_tensor(tf.keras.utils.load_img(path=path, grayscale=False, color_mode="rgb"), dtype=tf.float64)


def plot_recall_precision(ax, precision_score, recall_score):
    y_bar = [round(precision_score * 100, 3), round(recall_score * 100, 3)]
    ax.bar(["Precision", "Recall"], y_bar, width=0.8)
    ax.set_ylim(0, 100)
    ax.set_title("Precision and Recall score")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Precision and Recall")
    xlocs, _ = plt.xticks()
    for i, v in enumerate(y_bar):
        ax.text(xlocs[i] - 0.25, v + 0.01, str(v))
