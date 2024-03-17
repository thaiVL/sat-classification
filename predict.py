import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import util
from pipeline import SatClassModel, Label


def parsearg():
    parser = argparse.ArgumentParser(description="Predict using by loading in weights of the trained model")
    parser.add_argument("--weight_path", help="Path to the saved weights of the model", required=True, type=str)
    parser.add_argument("--batch_size", help="Batch size used in prediction", required=True, type=int)
    parser.add_argument("--image_file", help="Image to predict", type=str)
    parser.add_argument("--data_path", help="Path to dataset", type=str)
    parser.add_argument("--dataset_file", help="File to indicate which images to process", type=str)

    args = parser.parse_args()

    if args.image_file and (args.data_path or args.dataset_file):
        parser.error("--image_file argument is mutually exclusive with --data_path and --dataset_file")
    return args


def load_data(data_path: str, dataset_file: str):
    df = pd.read_csv(f"{data_path}/{dataset_file}")
    df["Filename"] = f"{data_path}/" + df["Filename"].astype(str)
    df["Image"] = df["Filename"].apply(util.read_img)
    return util.pd_series_to_tensor(df["Image"])


def load_image(image_file: str):
    return util.read_img(image_file)


if __name__ == '__main__':
    INPUT_SHAPE = (64, 64, 3)
    UNITS = 10
    PREDICTION_OUTPUT_FILE = "./prediction.csv"
    args = parsearg()

    label_mappings = Label.label_mappings(encoded=True)

    def int_to_label(label: int):
        return label_mappings[label]

    df = None
    x = None
    if (args.image_file):
        x = tf.reshape(load_image(args.image_file), [1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        df = pd.DataFrame(data={"Filename": [args.image_file]})
    else:
        x = load_data(args.data_path, args.dataset_file)
        df = pd.read_csv(f"{args.data_path}/{args.dataset_file}", usecols=["Filename"])

    if x is None or df is None:
        raise Exception("Failed to load dataset")

    model = SatClassModel(UNITS, INPUT_SHAPE)
    model.build((None,) + INPUT_SHAPE)
    model.load_weights(args.weight_path)
    pred = model.predict(x, batch_size=args.batch_size, use_multiprocessing=True)
    predicted_labels = np.argmax(pred, 1)
    df["Prediction"] = pd.Series(predicted_labels).apply(int_to_label)

    df.to_csv(PREDICTION_OUTPUT_FILE)

