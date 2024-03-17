# Overview

The goal of this project is to build a pipeline to train a model and update weights whenever needed. Then the model is used to predict
new samples/images given by the user.

NOTE: To simplify this project and fit the outlined requirements of the email sent to me, I've removed files 
related to other functions (such as finding hyperparameters) and combined many python files into one in order
to reduce the number of files present. Many of the functions and classes in pipeline.py were originally belonging
to other python files.

The pipeline.py and predict.py are the core files of my project.

# Script documentation

## pipeline.py

pipeline.py is used to train and update the model based on a set of given data and save the weights of the newly updated model.
The hyperparameters are adjustable via commandline arguments when running the python script.

### ARGS
--data_path: the path to where all the data is located
--batch_size: batch size to use for training and validation
--epochs: number of epochs to train on
--lr: learning rate at which the model is training at
--result_directory: output location of the graphs detailing training/validation performance
--saved_model_location: location at which the weights of the trained model is saved to. Must include the name of the file and extension.


## predict.py

predict.py uses the weights from the trained model generated from pipeline.py to predict the labels of images.

### ARGS
--weight_path: the path to the saved weights to be loaded into the model
--batch_size: batch size to use during prediction
--image_file: a single image to run the prediction on
--data_path: path to the dataset containing all the images
--dataset_file: a (csv) file containing all images to predict

NOTE: --data_path and --dataset_file are mutually exclusive with --image_file
So for input, you can only have either:
	--image_file OR
	--data_path and --dataset_file


The output of predict.py is a csv file where the headers are [Filename, Prediction]
The Filename column is the name of the image and Prediction column holds the predicted label corresponding to the image.

Example arg for predicting multiple images:
--weight_path="./data/trained_model.h5"
--batch_size=16
--data_path="./data/EuroSatData"
--dataset_file="test.csv"

Example arg for predicting a single image:
--weight_path="./data/trained_model.h5"
--batch_size=16
--image_file="./data/EuroSatData/AnnualCrop/AnnualCrop_11.jpg"