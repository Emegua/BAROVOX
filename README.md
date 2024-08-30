## A Fly on the Wall - Exploiting Acoustic Side-Channels in Differential Pressure Sensors


This repository contains the code for a project that investigates the potential of exploiting acoustic side-channels in
differential pressure sensors. The project aims to convert pressure data into audio signals and analyze them for speech command classification tasks.
### Contents:
1. Dataset_Generator: This folder includes the code to convert pressure data to audio. It provides functions and utilities
   to preprocess the pressure data and generate corresponding audio files.

2. classification_model: This folder contains the code to train a ResNet-based model using the dataset generated
   from the pressure data. The model is trained to classify audio signals based on the pressure information they represent.
### Usage:
- To convert pressure data to audio, navigate to the Dataset_Generator folder and run the appropriate scripts. Make sure to provide the necessary input data and specify the desired audio format.

    Please note that at the moment, we are unable to provide the dataset for the pressure sensor due to the lack of permission from the relevant authority. As such we acknoledge that running the experiments would be hard unless someone has access to the pressure sensors directly. We are actively working to find a possible and secure way to share the data.


- To train the ResNet-based model, navigate to the classification_model folder and execute the provided scripts.
  Ensure that the dataset generated from the pressure data is available in the correct format.

#

Note: This project is for research and educational purposes only. 
