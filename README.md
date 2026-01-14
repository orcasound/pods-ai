# Programmatic Orca Detection System using Artificial Intelligence (PODS-AI)

There are two projects in this repository.

## ModelTraining

The `ModelTraining/src` directory has the following scripts for different steps meant to be run in the order listed:

* make_csv.py: Create `output_segments/detections.csv` with a set of detections
* extract_training_samples.py: Use `output_segments/detections.csv` to create `output_segments/training_samples.csv`
* download_wavs.py: Use `output_segments/training_samples.csv` and download wav files into subdirectories under `output_segments`
* make_spectrograms.py: Create a png file alongside each wav file in a subdirectory of `output_segments`

It also contains some helper scripts:

* spectrogram_visualizer.py: This is adapted from <https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py>

## PictureRecognition

This project is along the lines of an exercise from "Deep Learning for Coders with fastai & PyTorch".
It downloads samples of orcas, etc. and builds and runs an AI model to recognize them from pictures.

This code is just an example resource to be used when constructing a model for the audio recognition.
