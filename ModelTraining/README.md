# ModelTraining

This directory contains scripts for preparing training data for orca detection models.

## Overview

The `ModelTraining/src` directory has the following scripts for different steps meant to be run in the order listed:

1. **make_csv.py**: Create `output_segments/detections.csv` with a set of detections
2. **extract_training_samples.py**: Use `output_segments/detections.csv` to create `output_segments/training_samples.csv`
   - For `tp_human_only` detections, runs model inference on preceding 60 seconds to find correct timestamp
   - For other detections, subtracts 2 seconds from timestamp
3. **download_wavs.py**: Use `output_segments/training_samples.csv` and download wav files into subdirectories under `output_segments`
4. **make_spectrograms.py**: Create a png file alongside each wav file in a subdirectory of `output_segments`

## Model-Based Timestamp Correction for tp_human_only

The `extract_training_samples.py` script now implements intelligent timestamp correction for `tp_human_only` detections:

### How it Works

For detections marked as `tp_human_only`:
1. Downloads 60 seconds of audio preceding the detection timestamp
2. Runs model inference to score each 2-second segment
3. Finds the highest scoring segment
4. Adjusts the timestamp based on the offset of the highest scoring segment (between 2-60 seconds)

This matches the behavior described in the issue and follows the approach used in [aifororcas-livesystem's LiveInferenceOrchestrator.py](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/LiveInferenceOrchestrator.py).

### Using the FastAI Model

By default, `extract_training_samples.py` uses the FastAI model with automatic download enabled.

#### Option 1: Default behavior (recommended)

Install FastAI dependencies and run the script:

```bash
pip install -r requirements-fastai.txt
cd ModelTraining/src
python extract_training_samples.py
```

This will automatically download the default model from:
https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip

The model will be cached in `./model` directory for future runs.

#### Option 2: Customize model version

To use a different model version, set the `MODEL_URL` environment variable:

```bash
pip install -r requirements-fastai.txt
export MODEL_URL=https://trainedproductionmodels.blob.core.windows.net/dnnmodel/YOUR-MODEL-VERSION.zip
cd ModelTraining/src
python extract_training_samples.py
```

#### Option 3: Use pre-downloaded model

If you've already downloaded the model manually:

```bash
pip install -r requirements-fastai.txt
# Download and extract model
mkdir -p ModelTraining/model
curl -o ModelTraining/model.zip https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip
unzip ModelTraining/model.zip -d ModelTraining

# Run with pre-downloaded model (no auto-download needed)
cd ModelTraining/src
export MODEL_AUTO_DOWNLOAD=false
export MODEL_PATH=../model
python extract_training_samples.py
```

#### Option 4: Use dummy model (for testing)

For testing without FastAI dependencies or model download:

```bash
cd ModelTraining/src
export MODEL_TYPE=dummy
python extract_training_samples.py
```

The dummy model will generate mock predictions suitable for testing the timestamp correction logic.

### Model Configuration

The model behavior can be configured using environment variables:

- `MODEL_TYPE`: Type of model to use (`dummy` or `fastai`, default: `fastai`)
- `MODEL_PATH`: Path to the model directory (default: `./model`)
- `MODEL_AUTO_DOWNLOAD`: Whether to auto-download the model if not found (default: `true` for fastai, `false` for dummy)
- `MODEL_URL`: Custom URL for model zip file (optional, default: `https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip`)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `boto3`: For accessing S3 audio files
- `ffmpeg-python`: For audio processing
- `librosa`: For audio analysis
- `m3u8`: For HLS stream parsing
- `pytz`: For timezone handling

### Optional: FastAI Model Support

For FastAI model support (when using `MODEL_TYPE=fastai`), install additional dependencies:

```bash
pip install -r requirements-fastai.txt
```

This installs:
- `fastai`
- `torch`
- `torchaudio`
- `soundfile`

Note: The script works with the dummy model by default without these dependencies.

## Helper Scripts

- **spectrogram_visualizer.py**: Adapted from [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py)
- **model_inference.py**: Provides model inference interface for scoring audio samples

## Architecture

The timestamp correction implementation follows the architecture described in the [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem):

- Uses `DateRangeHLSStream` approach to download audio from specific time ranges
- Downloads from Orcasound S3 buckets: `s3-us-west-2.amazonaws.com/audio-orcasound-net/`
- Processes HLS streams with m3u8 playlists
- Uses FFmpeg for audio format conversion
- Returns `local_confidences` array with scores for each 2-second segment

## Example Configuration

Similar to [aifororcas-livesystem config files](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/config/Test/Positive/FastAI_DateRangeHLS_AndrewsBay.yml):

```yaml
model_type: "FastAI"
model_local_threshold: 0.5
model_global_threshold: 3
model_path: "./model"
model_name: "model.pkl"
```
