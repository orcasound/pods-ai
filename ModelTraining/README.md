# ModelTraining

This directory contains scripts for preparing training data for orca detection models.

## Overview

The `ModelTraining/src` directory has the following scripts for different steps meant to be run in the order listed:

1. **make_csv.py**: Create a CSV file (`output/csv/detections.csv`) with a set of detections.
   The CSV file has the following columns: Category, NodeName, Timestamp, URI, Description, and Notes.
2. **process_humpback_wavs.py**: Process files from the humpback submodule into the humpback subdirectory under `output/wav`.
   A custom segment duration can be specified with `--duration _seconds_` (default: 3 seconds).
3. **extract_training_samples.py**: Use an input CSV file (`output/csv/detections.csv` by default)
   to create `output/csv/training_samples.csv` and `output/csv/testing_samples.csv`. An alternate input filename can be specified with
   `--input _filename_`. A custom segment duration can be specified with `--duration _seconds_` (default: 3 seconds).
   - For `tp_human_only` detections, runs model inference on preceding 60 seconds to find correct timestamp
   - For other detections, subtracts the segment duration from the timestamp
   - `testing_samples.csv` uses detections-format rows, excludes training rows, and includes up to 10 eligible samples per category
4. **download_wavs.py**: Use `output/csv/training_samples.csv` and `output/csv/testing_samples.csv` to download wav files
   - Training samples are written to subdirectories under `output/wav`
   - Testing samples are written to subdirectories under `output/testing-wav`
   - For testing samples, all rows download 60-second wav files (`tp_human_only` uses the row timestamp; others are centered on the row timestamp)
5. **make_spectrograms.py**: Create a png file for each wav file in a subdirectory of `output/png`
6. **train_huggingface_model.py**: A script to train a HuggingFace model on the generated training samples.

## Model-Based Timestamp Correction for tp_human_only

The `extract_training_samples.py` script now implements intelligent timestamp correction for `tp_human_only` detections:

### How it Works

For detections marked as `tp_human_only`:
1. Downloads 60 seconds of audio preceding the detection timestamp
2. Runs model inference to score each segment
3. Finds the highest scoring segment
4. Adjusts the timestamp based on the offset of the highest scoring segment

This matches the behavior described in the issue and follows the approach used in [aifororcas-livesystem's LiveInferenceOrchestratorV1.py](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/LiveInferenceOrchestratorV1.py).

### Using the FastAI Model

By default, `extract_training_samples.py` uses the FastAI model with automatic download enabled.

#### Option 1: Default behavior (recommended)

Install dependencies and run the script:

```bash
pip install -r requirements.txt

# For Python 3.11+, apply compatibility patch
bash patch_fastai_audio.sh

cd src
python extract_training_samples.py
```

This will automatically download the default model from:
https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip

The model will be cached in `./model` directory for future runs.

**Note**: Python 3.11+ requires a patch to fastai_audio for compatibility. The `patch_fastai_audio.sh` script applies this fix automatically.

#### Option 2: Customize model version

To use a different model version, set the `MODEL_URL` environment variable:

```bash
pip install -r requirements.txt
bash patch_fastai_audio.sh  # For Python 3.11+
export MODEL_URL=https://trainedproductionmodels.blob.core.windows.net/dnnmodel/YOUR-MODEL-VERSION.zip
cd src
python extract_training_samples.py
```

#### Option 3: Use pre-downloaded model

If you've already downloaded the model manually:

```bash
pip install -r requirements.txt
bash patch_fastai_audio.sh  # For Python 3.11+

# Download and extract model
mkdir -p model
curl -o model.zip https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip
unzip model.zip -d .

# Run with pre-downloaded model (no auto-download needed)
cd src
export MODEL_AUTO_DOWNLOAD=false
export MODEL_PATH=../model
python extract_training_samples.py
```

#### Option 4: Use dummy model (for testing)

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
- `librosa>=0.10.0`: For audio analysis
- `m3u8`: For HLS stream parsing
- `pytz`: For timezone handling
- `fastai==1.0.61`: For FastAI model support
- `torch>=2.1.0`: PyTorch deep learning framework
- `torchvision>=0.16.0`: Computer vision models and utilities
- `torchaudio>=2.1.0`: Audio processing for PyTorch
- `soundfile`: Audio file I/O
- `fastai_audio`: FastAI audio extensions (from GitHub)
- `pandas`, `pydub`: Data processing and audio manipulation

## Helper Scripts

- **spectrogram_visualizer.py**: Adapted from [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py)
- **model_inference.py**: Provides model inference interface for scoring audio samples
- **run_inference.py**: Runs a model on a wav file and prints the global prediction,
  confidence, and per-class probabilities.
- **compare_models.py**: Evaluates and compares fastai, orcahello, and huggingface models
  on the test set generated by `extract_training_samples.py` and downloaded by `download_wavs.py`.
  Reports correct identifications, false positives, and false negatives for each model.
- **get_best_timestamp.py**: Given a node slug and a detection timestamp, runs
  `process_sample()` and prints the corrected URI with the best timestamp.

### run_inference.py

Run model inference on a wav file and display the global prediction, confidence score,
and per-class probabilities.  For HuggingFace models the per-class probability is the
mean of all `local_confidence` values (from windows predicting that class) that exceed
the model's threshold — the same statistic used for `global_confidence`.  For the FastAI
binary model, `whale = global_confidence` and `other = 1 - global_confidence`.

```
usage: python run_inference.py <wav_file> [--model {huggingface,fastai,orcahello}] [--model-path PATH]
```

| Argument | Description |
|---|---|
| `wav_file` | Path to the wav file to score |
| `--model` | Model type: `huggingface` (default), `fastai`, or `orcahello` |
| `--model-path` | Path to model directory or HuggingFace Hub model ID. Required for `huggingface`; defaults to `./model` for `fastai`; defaults to `orcasound/orcahello-srkw-detector-v1` for `orcahello` |

**Example — HuggingFace model**

```bash
cd src
python run_inference.py sample.wav --model huggingface --model-path /path/to/hf-model
```

Output:
```
Model type: huggingface
Global prediction: resident (confidence: 0.7000)

Per-class probabilities:
  humpback: 0.0000
  human: 0.0000
  jingle: 0.0000
  resident: 0.7000
  transient: 0.0000
  vessel: 0.0000
  water: 0.0000
```

**Example — FastAI model**

```bash
cd src
python run_inference.py sample.wav --model fastai --model-path ../model
```

Output:
```
Model type: fastai
Global prediction: whale (confidence: 0.7500)

Per-class probabilities:
  other: 0.2500
  whale: 0.7500
```

**Example — OrcaHello SRKW Detector**

Uses the [`orcasound/orcahello-srkw-detector-v1`](https://huggingface.co/orcasound/orcahello-srkw-detector-v1)
model from HuggingFace Hub. This is a binary SRKW (Southern Resident Killer Whale) detector
based on the new OrcaHello inference pipeline (ResNet50 + mel spectrograms, no fastai_audio dependency).

The model implementation is loaded from the `orcasound/orcahello` submodule. Initialize it first:

```bash
git submodule update --init external/orcahello
```

Then run inference:

```bash
cd src
python run_inference.py sample.wav --model orcahello
```

Output:
```
Model type: orcahello
Global prediction: whale (confidence: 0.8000)

Per-class probabilities:
  other: 0.2000
  whale: 0.8000
```

You can compare results between the FastAI model and the OrcaHello model by running both on the
same file and comparing the output.

### compare_models.py

Evaluate and compare fastai, orcahello, and huggingface models on the same test set of
60-second audio samples.  Reads `output/csv/testing_samples.csv` (generated by
`extract_training_samples.py`) and the corresponding WAV files under `output/testing-wav/`
(downloaded by `download_wavs.py`), runs each enabled model, and reports a summary table
with correct identifications, false positives, and false negatives.

Evaluation uses a binary resident-vs-other framing that works across all three models:
- **Correct** – model predicted "resident" (whale) when the label is "resident", or
  anything other than "resident" when the label is not "resident".
- **False positive** – model predicted "resident" when the correct label is not "resident".
- **False negative** – model predicted something other than "resident" when the label is "resident".

```
usage: python compare_models.py [--testing-csv PATH] [--wav-dir PATH]
                                [--models MODEL_LIST]
                                [--fastai-model-path PATH]
                                [--orcahello-model-path PATH]
                                [--huggingface-model-path PATH]
```

| Argument | Description |
|---|---|
| `--testing-csv` | Path to `testing_samples.csv` (default: `../output/csv/testing_samples.csv`) |
| `--wav-dir` | Root directory of testing WAV files (default: `../output/testing-wav`) |
| `--models` | Comma-separated list of models to evaluate (default: `fastai,orcahello,huggingface`) |
| `--fastai-model-path` | Path to FastAI model directory. Defaults to `./model` (the `run_inference.py` default) when not specified |
| `--orcahello-model-path` | HuggingFace Hub ID or path for OrcaHello model. Defaults to `orcasound/orcahello-srkw-detector-v1` when not specified |
| `--huggingface-model-path` | Path or Hub ID for HuggingFace model. Required when `huggingface` is in `--models` |

**Example — compare all three models**

```bash
cd src
python compare_models.py \
    --models fastai,orcahello,huggingface \
    --fastai-model-path ../model \
    --huggingface-model-path /path/to/hf-model
```

Output:
```
Loaded 62 testing samples from ../output/csv/testing_samples.csv
WAV directory: ../output/testing-wav
Models to evaluate: fastai, orcahello, huggingface

Evaluating model: fastai
  [fastai] resident/rpi_orcasound_lab/2023_08_18_00_59_53_PST: predicted='whale' → correct
  ...

======================================================================
Model Comparison Summary
======================================================================
Model           Evaluated   Correct  Accuracy     FP     FP%     FN     FN%
----------------------------------------------------------------------
fastai                 55        42    76.4%      8   14.5%      5    9.1%
orcahello              55        40    72.7%     10   18.2%      5    9.1%
huggingface            55        45    81.8%      5    9.1%      5    9.1%
======================================================================

Definitions:
  Correct      = predicted resident when expected, or non-resident when expected
  FP (false+)  = predicted resident when correct class was non-resident
  FN (false-)  = predicted non-resident when correct class was resident
```

**Example — compare only fastai and orcahello**

```bash
cd src
python compare_models.py --models fastai,orcahello --fastai-model-path ../model
```

### get_best_timestamp.py

```
usage: python get_best_timestamp.py <node_slug> <timestamp_str> [--no-model] [--duration N]
```

| Argument | Description |
|---|---|
| `node_slug` | Node URL slug, e.g. `orcasound-lab` |
| `timestamp_str` | PST timestamp, e.g. `2023_08_18_00_59_53_PST` |
| `--no-model` | Skip model inference; apply a fixed-offset correction instead |
| `--duration N` | Segment duration in seconds (default: 3) |

The script uses the same model-based timestamp correction logic as
`extract_training_samples.py` (see [Model-Based Timestamp Correction](#model-based-timestamp-correction-for-tp_human_only)
above).  The same `MODEL_TYPE`, `MODEL_PATH`, `MODEL_AUTO_DOWNLOAD`, and
`MODEL_URL` environment variables apply.

**Example**

```bash
cd src
python get_best_timestamp.py orcasound-lab 2023_08_18_00_59_53_PST
# https://live.orcasound.net/bouts/new/orcasound-lab?time=2023-08-18T07%3A59%3A50.000Z
```

## Architecture

The timestamp correction implementation follows the architecture described in the [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem):

- Uses `DateRangeHLSStream` approach to download audio from specific time ranges
- Downloads from Orcasound S3 buckets: `s3-us-west-2.amazonaws.com/audio-orcasound-net/`
- Processes HLS streams with m3u8 playlists
- Uses FFmpeg for audio format conversion
- Returns `local_confidences` array with scores for each segment

## Example Configuration

Similar to [aifororcas-livesystem config files](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/config/Test/Positive/FastAI_DateRangeHLS_AndrewsBay.yml):

```yaml
model_type: "FastAI"
model_local_threshold: 0.5
model_global_threshold: 3
model_path: "./model"
model_name: "model.pkl"
```
