# download-wavs.py

Uses `output/csv/training_3s_samples.csv`, `output/csv/testing_60s_samples.csv`,
and `output/csv/dclde_60s_samples.csv` to download wav files. It keeps
`output/wav/humpback/signals-humpback_*.wav` segments from the `signals-humpback`
submodule (those rows are not in the CSVs).

```
usage: download_wavs.py [-h] [--validate-only] [--training-csv-path TRAINING_CSV_PATH]
                        [--testing-csv-path TESTING_CSV_PATH] [--dclde-manifest DCLDE_MANIFEST]
                        [--dclde-wav-root DCLDE_WAV_ROOT]

Download PODS-AI training, testing, and optional DCLDE Orcasound WAVs. Run from the repository root as python
src/download_wavs.py.

options:
  -h, --help            show this help message and exit
  --validate-only
  --training-csv-path TRAINING_CSV_PATH
                        Training manifest (default: output/csv/training_3s_samples.csv).
  --testing-csv-path TESTING_CSV_PATH
                        Testing manifest (default: output/csv/testing_60s_samples.csv).
  --dclde-manifest DCLDE_MANIFEST
                        DCLDE Orcasound manifest (default: output/csv/dclde_60s_samples.csv). If absent, DCLDE
                        downloading is skipped.
  --dclde-wav-root DCLDE_WAV_ROOT
                        Override DCLDE WAV output root (default: output/testing-wav, shared with Orcasound testing
                        clips).
```
