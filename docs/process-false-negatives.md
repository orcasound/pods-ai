# process_false_negatives.py

Re-checks confirmed OrcaHello detections by
downloading the 60-second WAV, re-running PODS-AI and OrcaHello segment inference,
and appending segments where OrcaHello predicts resident but PODS-AI does not to
`output/csv/new_manual_samples.csv` with corrected class `resident`. Supports
`--category CATEGORY` to process only detections whose PODS-AI predicted category
matches the provided value.

```
usage: process_false_negatives.py [-h] [--feed FEED] [--start YYYY_MM_DD_HH_MM_SS_PST]
                                  [--end YYYY_MM_DD_HH_MM_SS_PST|now] [--manual-samples-csv MANUAL_SAMPLES_CSV]
                                  [--output-dir OUTPUT_DIR] [--model-path MODEL_PATH]
                                  [--orcahello-model-path ORCAHELLO_MODEL_PATH] [--detections-csv DETECTIONS_CSV]
                                  [--category CATEGORY]

Process confirmed OrcaHello detections, find 60-second false negatives where PODS-AI misses resident calls, and append
corrected resident sub-segments to new_manual_samples.csv.

options:
  -h, --help            show this help message and exit
  --feed FEED           Process only this feed (by node_name, e.g., rpi_sunset_bay).
  --start YYYY_MM_DD_HH_MM_SS_PST
                        Include only detections with timestamp >= this value.
  --end YYYY_MM_DD_HH_MM_SS_PST|now
                        Include only detections with timestamp <= this value. Use 'now' to remove the upper bound.
  --manual-samples-csv MANUAL_SAMPLES_CSV
                        Path to new_manual_samples.csv.
  --output-dir OUTPUT_DIR
                        Directory where add_samples.py should write segment WAV files.
  --model-path MODEL_PATH
                        PODS-AI model path or HuggingFace model ID.
  --orcahello-model-path ORCAHELLO_MODEL_PATH
                        OrcaHello model path or HuggingFace model ID.
  --detections-csv DETECTIONS_CSV
                        Path to detections.csv for add_samples.py metadata lookups.
  --category CATEGORY   Process only detections whose PODS-AI predicted category matches this value.
```
