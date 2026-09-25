# CSV Manifests

The ongoing sample CSVs are:

- `output/csv/training_3s_samples.csv`
- `output/csv/testing_60s_samples.csv`
- `output/csv/dclde_60s_samples.csv`

These files can be updated manually by editing rows directly, or via scripts (for example
`add_samples.py`, `process_false_positives.py`, and `process_false_negatives.py`).

The columns are as follows:

* Category: The model category (e.g., resident, transient, humpback, etc.) according to a trained human moderator.
* NodeName: The unique name of the hydrophone (e.g., rpi_andrews_bay)
* StartTimestamp: The start time, in the time zone of the node, of the audio sample, in the form YYYY_MM_DD_HH_MM_SS_TZ (e.g., 2026_07_09_07_13_36_PST).
* URI: A URI where a human can listen to the audio.  The URI must contain a variation of the non-"rpi_" portion o the NodeName, either with underscores ("andrews_bay") or hyphens ("andrews-bay") somewhere in the URI path.
* Description: Comments provided by a trained human moderator.
* Notes: fp_machine_only if the sample came from a set of (typically false positives) AI detections, tp_human_only if the sample came from a set of human positives (thus false negatives for AI models), or tp_both if the sample was correctly detected by both humans and AI.
* Confidence: The confidence of an AI model at the time.  A value of 100 can be used in the absence of any other information.

Only the first three fields are used by scripts; the others are for ease of human use of the CSV files.

