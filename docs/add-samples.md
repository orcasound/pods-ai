# add_samples.py

Split a WAV recording into 3-second segments (with a 2-second hop — the same settings
used by `run_inference.py`), save each segment to a `new/` directory using the standard
filename convention, and print the predicted class for each segment.  The timestamp
encoded in each filename reflects the **actual start time** of that sample inside the
original recording.

Output files follow the same naming convention as `output/wav/humpback/` etc.:

```
{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
```

Inference always uses the **PODS-AI (podsai)** model type.  The default model is
`davethaler/whale-call-detector` on HuggingFace Hub; override with `--model-path`.

If `--node-name` and `--timestamp` are omitted, the script infers them from the input
filename.  The filename must follow the same convention:
`{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav`
(e.g. `rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav` → node `rpi_orcasound_lab`,
timestamp `2025_12_17_22_34_03_PST`).

After reviewing the predictions you can move the segments into the appropriate
`output/wav/<category>/` directory to add them to the training set.

```
usage: python add_samples.py <wav_file> [--node-name NAME] [--timestamp TIMESTAMP]
                             [--output-dir DIR] [--model-path PATH] [--uri URI]
```

| Argument | Description |
|---|---|
| `wav_file` | Path to the input WAV file to segment |
| `--node-name` | Hydrophone node name (e.g. `rpi_orcasound_lab`). Underscores are replaced with hyphens in output filenames. **Inferred from the input filename if omitted.** |
| `--timestamp` | PST timestamp of the **start** of the recording (e.g. `2025_01_15_12_30_00_PST`). **Inferred from the input filename if omitted.** |
| `--output-dir` | Directory to save segments (default: `new`) |
| `--model-path` | HuggingFace Hub model ID or path to a local podsai model directory (default: `davethaler/whale-call-detector`) |
| `--uri` | Optional custom URI to use for all segments. If provided, all output rows will use this URI instead of generating one per segment. Useful when all segments come from the same detection. |

**Example — node name and timestamp inferred from filename**

```bash
cd src
python add_samples.py rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav
```

**Example — explicit node name and timestamp with custom model**

```bash
cd src
python add_samples.py /path/to/recording.wav \
    --node-name rpi_orcasound_lab \
    --timestamp 2025_01_15_12_30_00_PST \
    --model-path /path/to/local-model
```


**Example — use custom URI for all segments**

When all segments come from the same detection event, you can specify a single URI
to use for all output rows:

```bash
cd src
python add_samples.py /path/to/recording.wav \
    --node-name rpi_orcasound_lab \
    --timestamp 2025_01_15_12_30_00_PST \
    --uri "https://live.orcasound.net/bouts/new/rpi_orcasound_lab?time=2025-01-15T20%3A30%3A00.000Z"
```

Output:
```
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_04_PST.wav
...

Loading podsai model from /path/to/local-model...

Segment predictions:
  rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav: water
  rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav: resident
  rpi-orcasound-lab_2025_01_15_12_30_04_PST.wav: resident
  ...
```
