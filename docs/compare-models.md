# `compare_models.py`

## Scope

`compare_models.py` evaluates model predictions for WAV files below
`output/testing-wav`. Each immediate subdirectory is treated as the expected
category. The command reports accuracy, macro F1 for whale classes,
per-class false-positive and false-negative rates, prediction time, and a
primary-label confusion matrix.

The command performs end-to-end inference on 60-second WAV files. Its results
are not the trainer's held-out split metrics.

## Command syntax

```text
python src/compare_models.py [options]
```

| Option | Description |
| --- | --- |
| `--wav-dir PATH` | Root directory of category subdirectories. Default: `output/testing-wav`. |
| `--dataset-metadata JSON_PATH` | Optional JSON sidecar. The default test set looks for `output/csv/testing_60s_samples.metadata.json`; custom WAV directories look for `dataset.json` inside the WAV directory. |
| `--dataset-manifest CSV_PATH` | Optional CSV manifest to read dataset version and provenance columns from. The default test set uses `output/csv/testing_60s_samples.csv`. |
| `--models MODEL_LIST` | Comma-separated built-in names. Include `custom` to evaluate the custom PODS-AI model. Default: `orcahello,oldpodsai,podsai`. |
| `--custom-model-path PATH_OR_HF_ID` | Local custom model file/directory, PODS-AI-compatible ONNX file, or Hugging Face repository ID. Required when `custom` is in `--models`; supplying it without `custom` adds `custom` automatically. |
| `--custom-model-revision REVISION` | Optional revision for a custom Hugging Face model. Ignored for local paths. |
| `--orcahello-model-path PATH_OR_HF_ID` | OrcaHello model path, PODS-AI-compatible ONNX file, or Hub ID. Default: `orcasound/orcahello-srkw-detector-v1`. |
| `--orcahello-model-revision REVISION` | Optional Hugging Face revision for OrcaHello. Used only when `--orcahello-model-path` is a Hub ID. |
| `--podsai-model-path PATH_OR_HF_ID` | PODS-AI model path, ONNX file, or Hub ID for `oldpodsai` and `podsai`. Default: `davethaler/whale-call-detector`. |
| `--podsai-model-revision REVISION` | Revision for the built-in `podsai` model. |
| `--max-samples N` | Evaluate at most `N` WAV files. `N` must be positive. |
| `--category NAME` | Evaluate only the named category. |
| `--orcahello-variant` | Optional configuration to compare orcahello model with different aggregation strategies. |

Built-in model names map to these inference backends:

| Name | Backend | Revision behavior |
| --- | --- | --- |
| `orcahello` | OrcaHello | `--orcahello-model-revision` |
| `oldpodsai` | PODS-AI | Pinned previous revision |
| `podsai` | PODS-AI | `--podsai-model-revision` |

Unknown names cause an error before model loading. The WAV directory must
exist and contain at least one category WAV file. Supported background
categories include `other`, `abiotic`, `water`, `vessel`, `jingle`, `human`,
and `bird`.

Each run prints `dataset_version` in its log and comparison summary. The
default is `unversioned`, so older datasets and manifests continue to run
without changes. A JSON sidecar may declare `dataset_version` (or `version`),
`git_tag`, `source_id`, and `hydrophone_site`; the CSV manifest can also include
`dataset_version` or `version`, `git_tag`, `source_id`, and `hydrophone_site`
columns. Source and site values are optional and are reported when available.

Use semantic versions in `MAJOR.MINOR.PATCH` form (optionally prefixed with
`v`). To bind a dataset version to a repository commit, create an annotated
Git tag named `TestingDataset.vMAJOR.MINOR.PATCH` on the commit containing
that dataset. For example, `TestingDataset.v1.0.0` identifies the commit
containing version `v1.0.0`. When a version is stored in JSON, include the
matching `git_tag` so evaluation reports both values. An exact dataset tag at
the checked-out commit is also detected automatically when no version is
declared. Content hashes are not part of the dataset version identifier.

Example sidecar:

```json
{
  "dataset_version": "v1.0.0",
  "git_tag": "TestingDataset.v1.0.0",
  "source_id": "orcasound-lab-2026",
  "hydrophone_site": "rpi_orcasound_lab"
}
```

## Custom model specification

The `custom` model name selects the PODS-AI backend. Its value is supplied by
`--custom-model-path`; the option accepts one local model path, a compatible
`.onnx` file, or one Hugging Face repository ID. A command that lists `custom`
without a path is rejected. A path supplied without `custom` adds `custom` to
the model list. Inline values such as `custom:./models/my-detector` are not
supported.

For ONNX inference, install the optional dependencies from
`requirements-onnx.txt`. For example, to compare the standard PODS-AI model
against an ONNX model:

```text
python src/compare_models.py --models podsai,custom --custom-model-path podsai.onnx
```

Custom models are PODS-AI-compatible multi-class models. They use the same
category-membership evaluation as `oldpodsai` and `podsai`,
including the confusion matrix, F1, and FP/FN calculations.

### Inference contract

The model artifact must be loadable by:

```python
get_model_inference(model_type="podsai", model_path=VALUE)
```

The loaded PODS-AI backend must expose a callable `predict()` method. Its
input and result contract is:

- Input audio is decoded as mono at 16 kHz.
- Analysis uses 3-second windows with a 2-second hop.
- Inference uses a confidence threshold of `0.5` by default. The default
  positive-event threshold is 2, scaled down for shorter clips.
- The Hugging Face audio-classification configuration provides `id2label` and
  `label2id` and includes at least one supported background label: `other`,
  `abiotic`, `water`, `vessel`, `jingle`, `human`, or `bird`.
- The prediction mapping contains `global_prediction_label`,
  `global_confidence`, `local_predictions`, `local_confidences`, and
  `per_class_probabilities`. `global_prediction_labels` may also be present.
- Confidence values are numeric values in the inclusive range `[0, 1]`.

An artifact that cannot be loaded by this backend or a factory result without
`predict()` is rejected before any test sample is evaluated.

### Model value resolution

1. If `VALUE` resolves to an existing file or directory, it is passed to the
   loader as a local model path. The command expands `~` before loading.
2. A missing value beginning with `./`, `../`, `~`, `/`, `\`, or a Windows
   drive letter is treated as a missing local path and rejected before loading.
3. Other missing values are passed to the Hugging Face loader as repository
   IDs. `--custom-model-revision` is forwarded for these values; it is ignored
   for local paths.

Invalid repository IDs, unavailable repositories, permission failures, and
incompatible artifacts are reported as a single custom-model loading error
that includes the loader exception type and message. A Python traceback is
not printed by the CLI.

## Evaluation definitions

- `orcahello` is binary resident-versus-other models. A sample
  is correct when resident status matches the expected category.
- `oldpodsai`, `podsai`, and `custom` use the expected category's
  membership in `global_prediction_labels`; older output falls back to
  `global_prediction_label`.
- For an `abiotic` sample, `water`, `vessel`, and `jingle` predictions also
  count as correct. Specific ground-truth categories still require a matching
  prediction. The confusion matrix and whale metrics retain the original labels.
- Macro F1 covers `resident`, `transient`, and `humpback` classes present in
  the evaluated samples, either as the correct class or the predicted class.
- `[R|T|H]FP%` is the fraction of non-class samples predicted as that class.
- `[R|T|H]FN%` is the fraction of class samples predicted as another class.
- The displayed confusion matrix uses `global_prediction_label`. Multi-label
  F1 and FP/FN calculations use every emitted global label.

## Examples

Evaluate the built-in models on all available samples:

```bash
python src/compare_models.py
```

Evaluate a local custom model:

```bash
python src/compare_models.py --models custom --custom-model-path ./models/my-detector
```

Evaluate a Hugging Face custom model with a pinned revision:

```bash
python src/compare_models.py \
    --models podsai,custom \
    --custom-model-path my-org/my-whale-detector \
    --custom-model-revision d66c75205e3d229aa5fe875cb27bc90304654fcd
```

Evaluate OrcaHello at a specific Hugging Face revision:

```bash
python src/compare_models.py \
    --models orcahello \
    --orcahello-model-revision 0123456789abcdef0123456789abcdef01234567
```

Limit evaluation to resident samples and ten files:

```bash
python src/compare_models.py --category resident --max-samples 10
```
