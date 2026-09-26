# compare_models.py

Evaluate and compare fastai, orcahello, podsai (AST), and oldpodsai (Wav2Vec2) models on the same test set of
60-second audio samples.  Loads the test set directly from `output/csv/testing_60s_samples.csv`, then runs each enabled model on the
corresponding WAV files under `output/testing-wav/`
(downloaded by `download_wavs.py`), and reports a summary table
with correct identifications, whale-class F1, per-whale-class false positive/false negative rates,
and average prediction time.

Evaluation uses model-specific correctness plus per-whale-class error counts:
- **Correct** – for `fastai` and `orcahello`, model predicted "resident" (SRKW) when the label is
  `resident`, or anything other than `resident` when the label is not `resident`; for
  `oldpodsai` and `podsai`, a whale category is correct when it is included in
  `global_prediction_labels`; non-whale categories use the primary
  `global_prediction_label`. Older model results use that primary-label fallback.
- **F1** – macro F1 over the whale classes `humpback`, `resident`, and `transient` that are
  present in the evaluated samples.
- **R/T/H false positive** – model predicted `resident`, `transient`, or `humpback`
  when the correct label was a different class.
- **R/T/H false negative** – the correct label was `resident`, `transient`, or `humpback`,
  but the model predicted a different class. Because `fastai` and `orcahello` are binary
  resident-vs-other models, their transient/humpback FP% values stay at `0.0%` and their
  transient/humpback FN% values are `100.0%` whenever those classes are present.
- For multi-label PODS-AI output, per-class F1 and FP/FN counts use every emitted global
  label; the displayed confusion matrix keeps the primary label for backwards-compatible
  tabular output.
- `compare_models.py` evaluates end-to-end 60-second WAV inference from `output/testing-wav`, so
  its results will differ from the training workflow's held-out evaluation metrics, which score the
  model directly on the trainer's test split.

```
usage: python compare_models.py [--testing-csv PATH] [--max-samples N]
                                [--wav-dir PATH] [--models MODEL_LIST]
                                [--fastai-model-path PATH]
                                [--orcahello-model-path PATH]
                                [--podsai-model-path PATH]
                                [--category CATEGORY]
```

| Argument | Description |
|---|---|
| `--testing-csv` | Path to `testing_60s_samples.csv` (default: `output/csv/testing_60s_samples.csv`) |
| `--max-samples` | Maximum number of test samples to process. If not specified, all samples are processed |
| `--wav-dir` | Root directory of testing WAV files (default: `output/testing-wav`) |
| `--models` | Comma-separated list of models to evaluate (default: `fastai,orcahello,podsai,oldpodsai`) |
| `--fastai-model-path` | Path to FastAI model directory. Defaults to `model` when not specified |
| `--orcahello-model-path` | HuggingFace Hub ID or path for OrcaHello model. Defaults to `orcasound/orcahello-srkw-detector-v1` when not specified |
| `--podsai-model-path` | Path or Hub ID for PODS-AI model. Used by both `podsai` (AST) and `oldpodsai` (Wav2Vec2). Defaults to `davethaler/whale-call-detector` when not specified |
| `--category` | Only evaluate samples from this category (e.g. `resident`, `humpback`, `water`). If not specified, all categories are evaluated |

**Example — compare all four models**

```bash
python src/compare_models.py \
    --models fastai,orcahello,podsai,oldpodsai \
    --fastai-model-path model \
    --podsai-model-path /path/to/podsai-model
```

Example output layout (actual metric values vary with the evaluated dataset):
```
Loaded 144 test samples from output\csv\testing_60s_samples.csv
WAV directory: output/testing-wav
Models to evaluate: fastai, orcahello, podsai, oldpodsai

  ...

================================================================================================================
Model Comparison Summary
================================================================================================================
Model           Evaluated   Correct  Accuracy      F1    RFP%    RFN%    TFP%    TFN%    HFP%    HFN%   Avg Time
----------------------------------------------------------------------------------------------------------------
fastai                144        58     40.3%   0.116   62.0%   55.8%    0.0%  100.0%    0.0%  100.0%     13.50s
orcahello             144        36     25.0%   0.119   93.5%   42.3%    0.0%  100.0%    0.0%  100.0%      4.89s
oldpodsai             144        72     50.0%   0.462   26.1%   46.2%   14.9%   63.3%   14.3%   38.9%      4.84s
podsai                144        68     47.2%   0.507   16.3%   42.3%    4.4%   36.7%    0.0%   88.9%      7.52s
================================================================================================================

Definitions:
  Accuracy     = Correct / Evaluated
  Correct      = fastai/orcahello: resident vs other; oldpodsai/podsai: category in prediction set
  F1           = macro F1 over humpback, resident, and transient classes that are present
  [R|T|H]FP%   = among non-[R|T|H] samples, fraction predicted as that class
  [R|T|H]FN%   = among actual samples of that class, fraction predicted as another class
  Avg Time     = average time spent in model predict() per 60-second WAV file
  Note         = compares end-to-end 60-second inference on testing_60s_samples.csv

Confusion Matrix for fastai (rows=actual, cols=predicted):
                other  resident     total
       bird         6         4        10
      human         6         4        10
   humpback        10         8        18
     jingle         7         0         7
   resident        29        23        52
  transient         4        26        30
     vessel         2         5         7
      water         0        10        10

Confusion Matrix for orcahello (rows=actual, cols=predicted):
                other  resident     total
       bird         2         8        10
      human         0        10        10
   humpback         4        14        18
     jingle         0         7         7
   resident        22        30        52
  transient         0        30        30
     vessel         0         7         7
      water         0        10        10

Confusion Matrix for oldpodsai (rows=actual, cols=predicted):
                 human   humpback     jingle   resident  transient     vessel      water      total
       bird          0          0          1          8          0          1          0         10
      human          7          1          0          1          1          0          0         10
   humpback          1         11          0          4          2          0          0         18
     jingle          0          7          0          0          0          0          0          7
   resident          5          1          0         28         14          1          3         52
  transient          1          9          0          9         11          0          0         30
     vessel          0          0          0          2          0          5          0          7
      water          0          0          0          0          0          0         10         10

Confusion Matrix for podsai (rows=actual, cols=predicted):
                 human   humpback   resident  transient     vessel      water      total
       bird          0          0          3          0          7          0         10
      human          9          0          0          1          0          0         10
   humpback          0          2          3          1         11          1         18
     jingle          0          0          0          0          7          0          7
   resident          1          0         30          3         16          2         52
  transient          0          0          9         19          2          0         30
     vessel          0          0          0          0          7          0          7
      water          0          0          0          0          9          1         10
```

Note: the potential of the podsai model is greater than shown above.  The same version used in the
podsai matrix above showed the above when trained:

```
============================================================
DETAILED EVALUATION METRICS
============================================================
Dataset: trainer test split from output/wav (80/20 split of training samples).

Class Distribution:
  water        - True:   9, Predicted:  12
  resident     - True:  23, Predicted:  25
  transient    - True:  12, Predicted:  10
  humpback     - True:  12, Predicted:  12
  vessel       - True:  11, Predicted:   9
  jingle       - True:   6, Predicted:   4
  human        - True:   9, Predicted:  10
  bird         - True:  10, Predicted:  10

Per-Class Performance:
Class        Precision    Recall       F1          
------------------------------------------------
water        0.750        1.000        0.857       
resident     0.920        1.000        0.958       
transient    1.000        0.833        0.909       
humpback     1.000        1.000        1.000       
vessel       0.889        0.727        0.800       
jingle       1.000        0.667        0.800       
human        0.900        1.000        0.947       
bird         1.000        1.000        1.000       

Confusion Matrix (rows=true, cols=predicted):
                 water  resident  transien  humpback    vessel    jingle     human      bird
       water         9         0         0         0         0         0         0         0
    resident         0        23         0         0         0         0         0         0
   transient         0         2        10         0         0         0         0         0
    humpback         0         0         0        12         0         0         0         0
      vessel         3         0         0         0         8         0         0         0
      jingle         0         0         0         0         1         4         1         0
       human         0         0         0         0         0         0         9         0
        bird         0         0         0         0         0         0         0        10
============================================================
```

**Example - compare only fastai and orcahello**

```bash
python src/compare_models.py --models fastai,orcahello --fastai-model-path model
```

**Example - limit to 10 test samples**

```bash
python src/compare_models.py --max-samples 10 --fastai-model-path model
```

**Example - evaluate only resident samples**

```bash
python src/compare_models.py --category resident --fastai-model-path model
```
