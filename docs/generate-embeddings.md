# generate_embeddings.py

Generate AST embeddings for a collection of WAV files and save them to a CSV file for
visualization and analysis. The script loads samples from
`output/csv/testing_60s_samples.csv`, runs PODS-AI AST inference on each corresponding
60-second WAV file, extracts the CLS-token embedding from the Audio Spectrogram
Transformer (AST), and writes one row per analyzed segment.

The output CSV contains:

- Original sample metadata (category, node name, timestamp, URI, notes, etc.)
- Segment timing information
- Local and global model predictions
- Confidence scores
- AST embedding dimensions (`embedding_0`, `embedding_1`, ...)

These embeddings can be used with dimensionality-reduction techniques such as UMAP
or t-SNE to visualize how the AST model organizes different whale call classes and
background sounds in embedding space.

Inference uses the PODS-AI AST model (`davethaler/whale-call-detector`) by default,
but a different model or local checkpoint can be specified with `--model-path`.

```bash
usage: python generate_embeddings.py
       [--testing-csv PATH]
       [--wav-dir PATH]
       [--output-csv PATH]
       [--model-path PATH]
       [--model-revision REVISION]
       [--category CATEGORY]
       [--max-samples N]
```
| Argument | Description |
|---|---|
| `--testing-csv` | Path to `testing_60s_samples.csv` (default: `output/csv/testing_60s_samples.csv`) |
| `--wav-dir` | Root directory containing downloaded testing WAV files (default: `output/testing-wav`) |
| `--output-csv` | Output CSV file containing embeddings and metadata (default: `output/csv/embeddings.csv`) |
| `--model-path` | HuggingFace Hub model ID or local PODS-AI model directory (default: `davethaler/whale-call-detector`) |
| `--model-revision` | Specific model revision to load from HuggingFace. Defaults to the pinned AST model revision used by the repository. |
| `--category` | Only process samples from the specified category (e.g. `resident`, `transient`, `humpback`, `water`) |
| `--max-samples` | Maximum number of samples to process. If omitted, all matching samples are processed. |  
