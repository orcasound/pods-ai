# generate_umaps.py

Generate a two-dimensional UMAP visualization from the embeddings produced by
`generate_embeddings.py`.

The script loads an embeddings CSV containing AST embedding vectors
(`embedding_0`, `embedding_1`, …), projects them into two dimensions using
UMAP (Uniform Manifold Approximation and Projection), and colors each point
according to a selected label column.

Supported label types are:

- `ground_truth_label`
- `predicted_label`
- `global_prediction_label`

This makes it possible to visually compare:

- Ground-truth class separation
- Local window predictions
- Global clip predictions

UMAP visualizations are useful for evaluating how well the learned AST embedding
space separates whale vocalizations from background sounds, identifying classes
that overlap in embedding space, and diagnosing distribution shifts between
training, validation, and real-world datasets.

```
usage: python generate_umaps.py
       --embeddings_csv PATH
       --label_type {ground_truth_label,predicted_label,global_prediction_label}
       --output_file OUTPUT.png
```

| Argument | Description |
|---|---|
| `--embeddings_csv` | CSV file produced by `generate_embeddings.py` containing embedding vectors and labels |
| `--label_type` | Column used to color the UMAP (`ground_truth_label`, `predicted_label`, or `global_prediction_label`) |
| `--output_file` | Output PNG filename |

**Example — visualize ground-truth labels**

```bash
cd src
python generate_umaps.py \
    --embeddings_csv ../output/csv/test_embeddings.csv \
    --label_type ground_truth_label \
    --output_file ground_truth_umap.png
```
