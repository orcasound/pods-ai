# Helper Scripts

- **spectrogram_visualizer.py**: Adapted from [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py)
- **model_inference.py**: Provides model inference interface for scoring audio samples
- **orcasite_feeds.py**: Lightweight module providing the `OrcasiteFeed` dataclass and
  `get_orcasite_feeds()` helper. Depends only on `requests` — no `azure-cosmos` — so
  scripts that only need the feeds REST API (e.g. `add_samples.py`) can import it
  without pulling in the full `make_csv` dependency tree.
- [**add_samples.py**](docs/add-samples.md): Splits a WAV file into 3-second segments (2-second hop), saves each
  segment to a `new/` directory using the standard filename convention, and prints the
  predicted class for each segment. Useful for labelling new recordings and adding them
  to the training set.

