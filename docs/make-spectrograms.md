# make_spectrograms.py

Generate PNG spectrograms for WAV files under `output/wav`. The script searches
recursively and saves each spectrogram under `output/png` in the matching
relative subdirectory. Existing spectrograms are skipped.

Run from the repository root:

```bash
python src/make_spectrograms.py
```

Display command-line usage:

```bash
python src/make_spectrograms.py --help
```
