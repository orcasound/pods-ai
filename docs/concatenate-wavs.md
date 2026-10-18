# concatenate_wavs.py

Concatenate all WAV files in a directory into a single WAV file with a short beep
between clips.  The primary use case is to take a set of 3-second WAV files for
training a given category, and put them into one long file that can be played
to verify that all samples do sound like a given category, typically to check
that the node name and timestamps were correct when the WAVs were downloaded.

```bash
cd src
python concatenate_wavs.py <directory> [--output OUTPUT_FILENAME]
```

Example:

```bash
cd src
python concatenate_wavs.py ../output/wav/resident --output concatenated.wav
