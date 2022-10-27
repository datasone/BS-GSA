# BS-GSA: Speech Emotion Recognition via Blend-Sample Empowered Global & Spectral Attentive Paradigm

## Replicating our results

- Download our pre-extracted feature file (tracked by git LFS) from git repository.
  - The file is split to zip files to bypass GitHub's 2GB file limit, so the `partitioned_features` file will need to be decompressed
- Create a experiment directory for storing log, data and models
- Run `train_test.sh` in the experiment directory
  - `../train_test.sh <FEATURE_FILE>`

---

- For more usage, see full code for model structure, data processing and training.
