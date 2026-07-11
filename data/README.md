# Data

This project uses only the small toy datasets that ship with scikit-learn or are
generated synthetically at runtime:

- `load_diabetes` and `load_iris` (bundled with scikit-learn, no download needed)
- `load_digits` (bundled with scikit-learn)
- `make_blobs` and `make_moons` (generated in memory)

Nothing needs to be placed in this directory. It exists only to keep the standard
project layout, and it is gitignored except for this file.

Run `python scripts/download_data.py` to verify that all datasets are available in
your environment.
