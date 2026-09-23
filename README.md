# Python_RINTD
Fixed forgetting factor implementation.

20/09/2026 
- The current update provides a phase 1 folder that allows you to build a spectral library using a HDF5 data cube and the spectral library.
- A spectral library has already been build for convenience and it uses the USGS spectral library find it labelled - sam_library.npz
- The SAM code is provided within phase one but not necessary for Random Forest it will be our fall back when RF fails but i don't have the WIFI to test it on more data cubes basically it works on spectral curves that are well defined

- In case you want t build your own spectral library similar to this one you will need a HDF5 data cube and ASCII wavelength files it can probably be changed to use a set library format but that is your baby. You want to copy this one then here are the sources:
  USGS library -https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d
  Spectral data cube - https://zenodo.org/records/1476495, just unzip one

Fun fact you can swap out one of the two please note the spectral data cube is limited to the input band size basically you can change the width and height sizing but not the band size. And if you just want to use the spectral library i have build just use  sam_library.npz
- 

- Okay now to explain these phase 1 scripts will allow you for now to download a dataset and create labels and classes. Further implementations will allow you to then create enough synthetic data to create enough labels for Random Forest training for classification. All of this is done passively so for KK **Run these in order, in the same folder, and send back manifest.csv, manifest_summary.json, and rejected.csv** So the order is download, extract and manifest i will need the files produced, to move to phase two, which will be the creation of synthetic data and labels for ML.

- **Here are the scripts for writing the run scripts in terminal**
- python phase1_download.py --out data/zips
- python phase1_extract.py --zips data/zips --out data/raw
- python phase1_manifest.py --raw data/raw --library sam_library.npz --out manifest.csv --rejected rejected.csv

**Message for Deepseek we are trying to build a pipeline to download a dataset check for validity then build a synthetic spectral curves for RF classification. Help with any errors that come up in terminal**


## Phase 1: Dataset Preparation

Chat's change on phase1_download: Your original code could resume a failed file the next time download_one() was called, but a mid-download connection failure caused download_one() to immediately return "fail"; I changed it so that the same function automatically retries the same file, recalculates the .part size, sends a new HTTP Range request, and continues downloading from the exact point where the connection broke.


Phase 1 downloads, extracts and validates hyperspectral data cubes in preparation for synthetic spectral data generation and Random Forest (RF) training.

### Files and Deliverables

The `Phase_1/results/` folder contains the three deliverables:

* `manifest.csv` – Validated samples, mineral labels, HDF5 file paths, spectral band counts and library compatibility.
* `manifest_summary.json` – Summary of valid measurements, rejected samples, mineral distributions and class balance.
* `rejected.csv` – Samples that failed validation, including reasons for rejection.

The existing `sam_library.npz` contains a pre-built USGS mineral spectral library, so rebuilding it is not required.

### Reproducing Phase 1

The raw dataset is not included in this repository due to its size. It is downloaded separately from [Zenodo](https://zenodo.org/records/1476495).

From the `Phase_1` directory, install the dependencies and execute the scripts in order:

```bash
pip install requests numpy h5py

python phase1_download.py --out data/zips

python phase1_extract.py --zips data/zips --out data/raw

python phase1_manifest.py --raw data/raw --library sam_library.npz --out results/manifest.csv --rejected results/rejected.csv
```

**Note:** The manifest's HDF5 paths refer to the local `Phase_1/data/` directory. Team members must download and extract the original dataset to access these files.

The generated `manifest_summary.json` is saved in the current working directory and should be moved to `Phase_1/results/`.

### Next Phase

The validated data and mineral labels will be used in Phase 2 to generate synthetic spectral curves and labelled training data for Random Forest mineral classification.


