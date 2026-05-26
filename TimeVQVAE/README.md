# TimeVQVAE Wind Simulator

This repository contains a two-stage TimeVQVAE pipeline for stochastic wind time-series simulation. The model first learns a discrete vector-quantized representation of wind sequences, then trains a MaskGIT-style transformer prior to generate new wind trajectories from those learned tokens.

This implementation is modified from the TimeVQVAE model and codebase by Lee, Malacarne, and Aune, released at [ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) for the AISTATS 2023 paper "Vector Quantized Time Series Generation with a Bidirectional Prior Model."

The repository includes prepared wind datasets, training scripts, pretrained checkpoints, notebooks, and example synthetic outputs.

## Repository layout

```text
.
|-- configs/                 # YAML configuration files for training and sampling
|-- datasets/                # Prepared NumPy wind datasets
|   `-- data preprocess.ipynb # Notebook used to create the prepared .npz datasets
|-- encoder_decoders/        # VQ-VAE encoder and decoder modules
|-- evaluation/              # Sampling/evaluation wrapper around trained models
|-- experiments/             # PyTorch Lightning experiment modules
|-- generators/              # MaskGIT priors and sampling utilities
|-- output_synthetic/        # Example generated wind time-series outputs
|-- preprocessing/           # Dataset importers and DataLoader construction
|-- saved_models/            # Pretrained stage 1 and stage 2 checkpoints
|-- utils/                   # Shared utilities, schedulers, and helper functions
|-- vector_quantization/     # Vector quantization implementation
|-- stage1.py                # Stage 1 VQ-VAE training entry point
|-- stage2.py                # Stage 2 MaskGIT prior training entry point
|-- simulation.ipynb         # Notebook for simulation workflows
`-- timevqvae.ipynb          # Notebook for model usage and experimentation
```

## Model workflow

The training pipeline has two stages:

1. **Stage 1: VQ-VAE training**
   Learns an encoder, codebook, and decoder for wind time-series windows.

2. **Stage 2: MaskGIT prior training**
   Trains a bidirectional transformer over the VQ-VAE token sequences so new wind trajectories can be sampled iteratively.

The code supports several data/model variants:

- Standard wind-feature generation.
- Weather-state-conditioned generation.
- Consecutive two-day generation.
- Consecutive generation with weather states.
- Optional masking of weather-state tokens during prior training.

## Modifications From Original TimeVQVAE

This repository adapts the original TimeVQVAE implementation for stochastic wind simulation. Main modifications include:

- Removing the original high-frequency (HF) and low-frequency (LF) preprocessing used in TimeVQVAE.
- Supporting weather-state information as either additional features or conditioning embeddings.
- Adding consecutive multi-day wind sequence generation.
- Adding weather-state masking for conditional/consecutive generation experiments.

## Data

Prepared datasets are stored in `datasets/` as `.npz` files. The preprocessing notebook
`datasets/data preprocess.ipynb` creates these files from minute-level wind-component
CSV data.

The notebook keeps the wind vector components `Easterly` and `Northerly`, parses the
`time` column as a datetime index, converts the selected columns to `float32`, and
stores arrays under the `data` key inside each `.npz` archive.

| Mode | Preprocessed file | Shape |
| --- | --- | --- |
| Standard one-day wind features | `datasets/training_imputed_full_days.npz` | `(474, 2, 1440)` |
| Standard one-day wind features with weather states | `datasets/training_imputed_full_days_ws.npz` | `(474, 3, 1440)` |
| Consecutive two-day wind features | `datasets/training_imputed_full_2days.npz` | `(444, 2, 2880)` |
| Consecutive two-day wind features with weather states | `datasets/training_imputed_full_2days_ws.npz` | `(444, 3, 2880)` |

The dimensions are ordered as `(samples, channels, time_steps)`. For the one-day files,
each sample contains 1440 one-minute observations. For the consecutive files, each
sample contains two adjacent days concatenated along the time axis, giving 2880
one-minute observations.

### Preprocessing details

The standard one-day dataset is built from `training_474.csv` by reshaping the selected
wind components to `(474, 2, 1440)`.

The consecutive two-day dataset is built from `training_imputed.csv` by reshaping 483
daily records into 23 years with 21 days per year, concatenating adjacent days, and
dropping any two-day windows that contain missing values. This reduces the dataset from
460 candidate windows to 444 clean samples. The missing-value filter covers days with
unimputed minutes in 2001, 2002, 2005, 2011, and 2018.

Weather-state datasets are built from `training_weather_state_10min.csv`. For the
one-day weather-state dataset, timestamps are aligned to `training_imputed_full_days.csv`
before reshaping. Weather states are encoded from sign strings into integer binary
codes:

```text
(+, -, +, +) -> 1011 binary -> 11
```

The same one-day and consecutive two-day reshaping steps are then applied, producing
`training_imputed_full_days_ws.npz` and `training_imputed_full_2days_ws.npz`.

The notebook also contains an exploratory `marginal` wind-speed calculation, but that
column is not part of the prepared training arrays.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The training scripts use PyTorch, PyTorch Lightning, and Weights & Biases. If you do not want online W&B logging, set:

```bash
export WANDB_MODE=offline
```

On Windows PowerShell:

```powershell
$env:WANDB_MODE = "offline"
```

## Configuration

Default hyperparameters are defined in `configs/config.yaml`.

The alternate `configs/config_consecutive_wsm.yaml` increases the MaskGIT hidden dimension for consecutive weather-state-masked training.

Important settings include:

- `trainer_params.max_steps.stage1`: number of VQ-VAE training steps.
- `trainer_params.max_steps.stage2`: number of MaskGIT prior training steps.
- `dataset.batch_sizes.stage1`: stage 1 batch size.
- `dataset.batch_sizes.stage2`: stage 2 batch size.
- `VQ-VAE.codebook_sizes`: number of discrete codes.
- `MaskGIT.T`: number of iterative decoding steps.
- `MaskGIT.choice_temperatures`: sampling diversity temperature.
- `MaskGIT.connect_period`: connection period used for consecutive sampling.

## Training

### Stage 1: train the VQ-VAE

```bash
python stage1.py --config configs/config.yaml --dataset_name Wind
```

The checkpoint is saved to:

```text
saved_models/stage1-Wind.ckpt
```

You can select GPU devices with:

```bash
python stage1.py --gpu_device_ind 0
```

If CUDA is not available, the script falls back to CPU.

### Stage 2: train the MaskGIT prior

Train the standard prior:

```bash
python stage2.py --config configs/config.yaml --dataset_name_stage1 Wind --dataset_name Wind
```

The checkpoint is saved to:

```text
saved_models/stage2-Wind.ckpt
```

Train with weather states:

```bash
python stage2.py --config configs/config.yaml --dataset_name_stage1 Wind_wsm --dataset_name Wind_wsm --use_weather_states true
```

Train consecutive generation:

```bash
python stage2.py --config configs/config.yaml --dataset_name_stage1 Wind_consecutive --dataset_name Wind_consecutive --consecutive true
```

Train consecutive generation with weather-state masking:

```bash
python stage2.py --config configs/config_consecutive_wsm.yaml --dataset_name_stage1 Wind_consecutive_wsm --dataset_name Wind_consecutive_wsm --use_weather_states true --consecutive true --weather_states_mask true
```

Make sure the `dataset_name_stage1` value matches an available stage 1 checkpoint in `saved_models/`.

## Sampling and evaluation

Sampling helpers are implemented in `generators/sample.py`, and the higher-level evaluation interface is in `evaluation/evaluation.py`.

The included notebooks provide the most convenient entry points:

- `timevqvae.ipynb`
- `simulation.ipynb`

Generated examples are stored in:

```text
output_synthetic/
|-- Weather as Embeddings/
`-- Weather as Features/
```

## Pretrained checkpoints

The repository includes pretrained checkpoints in `saved_models/`, including:

- `stage1-Wind_wf.ckpt`
- `stage1-Wind_wsm.ckpt`
- `stage2-Wind_wf.ckpt`
- `stage2-Wind_wsm.ckpt`
- `stage2-Wind_consecutive_wf.ckpt`
- `stage2-Wind_consecutive_wsm.ckpt`

These can be used directly from the notebooks or through the `Evaluation` class.

## Notes for GitHub publication

Before publishing, consider adding:

- Citation information, if this repository accompanies a paper or Zenodo release.
- A smaller `requirements.txt`, if you want users to install only the packages required by this project rather than a full exported environment.
- Dataset provenance and usage restrictions, if the included `.npz` files come from external sources.

## Citation

This repository builds on TimeVQVAE paper:

```bibtex
@InProceedings{pmlr-v206-lee23d,
  title = {Vector Quantized Time Series Generation with a Bidirectional Prior Model},
  author = {Lee, Daesoo and Malacarne, Sara and Aune, Erlend},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = {7665--7693},
  year = {2023},
  editor = {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = {206},
  series = {Proceedings of Machine Learning Research},
  month = {25--27 Apr},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v206/lee23d.html}
}
```

Original implementation:

```text
https://github.com/ML4ITS/TimeVQVAE
```

## Acknowledgments

This implementation is based on the TimeVQVAE architecture and MaskGIT-style iterative token generation for time-series synthesis, adapted here for stochastic wind simulation.
