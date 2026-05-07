# State Estimation MVP

This folder contains the current state-estimation work for the `text-clsp` repo.
It includes a text-only baseline and a lightweight multimodal scaffold for future PPG work.

## What’s included

- Structured zero-shot text generation and leakage-aware context construction
- Text-only state estimation for `arousal`, `valence`, and `cognitive_load_proxy`
- Synthetic session-state schema for the multimodal branch
- Text + PPG fusion model and training loop

## Current files

- `dataset.py`: text-only manifest builder, zero-shot split, dataset loader
- `text_context.py`: structured context schema, template builder, leakage checks
- `model.py`: text-only encoder + regression head
- `train.py`: text-only training loop
- `config.yaml`: text-only experiment config
- `step1_prepare_eevr_data.py`: build text-only structured manifest from `Textdata.csv` and `VADS.csv`
- `state_schema.py`: session-level categorical state schema and signature helper
- `step0_generate_synthetic_states.py`: generate synthetic session-state rows
- `dataset_multimodal.py`: multimodal dataset joining text and PPG features
- `model_multimodal.py`: text branch, PPG branch, and fusion head
- `train_multimodal.py`: multimodal training loop with alignment loss
- `config_multimodal.yaml`: multimodal experiment config

## Text-only data format

The text-only CSVs used by `train.py` should contain:

- `sample_id`
- `text`
- `arousal`
- `valence`
- `cognitive_load_proxy`
- `context_signature` is optional but useful for zero-shot evaluation

## Text-only run

From the repository root:

`python state_estimation_mvp/step1_prepare_eevr_data.py --split zero-shot`

`python state_estimation_mvp/train.py --config state_estimation_mvp/config.yaml`

The text-only config trains a regression model with MSE loss over the three targets.

## Multimodal scaffold

The multimodal branch is intentionally separated from the text-only baseline.
It currently uses a synthetic session-state table and a fusion model that combines text and PPG features.

### Files

- `Data_files/session_state_synth.csv`: synthetic session-state examples aligned by `sample_id`
- `config_multimodal.yaml`: paths and hyperparameters for the multimodal setup
- `dataset_multimodal.py`: loads text rows, merges PPG features, and returns tensors
- `model_multimodal.py`: DistilBERT text branch plus MLP PPG branch
- `train_multimodal.py`: trains the fused model with optional CLSP-style alignment loss

### Multimodal run

From the repository root:

`python state_estimation_mvp/step0_generate_synthetic_states.py`

`python state_estimation_mvp/train_multimodal.py --config state_estimation_mvp/config_multimodal.yaml --subset-ratio 0.1 --epochs 3`

If you want to point the multimodal branch at a different PPG feature CSV, update `state_estimation_mvp/config_multimodal.yaml`.

## Notes

- `cognitive_load_proxy` is derived from VADS `significance` via `(significance - 1) / 4`, clipped to `[0, 1]`
- The text-only and multimodal branches are kept separate on purpose
- The synthetic state schema uses one categorical value per session-state component
