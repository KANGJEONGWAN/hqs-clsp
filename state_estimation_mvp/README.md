# State Estimation MVP (Text-Only, Zero-Shot Context)

This folder provides a text-only implementation for:

- Structured zero-shot context text generation
- Label-leakage filtering and context validation
- Text-only state estimation (`arousal`, `valence`, `cognitive_load_proxy`)
- Zero-shot split by context signature

## Files

- `dataset.py`: text-only manifest builder, zero-shot split, dataset loader
- `text_context.py`: structured context schema + template builder + leakage checks
- `model.py`: text-only encoder + regression head
- `train.py`: training loop
- `config.yaml`: experiment config
- `step1_prepare_eevr_data.py`: build text-only structured manifest from Textdata/VADS
- `step2_validate_text_context.py`: validate text quality and label leakage
- `step3_model_sanity_check.py`: one-batch forward/shape/NaN/loss check
- `step5_quick_experiment.py`: 10% data, 2-3 epoch quick run
- `step6_compare_text_ablation.py`: compare blank-text baseline vs structured text
- `step6b_threeway_ablation.py`: compare structured vs minimal vs blank text
- `step7_text_quality_ablation.py`: compare full vs minimal vs random text context
- `step8_alignment_check.py`: context-signature cosine stats + t-SNE export

## Expected train/val CSV format (text-only)

Required columns:

- `sample_id`
- `text`
- `arousal`
- `valence`
- `cognitive_load_proxy`
- `context_signature` (recommended for zero-shot evaluation)

## Run

From repository root:

`python state_estimation_mvp/train.py --config state_estimation_mvp/config.yaml`

## Required execution order

1) Data 연결

`python state_estimation_mvp/step1_prepare_eevr_data.py --split zero-shot`

2) Text context 검증

`python state_estimation_mvp/step2_validate_text_context.py --csv Data_files/text_only_manifest.csv`

3) Model sanity check

`python state_estimation_mvp/step3_model_sanity_check.py --config state_estimation_mvp/config.yaml`

4) Loss structure

Default:

- total loss = regression MSE

5) 빠른 실험

`python state_estimation_mvp/step5_quick_experiment.py --config state_estimation_mvp/config.yaml --subset-ratio 0.1 --epochs 3`

6) 핵심 검증 (Text ablation)

`python state_estimation_mvp/step6_compare_text_ablation.py --config state_estimation_mvp/config.yaml --subset-ratio 0.1 --epochs 3`

6b) 3-way 검증 (structured / minimal / blank)

`python state_estimation_mvp/step6b_threeway_ablation.py --config state_estimation_mvp/config.yaml --subset-ratio 0.1 --epochs 3`

7) Text quality 검증

`python state_estimation_mvp/step7_text_quality_ablation.py --config state_estimation_mvp/config.yaml --subset-ratio 0.1 --epochs 3`

8) CLSP alignment 확인

`python state_estimation_mvp/step8_alignment_check.py --config state_estimation_mvp/config.yaml --checkpoint outputs/state_estimation_mvp/best.pt`

## Notes

- This branch is text-only by design.
- Text should describe conditions only (no direct label leakage).
- Zero-shot default split is context-signature holdout.
- `cognitive_load_proxy` is derived from VADS `significance` with min-max style normalization on 1~5:
	- `cognitive_load_proxy = (significance - 1) / 4`, clipped to `[0, 1]`

## Multimodal extension (Text + PPG)

New files for attaching a PPG branch without breaking text-only baseline:

- `state_schema.py`: session-level single-value state schema + validator/signature
- `step0_generate_synthetic_states.py`: synthetic session state generation
- `ppg_features.py`: synthetic/engineered PPG feature builders
- `step0_prepare_ppg_features.py`: create `Data_files/ppg_feature_manifest.csv`
- `dataset_multimodal.py`: text + ppg feature dataset
- `model_multimodal.py`: text branch + ppg branch + fusion head
- `train_multimodal.py`: multimodal training with optional alignment loss
- `config_multimodal.yaml`: multimodal config
- `step9_multimodal_sanity_check.py`: one-batch multimodal sanity test

Recommended multimodal run order:

1. `python state_estimation_mvp/step0_generate_synthetic_states.py`
2. `python state_estimation_mvp/step0_prepare_ppg_features.py --mode auto`
3. `python state_estimation_mvp/step9_multimodal_sanity_check.py --config state_estimation_mvp/config_multimodal.yaml`
4. `python state_estimation_mvp/train_multimodal.py --config state_estimation_mvp/config_multimodal.yaml --subset-ratio 0.1 --epochs 3`
