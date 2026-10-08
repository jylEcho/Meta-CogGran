# Meta-CogGran

Meta-CogGran is a multimodal research codebase for studying **cognitive granularity and memory-guided visual reasoning**. It builds on visual-language model training with semantic memory banks and adds spatial memory components and configurable multi-step Meta-Cog control. The repository contains a range of model, bank, and training variants, along with data processing and evaluation utilities.


## Method at a Glance

The training code integrates a vision-language model with a semantic memory bank. In addition to global and entity-level information, selected BankV5-based configurations include layout and relation memories. Meta-Cog settings control iterative reasoning behavior, including the maximum and minimum number of steps, a stopping threshold, a stability threshold, and the number of state tokens.

These options are exposed by particular training scripts. Check the implementation and the selected bank builder before transferring settings to another model or experiment.

## Repository Layout

| Path                             | Description                                           |
| -------------------------------- | ----------------------------------------------------- |
| `pretrain_*.py`, `pretrain_*.sh` | Training code and experiment launch configurations    |
| `custom_llava_*.py`              | Custom multimodal model and processor implementations |
| `Bank/`                          | Semantic and spatial memory bank variants             |
| `data_process/`                  | Dataset download and preprocessing utilities          |
| `eval.py`, `eval_*.py`           | General and benchmark-specific evaluation scripts     |
| `meta_cog_supervisor.sh`         | Optional staged training supervisor                   |
| `meta_cog_stage_runner.sh`       | Records stage start/end and exit status               |
| `meta_cog_summarize.py`          | Training status/report helper                         |
| `launch_meta_cog_supervisor.sh`  | Background launcher for the supervisor                |

## Environment and Setup

The repository does not provide one unified dependency lockfile. Training launchers use Linux Bash, Conda, CUDA, and multi-GPU DeepSpeed. Install a compatible PyTorch/Transformers/DeepSpeed stack and any model-specific data and evaluation packages in the environment used by the chosen experiment.

```bash
git clone https://github.com/jylEcho/Meta-CogGran.git
cd Meta-CogGran
```

Several scripts contain machine-specific paths, including references to `external/miniconda3`, `external/granulon_Codex`, and external model/data workspaces. Update the environment name, repository/data roots, model checkpoint, bank path, GPU IDs, and output directories before launching training. Model weights, datasets, and some bank artifacts are external prerequisites.

## Training

`pretrain_10_reason_meta_cogV1.sh` is an example of a Meta-Cog-enabled BankV5 configuration:

```bash
bash pretrain_10_reason_meta_cogV1.sh
```

The script invokes a BankV5-based training implementation and sets semantic-bank retrieval parameters together with Meta-Cog controls. Confirm that every referenced path exists and that the data and bank match the selected base model. Other `pretrain_*.sh` files may use different backbones, datasets, iteration counts, or output locations.

## Optional Staged Training

The supervisor scripts can manage primary and refinement stages, archive logs, and summarize run status. Their current configuration expects stage scripts such as `train_meta_cog_primary.sh`, `train_meta_cog_fallback.sh`, and `train_meta_cog_refine.sh`, plus configured output directories. These names are not all present at the repository root in every checkout.

Review and adapt `meta_cog_supervisor.sh` and its referenced scripts before using the background launcher:

```bash
bash launch_meta_cog_supervisor.sh
```

Runtime state and logs are written under the configured `runtime/meta_cog_supervisor/` directory.

## Data and Evaluation

`data_process/` contains utilities for preparing datasets such as FLUX-Reason, ImageNet-Think, OCR-VQA, RefCOCO, and SEED. Data formats and output paths are script-specific. Evaluation is provided through `eval.py` and benchmark-specific scripts; check each entry point for the expected model, index, image, and output paths.

## Reproduction Notes

Keep the model, training implementation, bank artifact, and evaluation pipeline from the same experiment configuration. Validate a small data sample before starting a long multi-GPU run, and record software versions, command-line parameters, input data versions, and checkpoints. The repository scripts have not been validated here in a clean, general-purpose environment.
