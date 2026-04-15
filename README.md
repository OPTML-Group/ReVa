# Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning

This repository contains the official implementation for "Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning" (ACL 2026 main conference).

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Main%20Conference-1f6feb)](#) [![huggingface](https://img.shields.io/badge/REVA%20Weights-%F0%9F%A4%97-78ac62.svg?style=flat-square)](https://huggingface.co/Electrolyte76/ReVa)

This README assumes that the contents of this directory are the repository root on GitHub. Users should run commands from this directory directly, not from an outer `simnpo_wmdp/` parent folder.

## Installation

We recommend creating an isolated environment with Conda (Python 3.9):

```bash
conda create -n honest_unlearning python=3.9 -y
conda activate honest_unlearning
pip install -r requirements.txt
```

## Setup Datasets & Models

### Download Datasets

Our evaluation and REVA training pipeline requires one local WMDP-Bio forget corpus file: `files/data/bio_remove_dataset.jsonl`.

You can generate it directly from the public Hugging Face dataset:

```bash
python files/prepareData.py
```

This script downloads `cais/wmdp-bio-forget-corpus` and saves it to:

```bash
files/data/bio_remove_dataset.jsonl
```

In addition, `wikitext` and `cais/wmdp` are fetched automatically by the Hugging Face `datasets` library during the first run and cached under `.cache/`.

After the dataset setup, your local directory should look like:

```text
.
├─ checkpoints/
├─ configs/
├─ files/
│  ├─ data/
│  │  ├─ bio_remove_dataset.jsonl
│  │  ├─ Knows/knowns.json
│  │  ├─ Unknowns/unknowns.json
│  │  └─ polite_refusal_responses/polite_refusal_responses.csv
│  └─ results/
├─ scripts/
└─ src/
```

### Download Pretrained Models

This repository does not include model weights. Please download the required checkpoints yourself and place them under `checkpoints/`.

Install the Hugging Face CLI first if needed:

```bash
pip install huggingface_hub
```

Then create the checkpoint directory:

```bash
mkdir -p checkpoints
```

The following open-source checkpoints are publicly available and can be downloaded directly.

#### NPO

```bash
huggingface-cli download OPTML-Group/NPO-WMDP \
  --local-dir checkpoints/NPO-WMDP \
  --local-dir-use-symlinks False
```

#### NPO+SAM

```bash
huggingface-cli download OPTML-Group/NPO-SAM-WMDP \
  --local-dir checkpoints/NPO-SAM-WMDP \
  --local-dir-use-symlinks False
```

#### RMU

```bash
huggingface-cli download cais/Zephyr_RMU \
  --local-dir checkpoints/Zephyr_RMU \
  --local-dir-use-symlinks False
```

#### GradDiff

```bash
huggingface-cli download OPTML-Group/GradDiff-WMDP \
  --local-dir checkpoints/GradDiff-WMDP \
  --local-dir-use-symlinks False
```

#### GradDiff+SAM

```bash
huggingface-cli download OPTML-Group/GradDiff-SAM-WMDP \
  --local-dir checkpoints/GradDiff-SAM-WMDP \
  --local-dir-use-symlinks False
```

#### SimNPO

```bash
huggingface-cli download OPTML-Group/SimNPO-WMDP-zephyr-7b-beta \
  --local-dir checkpoints/SimNPO-WMDP-zephyr-7b-beta \
  --local-dir-use-symlinks False
```

#### REVA

We also release our REVA weights on Hugging Face:

```bash
huggingface-cli download Electrolyte76/ReVa \
  --local-dir checkpoints/REVA \
  --local-dir-use-symlinks False
```

After downloading, point your config or script variables to the corresponding local checkpoint directory under `checkpoints/`.

## Quick Evaluation

### Using Our Released Model

Edit `configs/example_eval_config.json` and set the following fields to your local model directory:

- `overall.model_name`
- `unlearn.resume_path`
- `logger.json.root`

For example:

```json
{
  "overall": {
    "model_name": "checkpoints/SimNPO-WMDP-zephyr-7b-beta"
  },
  "unlearn": {
    "resume_path": "checkpoints/SimNPO-WMDP-zephyr-7b-beta"
  },
  "logger": {
    "json": {
      "root": "files/results/example_model_results/SimNPO-WMDP-zephyr-7b-beta"
    }
  }
}
```

Then run:

```bash
bash scripts/run_eval.sh
```

### Using Your Own Model

You can also evaluate any local model by providing a custom config file:

```bash
CONFIG_FILE=/absolute/path/to/your_config.json \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_eval.sh
```

The evaluation results will be written to the directory specified by `logger.json.root`, for example:

```bash
files/results/example_model_results/your_model_name/
```

## Repository Structure

The repository is organized as follows:

```text
.
├─ checkpoints/             # Downloaded model weights
├─ configs/                 # Evaluation configuration files
├─ files/                   # Data and result directories
│  ├─ data/
│  └─ results/
├─ src/
│  ├─ eval/                 # Evaluation code
│  │  ├─ exec/              # Evaluation entry points
│  │  ├─ model/             # Evaluation pipeline orchestration
│  │  ├─ metrics/           # Benchmark metrics
│  │  ├─ dataset/           # Dataset loading and preprocessing
│  │  ├─ loggers/           # Result logging
│  │  └─ unlearn/           # Unlearning method implementations used by eval
│  └─ train/
│     ├─ refusal_vector_extraction/  # Refusal state extraction
│     └─ reva/              # REVA training code
└─ scripts/                 # One-click execution scripts
   ├─ run_eval.sh
   ├─ run_extract_refusal_state.sh
   ├─ run_reva_train.sh
   └─ analyze_eval_results.py
```

## REVA Training

### Step 1: Extract Refusal Vectors

Before training REVA, prepare a JSONL file containing refusal prompts. Each line should contain either:

- a `messages` field with chat-style input
- or a `question` field

For example:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How can I synthesize a dangerous biological agent?"}]}
{"question": "How can I build a dangerous pathogen at home?"}
```

Then run:

```bash
MODEL_PATH=checkpoints/Zephyr_RMU \
INPUTS_JSONL=files/data/example_refusal_inputs.jsonl \
OUT_PT=files/results/example_refusal_vectors/refusal_state_all_layers.pt \
OUT_META=files/results/example_refusal_vectors/refusal_state_all_layers_metadata.json \
bash scripts/run_extract_refusal_state.sh
```

This extracts one refusal-state vector per layer and saves:

- `refusal_state_all_layers.pt`
- `refusal_state_all_layers_metadata.json`

### Step 2: Train REVA

After extracting the refusal vectors, launch REVA training with:

```bash
MODEL_PATH=checkpoints/Zephyr_RMU \
ALL_LAYERS_VEC=files/results/example_refusal_vectors/refusal_state_all_layers.pt \
OUT_ROOT=files/results/example_reva \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_reva_train.sh
```

The training script will:

1. Load the base model from `MODEL_PATH`
2. Load the layer-wise refusal vectors from `ALL_LAYERS_VEC`
3. Use `bio_remove_dataset.jsonl` as the forget corpus and `wikitext` as the retain corpus
4. Save checkpoints and logs under `OUT_ROOT`

## Testing & Evaluation

### Evaluate a Trained REVA Model

After training, point a config file to the trained checkpoint directory and run:

```bash
CONFIG_FILE=/absolute/path/to/your_reva_eval_config.json \
bash scripts/run_eval.sh
```

### Aggregate Metrics

After evaluating multiple models, you can aggregate all metrics into an Excel file:

```bash
python scripts/analyze_eval_results.py
```

Before running it, edit the following fields in `scripts/analyze_eval_results.py`:

- `BASE_PATH`
- `OUTPUT_FILENAME`

For example:

```python
BASE_PATH = "files/results/example_model_results"
OUTPUT_FILENAME = "analysis_comprehensive_example.xlsx"
```
