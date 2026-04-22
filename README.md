# Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning

This repository contains the official implementation for "Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning" (ACL 2026 main conference).

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Main%20Conference-1f6feb)](#) [![huggingface](https://img.shields.io/badge/REVA%20Weights-%F0%9F%A4%97-78ac62.svg?style=flat-square)](https://huggingface.co/Electrolyte76/ReVa)


## Installation

We recommend creating an isolated environment with Conda (Python 3.9):

```bash
conda create -n honest_unlearning python=3.9 -y
conda activate honest_unlearning
pip install -r requirements.txt
```

## Setup Datasets & Models

### Download Datasets

Before running the full evaluation pipeline, users need to prepare the following datasets manually.

1. **`bio_remove_dataset.jsonl`**

This file is required by both evaluation and REVA training.
It can be generated from the public Hugging Face dataset **`cais/wmdp-bio-forget-corpus`** by running:

```bash
python files/prepareData.py
```

This script saves the file to:

```bash
files/data/bio_remove_dataset.jsonl
```

2. **`bbh/`**

For BBH consistency evaluation, download the **`bbh`** data folder from:

- `https://github.com/milesaturpin/cot-unfaithfulness/tree/main/data/bbh`

and place it under:

```bash
files/data/bbh
```

After downloading, the structure should look like:

```text
files/data/bbh/
├─ causal_judgment/
├─ date_understanding/
├─ disambiguation_qa/
├─ ...
└─ web_of_lies/
```


The repository already includes the following local evaluation files, so users do not need to download them separately:

- **`files/data/Knows/knowns.json`**
  Used for knowledge-retention evaluation. 
- **`files/data/Unknowns/unknowns.json`**
  Used for unknown-question refusal evaluation. 
- **`files/data/csqa_open.json`**
  Used for the Open-Form Consistency evaluation.
- **`files/data/polite_refusal_responses/polite_refusal_responses.csv`**
  Used as auxiliary refusal-response templates in parts of the evaluation pipeline.

The following public datasets are downloaded automatically during the first run and cached under `.cache/`:

- **`wikitext`**
- **`cais/wmdp`** including **`wmdp-bio`**
- **`mmlu`** task data used by **`lm_eval`**

Users therefore do not need to manually download these datasets in advance, but they do need:

- a working internet connection for the first run
- the required evaluation dependencies installed, especially `datasets` and `lm_eval`
- enough local disk space for the cache directory

After the dataset setup, your local directory should look like:

```text
.
├─ checkpoints/
├─ configs/
├─ files/
│  ├─ data/
│  │  ├─ bbh/
│  │  ├─ bio_remove_dataset.jsonl
│  │  ├─ csqa_open.json
│  │  ├─ Knowns/knowns.json
│  │  ├─ Unknowns/unknowns.json
│  │  └─ polite_refusal_responses/polite_refusal_responses.csv
│  └─ results/
├─ scripts/
└─ src/
```

### Download Unlearned Models

This repository does not include model weights. Please download the required checkpoints yourself and place them under **`checkpoints/`**.

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

After downloading, point your config or script variables to the corresponding local checkpoint directory under **`checkpoints/`**.

## Quick Evaluation

### Using Existing Models

Edit **`configs/example_eval_config.json`** and set the following fields to your local model directory:

- **`overall.model_name`**
- **`unlearn.resume_path`**
- **`logger.json.root`**

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

During evaluation, the script will automatically download public benchmark data if it is not already cached, including:

- **`cais/wmdp`** for **`wmdp-bio`**
- **`mmlu`** task data through **`lm_eval`**

No separate manual dataset download is required for these benchmarks.

### Using Your Own Model

You can also evaluate any local model by providing a custom config file:

```bash
CONFIG_FILE=/absolute/path/to/your_config.json \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_eval.sh
```

The evaluation results will be written to the directory specified by **`logger.json.root`**, for example:

```bash
files/results/example_model_results/your_model_name/
```

### Aggregate Metrics

After evaluating multiple models, you can aggregate all metrics into an Excel file:

```bash
python scripts/analyze_eval_results.py
```

Before running it, edit the following fields in **`scripts/analyze_eval_results.py`**:

- **`BASE_PATH`**
- **`OUTPUT_FILENAME`**

For example:

```python
BASE_PATH = "files/results/example_model_results"
OUTPUT_FILENAME = "analysis_comprehensive_example.xlsx"
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

- a **`messages`** field with chat-style input
- or a **`question`** field

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

- **`refusal_state_all_layers.pt`**
- **`refusal_state_all_layers_metadata.json`**

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

1. Load the base model from **`MODEL_PATH`**
2. Load the layer-wise refusal vectors from **`ALL_LAYERS_VEC`**
3. Use **`bio_remove_dataset.jsonl`** as the forget corpus and **`wikitext`** as the retain corpus
4. Save checkpoints and logs under **`OUT_ROOT`**

If **`wikitext`** is not already cached locally, it will be downloaded automatically during the first run.

## Testing & Evaluation

### Evaluate a Trained REVA Model

After training, point a config file to the trained checkpoint directory and run:

```bash
CONFIG_FILE=/absolute/path/to/your_reva_eval_config.json \
bash scripts/run_eval.sh
```

## Acknowledgements

This project builds upon open-source benchmarks, datasets, and evaluation resources from (including but not limited to):

- [`MMLU`](https://github.com/hendrycks/test)
- [`WMDP`](https://github.com/centerforaisafety/wmdp)
- [`CSQA`](https://github.com/jonathanherzig/commonsenseqa)
- [`BeHonest`](https://github.com/GAIR-NLP/BeHonest)
- [`cot-unfaithfulness`](https://github.com/milesaturpin/cot-unfaithfulness)
- [`BBH`](https://github.com/suzgunmirac/BIG-Bench-Hard)

We sincerely thank the respective authors for releasing their codebases, datasets, and evaluation resources.
