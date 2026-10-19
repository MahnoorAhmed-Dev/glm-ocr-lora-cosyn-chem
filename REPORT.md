# Fine-Tuning GLM-OCR with LoRA: a Reusable Guide

This is a working guide for fine-tuning **GLM-OCR** with **LoRA** using **LLaMA-Factory**, written from the experiment in this repo (chemical-structure images to SMILES on CoSyn data) so it can be reused on a different project with its own dataset.

Everything marked **verified** was run or read in this repo's code and files. Anything marked **estimate** or **untested** is reasoning, not a measurement, so check it before relying on it.

## Contents

1. [The short version](#1-the-short-version)
2. [What you need](#2-what-you-need)
3. [Setting up a new project](#3-setting-up-a-new-project)
4. [Preparing your dataset](#4-preparing-your-dataset)
5. [Writing the training config](#5-writing-the-training-config)
6. [Running training](#6-running-training)
7. [Evaluating: base versus fine-tuned](#7-evaluating-base-versus-fine-tuned)
8. [After training: adapter, merging, inference](#8-after-training-adapter-merging-inference)
9. [Choosing hyperparameters and fixing weak results](#9-choosing-hyperparameters-and-fixing-weak-results)
10. [Problems hit in this project and their fixes](#10-problems-hit-in-this-project-and-their-fixes)
11. [Reuse map: what to copy from this repo](#11-reuse-map-what-to-copy-from-this-repo)
12. [Checklist for a new project](#12-checklist-for-a-new-project)

---

## 1. The short version

The workflow, in order:

1. **Define the task as one prompt and one exact output format.** Image(s) in, text out.
2. **Build the dataset** as a ShareGPT-style JSON plus an image folder, with a **train** split and a separate **held-out eval** split.
3. **Evaluate the untouched base model first** on the eval split. Without that baseline you cannot say the fine-tune helped.
4. **Train a LoRA adapter** with LLaMA-Factory and a YAML config.
5. **Evaluate base + adapter** on the same eval split, with the same prompt and decoding settings.
6. **Compare**, keep the small adapter, and leave the 2.7 GB base model out of git.

What this repo got on its task (30 held-out molecules, one run): exact match 6.7% to 23.3%, valid output 50% to 70%, mean character similarity 31.8% to 61.1%. Most of the gain was **output behavior** (the base model produced repetition loops and LaTeX/HTML; the fine-tuned one mostly did not). That pattern, where LoRA teaches the model your output format and style before it teaches new knowledge, is typical for small datasets.

---

## 2. What you need

### 2.1 From GLM-OCR (the base model)

Source: [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) on Hugging Face. Download the whole repository into a local folder. The files present in this repo's copy (**verified**):

| File | Why it is needed |
|---|---|
| `model.safetensors` | The weights, about 2.7 GB in bf16 (roughly 1.3 billion parameters, an estimate from file size). |
| `config.json` | Architecture: a 24-layer vision encoder (hidden 1024), a projector that merges 2x2 patches, and a 16-layer language decoder (hidden 1536, 16 heads, 8 key/value heads, vocab 59,392, context 131,072). |
| `tokenizer.json`, `tokenizer_config.json` | The tokenizer, including the special image tokens. |
| `preprocessor_config.json`, `processor_config.json` | How images are resized and turned into patches. |
| `chat_template.jinja` | How user/assistant turns and the image placeholder are rendered into the prompt. |
| `generation_config.json` | Default generation settings and end-of-sequence token IDs. |
| `configuration.json` | Extra metadata file shipped with the model; keep it with the rest. |

Download command:

```bash
pip install -U "huggingface_hub>=1.5,<2.0"
hf download zai-org/GLM-OCR --local-dir GLM-OCR
```

Use `huggingface_hub` 1.x, not 2.x: with the transformers version installed here, 2.x raised an import error (see section 10). The model must be loaded with `trust_remote_code: true` as set in the configs, and you need a `transformers` release that knows the `glm_ocr` architecture (**verified** with 5.20.1 locally).

### 2.2 From LLaMA-Factory (the training framework)

Training itself needs only the framework code, not the demo data, docs or tests. The minimal set (**verified**, this is what this repo's trimmed copy contains and it ran):

| Path | Purpose |
|---|---|
| `src/llamafactory/` | The framework: data loading, templates, model loading, the LoRA and trainer code. |
| `pyproject.toml` | Defines the package and its dependencies, so `pip install -e .` works. It also expects `README.md` to exist (line `readme = "README.md"`). |
| `README.md` | Required only because `pyproject.toml` references it. |
| `LICENSE` | Apache-2.0. Keep it if you redistribute the source. |
| `requirements/` | Optional extras. Only `metrics.txt` (BLEU/ROUGE-style scoring) and `deepspeed.txt` (multi-GPU) are kept here. |

Two details that matter for GLM-OCR specifically:

- The template is named **`glm_ocr`** (defined in `src/llamafactory/data/template.py`). Your YAML must say `template: glm_ocr`.
- The multimodal registration for `glm_ocr` is in `src/llamafactory/model/model_utils/visual.py`. It marks `visual.merger` as the projector, `visual.patch_embed` and `visual.blocks` as the vision tower, and `language_model` plus `lm_head` as the language side. This is how LLaMA-Factory knows what to freeze.

**Version caution:** the `glm_ocr` template and registration exist in the LLaMA-Factory source in this repo (version `0.9.6.dev0`). An older release from PyPI may not include them. If you install a different copy, confirm both strings exist before you start:

```bash
grep -n 'name="glm_ocr"' src/llamafactory/data/template.py
grep -n 'model_type="glm_ocr"' src/llamafactory/model/model_utils/visual.py
```

The simplest safe route is to copy this repo's `LLaMA-Factory/` folder into the new project.

### 2.3 Python packages

Installed by `pip install -e LLaMA-Factory` plus `rdkit` for this task (**verified** locally: Python 3.12, torch 2.13 CPU build, transformers 5.20.1, peft 0.18.1, datasets 4.0.0, accelerate 1.11.0). On Colab it ran on Python 3.13 after the `huggingface_hub` pin. For a different project replace `rdkit` with whatever your metric needs.

### 2.4 Hardware

| Setup | Status |
|---|---|
| **Colab T4 (16 GB), LoRA** | **Verified.** 200 samples, batch 4, 3 epochs, 75 steps took about 333 seconds. The T4 has no bf16, so use `fp16: true`. |
| **CPU only** | A 2-sample inference smoke test of the base model ran on a CPU-only machine (**verified**); speed was not measured. Training on CPU is not realistic. For reference, evaluation on the Colab T4 averaged about 24 seconds per sample for the base model (many of those were long runaway outputs hitting the 256-token cap). |
| **Full fine-tuning** | **Untested here.** Rough rule: around 16 bytes per trainable parameter for weights, gradients and Adam state, so likely 15 to 20 GB or more. A T4 is probably too small. |

---

## 3. Setting up a new project

Recommended layout, matching this repo:

```
my-project/
├── README.md
├── .gitignore               # GLM-OCR/, LLaMA-Factory/saves/, .claude/, __pycache__/
├── GLM-OCR/                 # base model, downloaded, not committed
├── LLaMA-Factory/           # trimmed copy (see 2.2)
│   └── data/
│       ├── dataset_info.json
│       ├── my_train.json  + my_train/   # images
│       ├── my_eval.json   + my_eval/
│       └── my_lora.yaml                 # training config
├── scripts/
│   ├── prepare_data.py      # builds the splits
│   ├── evaluate.py          # scores a model, optionally + adapter
│   └── make_summary.py      # base vs fine-tuned table
├── notebooks/colab_run.ipynb
└── results/                 # base.json, tuned.json, summary.md, adapter/
```

### What goes in git and what does not

| Commit | Do not commit |
|---|---|
| Source, configs, scripts, notebook | `GLM-OCR/model.safetensors` (2.7 GB) |
| Datasets that are small and shareable | `LLaMA-Factory/saves/` (checkpoints, merged models) |
| `results/` including the adapter (about 15 MB here) | `.claude/`, `__pycache__/`, `*.egg-info/`, `wandb/` |

Why the model stays out: it is 2.7 GB, GitHub rejects normal files over 100 MB, and **Git LFS has a 2 GB per-file limit**, so the push failed even with LFS (verified: the error was `Size must be less than or equal to 2147483648`). The base weights never change during LoRA training anyway, so people can download them from Hugging Face. If you ever need the model in git, re-save it as shards under 2 GB (untested here).

If your dataset images are private or large, keep them out of git too and document how to obtain them.

---

## 4. Preparing your dataset

### 4.1 Record format

LLaMA-Factory reads **ShareGPT-style** records. One record is one training example:

```json
{
  "messages": [
    { "role": "user", "content": "<image>Your instruction here" },
    { "role": "assistant", "content": "the exact target text" }
  ],
  "images": ["my_train/1.png"]
}
```

Rules that matter:

- The number of `<image>` placeholders in the messages must equal the number of paths in `images`.
- The `<image>` token goes where the image should sit in the prompt. This repo puts it first, directly before the instruction.
- Multi-turn conversations are possible by adding more user/assistant pairs. The loss is computed only on assistant replies.
- Image paths are resolved against `media_dir`, which defaults to `dataset_dir` (the `data/` folder). **Gotcha:** if your images live in a subfolder such as `data/finetune/`, you must either move them up or set `media_dir: data/finetune` in the YAML. This repo hit exactly that mismatch.

### 4.2 Registering the dataset

Add an entry to `data/dataset_info.json` (**verified** format):

```json
"my_train": {
  "file_name": "my_train.json",
  "formatting": "sharegpt",
  "columns": { "messages": "messages", "images": "images" },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

The key (`my_train`) is what you put in the YAML's `dataset:` field. `file_name` is relative to `data/`. Register the eval split too, so you can point the YAML at it later if you add an eval loop.

### 4.3 Designing the task and the target

This is where most of the result is decided.

- **One prompt, used identically in training and evaluation.** Changing the wording at test time costs accuracy.
- **A single, rigid output format.** This repo's target was a SMILES string inside an ` ```rdkit ` fence. The base model did not use the fence at all, so the first thing LoRA learns is the format. A format that is easy to parse also makes scoring easy.
- **Keep targets within `cutoff_len`.** The default here is 2048 tokens for prompt plus image tokens plus answer. Long documents with long transcriptions may need a higher value or cropping. Samples longer than the cutoff may be truncated or dropped, so check your longest targets.
- **Image size matters.** The processor converts the image into patch tokens, so very large images cost many tokens. Check `preprocessor_config.json` for the resizing rules before you decide how to render or crop your inputs.
- **Clean labels beat many labels.** Check a few dozen samples by eye. A systematic labeling error is learned faithfully.

### 4.4 Splits

- **Always hold out an eval set that is never trained on.** With the original 10 samples every example was in training, so any score would have measured memorization.
- **Make the splits disjoint on the thing that could leak.** Here that was the target molecule, so `prepare_data.py` deduplicates by SMILES before splitting. For documents, split by document, not by page. For forms, split by template or by issuer if the test should generalize across them.
- **Fix the random seed** so the split can be recreated (this repo uses 42).
- **Sizes used here:** 200 train and 30 eval. That was enough to see the effect but too small for tight conclusions (see section 9).

### 4.5 Where to get data

This project pulled molecules from the **CoSyn-400K** dataset (`allenai/CoSyn-400K`, subset `chemical`, about 9,000 rows with `image` and `code` columns). For your own project the equivalent step is a `prepare_data.py` that reads your source (a folder, a database, a Hugging Face dataset), writes the images as files, and emits the JSON. `scripts/prepare_data.py` in this repo is a working template using `datasets` streaming.

---

## 5. Writing the training config

LLaMA-Factory is driven by one YAML file. This is the config that produced this repo's results (`glm_ocr_lora_colab.yaml`, **verified** to run), with notes on what to change for a new project.

```yaml
### model
model_name_or_path: ../GLM-OCR       # path to the downloaded base model
trust_remote_code: true

### method
stage: sft                           # supervised fine-tuning
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: cosyn_chem_train            # CHANGE: key from dataset_info.json
template: glm_ocr                    # keep for GLM-OCR
cutoff_len: 2048                     # CHANGE if your samples are longer
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: saves/glm-ocr/lora/colab # CHANGE per experiment
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: true                # keep only the adapter, not optimizer state
report_to: none

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 2       # effective batch = 4 x 2 = 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true                           # T4 has no bf16; use bf16: true on A100/L4 or newer
```

### 5.1 What each group does

**Model.** `model_name_or_path` can be a Hugging Face ID (`zai-org/GLM-OCR`) or a local path. Relative paths are resolved from the directory you run the command in, which is why every command in this guide is run from inside `LLaMA-Factory/`.

**Method.** `stage: sft` trains on input/target pairs. `finetuning_type: lora` freezes the base weights and trains small adapters. `lora_rank` sets the adapter width.

**What LoRA actually touches here (verified from the source).** With `lora_target: all`, LLaMA-Factory attaches adapters to every linear layer except `lm_head`, the projector (`visual.merger`), and, because `freeze_vision_tower` defaults to `true`, the whole vision tower. So only the **language decoder's** linear layers are adapted. The vision encoder and projector never change. The learned update for a weight `W` is `(alpha / rank) * B * A`, with `lora_alpha` defaulting to `2 x rank` (so a scale of 2) and `lora_dropout` defaulting to `0`. `B` starts at zero, so training begins from exactly the base model's behavior.

**Dataset.** `dataset` takes a comma-separated list of keys if you want to mix several. `max_samples` caps the number used (the original config had 1000).

**Train.**

| Setting | What it does | Typical move |
|---|---|---|
| `per_device_train_batch_size` | Samples per forward pass per GPU | Lower it if you run out of memory. |
| `gradient_accumulation_steps` | Batches summed before one update | Raises the effective batch without more memory. |
| `learning_rate` | Step size | `1e-4` is a common LoRA start. Full fine-tuning would use about `1e-5`. |
| `num_train_epochs` | Passes over the data | 3 here. More risks overfitting on a small set. |
| `lr_scheduler_type: cosine` | Decays the rate smoothly to zero | Fine as is. |
| `warmup_ratio` | Fraction of steps spent ramping the rate up | 0.1 is fine. |
| `bf16` / `fp16` | Precision | `bf16` on newer GPUs, `fp16` on a T4. |

**Optimizer steps.** `steps = samples / (batch x accumulation) x epochs`. With 200 samples, an effective batch of 8 and 3 epochs, that is 75. With only 10 samples and an effective batch of 16 it is about 3, which is why the original config could not learn anything and logged no loss.

### 5.2 Optional additions

- **Validation loss during training.** Uncomment or add `val_size: 0.1`, `per_device_eval_batch_size: 1`, `eval_strategy: steps`, `eval_steps: 25`. This repo did not use it, so it could not tell whether the loss plateau meant "fitted" or "still improving". Adding it is the first upgrade I would make.
- **Unfreezing the vision tower.** `freeze_vision_tower: false` also adapts the image encoder. Consider this only if your images look very different from what GLM-OCR was trained on. It costs more memory and is **untested here**.
- **Higher rank or `lora_alpha`.** See section 9.

---

## 6. Running training

All commands run from inside `LLaMA-Factory/`.

```bash
cd LLaMA-Factory
pip install -e .
llamafactory-cli train data/my_lora.yaml
```

### 6.1 What gets written

To `output_dir`, with `save_only_model: true` (**verified** file list from this repo's adapter folder):

| File | Contents |
|---|---|
| `adapter_model.safetensors` | The trained LoRA weights, about 15 MB here. This is the deliverable. |
| `adapter_config.json` | Rank, alpha, target modules and the base-model reference. |
| `training_loss.png` | Loss curve (from `plot_loss: true`). |
| `trainer_log.jsonl` | Loss per logged step. |
| `all_results.json` | Final train loss, runtime, samples per second. |

### 6.2 Running on Colab

`notebooks/colab_run.ipynb` automates the whole experiment. Steps:

1. Open it from GitHub in Colab (File, Open notebook, GitHub tab).
2. Set the runtime to a **T4 GPU** (Runtime, Change runtime type).
3. Run the cells in order: setup, baseline evaluation, training, fine-tuned evaluation, summary, download.
4. Unzip the downloaded `results_bundle.zip` into the repo and commit the contents.

Colab rules learned the hard way:

- **Do not use "Save a copy in GitHub" from Colab.** It overwrites the notebook with its outputs and pushes a commit that conflicts with your local work. Download the results zip instead.
- After fixing a failed install, **restart the session** (Runtime, Restart session) before re-running setup, otherwise the broken package versions stay loaded.
- Nothing survives a disconnect. Download results before closing the tab.

---

## 7. Evaluating: base versus fine-tuned

### 7.1 Principles

- **Baseline first.** Score the untouched model with the same script before any training.
- **Same everything else.** Same eval images, same prompt, same decoding settings (this repo uses greedy decoding and a 256-token cap).
- **Held-out data only.**
- **Report more than one metric**, and read the raw outputs, not just the table. Here the biggest effect (the base model's repetition loops and LaTeX/HTML answers) was invisible in the headline percentages and obvious in the samples.

### 7.2 What `evaluate.py` does

`scripts/evaluate.py` loads the model with `AutoModelForImageTextToText` and `AutoProcessor`, optionally applies a LoRA adapter with `peft.PeftModel.from_pretrained`, builds a chat message with the image and the prompt, generates, and scores each sample. Usage:

```bash
python scripts/evaluate.py --data LLaMA-Factory/data/my_eval.json \
  --data-root LLaMA-Factory/data --model GLM-OCR --out results/base.json

python scripts/evaluate.py --data LLaMA-Factory/data/my_eval.json \
  --data-root LLaMA-Factory/data --model GLM-OCR \
  --adapter LLaMA-Factory/saves/glm-ocr/lora/colab --out results/tuned.json
```

### 7.3 Adapting the metrics to your task

The generation code is generic; **the scoring is chemistry-specific** and is the part to replace.

| Your task | Suggested metrics |
|---|---|
| Plain text transcription | Character error rate and word error rate (edit distance over characters or words), normalized. |
| Tables | Cell-level exact match, or table structure similarity. |
| Forms / key-value extraction | Per-field exact match and field-level F1 over a parsed JSON output. |
| Math / LaTeX | Normalized edit distance, plus exact match after whitespace normalization. |
| Any structured output | Validity rate (does it parse?) alongside correctness. |

In `evaluate.py` change `extract_smiles` (pull the answer out of the model text), `canonical` (normalize before comparing) and the metric block in `main`. Keep the per-sample JSON output so you can inspect failures.

### 7.4 How much eval data

Thirty samples gave a clear direction (2 to 7 exact matches) but no confidence interval. For a result you will quote, use at least 100 to 200 eval samples, or repeat the run with a few seeds and report the spread.

---

## 8. After training: adapter, merging, inference

### 8.1 Use the adapter without merging

Load the base model, then apply the adapter. This is what `evaluate.py --adapter` does. The adapter is only a few MB, so you can keep several and swap them on the same base.

### 8.2 Merge into a standalone model

```bash
llamafactory-cli export \
  --model_name_or_path ../GLM-OCR \
  --adapter_name_or_path saves/glm-ocr/lora/colab \
  --template glm_ocr \
  --export_dir saves/glm-ocr/lora/colab/merged \
  --trust_remote_code true
```

This writes a full-size model (about 2.7 GB) to `export_dir`; it does not modify `GLM-OCR/`. Merging removes the adapter's inference overhead. **Untested in this repo**: the command comes from the upstream GLM-OCR fine-tuning guide, so run it once and evaluate the merged model before trusting it.

### 8.3 What gets published

Publish the adapter (and optionally the merged model on Hugging Face, not on GitHub). Keep the base model as a download instruction.

---

## 9. Choosing hyperparameters and fixing weak results

### 9.1 Starting values (those used here)

`lora_rank: 8`, `lora_target: all`, `learning_rate: 1e-4`, `3 epochs`, effective batch 8, cosine schedule, warmup 0.1. They worked well enough on 200 samples; there was no tuning, so do not treat them as optimal.

### 9.2 Reading the training loss

In this run the loss fell from about 3.0 (step 5) to about 0.4 (step 30), then stayed between roughly 0.32 and 0.51 until step 75. Interpretations that fit that curve:

- The format and basic mapping were learned quickly.
- Further epochs on the same 200 samples may add little, or start to overfit. Without a validation loss this cannot be separated, which is the reason to add `val_size` next time.

### 9.3 If results are weak

| Symptom | Likely cause | What to try |
|---|---|---|
| Output is in the wrong format | Not enough training, or the format varies in the data | Make targets uniform, check the prompt matches between train and eval. |
| Format right, content wrong | Not enough data or capacity | More data first; then a higher `lora_rank` (16 or 32) and a matching `lora_alpha`. |
| Good train loss, bad eval | Overfitting or a leaky/unrepresentative split | More data, fewer epochs, lower learning rate, check the split. |
| Long repeated output | Decoding runs on | Cap `max_new_tokens`; check the model learned an end-of-answer pattern; check targets end cleanly. |
| Fine on easy cases, fails on hard ones | Distribution gap | Add hard examples; consider unfreezing the vision tower. |
| Out of memory | Batch too large | Lower `per_device_train_batch_size` and raise accumulation; shorten `cutoff_len`. |
| Loss does not move | Learning rate too low, or too few steps | Check the step count with the formula in 5.1; raise the rate or epochs. |

### 9.4 Rules of thumb that held up here

- Data quality and quantity moved the result more than any hyperparameter. The jump from 10 to 200 samples turned a pipeline test into a measurable result.
- Change one thing per run, keep each run's `results/` so you can compare.
- LoRA is not a good way to teach the model something radically new. It is good for adapting format, style and a narrow task.

---

## 10. Problems hit in this project and their fixes

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: huggingface-hub>=1.5.0,<2.0 is required ... found 2.2.0` in Colab | `pip install -U huggingface_hub` upgraded past what transformers allows | Pin `huggingface_hub>=1.5,<2.0`, restart the session, and use `hf download` (the old `huggingface-cli` is deprecated in 1.x). |
| Dataset not found / images not loading | Images were in `data/finetune/`, but the loader resolves them against `data/` | Move files up into `data/`, or set `media_dir`. Keep `file_name` in `dataset_info.json` relative to `data/`. |
| Push of the 2.7 GB model failed even with LFS | GitHub LFS limit is 2 GB per file | Do not commit the model; ignore `GLM-OCR/` and download from Hugging Face. |
| `bf16` errors or slow training on Colab | T4 has no bf16 | Use `fp16: true`. |
| `pip install -e .` fails | `pyproject.toml` requires a `README.md` next to it | Keep a README in the LLaMA-Factory folder. |
| Push rejected as non-fast-forward | Colab's "Save a copy in GitHub" had pushed a commit | `git fetch`, then rebase your commit on top and resolve the notebook conflict. Avoid that Colab button. |
| Relative paths broke | Commands were run from the wrong directory | Run training from inside `LLaMA-Factory/`, and evaluation from the repo root. |
| Base model outputs stray tokens such as `<sep>` or HTML | Genuine base-model behavior | Leave it in the baseline (it is a fair measurement); do not post-process it away for the base run only. |
| Windows warnings about `LF will be replaced by CRLF` | Git line-ending normalization | Harmless; ignore or set `core.autocrlf`. |
| Summary table broke on GitHub | Model outputs contained newlines and hundreds of characters | `make_summary.py` now collapses whitespace, escapes `|` and truncates cells. |

---

## 11. Reuse map: what to copy from this repo

| File | Reuse as is | Change for a new project |
|---|---|---|
| `LLaMA-Factory/` (src, pyproject, LICENSE, README stub) | Yes | Nothing. |
| `LLaMA-Factory/data/glm_ocr_lora_colab.yaml` | Template | `dataset`, `output_dir`, `cutoff_len`, hyperparameters, precision. |
| `LLaMA-Factory/data/dataset_info.json` | Format | Replace the CoSyn entries with yours. |
| `scripts/prepare_data.py` | Pattern | Replace the CoSyn streaming with your data source; keep dedupe, seeded shuffle, disjoint splits. |
| `scripts/evaluate.py` | Generation and loading code | Replace `extract_smiles`, `canonical` and the metrics. |
| `scripts/make_summary.py` | Layout | Change metric names and the per-sample columns. |
| `notebooks/colab_run.ipynb` | Structure | Repo URL, paths, config file name, dataset prep cell. |
| `.gitignore` | Yes | Add your own large or private folders. |

---

## 12. Checklist for a new project

**Before training**

- [ ] Task defined as one prompt and one strict output format
- [ ] Dataset in ShareGPT JSON, with `<image>` count equal to the `images` count per record
- [ ] Train and eval splits are disjoint, with a fixed seed
- [ ] Dataset registered in `dataset_info.json`; image paths resolve (check a few by hand)
- [ ] `GLM-OCR/` downloaded; `glm_ocr` template present in your LLaMA-Factory copy
- [ ] Training YAML set: dataset, `template: glm_ocr`, `cutoff_len`, precision matching your GPU
- [ ] Step count computed (`samples / effective batch x epochs`) and sensible
- [ ] `.gitignore` excludes the model and `saves/`
- [ ] Scoring script adapted to your task and tested on a couple of samples

**Training and evaluation**

- [ ] Baseline evaluation run and saved
- [ ] Training run from inside `LLaMA-Factory/`
- [ ] Fine-tuned evaluation run with identical settings
- [ ] Per-sample outputs read, not just the summary numbers
- [ ] Results, adapter and loss plot saved

**Wrapping up**

- [ ] Adapter and `results/` committed; model weights not committed
- [ ] README states what was measured, on how many samples, and the caveats
- [ ] Optional: merged model exported and evaluated before use

---

## Reference commands

```bash
# one-time setup
hf download zai-org/GLM-OCR --local-dir GLM-OCR
cd LLaMA-Factory && pip install -e . && cd ..

# build data
python scripts/prepare_data.py --n-train 200 --n-eval 30 --seed 42

# baseline
python scripts/evaluate.py --data LLaMA-Factory/data/cosyn_chem_eval.json \
  --data-root LLaMA-Factory/data --model GLM-OCR --out results/base.json

# train (from inside LLaMA-Factory/)
cd LLaMA-Factory && llamafactory-cli train data/glm_ocr_lora_colab.yaml && cd ..

# fine-tuned evaluation
python scripts/evaluate.py --data LLaMA-Factory/data/cosyn_chem_eval.json \
  --data-root LLaMA-Factory/data --model GLM-OCR \
  --adapter LLaMA-Factory/saves/glm-ocr/lora/colab --out results/lora.json

# summary
python scripts/make_summary.py
```
