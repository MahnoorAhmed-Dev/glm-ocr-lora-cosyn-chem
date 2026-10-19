# GLM-OCR LoRA Fine-Tune on CoSyn Chemical Data

A small, hands-on experiment: fine-tune the **GLM-OCR** vision-language model with **LoRA**, using **LLaMA-Factory** as the training framework, on a tiny example set of chemical-structure images from the **CoSyn** dataset.

The goal is **not** to ship a production model. The goal is to understand how LoRA works end to end: what gets trained, what stays frozen, what the config knobs do, and what comes out the other side. Ten samples is enough to exercise the whole pipeline and see every moving part.

## Contents

1. [What is GLM-OCR](#1-what-is-glm-ocr)
2. [What is LLaMA-Factory and how it is used here](#2-what-is-llama-factory-and-how-it-is-used-here)
3. [The data](#3-the-data)
4. [What LoRA does](#4-what-lora-does)
5. [The training config, line by line](#5-the-training-config-line-by-line)
6. [Repository layout](#6-repository-layout)
7. [Running it](#7-running-it)
8. [After training: using the adapter](#8-after-training-using-the-adapter)
9. [Limitations](#9-limitations)

---

## 1. What is GLM-OCR

GLM-OCR is a compact **multimodal (vision-language) model** from Z.ai (Zhipu AI), part of the GLM family, built for optical character recognition and document understanding. Instead of the classic OCR pipeline (detect text regions, then recognise characters, then post-process), a model like this takes an **image plus a text instruction** and generates the answer as text, token by token, the same way a language model writes a sentence. That lets one model read plain text, tables, formulas and layout, and lets you steer it with a prompt.

### Architecture

Everything below is read directly from `GLM-OCR/config.json` (the local copy of the model), so it describes the exact checkpoint used here.

The model has three parts that run in sequence:

| Part | What it does | Key numbers from the config |
|---|---|---|
| **Vision encoder** (`glm_ocr_vision`) | Splits the image into small square patches and turns them into feature vectors with a Vision Transformer | 24 layers, hidden size 1024, 16 heads, patch size 14, 336 px base image size |
| **Projector / merger** | Merges neighbouring patches (`spatial_merge_size: 2`, so 2×2 patches become one token) and maps them into the language model's embedding space (`out_hidden_size: 1536`) | Reduces the number of image tokens the decoder must read |
| **Language decoder** (`glm_ocr_text`) | A causal transformer that reads image tokens and the prompt, then writes the answer | 16 layers, hidden size 1536, 16 attention heads with 8 key/value heads (grouped-query attention), vocab 59,392, context up to 131,072 tokens |

A few details worth knowing:

- **Image tokens live in the same sequence as text.** The config reserves special tokens (`image_start_token_id`, `image_token_id`, `image_end_token_id`) that mark where visual features sit inside the prompt. This is why the training data uses an `<image>` placeholder in the user message.
- **Multimodal rotary positions (`mrope_section: [16, 24, 24]`).** Position encoding is split across several axes so the decoder can tell where a token sits in the text order and in the image grid.
- **Multi-token prediction head (`num_nextn_predict_layers: 1`).** The model carries an extra prediction layer that predicts a token further ahead, which can speed up generation.
- **Small and fast.** The weights are stored in bfloat16 and the file is about 2.7 GB, which works out to roughly 1.3 billion parameters. That is why it can be fine-tuned on a single modest GPU.
- **Grouped-query attention** (8 KV heads for 16 query heads) keeps memory use lower at inference time.

Hugging Face: [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR). Chat template used by LLaMA-Factory: `glm_ocr`.

---

## 2. What is LLaMA-Factory and how it is used here

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is an open-source framework that wraps Hugging Face Transformers, PEFT and TRL behind a **config-driven training interface**. Instead of writing a training loop, you describe the run in a YAML file and launch it:

```bash
llamafactory-cli train data/glm_ocr_lora_sft.yaml
```

What LLaMA-Factory handles for this project:

- **Loading the model** from the local `GLM-OCR/` folder and picking the right chat template (`template: glm_ocr`), which controls how the system/user/assistant turns and the image tokens are formatted.
- **Loading and registering the dataset** through `data/dataset_info.json`, converting each ShareGPT-style record into token IDs and image features.
- **Attaching LoRA adapters** to the model via PEFT (`finetuning_type: lora`).
- **Running supervised fine-tuning (SFT)**: the loss is computed only on the assistant's reply, so the model learns to produce the target answer given the image and prompt.
- **Logging, checkpointing and loss plots** into `saves/`.

### What was trimmed

This repo contains a **trimmed copy** of LLaMA-Factory. Tests, docs, Docker files, example configs, demo datasets and optional extras were removed, and the source under `LLaMA-Factory/src/` is otherwise unmodified. The Apache-2.0 `LICENSE` is kept because the upstream source is redistributed here.

---

## 3. The data

### Source: CoSyn

**CoSyn** (Code-Guided Synthetic data, from the paper *Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation*) is a synthetic dataset for text-rich images. Its idea is that a language model writes **code** that renders an image (a chart, a table, a diagram, a molecule), and because the code is the ground truth, every image comes with exact, free labels. The chemical subset renders molecules.

### What is used here

Ten examples from the chemical subset, stored in `LLaMA-Factory/data/`:

```
data/cosyn_chemical.json      # the 10 records
data/cosyn_chemical/1.png … 10.png   # the 10 molecule images
```

This is deliberately tiny. It is an **example set to learn how LoRA behaves**, not a dataset to train a useful model on.

### Record format

Each record is ShareGPT-style: one user turn with an image placeholder, one assistant turn with the target, and the image path.

```json
{
  "messages": [
    { "role": "user", "content": "<image>Code Generation:" },
    { "role": "assistant", "content": "```rdkit\nCCNc1nc(Cl)nc(NC(C)C)n1\n```" }
  ],
  "images": ["cosyn_chemical/2.png"]
}
```

- **Input:** a picture of a molecule plus the instruction `Code Generation:`.
- **Target:** the molecule written as a **SMILES string** (a compact text encoding of a chemical structure) inside an `rdkit` code fence. RDKit is the standard open-source cheminformatics library, so the output is something you could paste straight into RDKit to rebuild the molecule.

So the task being taught is **image-to-structure transcription**: read a drawn molecule and write the text that describes it. The examples range from tiny molecules (`FC(F)(Cl)Cl`, `c1ccccc1C#N`) to larger ones (a branched sugar with many hydroxyl groups), which makes it a good stress test for an OCR model, since chemical notation is dense, unusual text.

### How it is registered

`data/dataset_info.json` tells LLaMA-Factory where the file is and how to read it:

```json
"cosyn_chemical": {
  "file_name": "cosyn_chemical.json",
  "formatting": "sharegpt",
  "columns": { "messages": "messages", "images": "images" },
  "tags": {
    "role_tag": "role", "content_tag": "content",
    "user_tag": "user", "assistant_tag": "assistant"
  }
}
```

Image paths in the JSON are resolved relative to `data/` (LLaMA-Factory's default media directory).

---

## 4. What LoRA does

### The problem it solves

Fine-tuning normally updates **every weight** in the model. For a model with about 1.3 billion parameters that means storing gradients and optimizer state for all of them, which needs a lot of GPU memory, produces a full-size copy of the model for every experiment, and risks overwriting what the model already knows (catastrophic forgetting).

### The idea

**LoRA (Low-Rank Adaptation)** freezes the original weights completely and learns a small **additive correction** instead.

Take one weight matrix `W` in the model, of shape `d × k`. Full fine-tuning would learn a full update `ΔW` of the same shape. LoRA observes that the useful update usually has low intrinsic rank, so it writes the update as the product of two thin matrices:

```
W' = W + ΔW = W + (alpha / r) · B · A

W : d × k   frozen, never changes
A : r × k   trainable, initialised with small random values
B : d × r   trainable, initialised to zero
r           the rank, much smaller than d and k
```

Because `B` starts at zero, `ΔW` is zero at step 0, so the adapted model begins as exactly the original model and drifts away only as training pushes it.

### Why that is cheaper

A `d × k` matrix has `d·k` numbers. The pair `A`, `B` has only `r·(d + k)`. For a 1536 × 1536 matrix and rank 8:

- Full update: 1536 × 1536 = **2,359,296** trainable numbers
- LoRA update: 8 × (1536 + 1536) = **24,576** trainable numbers, about **1%** of the matrix

Only those small matrices receive gradients and optimizer state, so memory drops sharply, and the saved result is a tiny **adapter file** (a few megabytes) instead of a multi-gigabyte model copy.

### Step by step

1. **Freeze** every original weight.
2. **Insert** a pair of small matrices `A` and `B` next to each chosen weight matrix.
3. **Forward pass:** the layer computes `W·x + (alpha/r)·B·A·x`, i.e. the original output plus the learned correction.
4. **Backward pass:** gradients flow only into `A` and `B`.
5. **Save** just the adapter weights.
6. **Use:** load the base model and apply the adapter on the fly, or **merge** `B·A` into `W` once to get a standalone model with no extra inference cost.

### The knobs

| Setting | Meaning | Effect |
|---|---|---|
| `lora_rank` (r) | Width of the thin matrices | Higher means more capacity and more trainable parameters; 8 is a common small default |
| `lora_alpha` | Scaling of the update (`alpha / r`) | Controls how strongly the adapter changes the output; LLaMA-Factory defaults it from the rank when unset |
| `lora_target` | Which layers get adapters | `all` attaches adapters to all linear layers; fewer targets means fewer parameters |
| `learning_rate` | Step size | LoRA tolerates larger rates than full fine-tuning, hence `1e-4` here, where full fine-tuning would typically use around `1e-5` |

### Trade-offs

- **Pros:** far less memory, fast to train, small files, the base model is untouched, and you can keep several adapters for one base model and swap them.
- **Cons:** limited capacity compared with full fine-tuning, so it is less suited to teaching a model something radically new; and the result is only as good as the data.

---

## 5. The training config, line by line

`LLaMA-Factory/data/glm_ocr_lora_sft.yaml`:

```yaml
### model
model_name_or_path: ../GLM-OCR     # local copy of the base model
trust_remote_code: true

### method
stage: sft                         # supervised fine-tuning
do_train: true
finetuning_type: lora              # train adapters, freeze the base
lora_rank: 8
lora_target: all                   # adapters on all linear layers

### dataset
dataset: cosyn_chemical            # key in data/dataset_info.json
template: glm_ocr                  # chat/image-token formatting for this model
cutoff_len: 2048                   # max tokens per sample
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/glm-ocr/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4     # effective batch size = 4 × 4 = 16
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine          # decays smoothly to zero
warmup_ratio: 0.1                  # first 10% of steps ramp the LR up
bf16: true                         # bfloat16 training
```

### What to expect with 10 samples

- With a batch of 4 and 4 accumulation steps, 10 samples give roughly **one optimizer update per epoch**, so about 3 updates in total. That is enough to confirm the pipeline works but not enough to learn the task.
- `logging_steps: 10` means no loss line may be printed in so few steps. Lower it to `1` to watch the loss.
- To see LoRA actually fit the data, reduce `gradient_accumulation_steps`, raise `num_train_epochs`, or both.

---

## 6. Repository layout

```
.
├── README.md
├── .gitignore
├── GLM-OCR/                    # base model (not tracked, see below)
└── LLaMA-Factory/              # trimmed LLaMA-Factory
    ├── LICENSE                 # Apache-2.0, kept for the redistributed source
    ├── README.md
    ├── pyproject.toml
    ├── requirements/           # deepspeed.txt, metrics.txt
    ├── src/                    # LLaMA-Factory source
    └── data/
        ├── dataset_info.json
        ├── cosyn_chemical.json
        ├── cosyn_chemical/     # 1.png … 10.png
        ├── glm_ocr_lora_sft.yaml
```

`GLM-OCR/` holds the base model. The 2.7 GB `model.safetensors` is too large for a normal GitHub push (and over the 2 GB Git LFS limit), so `GLM-OCR/` is git-ignored and downloaded from Hugging Face instead. `LLaMA-Factory/saves/` (training output) is git-ignored too.

---

## 7. Running it

### 1. Get the base model

Download [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) into `GLM-OCR/` at the repo root:

```bash
hf download zai-org/GLM-OCR --local-dir GLM-OCR
```

### 2. Install

```bash
cd LLaMA-Factory
pip install -e .
```

Use a recent `transformers` release that supports the `glm_ocr` architecture.

### 3. Train (LoRA)

Run from inside `LLaMA-Factory/` so the relative paths (`../GLM-OCR`, `data/`) resolve:

```bash
llamafactory-cli train data/glm_ocr_lora_sft.yaml
```

Outputs land in `LLaMA-Factory/saves/glm-ocr/lora/sft/`: the adapter weights, checkpoints, trainer state and a loss plot.

---

## 8. After training: using the adapter

Merge the adapter into the base weights to get a standalone model:

```bash
llamafactory-cli export \
  --model_name_or_path ../GLM-OCR \
  --adapter_name_or_path saves/glm-ocr/lora/sft \
  --template glm_ocr \
  --export_dir saves/glm-ocr/lora/sft/merged \
  --trust_remote_code true
```

Or skip merging and load the base model with the adapter applied at inference time.

---

## 9. Limitations

- **Ten samples is a demonstration, not a training set.** Do not expect the model to generalise from it, and do not evaluate on the same images it trained on.
- There is **no validation split** configured (the eval block in the YAML is commented out). Add `val_size` and an eval strategy when you move to a real dataset.
- The task here (image to SMILES) is narrow. Results say little about GLM-OCR's general document-OCR ability.

## Credits

- [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) by Z.ai
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga (Apache-2.0)
- CoSyn: *Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation*
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
