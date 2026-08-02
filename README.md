# Laziness @ KnowledgeGraphEval 2026 — Cross-Domain Arabic Named Entity Recognition

Multi-head AraBERT with Domain Adversarial Training and Self-Training for nested, cross-domain Arabic NER.

Official submission of team **Laziness** to the **AdaptNER shared task (Subtask 1: Cross-Domain Named Entity Recognition)**, hosted at **KnowledgeGraphEval 2026**.

> Ho, Q. H. "Laziness at KnowledgeGraphEval 2026: Multi-Head AraBERT with Domain Adversarial Training and Self-Training for Cross-Domain Arabic Named Entity Recognition."

---

## Overview

This repository implements a single-encoder, multi-head token classification system for **nested** Arabic Named Entity Recognition under **unsupervised domain adaptation**. The model is trained on the source-domain **Wojood** corpus and adapted, without target labels, to 10 unseen domains drawn from the **Konooz** corpus (Agriculture, Art, Economics, Finance, Health, History, Law, Politics, Science, Sport).

The system combines four components on top of an AraBERT backbone:

1. **21 parallel linear classification heads** — one per Wojood entity type, each predicting an independent `{B, I, O}` distribution per token, allowing overlapping/nested spans without collapsing them into a flat tag set.
2. **Domain Adversarial Neural Network (DANN)** with a Gradient Reversal Layer (GRL) on the `[CLS]` representation, encouraging domain-invariant features between Wojood (source) and Konooz (target).
3. **Confidence-gated teacher-student self-training** on unlabeled Konooz text, using a frozen Phase 1 checkpoint as the teacher and a threshold of `τ = 0.90` on teacher softmax confidence.
4. **Stochastic Weight Averaging (SWA)** over the final adaptation checkpoints, exploiting AraBERT's LayerNorm (rather than BatchNorm) to allow direct weight averaging.

### Official results

| Metric      | Validation (Wojood, in-domain) | Official Test (Konooz, out-of-domain) |
|-------------|:-------------------------------:|:--------------------------------------:|
| Precision   | 0.9089                          | 0.7548                                 |
| Recall      | 0.9330                          | 0.7121                                 |
| Micro F1    | 0.9208                          | 0.7329                                 |
| Macro F1    | 0.8151                          | 0.6337                                 |

Final ranking: **6th of 9** participating teams on the blinded test set (1,645 sentences across 10 domains).

---

## Repository structure

This repository ships the pipeline code only. It does not redistribute the Wojood or Konooz corpora, or the shared task's blinded test set — see [Data](#data) for how to obtain them.

```
.
├── pipeline.ipynb          # End-to-end notebook: setup, training, adaptation, eval, inference
├── README.md
├── requirements.txt
├── Wojood/                 # (not included) source-domain labeled corpus
│   └── Wojood1_1_nested/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── dev-konooz/              # (not included) unlabeled target-domain dev files, one per domain
│   ├── Agriculture.txt
│   ├── Art.txt
│   ├── ...
│   └── Sport.txt
├── blinded-test-data/       # (not included) organizer-provided blinded test files
│   ├── Agriculture.txt
│   ├── ...
│   └── Sport.txt
└── B1/                       # generated at runtime: checkpoints, logs, plots, predictions
    ├── checkpoints/
    ├── args.json
    ├── tag_vocab.pkl
    ├── history.json
    └── ...
```

The notebook is organized into nine numbered sections, executed top to bottom:

| Section | Content |
|---|---|
| 0 | Installation |
| 1 | Setup and global configuration |
| 2 | CoNLL parsing and vocabulary construction |
| 3 | Dataset and subword-alignment transform |
| 4 | Model architecture (`BertNestedTagger`) |
| 5 | Trainer (`BaseTrainer`, `BertNestedTrainer`) |
| 6 | Phase 1 — supervised training on Wojood |
| 6b | Phase 2 — domain adaptation (DANN + self-training) and SWA |
| 7 | Training dashboard and reporting exports |
| 8 | Evaluation |
| 9 | Inference and submission packaging |

---

## Requirements

The reference run was executed on a **Kaggle Notebook environment (2026-03-20 image)** with **2 × NVIDIA Tesla T4 GPUs** and 30 GB system RAM. A single GPU with at least 16 GB of memory is sufficient to reproduce the pipeline at the configured batch sizes; adjust batch sizes accordingly for smaller GPUs.

Core dependencies:

```
torch
transformers
seqeval
natsort
numpy
matplotlib
huggingface_hub
```

Install with:

```bash
pip install setuptools==69.5.1
pip install torch transformers seqeval natsort numpy matplotlib huggingface_hub
```

or, if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

The backbone checkpoint is pulled automatically from the Hugging Face Hub:

```
aubmindlab/bert-base-arabertv02
```

If the model or tokenizer requires an authenticated download, export a Hugging Face access token before running the notebook (Section 0 of `pipeline.ipynb` performs this login):

```bash
export HF_TOKEN=your_token_here
```

---

## Data

This repository does not include any corpus data. To run the pipeline, obtain the following and place them under the paths shown in [Repository structure](#repository-structure):

- **Wojood** (Nested Arabic NER corpus, source domain) — Jarrar et al., 2022. Labeled, multi-column CoNLL format: each line is `token B-TYPE1 I-TYPE2 ... `, one column of tags per active entity type, blank lines separate sentences.
- **Konooz** (unlabeled target domains) — 10 domain files (`Agriculture.txt`, `Art.txt`, `Economics.txt`, `Finance.txt`, `Health.txt`, `History.txt`, `Law.txt`, `Politics.txt`, `Science.txt`, `Sport.txt`), one token per line, blank-line-delimited sentences, no gold tags. Used only for unsupervised adaptation — never for label supervision.
- **Blinded test data** — organizer-distributed, same per-domain file layout as `dev-konooz`, released for the official AdaptNER evaluation.

Both corpora must be obtained directly from the shared task organizers / original dataset authors; redistribution rights are not held by this repository.

---

## Configuration

All hyperparameters are set in a single `Namespace` object (`config`) in Section 1 of `pipeline.ipynb`. Key defaults:

**Backbone**

| Parameter | Value |
|---|---|
| `bert_model` | `aubmindlab/bert-base-arabertv02` |
| `dropout` | 0.1 |
| `max_seq_len` | 512 |
| `seed` | 42 |

**Phase 1 — supervised pretraining on Wojood**

| Parameter | Value |
|---|---|
| `max_epochs` | 50 (early stopping, patience 5) |
| `batch_size` | 8 |
| `lr` | 2e-5 (AdamW) |
| `gamma` | 0.95 (`ExponentialLR`) |
| `max_checkpoints` | 5 |

**Phase 2 — domain adaptation on Konooz**

| Parameter | Value |
|---|---|
| `adapt_epochs` | 5 |
| `adapt_batch_size` | 4 |
| `adapt_lr` | 2e-5 |
| `adapt_freeze_layers` | 6 (embeddings + encoder layers 0–5 frozen) |
| `adapt_conf_threshold` | 0.90 |
| `adapt_domain_weight` | domain-adversarial loss coefficient (`λ_dom` in the total loss) |
| `adapt_st_weight` | self-training loss coefficient (`λ_st` in the total loss) |
| `swa_n_average` | 4 (final checkpoints averaged) |

The total adaptation loss is:

```
L_total = L_ner_src + λ_st · L_self_train + λ_dom · L_domain
```

with the GRL scaling factor ramped over training as `λ = 2 / (1 + exp(-10 · step / total_steps)) − 1`.

Edit these values directly in the `config` Namespace at the top of the notebook before running; every downstream cell reads from this single object.

---

## Usage

Run `pipeline.ipynb` top to bottom in a single session — later sections depend on objects (`model`, `trainer`, `vocab`, `args`) created earlier.

### 1. Install and authenticate

Run Section 0. Provide a Hugging Face token if required by your environment.

### 2. Configure paths

In Section 1, update `config.train_path`, `config.val_path`, `config.test_path`, `config.adapt_konooz_dir`, and `config.blind_test_dir` to point at your local copies of Wojood, `dev-konooz`, and the blinded test set.

### 3. Phase 1 — supervised training

Sections 2–6 parse the Wojood CoNLL files, build per-type tag vocabularies, construct the `BertNestedTagger` model, and run supervised training with early stopping on validation Micro F1. Checkpoints, `history.json`, `tag_vocab.pkl`, and `args.json` are written to `config.train_output_path` (default `./B1`).

```python
trainer.train()
```

### 4. Phase 2 — domain adaptation

Section 6b reloads the best Phase 1 checkpoint, freezes the bottom `adapt_freeze_layers` encoder layers, instantiates the domain classifier and GRL, and jointly optimizes the supervised Wojood loss, confidence-gated self-training loss on unlabeled Konooz text, and the domain-adversarial loss. It then averages the final `swa_n_average` checkpoints into an SWA checkpoint (`checkpoint_9999.pt`), which is loaded preferentially at evaluation and inference time.

### 5. Dashboard and reporting

Section 7 renders loss/metric curves, a per-entity-type F1 bar chart, and a `results_summary.md` file from the saved training history — no retraining required, only a populated `B1/history.json`.

### 6. Evaluation

Section 8 reloads the best checkpoint and reports Micro F1, Macro F1, precision, recall, and a full seqeval classification report on `config.eval_data_path` (default: the Wojood validation split). Point `eval_data_path` at a labeled Konooz sample if you have one available.

### 7. Inference and submission packaging

Section 9 runs inference over each of the 10 Konooz domain files under `config.blind_test_dir`, writes token-level predictions in CoNLL format to `config.output_pred_file`, and packages the result into `config.output_zip_file` for submission.

```python
save_inference_to_txt(all_predicted_segments, output_pred_file)
```

---

## Model architecture notes

- The encoder produces a single shared `[CLS]`/token representation per input; 21 independent `nn.Linear(768, 3)` heads are applied on top, one per entity type, each outputting a `{B, I, O}` distribution. This lets multiple entity types be active on the same token simultaneously, which a flat single-head tagger cannot represent.
- Subword alignment assigns the gold label only to the first WordPiece of each token; continuation subwords receive `ignore_index=-100` and do not contribute to the loss.
- The domain classifier is a small MLP (`Linear(768, 256) → ReLU → Dropout(0.2) → Linear(256, 2)`) applied to the pooled `[CLS]` hidden state through a Gradient Reversal Layer, so that encoder gradients are reversed with respect to the domain-classification objective — pushing the encoder toward domain-invariant representations.
- SWA averages raw model weights directly (no batch-statistics recalculation pass), which is valid here because AraBERT uses LayerNorm rather than BatchNorm.

---

## Known limitations

- Unsupervised domain alignment does not fully close the gap for highly specialized vocabulary (for example, legal or scientific terminology largely absent from AraBERT's pretraining corpus).
- Because each entity type is scored independently, nested-span consistency across heads is not explicitly enforced; one head can recover a span while an overlapping head misses the corresponding nested mention.
- Minority classes with low support in Wojood (for example, `PRODUCT`, `QUANTITY`) receive little to no positive self-training signal on the target domain, since the teacher rarely exceeds the 0.90 confidence gate for these categories.

See Sections 6–7 of the accompanying paper for full error analysis and discussion.

---

## Citation

---

## License

---

## Acknowledgments

AdaptNER Shared Task organizers and the KnowledgeGraphEval 2026 workshop; the Wojood and Konooz corpus authors; the AraBERT team (Antoun et al., 2020).
