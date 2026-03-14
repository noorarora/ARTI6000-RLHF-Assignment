# ARTI6000 – Advanced Topics in AI and ML  
## Assignment 1 – RLHF Implementation

Student: Noor Arora  
Student id: A1963789
University: University of Adelaide  


## RLHF Reward Model Training

A reward model implementation for **Reinforcement Learning from Human Feedback (RLHF)** using the [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) and HuggingFace Transformers. The model learns human preferences between `chosen` and `rejected` responses, and can be integrated into downstream RLHF pipelines to guide language model behaviour.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Overview

Reinforcement Learning from Human Feedback (RLHF) is widely used to align large language models with human expectations. Instead of directly generating text, a **reward model** learns to score responses based on human preference data.

This project:
- Uses the **Anthropic HH-RLHF** dataset
- Trains a **binary preference reward model**
- Predicts which response humans prefer
- Demonstrates a simplified RLHF reward modelling pipeline

---

## Dataset

**Dataset:** [Anthropic Helpful–Harmless RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)

Each sample contains two responses:
| Field | Description |
|---|---|
| `chosen` | Response preferred by humans |
| `rejected` | Response rated lower by humans |

The model learns the relationship: `Reward(chosen) > Reward(rejected)`

**Example:**
```json
{
  "chosen": "Assistant response preferred by humans",
  "rejected": "Assistant response that is less preferred"
}
```

---

## Model Architecture

| Component | Details |
|---|---|
| Framework | HuggingFace Transformers |
| Base Model | `distilbert-base-uncased` |
| Head | Sequence classification |
| Output | Single scalar reward score |

The model outputs a **scalar reward score** indicating the quality of a response.

---

## Training Pipeline

1. Load the RLHF dataset
2. Preprocess `chosen` and `rejected` responses
3. Tokenize text using the transformer tokenizer
4. Construct pairwise preference training samples
5. Train the reward model
6. Evaluate model performance

---

## Installation

Install all required dependencies:
```bash
pip install transformers datasets torch accelerate sentencepiece
```

---

## Usage

**Run the training notebook:**
```bash
jupyter notebook 02_reward_model.ipynb
```

**Or train via script:**
```bash
python reward_model.py
```

**Example — Loading the dataset:**
```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")

chosen = dataset["train"]["chosen"]
rejected = dataset["train"]["rejected"]
```

---

## Results

The trained reward model learns to distinguish between helpful and less helpful responses, demonstrating how human preference signals can be converted into a trainable reward function.

---

## Repository Structure
```
RLHF-Assignment/
├── 01_data_loading.ipynb    # Dataset loading and exploration
├── 02_reward_model.ipynb    # Reward model training and evaluation
└── README.md
```

---

## Future Improvements

- [ ] Train with larger transformer models (e.g. `roberta-large`, `deberta`)
- [ ] Implement pairwise ranking loss
- [ ] Integrate the reward model into a PPO-based RLHF pipeline
- [ ] Evaluate on preference benchmarks

---

## References

- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- Ouyang et al., 2022 — [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
