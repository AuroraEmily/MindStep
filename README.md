# 🧠 MindStep: Structured Cognitive Prompting for Human Behavior Simulation

This repository contains the dataset, evaluation scripts, and benchmark details for our paper:

> **MindStep: Structured Cognitive Prompting for Human Behavior Simulation**

---

## 📌 Overview

**MindStep** is a cognitively‑grounded prompting framework for human behavior simulation. It structures reasoning into three stages:

- 🧩 **Evocation** – anchor a user‑specific persona from profile and memory  
- 👁️ **Perception** – filter situationally relevant cues from social context  
- ⚡ **Reaction** – infer the final action (type, object, content)  

We introduce **SocialAct**, a fine‑grained benchmark for behavioral prediction across four real‑world social events, and evaluate MindStep on **RecToM** for mental‑state reasoning.

---

## 📊 Dataset Statistics

| Benchmark | Question Type | Quantity | Options | Answer Type | Input Context | Avg. Chars |
|-----------|---------------|----------|---------|-------------|---------------|------------|
| **SocialAct** | BLM | 1,000 | 4 × 3 | Triplet | Profile, Memory, News, Twitter feed | 3,796 |
| | MeToo | 1,000 | 4 × 3 | Triplet | Profile, Memory, News, Twitter feed | 3,807 |
| | COVID‑19 | 1,000 | 4 × 3 | Triplet | Profile, Memory, News, Twitter feed | 3,901 |
| | CAL Fire | 1,000 | 4 × 3 | Triplet | Profile, Memory, News, Twitter feed | 3,599 |
| **RecToM** | Belief (Rec) | 1,762 | 7 | Single | Dialogue History | 725 |
| | Fine Intention (Rec) | 2,205 | 10 | Multi | Dialogue History | 1,462 |
| | Desire (Seek) | 1,448 | 2 | Single | Dialogue History | 723 |
| | Fine Intention (Seek) | 2,205 | 16 | Multi | Dialogue History | 785 |

> `Options` denotes the choice space. For SocialAct, `4 × 3` represents the structured prediction of the **Action Triplet** (type, object, content). Character count reflects the average length of the input context.

---

## 🔧 Evaluation
### 🐦 **SocialAct (Behavioral Prediction)**
 
```bash
# Generate model predictions
python generate_socialact_actions.py \
    --model qwen3-max-2026-01-23 \
    --topic Covid \
    --method MindStep \
    --num_samples 1000
# Evaluate results (object/type/content accuracy & F1)
python evaluate_socialact_actions.py
```

### 💬 RecToM (Mental‑State Reasoning)
```bash
#evaluate
python evaluate_tom_belief_desire.py \
    --dataset_type belief_rec.json \
    --model gpt-5-mini \
    --cot true \
    --num_samples 0
```
