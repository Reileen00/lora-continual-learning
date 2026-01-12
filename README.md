# LoRA-Based Continual Learning with Episodic Memory

This project implements a parameter-efficient continual learning system using Low-Rank Adaptation (LoRA) on top of a Transformer backbone. The goal is to enable models to learn new tasks sequentially without catastrophic forgetting, while remaining efficient enough for personalization at scale.

## Motivation
Large neural networks struggle to adapt to new tasks without overwriting previously learned knowledge. This is especially problematic for personalization systems, where models must continuously adapt to individual users over time.

LoRA solves this by freezing the backbone model and learning small, low-rank adapters that can be updated cheaply and safely. We combine this with episodic memory replay to further reduce forgetting.

This approach mirrors how modern LLM-based systems enable continual personalization without retraining the full model.

## Architecture

Frozen Transformer Backbone  
→ LoRA Adaptation Layers  
→ Task-Specific Output  
+ Episodic Memory Replay  

Only the LoRA parameters are trained. The base Transformer remains fixed, which stabilizes learning and prevents catastrophic forgetting.

## Methods

We implement:
- A Transformer encoder for sequence modeling
- LoRA low-rank adaptation layers for efficient continual updates
- Episodic memory replay for rehearsal-based continual learning
- Sequential task training

Each task is trained one after another. During training on a new task, samples from previous tasks are replayed from memory.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
