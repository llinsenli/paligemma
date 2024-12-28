# PaliGemma from Scratch in PyTorch

**PaliGemma** is a vision-language model built entirely from scratch in PyTorch.  
It follows a design inspired by PaLI and PaLI-3, integrating a Vision Transformer (SigLip) and a Language Model (Gemma) to provide powerful text and image understanding capabilities.  

This repository showcases an end-to-end implementation of **PaliGemma** with detailed modules for:
- Vision transformer encoder (SigLip)
- Language decoder (Gemma)
- Advanced transformer techniques (KV-Cache, RMSNorm, Grouped Query Attention, Rotary Positional Embedding)
- A unified processor for text and image (PaliGemmaProcessor)
- Inference pipeline for generating text conditioned on both image and text input

---

## Table of Contents
1. [Overview](#overview)  
2. [Architecture](#architecture)  
   - [SigLip (Vision Transformer)](#siglip-vision-transformer)  
   - [Gemma (Language Model)](#gemma-language-model)  
   - [PaliGemma (Combined System)](#paligemma-combined-system)  
3. [Advanced Transformer Techniques](#advanced-transformer-techniques)  
   - [KV-Cache](#kv-cache)  
   - [RMS Normalization](#rms-normalization)  
   - [Grouped Query Attention](#grouped-query-attention)  
   - [Rotary Positional Embedding](#rotary-positional-embedding)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Acknowledgements](#acknowledgements)  
7. [References](#references)  

---

## Overview
**PaliGemma**[@beyer2024paligemma] excels at interpreting and understanding both images and text. Tasks include:
- Generating descriptions of images  
- Answering questions based on visual content  
- Analyzing complex visuals (infographics, satellite images, etc.)  

The model is designed to accept image and/or text input to produce text output. Despite its smaller size compared to some large-scale models, **PaliGemma** handles a wide range of vision-language tasks efficiently.

---

## Architecture

### SigLip (Vision Transformer)
The **SigLip** component processes images and converts them into “soft tokens.” It uses a Vision Transformer architecture that:
1. Splits each image into patches via a 2D convolution (patch embedding).  
2. Applies positional embeddings to these patch embeddings.  
3. Encodes the patch embeddings through multiple Transformer encoder layers (multi-head self-attention and MLP blocks).  

> **File Reference:** [SigLip.py](./SigLip.py)

### Gemma (Language Model)
The **Gemma** component is a decoder-only language model:
1. Tokenizes input text (and special tokens for images).  
2. Processes the input sequence in a stack of transformer decoder blocks.  
3. Generates output text in an auto-regressive manner.  

> **File Reference:** [Gemma.py](./Gemma.py)

### PaliGemma (Combined System)
**PaliGemma** merges the SigLip encoder output and Gemma decoder:
1. Image inputs -> SigLip -> **soft tokens**.  
2. Text inputs -> tokenized via Gemma’s tokenizer -> **text tokens**.  
3. Soft tokens (image) + text tokens -> Gemma’s decoder -> **final text output**.  

> **Key Classes & Methods**: 
> - **`PaliGemmaForConditionalGeneration`** in [Gemma.py](./Gemma.py)  
> - **`_merge_input_ids_with_image_features`** merges the image features into placeholder `<image>` tokens.  

---

## Advanced Transformer Techniques

### KV-Cache
The **KV-Cache** mechanism reduces computation in auto-regressive decoding. It stores key-value tensors from previous tokens and reuses them to avoid re-computation.  
> Implemented in the `KVCache` class in [Gemma.py](./Gemma.py).

### RMS Normalization
**RMS Norm** is a normalization approach used in place of LayerNorm. It normalizes the vector by its root mean square and learns a scaling parameter.  
> Implemented in `GemmaRMSNorm` within [Gemma.py](./Gemma.py).

### Grouped Query Attention
In **Grouped Query Attention**, multiple query heads share the same key/value heads, improving memory efficiency. This design reduces overhead in large-scale models.  
> Implemented in `GemmaAttention` with `repeat_kv` logic.

### Rotary Positional Embedding
A positional encoding strategy that rotates the query/key embeddings by a position-dependent angle, helping the model better capture long-range dependencies.  
> Implemented in `GemmaRotaryEmbedding` within [Gemma.py](./Gemma.py).

---

## Getting Started

### Prerequisites
- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [Pillow](https://pypi.org/project/Pillow/)  
- [Hugging Face Transformers & Tokenizers](https://github.com/huggingface/transformers) (for tokenizer)  
- [safetensors](https://pypi.org/project/safetensors/) (optional: if loading weights from safetensors files)  
- [fire](https://github.com/google/python-fire) (for the `main()` CLI)  

### Installation
1. **Clone this repository**:
   ```bash
   git clone https://github.com/<your-username>/paligemma-from-scratch.git
   cd paligemma-from-scratch
   ```
2. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure your `requirements.txt` includes PyTorch, Pillow, Hugging Face libraries, and any other dependencies.)

### Usage

1. **Prepare Model Weights**  
   - By default, the project expects to load weights from a folder containing:
     - `*.safetensors` (checkpoint files)  
     - `config.json` (model configuration)  
     - Tokenizer files, etc.  
   - Update the path in [`launch_inference.sh`](./launch_inference.sh) if needed.

2. **Run the Inference Script**  
   - Update `MODEL_PATH`, `PROMPT`, and `IMAGE_FILE_PATH` in [launch_inference.sh](./launch_inference.sh).  
   - Then execute:
     ```bash
     bash launch_inference.sh
     ```
   - Alternatively, you can directly run:
     ```bash
     python inference.py \
       --model_path "<model-checkpoint-folder>" \
       --prompt "Describe this scene" \
       --image_file_path "/path/to/image.jpg" \
       --max_tokens_to_generate 100 \
       --temperature 0.8 \
       --top_p 0.9 \
       --do_sample False \
       --only_cpu True
     ```
   - Output text will be printed to the console.

---

## Project Structure
Below is a high-level view of the files included in this repository:

```
paligemma-from-scratch/
├── SigLip.py                    # Vision transformer (SigLip) for image encoding
├── PaliGemma_Processor.py       # Combined text/image processing for PaliGemma
├── Gemma.py                     # Language model (Gemma) + Transformer layers
├── inference.py                 # Inference loop & top-p sampling
├── launch_inference.sh          # Shell script to launch inference
├── utils.py                     # Utility functions to load model + tokenizer
├── requirements.txt             # Dependencies (PyTorch, HF transformers, etc.)
└── README.md                    # This file (user guide & documentation)
```

---

## Acknowledgements
Thanks to all contributors who have supported the creation of **PaliGemma**. Special appreciation to:
- The authors of **PaLI** and **PaLI-3** for the inspiring architecture.  
- PyTorch & Hugging Face communities for open-source tools and resources.

---

## References
- [@beyer2024paligemma]: Beyer, L., et al. (2024). *PaLI: A Jointly-Scaled Multilingual Language-Image Model*.  


