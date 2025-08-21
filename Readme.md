
<div align="center">
  <h1><img src="assets/logo.jpeg" alt="TokenCast logo" style="height: 1em; width: auto; vertical-align: -0.15em; margin-right: 0.4em;">TokenCast: An LLM-Driven Framework for Context-Aware Time Series Forecasting via Symbolic Discretization</h1> 
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/TokenCast/stargazers">
    <img src="https://img.shields.io/github/stars/Xiaoyu-Tao/TokenCast?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/TokenCast/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
  </a>

</div>

---

TokenCast is a novel framework that leverages **Large Language Models (LLMs)** for **context-aware time series forecasting**, by transforming continuous time series into discrete symbolic tokens. It enables a unified generative modeling over both temporal and textual modalities.

> 📝 “From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization”  
> **Under review** | [📄 Paper](https://arxiv.org/abs/2508.09191)

---

## 🔍 Overview

Traditional forecasting models struggle to effectively integrate heterogeneous contextual data like clinical notes, policy documents, or logs. TokenCast introduces a new paradigm:

- Converts time series into **discrete temporal tokens** via dynamic vector quantization.
- Embeds both temporal and textual tokens into a **shared semantic space** using a frozen pre-trained LLM.
- Performs **prompt-based generative forecasting** using autoregressive language modeling.

<p align="center">
  <img src="assets/main.png" width="700">
</p>

---

## ✨ Key Features

- ✅ **Discretized Temporal Modeling**: Learnable, reversible tokenizer for symbolic time series.
- 🔗 **Cross-Modality Alignment**: Unified vocabulary space for both time and text tokens.
- 📈 **Prompt-driven Generation**: Forecasting with LLM via token-level instruction generation.
- 📊 **Multi-domain Evaluation**: Benchmarked across economic, health, web, stock, and environmental domains.
- 🌡️ **Uncertainty Quantification**: Predictive intervals with temperature-controlled generation.

<!-- ---

## 📁 Project Structure
```

TokenCast/
 ├── tokenizer/               # Time series VQ-VAE tokenizer
 ├── models/                  # LLM backbone and embedding alignment
 ├── prompts/                 # Prompt templates for generation
 ├── datasets/                # Preprocessed benchmark datasets
 ├── evaluation/              # Evaluation scripts and metrics
 ├── scripts/                 # Training and inference scripts
 ├── configs/                 # YAML config files
 └── README.md

```
--- -->
---
## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Xiaoyu-Tao/TokenCast.git
cd TokenCast
```

### 2. Environment Setup

```bash
conda create -n tokencast python=3.10
conda activate tokencast
pip install -r requirements.txt
```

### 3. Prepare Data
TokenCast supports multiple publicly available datasets:
- **Economic (FRED-MD)**
- **Health (Covid-19 mobility)**
- **Web (Wikipedia pageviews)**
- **Stock-NY & Stock-NA (NYSE/NASDAQ)**
- **Nature (Environmental sensor data)**

First, the training and evaluation datasets used in our experiments can be found in [Google Drive](https://drive.google.com/drive/u/0/home). Then, create a directory named `datasets` and then download the necessary datasets into it.

```bash
mkdir datasets
```

### 4. Train the Time Series Tokenizer

```bash
sh Tokenizer/scripts/Czelan.sh 
```

### 5. Align Embeddings with LLM

```bash
sh scripts/pretrain/Czelan.sh  
```

### 6. Fine-tune Forecasting Model

```bash
sh scripts/finetune/Czelan.sh 
```

------

## 📊 Benchmark Results
**Full Results:**
![table1](assets/1-main-results.png)

**Ablation Results:**
![table2](assets/2-ablation-results.png)

------

## 📚 Citation

If you find this project useful, please consider citing our paper:

```bibtex
@inproceedings{tao2026tokencast,
  title={From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization},
  author={Tao, Xiaoyu and Zhang, Shilong and Cheng, Mingyue and Wang, Daoyu and Pan, Tingyue and Pan, Bokai and Zhang, Changqing and Wang, Shijin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

------

## 🤝 Acknowledgements

This project is developed by researchers from:

- 🧠 University of Science and Technology of China (USTC)
- 🧮 Tianjin University
- 🗣️ iFLYTEK Research

------

## 📬 Contact

For questions or collaborations, please contact:

- 🧑‍🏫 Mingyue Cheng ([mycheng@ustc.edu.cn](mailto:mycheng@ustc.edu.cn))
- 🤖 Xiaoyu Tao ([txytiny@mail.ustc.edu.cn](mailto:txytiny@mail.ustc.edu.cn))

------

## 📌 License

This project is released under the MIT License.


