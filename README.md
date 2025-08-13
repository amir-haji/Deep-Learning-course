# Deep Learning — Homework Implementations

This repository contains my implementations for the **Deep Learning** course.  
Each homework explores a different architecture or generative/transformer-based approach — from CNNs and RNNs to LLMs and diffusion models.

---

## 📂 Homeworks

### [HW1 — Basic Image Classification (ResNet18 from Scratch)](./HW1)
- Implemented **ResNet18** entirely from scratch.
- Trained and evaluated on the **CIFAR-10** dataset.
- Covered:
  - Residual connections
  - Data augmentation
  - Training with SGD + momentum

---

### [HW2 — Image Colorization (U-Net from Scratch)](./HW2)
- Built a **U-Net** architecture from scratch.
- Trained for **image colorization** from grayscale images.
- Used:
  - Skip connections for better spatial reconstruction
  - L2 reconstruction loss and perceptual loss

---

### [HW3 — RNN Series Prediction](./HW3)
- Trained an **RNN** for **oil price prediction** using time-series data.
- Explored:
  - Simple RNN and LSTM architectures
  - Sliding window sequence input
  - Forecasting evaluation metrics

---

### [HW4 — GPT-2 from Scratch](./HW4)
- Implemented a **GPT-2 style** transformer-based language model from scratch.
- Trained it to generate **Persian food service reviews**.
- Key topics:
  - Multi-head self-attention
  - Layer normalization
  - Byte Pair Encoding (BPE) tokenization

---

### [HW5 — Parameter-Efficient Fine-Tuning (PEFT)](./HW5)
- Experimented with **PEFT methods** in HuggingFace Transformers.
- Compared:
  - LoRA
  - Prefix Tuning
  - Adapter Tuning
- Benchmarked performance & efficiency trade-offs.

---

### [HW6 — Reasoning for LLMs](./HW6)
- Tested reasoning-enhanced LLM methods:
  - Chain-of-Thought (CoT)
  - Best-of-N sampling
  - Beam search
- Benchmarked results on the **Math-500** dataset.

---

### [HW7 — Diffusion Models (DDPM)](./HW7)
- Created a **DDPM (Denoising Diffusion Probabilistic Model)** from scratch.
- Implemented **classifier-free conditional generation** for MNIST digits.
- Generated class-specific digit images from pure noise.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x  
- **Libraries:** PyTorch, NumPy, Matplotlib, HuggingFace Transformers, torchvision  

---

## ⚠️ Disclaimer
All implementations are for **educational purposes only** and should be used responsibly.

