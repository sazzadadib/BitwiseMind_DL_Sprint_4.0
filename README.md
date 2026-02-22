# 🗣️ Bengali Long-Form Speech Recognition & Speaker Diarization

This repository contains our complete solution for:

* **Task 1:** Long-Form Automatic Speech Recognition (ASR)
* **Task 2:** Speaker Diarization

The system was developed and fine-tuned exclusively using the official competition dataset and optimized for long-form Bengali speech.

---

# 📌 Task 1 – Long-Form ASR Transcription

## 🔎 Overview

Our final ASR system is built using the Hugging Face Transformers implementation of **OpenAI Whisper**.

We initialized from the pretrained checkpoint:

* **Base Model:** Hugging Face Transformers implementation of OpenAI Whisper
* **Checkpoint:** Hugging Face model `tugstugi/bengaliai-regional-asr_whisper-medium`

The model was further fine-tuned exclusively on the competition dataset to adapt to:

* Long-form Bengali speech
* Domain-specific vocabulary
* Natural conversational patterns

---

## 🧠 Model Parameter Size

* **Base Model:** `tugstugi/bengaliai-regional-asr_whisper-medium`
* **Total Parameters:** ~769 Million

---

## ⏱️ Inference Time

Inference was performed on a **single T4 GPU (Kaggle environment)**.

| Configuration               | Time                    |
| --------------------------- | ----------------------- |
| Total (Full Test Set)       | ~3 hours 38 minutes     |
| Average per audio (default) | 8–9 minutes             |
| Optimized decoding          | 2.5–3 minutes per audio |

---

## 📦 Model Artifact

The fine-tuned ASR model is publicly available on Hugging Face:

👉 [https://huggingface.co/bitwisemind/sam_15000_clean_text_full_model](https://huggingface.co/bitwisemind/sam_15000_clean_text_full_model)

---

## 📊 Dataset Used

* Only the **official competition dataset**
* 90/10 train-validation split
* Segments: 1–15229

---

## ⚙️ Training Configuration

| Setting                | Value                      |
| ---------------------- | -------------------------- |
| Optimizer              | AdamW 8-bit                |
| Learning Rate          | 5e-6                       |
| Scheduler              | Cosine                     |
| Epochs                 | 6                          |
| Batch Size             | 8 × 2 = 16                 |
| Precision              | FP16                       |
| Gradient Checkpointing | Enabled                    |
| Evaluation/Save        | Every 6000 steps (limit 2) |
| Max Audio Length       | 30 seconds                 |
| Max Text Length        | 1000                       |
| Augmentation           | 0.3                        |
| Generation Max Length  | 225                        |

---

## 📒 Training / Fine-Tuning / Inference Notebooks

GitHub Repository:

👉 [https://github.com/sazzadadib/BitwiseMind_DL_Sprint_4.0/tree/main/Bengali%20Long-form%20Speech%20Recognition](https://github.com/sazzadadib/BitwiseMind_DL_Sprint_4.0/tree/main/Bengali%20Long-form%20Speech%20Recognition)

---

# 🎙️ Task 2 – Speaker Diarization

## 🔎 Overview

Our speaker diarization system is built using the **pyannote.ai diarization pipeline**, powered by the `pyannote.audio` framework.

We used the pretrained segmentation model:

* **Segmentation Model:** `pyannote/segmentation-3.0`

The pipeline performs:

1. Speaker change detection
2. Time-stamped segment generation
3. Clustering into distinct speaker identities

---

## 🧠 Model Parameter Size

* **Model:** `pyannote/segmentation-3.0`
* **Total Parameters:** ~1.5 Million

---

## ⏱️ Inference Time

Inference was performed on a **single T4 GPU (Kaggle environment)**.

| Configuration          | Time               |
| ---------------------- | ------------------ |
| Total (Full Test Set)  | ~1 hour 20 minutes |
| Average per audio file | 5–6 minutes        |

---

## 📦 Model Artifact

Fine-tuned segmentation checkpoint:

👉 [https://huggingface.co/datasets/Amanafif554/diarizaaaaaation_CKPT](https://huggingface.co/datasets/Amanafif554/diarizaaaaaation_CKPT)

Checkpoint used during inference:

```
segmentation-epoch=39.ckpt
```

---

## 📊 Dataset Used

* Only the **official competition dataset**
* Custom train/dev/test splits
* Protocol: `CustomData.SpeakerDiarization.train`

---

## ⚙️ Training Configuration

| Setting                | Value                                      |
| ---------------------- | ------------------------------------------ |
| Base Model             | `pyannote/segmentation-3.0`                |
| Pipeline               | `pyannote/speaker-diarization-community-1` |
| Chunk Size             | 10 seconds                                 |
| Max Speakers per Chunk | 3                                          |
| Training Epochs        | 50                                         |
| GPU                    | 1                                          |
| Framework              | PyTorch Lightning                          |
| Monitoring             | loss/train (min)                           |
| Checkpoints            | Best + Last                                |
| Evaluation Metric      | DER                                        |

---

## 📒 Training / Fine-Tuning / Inference Notebooks

GitHub Repository:

👉 [https://github.com/sazzadadib/BitwiseMind_DL_Sprint_4.0/tree/main/Bengali%20Speaker%20Diarization](https://github.com/sazzadadib/BitwiseMind_DL_Sprint_4.0/tree/main/Bengali%20Speaker%20Diarization)

---

# 🚀 System Summary

| Task        | Model                       | Params | GPU | Total Inference |
| ----------- | --------------------------- | ------ | --- | --------------- |
| ASR         | Whisper-medium (fine-tuned) | 769M   | T4  | ~3h 38m         |
| Diarization | pyannote segmentation-3.0   | 1.5M   | T4  | ~1h 20m         |

---

# ✅ Key Highlights

* Fully fine-tuned on official dataset only
* Optimized long-form decoding
* Efficient GPU utilization
* Public model artifacts available
* Reproducible training pipelines

---

# 📌 Citation

If you use this repository or models, please cite accordingly.

---

# 👥 Team

**BitwiseMind**

