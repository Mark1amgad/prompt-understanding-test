---
title: Prompt Understanding Test
emoji: 🧪
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🧪 PromptEval: LLM Understanding Benchmark

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange)](https://opensource.org/licenses/Apache-2.0)

**PromptEval** is a diagnostic tool designed to evaluate how effectively Large Language Models (LLMs) interpret and follow specific instructions. By utilizing semantic similarity analysis and structural validation, it provides a quantitative score for model responses, helping developers assess model performance on the fly.

## 🚀 Key Features

* **Multi-Model Interface:** Seamlessly toggle between different open-source models (e.g., Zephyr, Gemma, Mistral).
* **Semantic Scoring:** Uses `Sentence-Transformers` to calculate Cosine Similarity between the prompt intent and the generated response.
* **Format Validation:** Automatically detects if the model adhered to specific format requests (e.g., "Write code," "Make a list," "Translate").
* **Weighted Metrics:** Calculates a final weighted score based on similarity, formatting, and content richness.

---

## ⚙️ Setup & Configuration

### 1. Hugging Face Space Setup

If deploying to Hugging Face Spaces:

1.  Go to **Settings** → **Secrets**.
2.  Add a new secret key named `HF_TOKEN`.
3.  Paste your Hugging Face Access Token as the value (read/write permissions recommended).
4.  *(Optional)* Add a variable `MODEL_REPO` to override the default model.

### 2. Local Development

To run this application locally:

```bash
# Clone the repository
git clone [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

```
> **Note:** You will need to set your `HF_TOKEN` as an environment variable locally.

---

## 📊 Scoring Methodology

The "Understanding Score" is calculated using a weighted algorithm to balance accuracy with adherence to constraints:

| Metric | Weight | Description |
| :--- | :--- | :--- |
| **Semantic Similarity** | 60% | Measures intent alignment using vector embeddings. |
| **Format Compliance** | 20% | Checks for structural cues (JSON, Lists, Code blocks). |
| **Response Richness** | 20% | Evaluates if the detail level matches the query. |

### Grading Scale

* ✅ **Good (≥ 70):** High adherence and accurate content.
* ⚠️ **Average (50 – 69.9):** Correct intent but misses formatting/depth.
* ❌ **Poor (< 50):** Failed instructions or irrelevant content.

---

## 📝 Usage Guide

1. **Enter Prompt:** Type instructions (e.g., *"Write a Python function for Fibonacci"*).
2. **Select Model:** Choose an LLM from the dropdown.
3. **Adjust Parameters:** Set `max_new_tokens` and `temperature`.
4. **Analyze:** Click **"Evaluate"** to see the score and response.

---

## 📦 Recommended Models

For optimal benchmark results, use instruction-tuned models:
* `HuggingFaceH4/zephyr-7b-beta`
* `google/gemma-2-2b-it`
* `mistralai/Mistral-7B-Instruct-v0.2`
* `mistralai/Mistral-7B-Instruct-v0.2`
