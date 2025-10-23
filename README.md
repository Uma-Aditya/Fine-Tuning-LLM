# Fine-Tuning Phi-3-mini Model with Unsloth and LoRA

This repository provides a step-by-step guide to fine-tune the **Phi-3-mini-4k-instruct-bnb-4bit** model using **Unsloth**, **LoRA adapters**, and **TRL** in **Google Colab**.

## 🚀 Features

* Load a pre-trained Phi-3-mini model in 4-bit precision.
* Prepare custom datasets in JSON format.
* Apply **LoRA adapters** to efficiently fine-tune the model.
* Use **SFTTrainer** from TRL for supervised fine-tuning.
* Test and generate responses using the fine-tuned model.
* Save the model in **GGUF** format for fast inference and deployment.

## 📦 Installation

```bash
!pip install unsloth trl peft accelerate bitsandbytes datasets
```

## 🔧 GPU Check

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 📂 Dataset Preparation

* Load your dataset from JSON:

```python
import json
file = json.load(open("json_extraction_dataset_500.json", "r"))
```

* Format the prompts for fine-tuning:

```python
def format_prompt(example):
    return f"### Input: {example['input']}\n### Output: {json.dumps(example['output'])}<|endoftext|>"

formatted_data = [format_prompt(item) for item in file]
from datasets import Dataset
dataset = Dataset.from_dict({"text": formatted_data})
```

## ⚙️ Model Loading and LoRA Integration

```python
from unsloth import FastLanguageModel
model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(model, r=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_alpha=128, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
```

## 🏋️ Fine-Tuning

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
    max_seq_length=2048, dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        report_to="none",
    ))

trainer_stats = trainer.train()
```

## 🧪 Testing the Model

```python
FastLanguageModel.for_inference(model)
messages = [{"role": "user", "content": "Extract the product information:\niPad Air$1344audioDell"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True, temperature=0.7, do_sample=True, top_p=0.9)
response = tokenizer.batch_decode(outputs)[0]
print(response)
```

## 💾 Saving and Downloading Model

```python
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")

from google.colab import files
import os

gguf_files = [f for f in os.listdir("gguf_model") if f.endswith(".gguf")]
if gguf_files:
    files.download(os.path.join("gguf_model", gguf_files[0]))
```

## ⚡ Notes

* Optimized for **GPU training** and low-memory 4-bit fine-tuning.
* Supports **LoRA adapters** for parameter-efficient tuning.
* Saves model in **GGUF format** for fast inference.
* Compatible with **custom JSON datasets** and various instruction-generation tasks.

---

**Author:** Harsha Charan
**Date:** 03-08-2025
