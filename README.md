# llm-structured-data-extraction

This project demonstrates how to fine-tune [Phi-3 Mini](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) using [Unsloth](https://github.com/unslothai/unsloth) for efficient training and [TRL](https://github.com/huggingface/trl) for supervised fine-tuning (SFT). The workflow includes dataset preparation, LoRA-based fine-tuning, inference, and exporting the model to GGUF format.

---

## 🚀 Features
- Loads and tokenizes a JSON dataset of prompts and responses.
- Uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
- Trains with **SFTTrainer** from Hugging Face TRL.
- Generates responses post-fine-tuning.
- Exports the trained model to **GGUF** format for optimized inference.

---

## 📦 Requirements
Install the required dependencies:

```bash
pip install unsloth trl
pip install protobuf==3.20.3
```

---

## 📂 Dataset Format
The dataset should be a JSON file (`people_data.json`) structured as a list of dictionaries:

```json
[
  {
    "prompt": "While strolling through a botanical garden, Igor, now 20 earns a living as a tour guide. He is known among friends for conducting amateur astronomy observations in quiet solitude.",
    "response": {
      "name": "Igor",
      "age": "20",
      "job": "tour guide",
      "gender": "male"
    }
  },
  {
    "prompt": "On the edge of the desert, Nadine, currently 70 years old makes ends meet working as a psychologist. A surprising fact: she has been craftinging ceramic vases on a wheel since last year.",
    "response": {
      "name": "Nadine",
      "age": "70",
      "job": "psychologist",
      "gender": "female"
    }
  }
```
---

## ⚙️ Training Steps

### 1. Load the Model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```

### 2. Prepare Dataset
```python
import json
from datasets import Dataset

with open("people_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)
```

Each example is converted to a chat-style format using `tokenizer.apply_chat_template`.

### 3. Apply LoRA
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth"
)
```

### 4. Train with TRL
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        num_train_epochs=3
    )
)

trainer.train()
```

---

## 🧪 Inference
Switch the model to inference mode:

```python
FastLanguageModel.for_inference(model)
```

Example prompt:
```python
messages = [{"role": "user", "content": "Mike is 30 years old, loves hiking, and works as a coder."}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.batch_decode(outputs)[0]
print(response)
```

---

## 💾 Export Model to GGUF
Save the fine-tuned model in GGUF format:

```python
model.save_pretrained_gguf(
    "gguf_finetuned_model",
    tokenizer,
    quantization_method="q4_k_m",
    maximum_memory_usage=0.3
)
```

---

## 📈 Quick Start
A minimal cheat sheet for training and inference:

```python
# Load model
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# Load dataset
import json
from datasets import Dataset
with open("people_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
ds = Dataset.from_list(data)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model, r=64, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=128, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth"
)

# Train
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, train_dataset=ds, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=2048,
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=10, max_steps=60,
                   logging_steps=1, output_dir="outputs", optim="adamw_8bit", num_train_epochs=3)
)
trainer.train()

# Inference
FastLanguageModel.for_inference(model)
messages = [{"role": "user", "content": "Mike is 30 years old, loves hiking, and works as a coder."}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7, do_sample=True, top_p=0.9)
print(tokenizer.batch_decode(outputs)[0])

# Export to GGUF
model.save_pretrained_gguf("gguf_finetuned_model", tokenizer, quantization_method="q4_k_m", maximum_memory_usage=0.3)
```

---

## 📊 Key Hyperparameters
- **LoRA rank (`r`)**: 64
- **LoRA alpha**: 128
- **Batch size per device**: 2
- **Gradient accumulation**: 4 (effective batch size = 8)
- **Training steps**: 60
- **Learning rate warmup steps**: 10

---

## ✅ Notes
- **Unsloth** enables 4-bit quantized training to reduce GPU memory usage.
- The dataset is mapped to a chat format for compatibility with instruction-tuned models.
- **SFTTrainer** handles supervised fine-tuning with logging and checkpointing.
- GGUF export supports efficient inference with tools like **llama.cpp** or **Ollama**.

---

## 📌 References
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [TRL (Hugging Face)](https://github.com/huggingface/trl)
- [Phi-3 Mini Model Card](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)
