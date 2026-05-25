"""
LLM Fine-tuning for Turkish Legal QA - Qwen2.5-7B with QLoRA
Colab GPU Version - Optimized
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

print("="*70)
print("🚀 CENG493 Turkish Legal QA - LLM Fine-tuning (COLAB GPU)")
print("   Model: Qwen2.5-7B (QLoRA)")
print("   Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("="*70)

# ==================== DEVICE CHECK ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== CONFIG ====================
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # LoRA Configuration
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training Configuration (GPU optimized - Colab)
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "warmup_steps": 50,
    "eval_steps": 250,
    "save_steps": 250,
    "logging_steps": 50,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    
    # Paths
    "data_path": "llm.jsonl",
    "output_dir": "models/qwen_legal_lora",
    "logs_dir": "logs/qwen_finetuning",
    "device": device,
}

print(f"\n⚙️  Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Epochs: {CONFIG['num_train_epochs']}")

# ==================== SETUP ====================
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)

# ==================== DATA LOADING ====================
def load_and_prepare_dataset():
    """Load llm.jsonl and prepare for training"""
    print(f"\n📂 Veri yükleniyor: {CONFIG['data_path']}")
    
    from datasets import Dataset
    
    data_list = []
    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    dataset = Dataset.from_dict({
        "messages": [d["messages"] for d in data_list]
    })
    
    print(f"   ✅ Toplam örnek: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\n   📝 Örnek veri (messages format):")
    for msg in sample["messages"][:2]:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))[:80]
        print(f"      {role}: {content}...")
    
    # Split train/val (80/20)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"\n   ✅ Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

# ==================== TOKENIZATION ====================
def format_chat_template(messages, tokenizer):
    """Format messages using chat template"""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def tokenize_function(examples, tokenizer):
    """Tokenize examples"""
    texts = [format_chat_template(msgs, tokenizer) for msgs in examples["messages"]]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=CONFIG["max_seq_length"],
        padding="max_length",
        return_tensors=None,
    )
    
    return tokenized

# ==================== MODEL LOADING ====================
def load_model_and_tokenizer():
    """Load Qwen with 4-bit quantization"""
    print(f"\n🤖 Model yükleniyor: {CONFIG['model_name']}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load model with device_map="auto" (works on Colab)
    device_map = "auto" if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map=device_map,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   ✅ Model yüklendi")
    
    return model, tokenizer

# ==================== LORA SETUP ====================
def setup_lora(model):
    """Configure LoRA"""
    print(f"\n⚙️  LoRA konfigürasyonu")
    
    from peft import get_peft_model, LoraConfig, TaskType
    
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ LoRA hazır")
    print(f"      r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")
    print(f"      Trainable: {trainable_params/1e6:.1f}M / {total_params/1e9:.1f}B ({100*trainable_params/total_params:.2f}%)")
    
    return model

# ==================== TRAINING ====================
def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Fine-tune with Trainer"""
    print(f"\n🎓 Eğitim başlıyor...")
    
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # Tokenize
    print(f"   📝 Tokenizing datasets...")
    train_dataset_tokenized = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=["messages"],
        desc="Tokenizing train"
    )
    
    eval_dataset_tokenized = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=["messages"],
        desc="Tokenizing eval"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        fp16=True,
        optim=CONFIG["optim"],
        lr_scheduler_type="linear",
        
        logging_dir=CONFIG["logs_dir"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        
        gradient_checkpointing=True,
        seed=42,
    )
    
    print(f"   ✅ Training args hazır")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_dataset_tokenized,
        data_collator=data_collator,
    )
    
    print(f"\n▶️  Eğitim başladı...")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Estimated time: ~90 minutes on GPU")
    train_result = trainer.train()
    
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    
    return model, trainer

# ==================== SAVING ====================
def save_model(model, tokenizer, trainer):
    """Save adapters"""
    print(f"\n💾 Model kaydediliyor: {CONFIG['output_dir']}")
    
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Save config
    config_path = Path(CONFIG["output_dir"]) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    
    print(f"   ✅ LoRA adapters: {CONFIG['output_dir']}")
    print(f"   ✅ Config: {config_path}")

# ==================== MAIN ====================
print("\n" + "="*70)
print("Pipeline başlamak üzere...")
print("="*70)

try:
    # Data
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Model
    model, tokenizer = load_model_and_tokenizer()
    
    # LoRA
    model = setup_lora(model)
    
    # Train
    model, trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    # Save
    save_model(model, tokenizer, trainer)
    
    print("\n" + "="*70)
    print("✅ BAŞARILI! Training tamamlandı!")
    print("="*70)
    print("\n📥 Sonra şu kodu çalıştır:")
    print("""
from google.colab import files
import os
os.system('cd /content/colab_training && zip -r qwen_lora_models.zip models/')
files.download('qwen_lora_models.zip')
    """)
    
except Exception as e:
    print(f"\n❌ HATA: {e}")
    import traceback
    traceback.print_exc()
