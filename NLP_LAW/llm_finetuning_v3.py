"""
LLM Fine-tuning for Turkish Legal QA - Qwen2.5-7B with QLoRA
CENG493 Legal RAG Project

Using HuggingFace Transformers + PEFT (instead of Unsloth for Windows compatibility)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

print("="*70)
print("🚀 CENG493 Turkish Legal QA - LLM Fine-tuning v3")
print("   Model: Qwen2.5-7B (HF Transformers + QLoRA)")
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
    # Model - Hugging Face model ID
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # LoRA Configuration
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training Configuration
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,  # One epoch with full data
    "warmup_steps": 50,
    "eval_steps": 250,
    "save_steps": 250,
    "logging_steps": 50,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    
    # Paths
    "data_path": "data/raw/llm.jsonl",
    "output_dir": "models/qwen_legal_lora",
    "logs_dir": "logs/qwen_finetuning",
    
    # Device
    "device": device,
}

print(f"\n⚙️  Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Epochs: {CONFIG['num_train_epochs']}")

# ==================== SETUP DIRECTORIES ====================

def setup_directories():
    """Create necessary directories"""
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Dizinler hazır")

setup_directories()

# ==================== DATA LOADING ====================

def load_and_prepare_dataset():
    """Load llm.jsonl (messages format) and prepare for training"""
    print(f"\n📂 Veri yükleniyor: {CONFIG['data_path']}")
    
    from datasets import Dataset
    
    # Load JSONL directly
    data_list = []
    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    dataset = Dataset.from_dict({
        "messages": [d["messages"] for d in data_list]
    })
    
    print(f"   ✅ Toplam örnek: {len(dataset)}")
    
    # Show sample
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
    """Format messages to text using tokenizer's chat template"""
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

# ==================== MODEL & TOKENIZER LOADING ====================

def load_model_and_tokenizer():
    """Load Qwen model with 4-bit quantization"""
    print(f"\n🤖 Model yükleniyor: {CONFIG['model_name']}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load model
    device_map = "auto" if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map=device_map,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Add pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   ✅ Model yüklendi")
    
    return model, tokenizer

# ==================== LORA SETUP ====================

def setup_lora(model):
    """Configure LoRA parameters"""
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
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ LoRA hazır")
    print(f"      r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")
    print(f"      Trainable: {trainable_params/1e6:.1f}M / {total_params/1e9:.1f}B ({100*trainable_params/total_params:.2f}%)")
    
    return model

# ==================== TRAINING ====================

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Fine-tune model with SFTTrainer"""
    print(f"\n🎓 Eğitim başlıyor...")
    
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # Tokenize datasets
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
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        fp16=True if device == "cuda" else False,
        optim=CONFIG["optim"],
        lr_scheduler_type="linear",
        
        # Logging & Evaluation
        logging_dir=CONFIG["logs_dir"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        
        # Performance
        gradient_checkpointing=True,
        
        # Seed
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
    
    # Train
    print(f"\n▶️  Eğitim başladı...")
    print(f"   Device: {CONFIG['device']}")
    train_result = trainer.train()
    
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    
    return model, trainer

# ==================== SAVING ====================

def save_model(model, tokenizer, trainer):
    """Save LoRA adapters and training artifacts"""
    print(f"\n💾 Model kaydediliyor: {CONFIG['output_dir']}")
    
    # Save LoRA weights
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Save training config
    config_path = Path(CONFIG["output_dir"]) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    
    # Save training logs
    logs_path = Path(CONFIG["output_dir"]) / "training_logs.json"
    logs = {
        "timestamp": datetime.now().isoformat(),
        "model": CONFIG["model_name"],
        "num_train_epochs": CONFIG["num_train_epochs"],
        "learning_rate": CONFIG["learning_rate"],
        "batch_size_effective": CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"],
        "device": CONFIG["device"],
        "final_loss": float(trainer.state.best_metric) if trainer.state.best_metric else None,
    }
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"   ✅ LoRA adapters: {CONFIG['output_dir']}")
    print(f"   ✅ Config: {config_path}")
    print(f"   ✅ Logs: {logs_path}")

# ==================== INFERENCE TEST ====================

def test_inference(model_path, tokenizer_path, test_messages: List[List[Dict]]):
    """Test fine-tuned model on sample queries"""
    print(f"\n🧪 Inference test başlıyor ({len(test_messages[:3])} örnek)...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import AutoPeftModelForCausalLM
        
        # Load model with LoRA
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Test
        print(f"\n   📝 Test Örnekleri:")
        results = []
        
        for i, messages in enumerate(test_messages[:3], 1):
            print(f"\n   --- Test {i} ---")
            
            # Get user message
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            print(f"   Soru: {user_msg[:80]}...")
            
            # Generate
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant response
            if "<|im_start|>assistant" in answer:
                answer = answer.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            
            print(f"   Cevap: {answer[:100]}...")
            
            results.append({
                "question": user_msg,
                "answer": answer,
            })
        
        # Save results
        results_path = Path(CONFIG["output_dir"]) / "inference_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n   ✅ Test sonuçları kaydedildi: {results_path}")
        
    except Exception as e:
        print(f"   ❌ Inference hatası: {e}")
        import traceback
        traceback.print_exc()

# ==================== MAIN ====================

def main():
    """Main pipeline"""
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
        
        # Test
        test_messages = [d["messages"] for d in train_dataset.select(range(min(5, len(train_dataset)))).to_list()]
        test_inference(CONFIG["output_dir"], CONFIG["output_dir"], test_messages)
        
        print("\n" + "="*70)
        print("✅ TAMAMLANDI!")
        print(f"   LoRA Adapters: {CONFIG['output_dir']}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
