from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# ✅ Mutlak path - HuggingFace'e gitmeyi engeller
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(BASE_DIR, "models", "qwen_legal_lora")

# Kontrol
print(f"📁 LoRA Path: {LORA_PATH}")
assert os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")), \
    f"❌ adapter_config.json bulunamadı: {LORA_PATH}"
print("✅ adapter_config.json bulundu!\n")

# --- 1. Ana Model ---
print("🚀 1. Qwen2.5-7B Ana Modeli Yükleniyor (4-bit NF4)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# --- 2. LoRA Adaptörü ---
print("🧠 2. Hukuk LoRA Adaptörü Entegre Ediliyor...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# --- 3. Tokenizer ---
print("🔤 3. Tokenizer Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)

# --- 4. Chat Template ---
print("⚙️ 4. Chat Template Bağlanıyor...")
try:
    template_path = os.path.join(LORA_PATH, "chat_template.jinja")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    tokenizer.chat_template = template
    print("✅ chat_template.jinja yüklendi!")
except Exception as e:
    print(f"⚠️ chat_template.jinja yüklenemedi, varsayılan kullanılacak. Hata: {e}")

print("\n✅ Sistem Hazır! Test Başlıyor...")
print("=" * 50)

# --- 5. Test ---
prompt = "Türkiye'de avukat olmanın şartları nelerdir?"

messages = [
    {
        "role": "system",
        "content": "Sen uzman bir Türk Hukuku asistanısın. Soruları hukuki bir dille, net ve doğru şekilde cevapla."
    },
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f"Soru: {prompt}")
print("🤖 Hukuk Asistanı Düşünüyor...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )

input_length = inputs["input_ids"].shape[1]
cevap = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

print(cevap)
print("=" * 50)