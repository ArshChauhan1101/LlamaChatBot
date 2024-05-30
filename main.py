import transformers
from peft import LoraConfig, get_peft_model  
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

login() # Need access to the gated model.

# Load LLAMA 2 model
model_name = "meta-llama/Llama-2-7b-chat-hf" 

# Quantization configuration 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Load LoRA configuration
lora_config = LoraConfig.from_pretrained('harpyerr/archimedes-300s-7b-chat')
model = get_peft_model(model, lora_config) 

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define prompt  
text = "Can you tell me who made Space-X?"
prompt = "You are a helpful assistant. Please provide an informative response. \n\n" + text

# Generate response
device = "cuda:0" 
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
