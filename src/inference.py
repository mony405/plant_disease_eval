import torch
import json
import os
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel # Added to handle the LoRA adapter layers
from dotenv import load_dotenv
from huggingface_hub import login

# Load environmental tokens from .env
load_dotenv()

# Constants
ADAPTER_MODEL_ID = "ramyibrahim/gemma-plant-disease-it-alpha"
BASE_MODEL_ID = "google/gemma-3n-e2b-it" # Matches your 2B active parameter vision notes
TEST_JSON = "data/test_split.json"
OUTPUT_PATH = "outputs/predictions/results.json"

PROMPT_INSTRUCTION = """
Analyze the following image of a crop.
Identify the type of crop and its health status.
Output your findings strictly as a JSON object with 'crop' and 'status' keys.
""".strip()

def load_model_and_processor(adapter_id, base_id):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found! Ensure it is defined in your .env file.")
        
    login(token=hf_token)

    # 4-bit configuration to stay within your RTX 5060 8GB VRAM footprint
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"--- Step 1: Initializing Base Multimodal Model ({base_id}) ---")
    processor = AutoProcessor.from_pretrained(adapter_id, token=hf_token)
    
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    
    print(f"--- Step 2: Merging Your Fine-Tuned Adapter Weights ({adapter_id}) ---")
    # Wrap the quantized base model structural backbone with your adapter weights
    model = PeftModel.from_pretrained(base_model, adapter_id, token=hf_token)
    
    return processor, model

def run_evaluation():
    # Pass both IDs to the updated initialization function
    processor, model = load_model_and_processor(ADAPTER_MODEL_ID, BASE_MODEL_ID)
    
    if not os.path.exists(TEST_JSON):
        print(f"Error: {TEST_JSON} not found. Run prepare_data.py first.")
        return

    with open(TEST_JSON, "r") as f:
        test_metadata = json.load(f)
    
    results = []
    print(f"--- Running Inference on {len(test_metadata)} cached samples ---")
    
    for item in tqdm(test_metadata):
        local_path = item['local_path']
        if not os.path.exists(local_path):
            continue
            
        image = Image.open(local_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_INSTRUCTION},
                    {"type": "image"}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False,
                use_cache=True
            )
        
        generated_text = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()

        results.append({
            "index": item['original_index'],
            "ground_truth": {
                "crop": item.get("crop"),
                "status": item.get("status"),
                "full_label": item.get("label")
            },
            "prediction_raw": generated_text
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"--- Inference Complete. Raw predictions saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    run_evaluation()